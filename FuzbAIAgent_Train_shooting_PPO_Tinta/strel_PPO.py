import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import requests
import math
import json
import csv
from datetime import datetime
from pprint import pprint

from actor_critic.main import ActorCriticNet
from memory.main import PPOBuffer
from reward.single_bar_shoting import simple_reward


HOST_ADDRESS = '127.0.0.1:23336'  # IP or Host for your environment

def get_camera_state():
    """Gets the data from the camera"""
    cam_url = f"http://{HOST_ADDRESS}/Camera/State"
    response = requests.get(cam_url)
    return response.json()

def send_motor_commands(cmds):
    """Sends REST communication to the HW"""
    motors_url = f"http://{HOST_ADDRESS}/Motors/SendCommand?blue=False"
    requests.post(motors_url, json=cmds)

def calculate_player_positions_and_angles(camera, geometry):
    "Function that calculates the state parameters of the rods in the enviroment"
    # NOTE: Not used for single row training, activate later
    field = geometry["field"]
    rods = geometry["rods"]
    player_positions = []

    # We'll read from the camera the same way:
    cam_data = camera["camData"][0]
    if cam_data is None:
        cam_data = camera["camData"][1]

    for rod in rods:
        rod_id = rod["id"]
        team = rod["team"]
        rod_x = rod["position"]
        travel_range = rod["travel"]
        num_players = rod["players"]
        first_offset = rod["first_offset"]
        spacing = rod["spacing"]

        rod_position_calib = cam_data["rod_position_calib"][rod_id - 1]
        rod_angle = cam_data["rod_angle"][rod_id - 1]

        # If the calibration is stored as a list, just grab the first element
        if isinstance(rod_position_calib, list):
            rod_position_calib = rod_position_calib[0]

        # Convert the normalized rod_position_calib to actual table coordinates
        rod_y_base = rod_position_calib * travel_range

        for i in range(num_players):
            player_y = rod_y_base + first_offset + i * spacing
            player_positions.append({
                "rod_id": rod_id,
                "team": team,
                "position": (rod_x, player_y),
                "angle": rod_angle
            })

    return player_positions

def mlp_gaussian_likelihood(action, mean, log_std):
    """
    Compute log likelihood of a Gaussian distribution with diagonal covariance
    (log_std is a vector).
    """
    # 1. Compute σ from log_std
    std = torch.exp(log_std)  # σ = e^(log_std)

    # 2. Standardized residual: (action - mean) / σ
    z = (action - mean) / std

    # 3. Compute the log-probability for each action dimension
    #    Formula: -0.5 * z² - log(σ) - 0.5·log(2π)
    pre_sum = -0.5 * (z**2 + 2*log_std + np.log(2*np.pi))
    #pre_sum = -0.5 * (((action - mean) / (torch.exp(log_std)))**2 + 2*log_std + np.log(2*np.pi))
    return torch.sum(pre_sum, axis=1)


class PPOAgent:
    """
    PPO RL agent that:
      - Holds a policy network (actor-critic).
      - On each call to `process_data(camera)`, picks an action in [-1,1] for each rod.
      - Uses a buffer to accumulate experiences for PPO updates.
    """
    def __init__(self,
                 obs_dim=10, # ball(4) + rod(5) + ball_kicked(1)
                 act_dim=4, #act_dim=4,          # 1 rod × 4 numbers each
                 hidden_size=512,
                 steps_per_env=256,  # how many steps per iteration
                 gamma=0.99,
                 lam=0.95,
                 clip_ratio=0.2,
                 lr=1e-4,
                 train_iters=2,
                 target_kl=0.01,
                 delay_step=2,
                 save_model_every=100,   # save every N episodes
                 model_save_path="ppo_foos.pth",
                 l2_lambda=5e-4,
                 controlled_rod_id=4):      # L2 regularization strength

        self.controlled_rod_id = controlled_rod_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.save_model_every = save_model_every
        self.model_save_path = model_save_path
        self.l2_lambda = l2_lambda  # L2 regularization strength


        self.prev_vel = None
        self.prev_score = None
        self.active_regions = None  # Initialize active_regions
        self.MAX_EPISODE_STEPS = 100
        self.episode_steps = 0

        

        with open('geometry.json') as f:
            self.geometry = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Actor-Critic network
        self.ac = ActorCriticNet(obs_dim, act_dim, hidden_size)
        self.ac.to(self.device)

        # Separate or shared log_std for continuous actions
        self.log_std = nn.Parameter(-1*torch.zeros(act_dim, dtype=torch.float32, device=self.device), requires_grad=True)
        self.log_std = self.log_std.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(list(self.ac.parameters()) + [self.log_std], lr=lr)

        # PPO hyperparameters
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.target_kl = target_kl

        # Replay buffer
        self.steps_per_env = steps_per_env
        self.buf = PPOBuffer(obs_dim, act_dim, steps_per_env, gamma, lam)

        # Tracking
        self.episode_count = 0   # how many episodes finished so far
        self.current_step = 0    # how many steps in the current episode
        self.ep_reward = 0.0     # reward this episode
        self.last_obs = None     # store last observation
        self.episode_rewards = []
        self.reward = 0

        # Episode statistics
        self.episode_stats = []  # Store detailed stats for each episode
        self.current_episode_goals = 0
        self.current_episode_opponent_goals = 0
        self.current_episode_ball_kicks = 0
        self.current_episode_x_threshold_terminations = 0
        self.current_episode_step_rewards = []  # Track reward at each step

        # For training mode vs. inference mode
        self.training_enabled = True



    def save_model(self, path=None):
        if path is None:
            path = self.model_save_path
        torch.save(self.ac.state_dict(), path)
        print(f"[PPOAgent] Model saved to {path}")

    def load_model(self, path=None):
        if path is None:
            path = self.model_save_path
        if os.path.exists(path):
            self.ac.load_state_dict(torch.load(path))
            print(f"[PPOAgent] Model loaded from {path}")
        else:
            print("[PPOAgent] No saved model found, skipping load.")

    def normalize_observation(self, obs):
       
        obs[0] /= 1210
        obs[1] /= 700
        obs[2] /= 10
        obs[3] /= 10

        # Normalize the player positions and angles
        for i in range(4, len(obs), 4):
            obs[i + 1] /= 1210
            obs[i + 2] /=  700
            obs[i + 3] = (obs[i + 3] + 32) / 64

        return obs

    def compute_action(self, obs):
        """
        Given a single observation (numpy array),
        return an action in [-1,1], value estimate, and log probability.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, value_t = self.ac(obs_t)


        log_std = torch.clamp(self.log_std, min=-5.0, max=2.0).unsqueeze(0).expand_as(mean)
        std = torch.exp(log_std)

        # Sample from Gaussian
        action_raw = mean + std * torch.randn_like(mean)
        action_clipped = torch.clamp(action_raw, -1.0, 1.0)

        # Computing logp of the action executed
        logp = mlp_gaussian_likelihood(action_clipped, mean, log_std)

        # Squeeze out the batch dimension
        action = action_clipped.detach().cpu().numpy()[0]
        value  = value_t.detach().cpu().numpy()[0,0]
        logp   = logp.detach().cpu().numpy()[0]

        # Action in [-1,1], but the raw Gaussian might exceed that.
        #action = np.clip(action, -1.0, 1.0)
        return action, value, logp

    def train_on_buffer(self):
        """
        Run PPO update once we have a full buffer (N steps).
        """
        data = self.buf.get()  # get everything as torch tensors

        # Add validation
        if data is None:
            print("[Warning] No data in buffer to train on")
            return
    

        obs = data["obs"].to(self.device)
        act = data["act"].to(self.device)
        ret = data["ret"].to(self.device)
        adv = data["adv"].to(self.device)
        logp_old = data["logp"].to(self.device)

        if not torch.isfinite(logp_old).all():
            print("NaN in logp_old!")

        for i in range(self.train_iters):

            # Forward pass
            mean, value = self.ac(obs)
            log_std = self.log_std.expand_as(mean)
            std = torch.exp(log_std)

            # Compute log probability for the new actions
            logp_pi = mlp_gaussian_likelihood(act, mean, log_std)

            # Ratio for surrogate loss
            ratio = torch.exp(logp_pi - logp_old)

            obj = ratio * adv
            clipped_obj = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv

            loss_pi = -torch.mean(torch.min(obj, clipped_obj))
            loss_vf = torch.mean((ret - value.squeeze())**2)

            # Calculate L2 regularization
            l2_norm = sum(p.pow(2.0).sum() for p in self.ac.parameters())
            loss = loss_pi + 0.5 * loss_vf + self.l2_lambda * l2_norm

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("Backprop complete!")

            # Approximate KL divergence
            kl = torch.mean(logp_old - logp_pi).item()
            if kl > 1.5 * self.target_kl:
                print(f"[PPO] Early stopping at iter={i} due to reaching max kl.")
                break

    def process_data(self, camera):
        # Extract the current observation
        obs, bxy, vxy = self.extract_observation(camera)

        # Events from environment
        ball_kicked = bool(camera.get("ball_kicked", False))
        terminated_by_x_threshold = bool(camera.get("terminated_by_x_threshold", False))

        # Goal detection via score delta (more reliable than ball position thresholds)
        score = camera.get("score", None)
        goal_scored = False
        opponent_goal_scored = False
        if isinstance(score, (list, tuple)) and len(score) >= 2:
            if self.prev_score is not None:
                goal_scored = score[0] > self.prev_score[0]
                opponent_goal_scored = score[1] > self.prev_score[1]
            self.prev_score = list(score)

        # If not training, just run the policy forward pass
        if not self.training_enabled:
            action, _, _ = self.compute_action(obs)
            commands = self.scale_to_motor_commands(action)
            return commands

    
        # Training: collect the step
        if self.last_obs is not None:
            # Calculate the reward for the previous step (s_t-1, a_t-1 -> r_t)
            reward, reward_breakdown = simple_reward(
                goal_scored=goal_scored,
                ball_kicked=ball_kicked,
                terminated_by_x_threshold=terminated_by_x_threshold,
            )
            reward = np.clip(reward, -1000, 1000)       # Clippanje rewarda, se lahko potem še spreminja

            # NEW: Track step-level reward
            self.current_episode_step_rewards.append(reward)
            
            # NEW: Track specific events
            if goal_scored:
                self.current_episode_goals += 1
            if opponent_goal_scored:
                self.current_episode_opponent_goals += 1
            if ball_kicked:
                self.current_episode_ball_kicks += 1
            if terminated_by_x_threshold:
                self.current_episode_x_threshold_terminations += 1

            # Shranitev celotne tranzicije (s_t-1, a_t-1, r_t, V_t-1)
            stored = self.buf.store(self.last_obs, self.last_action, reward, self.last_val, self.last_logp)
            
            if not stored:
                # Buffer is full - must train immediately
                _, final_value, _ = self.compute_action(obs)
                self.buf.finish_path(last_val=final_value)
                print(f"[Training] Buffer full, training now...")
                self.train_on_buffer()


                # Reset episode tracking
                self.episode_count += 1
                self.current_step = 0
                self.episode_steps = 0
                self.ep_reward = 0.0
                self.last_obs = None
                self.last_action = None
                self.last_val = None
                self.last_logp = None
            else:
                self.ep_reward += reward

            # Early episode termination on x-threshold termination
            if terminated_by_x_threshold:
                self.finish_episode(last_value=0)
                self.episode_steps = 0

        # Preveritev terminacije epizode
        if not terminated_by_x_threshold:
            if self.episode_steps >= self.MAX_EPISODE_STEPS:
                # Zadnja napovedana vrednost za GAE izračun
                _, final_value, _ = self.compute_action(obs)
                self.finish_episode(last_value=final_value)
                self.episode_steps = 0
            else:
                self.episode_steps += 1

        # Izračun akcije za trenutno stanje
        action, value, logp = self.compute_action(obs)

        if not np.all(np.isfinite(action)):
            print("Nan or Inf detected in action:", action)
            action = np.zeros_like(action)  # fallback to safe value


        self.prev_vel = vxy[0]
        # store for next iteration
        self.last_obs = obs
        self.last_action = action
        self.last_val = value
        self.last_logp = logp
        self.current_step += 1

        commands = self.scale_to_motor_commands(action)
        return commands

    def extract_observation(self, camera):
        """
        Convert camera dict into a flat numpy array (obs_dim).
        Fill in whatever you need: ball pos, ball vel, rod pos, rod angles, etc.
        NOTE: Currently training only 
        """

        field = self.geometry["field"]
        rods = self.geometry["rods"]
        player_positions = []

        # Just a minimal example:
        CD0 = camera["camData"][0] if camera["camData"][0] is not None else camera["camData"][1]

        # Ball
        ball_x = (CD0["ball_x"] - 605) / 605  # Center around table middle [-1,1]
        ball_y = (CD0["ball_y"] - 350) / 350  # Center around table middle [-1,1]
        ball_vx = np.clip(CD0["ball_vx"] / 5.0, -2, 2)  # Velocity normalized
        ball_vy = np.clip(CD0["ball_vy"] / 5.0, -2, 2)

        #print(f"pos (x, y): {ball_x:.3f} {ball_y:.3f}, ball velocity (x, y): {ball_vx:.5f} {ball_vy:.5f}")

        # Find controlled rod in geometry
        controlled_rod_info = None
        for rod in self.geometry["rods"]:
            if rod["id"] == self.controlled_rod_id:
                controlled_rod_info = rod
                break
        
        if controlled_rod_info is None:
            raise ValueError(f"Rod {self.controlled_rod_id} not found in geometry")
        
        # Controlled rod state (5 values)
        rod_idx = self.controlled_rod_id - 1  # Convert to 0-based index
        rod_pos_calib = CD0["rod_position_calib"][rod_idx]
        rod_angle = CD0["rod_angle"][rod_idx]
        
        if isinstance(rod_pos_calib, list):
            rod_pos_calib = rod_pos_calib[0]
        
        # Team encoding (0=red, 1=blue)
        team = 0.0 if controlled_rod_info["team"] == "red" else 1.0
        
        # Rod position (already normalized 0-1)
        rod_y_normalized = rod_pos_calib
        
        # Relative positioning - KEY FOR SINGLE ROD LEARNING
        rod_x_world = controlled_rod_info["position"]
        ball_x_world = CD0["ball_x"]
        ball_y_world = CD0["ball_y"] 
        rod_y_world = rod_pos_calib * controlled_rod_info["travel"]
        
        # Distance from ball to controlled rod
        ball_rod_dist_x = (ball_x_world - rod_x_world) / 605  # Normalized [-1,1]
        ball_rod_dist_y = (ball_y_world - rod_y_world) / 350  # Normalized [-1,1]
        
        # Rod angle normalized
        angle_normalized = np.clip(rod_angle / 45.0, -1, 1)  # Assuming ±45° range
        
        # Combine: Ball(4) + Rod(5) = 9 total
        ball_kicked = float(camera.get("ball_kicked", False))

        # Combine: Ball(4) + Rod(5) + ball_kicked(1) = 10 total
        obs = np.array([
            ball_x, ball_y, ball_vx, ball_vy,           # Ball state (4)
            team,                                        # Rod team (1)
            ball_rod_dist_x, ball_rod_dist_y,           # Relative position (2)
            rod_y_normalized,                            # Rod position (1) 
            angle_normalized,                            # Rod angle (1)
            ball_kicked                                  # Ball contact flag (1)
        ], dtype=np.float32)
        
        # Verify size
        assert len(obs) == 10, f"Expected obs_dim=10, got {len(obs)}"
        
        return obs, (CD0["ball_x"], CD0["ball_y"]), (CD0["ball_vx"], CD0["ball_vy"])


    

    def scale_to_motor_commands(self, action):
        """
        We have 4 values in [-1,1]: for the forward-most rod, each rod has 4 values:
        [rot_target, rot_speed, trans_target, trans_speed]
        We'll scale them appropriately into the JSON commands expected by the simulator.
        """

        # Example scaling:
        rot_target   = 0.5 * action[0]  # in [-1,1]
        rot_velocity = 0.5 * (action[1] + 1) / 4  # scale [-1,1]→[0,1], then multiply by max 0.5
        trans_target = 0.8 * ((action[2] + 1) / 2)  # scale [-1,1]→[0,1], you might want full 0..1 0.5
        trans_velocity = 1.0 * (action[3] + 1) / 2   # scale [-1,1]→[0,1]

        cmd = {
            "driveID": 4,
            "rotationTargetPosition": rot_target,
            "rotationVelocity": rot_velocity,
            "translationTargetPosition": trans_target,
            "translationVelocity": trans_velocity
        }

        # Idle commands for the other rods
        idle_commands = [
            {
                "driveID": 2,
                "rotationTargetPosition": 0.5,
                "rotationVelocity": 0.1,
                "translationTargetPosition": 0.5,
                "translationVelocity": 0.0
            },
            {
                "driveID": 3,
                "rotationTargetPosition": 0.0,
                "rotationVelocity": 0.0,
                "translationTargetPosition": 0.5,
                "translationVelocity": 0.0
            },
            {
                "driveID": 1,
                "rotationTargetPosition": 0.0,
                "rotationVelocity": 0.0,
                "translationTargetPosition": 0.5,
                "translationVelocity": 0.0
            }
        ]

        commands = [cmd] + idle_commands
        return commands

    def finish_episode(self, last_value=0):
        """Fixed episode finishing with proper buffer managment"""

        # Only finish path if we have data in the buffer
        if self.buf.ptr > self.buf.path_start_idx:
            self.buf.finish_path(last_val=last_value)
            self.episode_rewards.append(self.ep_reward)

            # Train when buffer is sufficiently full (≥80% capacity)
            buffer_fill = self.buf.ptr / self.buf.max_size
            if buffer_fill >= 0.8:
                print(f"[Training] Buffer {buffer_fill*100:.1f}% full ({self.buf.ptr}/{self.buf.max_size}), Episode {self.episode_count}")
                self.train_on_buffer() # -> calls buf.get() which resets ptr internally



        self.episode_count += 1
        self.current_step = 0
        self.ep_reward = 0.0
        self.last_obs = None
        # Reset episode-specific variables
        self.last_action = None
        self.last_val = None
        self.last_logp = None

        self.current_episode_goals = 0
        self.current_episode_opponent_goals = 0
        self.current_episode_ball_kicks = 0
        self.current_episode_x_threshold_terminations = 0
        self.current_episode_step_rewards = []

        # Save model periodically
        if self.episode_count % self.save_model_every == 0:
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            print(f"\n{'='*70}")
            print(f"Checkpoint: Episode {self.episode_count}")
            print(f"Average Reward (last 100): {avg_reward:.2f}")
            print(f"{'='*70}\n")
            self.save_model()



        # Save model periodically
        if self.episode_count % self.save_model_every == 0:
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            print(f"Episode {self.episode_count}, Avg Reward (last 100): {avg_reward:.2f}")
            self.save_model()


# --------------------------------------------
# If you want to run standalone:
if __name__ == "__main__":
    # Create the PPO agent
    agent = PPOAgent()

    # Possibly load an existing model
    agent.load_model()

    episode =0

    try:
        while True:
            episode += 1
            time.sleep(0.02)

            # 1) Get the current camera state
            cam_data = get_camera_state()

            # 2) Process data and get motor commands
            motor_cmds = agent.process_data(cam_data)

            # 3) Send commands to environment
            send_motor_commands({'commands': motor_cmds})

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        agent.save_model("model.pth")
