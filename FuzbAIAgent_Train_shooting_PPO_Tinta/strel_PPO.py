import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import requests
import math
import json


HOST_ADDRESS = '127.0.0.1:23336'  # IP or Host for your environment

##############################
# 1) Online Functions (unchanged)
##############################

def get_camera_state():
    cam_url = f"http://{HOST_ADDRESS}/Camera/State"
    response = requests.get(cam_url)
    return response.json()


def send_motor_commands(cmds):
    motors_url = f"http://{HOST_ADDRESS}/Motors/SendCommand?blue=False"
    requests.post(motors_url, json=cmds)


def calculate_player_positions_and_angles(camera, geometry):

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



class ActorCriticNet(nn.Module):
    """
    A simple Actor-Critic network.
    It outputs both action_mean (the policy) and value (the critic).
    """
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Add dropout
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Add dropout
            nn.Linear(hidden_size, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Add dropout
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x is a batch of observations
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value

class PPOBuffer:
    """
    A simple buffer to store trajectories for PPO.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """Store one step of interaction."""
        assert self.ptr < self.max_size, "Buffer overflow!"
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory (episode).
        Computes advantage/returns for the path.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # Compute GAE-Lambda advantage
        adv = 0
        for i in reversed(range(len(rews) - 1)):
            delta = rews[i] + self.gamma * vals[i+1] - vals[i]
            adv = delta + self.gamma * self.lam * adv
            self.adv_buf[path_slice][i] = adv

        # Compute returns
        self.ret_buf[path_slice] = self.adv_buf[path_slice] + self.val_buf[path_slice]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Get all data from the buffer, then normalize advantages.
        """
        assert self.ptr == self.max_size, "Buffer has to be full before you get()"
        self.ptr, self.path_start_idx = 0, 0

        adv_mean = np.mean(self.adv_buf)
        adv_std  = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)

        data = dict(obs=self.obs_buf,
                    act=self.act_buf,
                    ret=self.ret_buf,
                    adv=self.adv_buf,
                    logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def mlp_gaussian_likelihood(action, mean, log_std):
    """
    Compute log likelihood of a Gaussian distribution with diagonal covariance
    (log_std is a vector).
    """
    pre_sum = -0.5 * (((action - mean) / (torch.exp(log_std)))**2 + 2*log_std + np.log(2*np.pi))
    return torch.sum(pre_sum, axis=1)

class PPOAgent:
    """
    PPO RL agent that:
      - Holds a policy network (actor-critic).
      - On each call to `process_data(camera)`, picks an action in [-1,1] for each rod.
      - Uses a buffer to accumulate experiences for PPO updates.
    """
    def __init__(self,
                 obs_dim=70,         # ball = x, y, vx, vy; player = 2 x 11 x 3
                 act_dim=16,         # 4 rods × 4 numbers each
                 hidden_size=128,
                 steps_per_env=2048, # how many steps per iteration
                 gamma=0.99,
                 lam=0.95,
                 clip_ratio=0.2,
                 lr=3e-4,
                 train_iters=10,
                 target_kl=0.01,
                 save_model_every=100,   # save every N episodes
                 model_save_path="ppo_foos.pth"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.save_model_every = save_model_every
        self.model_save_path = model_save_path

        with open('geometry.json') as f:
            self.geometry = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        # Actor-Critic network
        self.ac = ActorCriticNet(obs_dim, act_dim, hidden_size)
        self.ac.to(self.device)
        
        # Separate or shared log_std for continuous actions
        #self.log_std = nn.Parameter(-0.5*torch.ones(act_dim, dtype=torch.float32))
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

    def compute_action(self, obs):
        """
        Given a single observation (numpy array),
        return an action in [-1,1], value estimate, and log probability.
        """
        #obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, value_t = self.ac(obs_t)
        log_std = self.log_std.unsqueeze(0).expand_as(mean)
        std = torch.exp(log_std)

        # Sample from Gaussian
        action = mean + std * torch.randn_like(mean)
        logp = mlp_gaussian_likelihood(action, mean, log_std)

        # Squeeze out the batch dimension
        action = action.detach().cpu().numpy()[0]
        value  = value_t.detach().cpu().numpy()[0,0]
        logp   = logp.detach().cpu().numpy()[0]

        # Action in [-1,1], but the raw Gaussian might exceed that. 
        # We can clamp or just let it be. Usually we just let it be and clamp downstream.
        # We'll clamp here for safety:
        action = np.clip(action, -1.0, 1.0)
        return action, value, logp

    def train_on_buffer(self):
        """
        Run PPO update once we have a full buffer (N steps).
        """
        data = self.buf.get()  # get everything as torch tensors
        obs = data["obs"].to(self.device)
        act = data["act"].to(self.device)
        ret = data["ret"].to(self.device)
        adv = data["adv"].to(self.device)
        logp_old = data["logp"].to(self.device)

        #obs, act, ret, adv, logp_old = data['obs'], data['act'], data['ret'], data['adv'], data['logp']

        for i in range(self.train_iters):
            mean, value = self.ac(obs)
            log_std = self.log_std.expand_as(mean)
            std = torch.exp(log_std)

            # compute logp for the new actions
            logp_pi = mlp_gaussian_likelihood(act, mean, log_std)

            # ratio = exp(logp_new - logp_old)
            ratio = torch.exp(logp_pi - logp_old)

            # clipped surrogate
            clip_adv = torch.where(
                ratio > (1 + self.clip_ratio),
                adv,
                torch.where(ratio < (1 - self.clip_ratio), adv, adv)
            )  # note: the typical PPO formula is: min(ratio*adv, clip(ratio,1±eps)*adv)

            obj = ratio * adv
            clipped_obj = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv

            loss_pi = -torch.mean(torch.min(obj, clipped_obj))
            loss_vf = torch.mean((ret - value.squeeze())**2)
            loss = loss_pi + 0.5*loss_vf  # typical weighting

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # approximate KL
            kl = torch.mean(logp_old - logp_pi).item()
            if kl > 1.5 * self.target_kl:
                # early stopping
                print(f"[PPO] Early stopping at iter={i} due to reaching max kl.")
                break

    # ------------------------------------------------------------------
    # The main interface: "process_data(camera)" for each step in the sim
    # ------------------------------------------------------------------
    def process_data(self, camera):
        """
        Called by the simulator each step to get motor commands.

        1) Convert camera data to an observation array.
        2) If training, either handle the previous step's transition or store the new one.
        3) Compute an action using policy.
        4) Convert that action into final motor commands (scaled).
        5) Return the list of dicts for each rod.
        """
        # 1) Make an observation (example: ball pos, velocity, rod positions, rod angles).
        obs, col, bxy, vxy = self.extract_observation(camera)

        if col:
            print("colision")

        # If we are in the middle of an episode and have a 'last_obs', we can store
        # the transition from the previous step. But we also need the reward from the
        # previous step. That means you must keep track of reward signals externally,
        # or incorporate them here if you have the info. For simplicity, we’ll do it
        # in a separate function call "update_on_step(...)" that you can call from your
        # simulator. (Alternatively, store partial transitions here, etc.)
        # (See "reward_function(...)" placeholder below for how to compute it.)

        if not self.training_enabled:
            # If not training, just run the policy forward pass
            action, _, _ = self.compute_action(obs)
        else:
            # Training: collect the step
            if self.last_obs is not None:
                # We have a previous observation; store the reward for that step
                # until now. The reward must be computed from your environment logic:
                reward = 0.0  # or update from your environment’s collision/score trackers
                done   = False

                # We store (last_obs, act, rew, val, logp). But we need val & logp from last step.
                # So typically you'd store them as soon as you pick them. For brevity, we skip that detail.
                # You can do so with an internal "self.prev_val" etc.  
                # Or if you prefer, do a short-circuit approach: pick the action for next step, 
                # but store the (last_obs, last_act, reward, last_val, last_logp).
                pass

            reward = 0
            goal_x_range = (1200, 1210)
            goal_y_range = (250, 450)

            # 1. Collision Reward
            if col:
                reward += 100
                print("boom!!!!!!!")
            else:
                reward -= 5
                #print("NO colision")

            # 2. Direction towards the goal
            goal_center = (1205, 350)
            ball_vector = np.array([vxy[0], vxy[1]])
            direction_vector = np.array([goal_center[0] - bxy[0], goal_center[1] - bxy[1]])

            if np.linalg.norm(ball_vector) > 0:
                cosine_similarity = np.dot(ball_vector, direction_vector) / (
                    np.linalg.norm(ball_vector) * np.linalg.norm(direction_vector)
                )
                if cosine_similarity > 0:
                    directional_reward = cosine_similarity * 30
                    reward += directional_reward
                    #print(f"ball moving towards the goal, Reward: {directional_reward}")
                else:
                    reward -= 10
            else:
                reward -= 5

            # 3. Speed
            ball_speed = np.linalg.norm(ball_vector)
            reward += ball_speed * 5
            #print(f"Speed reward: {ball_speed}")

            # 4. Check goal
            if goal_x_range[0] <= bxy[0] <= goal_x_range[1] and goal_y_range[0] <= bxy[1] <= goal_y_range[1]:
                reward += 100
            elif bxy[0] < 1000 or bxy[1] > 1230:
                reward -= 50  # Own goal or out of bounds
                #print(f"Goal received, Reward -50")

            # 5. Slight penalty if ball is basically still
            if ball_speed < 0.01:
                reward -= 1
                #print(f"Slow ball spet penalty: -1")

            # Reward
            print("Reward:", reward)

            # Now pick the action for current step
            action, value, logp = self.compute_action(obs)

            # Typically you'd store that in your buffer *immediately*, e.g.:
            #self.buf.store(obs, action, reward, value, logp)

            if self.last_obs is not None:
                self.buf.store(self.last_obs, self.last_action, reward, self.last_val, self.last_logp)
            
            self.last_action = action
            self.last_val = value
            self.last_logp = logp



        # Keep track for next step
        self.last_obs = obs
        self.current_step += 1
        self.ep_reward += 0.0  # add reward from this step if you have it

        # 2) Scale the raw action in [-1,1] to your motor commands
        commands = self.scale_to_motor_commands(action)

        # 3) Return the motor commands so the simulator can drive the rods
        return commands


    def extract_observation(self, camera):
        """
        Convert camera dict into a flat numpy array (obs_dim).
        Fill in whatever you need: ball pos, ball vel, rod pos, rod angles, etc.
        """

        field = self.geometry["field"]
        rods = self.geometry["rods"]
        player_positions = []

        # Just a minimal example:
        CD0 = camera["camData"][0]
        if CD0 is None:
            # Fallback if the first camera is None
            CD0 = camera["camData"][1]

        # Ball
        bx = CD0["ball_x"]
        by = CD0["ball_y"]
        bvx = CD0["ball_vx"]
        bvy = CD0["ball_vy"]

        for rod in rods:
            rod_id = rod["id"]
            team = rod["team"]
            rod_x = rod["position"]
            travel_range = rod["travel"]
            num_players = rod["players"]
            first_offset = rod["first_offset"]
            spacing = rod["spacing"]

            rod_position_calib = CD0["rod_position_calib"][rod_id - 1]
            rod_angle = CD0["rod_angle"][rod_id - 1]

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
                    "position": (rod_x, player_y), # TODO: X - JA / NE?
                    "angle": rod_angle
                })

        # Flatten it all
        player_numeric_positions = [
            [p["position"][0], p["position"][1], p["angle"]] for p in player_positions
        ]

        # Convert to NumPy array
        player_positions_array = np.array(player_numeric_positions, dtype=np.float32)

        # Flatten and concatenate everything
        obs = np.concatenate(([bx, by, bvx, bvy], player_positions_array.flatten()), dtype=np.float32)

        ### Colision with a ball
        ball_radius=0.017
        player_radius=0.03

        for player in player_positions:
            if player["team"] == "red":
                distance = math.hypot(bx - player["position"][0], by - player["position"][1])
                col = distance <= (ball_radius + player_radius)

        return obs, col, (bx, by), (bx, by)

    def scale_to_motor_commands(self, action):
        """
        We have 16 values in [-1,1]: for rods 0,1,3,5, each rod has 4 values:
          [rot_target, rot_speed, trans_target, trans_speed]
        We'll scale them appropriately into the JSON commands expected by the simulator.
        """
        # Reshape to (4 rods, 4 dims)
        act_rod = action.reshape((4,4))

        commands = []
        # The rods we control (index in geometry): 0,1,3,5 => drive IDs = 1,2,3,4
        rod_map = [0,1,3,5]
        driveID = [1,2,3,4]
        for rod_i in range(4):
            # Raw from policy
            rot_target_raw  = act_rod[rod_i, 0]  # in [-1,1]
            rot_speed_raw   = act_rod[rod_i, 1]  # in [-1,1]
            trans_target_raw= act_rod[rod_i, 2]  # in [-1,1]
            trans_speed_raw = act_rod[rod_i, 3]  # in [-1,1]

            # print("rot_target_raw",rot_target_raw) 
            # print("rot_speed_raw",rot_speed_raw) 
            # print("trans_target_raw",trans_target_raw)
            # print("trans_speed_raw",trans_speed_raw) 

            # Example scaling:
            rot_target   = 0.8 * rot_target_raw     # we only want to rotate between -0.8..+0.8
            rot_velocity = 0.5 * (rot_speed_raw+1)/2  # scale [-1,1]→[0,1], then multiply by max
            trans_target = 0.5 * ((trans_target_raw+1)/2)  # scale [-1,1]→[0,1], you might want full 0..1
            trans_velocity = 1.0 * (trans_speed_raw+1)/2   # scale [-1,1]→[0,1]

            cmd = {
                "driveID": driveID[rod_i],
                "rotationTargetPosition": rot_target,
                "rotationVelocity": rot_velocity,
                "translationTargetPosition": trans_target,
                "translationVelocity": trans_velocity
            }
            commands.append(cmd)

        #print(commands)
        return commands

    def finish_episode(self, last_value=0):
        """
        Call this when the episode (i.e. round) ends so we can finish
        advantage calculation, do a PPO update, etc.
        """
        self.episode_count += 1
        # Let the buffer compute GAE, returns, etc.
        self.buf.finish_path(last_val=last_value)
        self.episode_rewards.append(self.ep_reward)

        # If we filled our buffer, we do a PPO update
        if self.buf.ptr == self.buf.max_size:
            self.train_on_buffer()

        # Housekeeping
        self.current_step = 0
        self.ep_reward = 0.0
        self.last_obs = None

        # Save model every N episodes
        if self.episode_count % self.save_model_every == 0:
            self.save_model()

    def reward_function(self, collision=False, goal_scored=False, own_goal=False, ball_dir_bonus=0.0):
        """
        Example reward shaping logic.  You or the simulator can call this each step
        or after collisions, etc.  Then feed the result to the buffer store(...).
        
        Some ideas from your request:
          - If ball collided with red player => +R
          - If ball is traveling toward the opponent => +r
          - If scored => big +R
          - If lost => big negative
          - Keep rods' legs down => small positive, etc.
        """
        reward = 0.0
        if collision:
            reward += 0.1
        reward += ball_dir_bonus
        if goal_scored:
            reward += 5.0
        if own_goal:
            reward -= 5.0
        # etc.
        return reward

# --------------------------------------------
# If you want to run standalone:
if __name__ == "__main__":
    # Create the PPO agent
    agent = PPOAgent()

    # Possibly load an existing model
    agent.load_model()

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
        agent.save_model("actor.pth", "critic.pth")
