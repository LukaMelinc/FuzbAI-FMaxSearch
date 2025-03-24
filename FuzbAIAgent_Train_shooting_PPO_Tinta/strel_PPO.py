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



def detect_collision_and_reward(yv, prev_ball_vx, current_ball_vx, ball_x, active_regions=None):
    
   # print("-------")
   # print(prev_ball_vx)
   # print(current_ball_vx)
   # print(yv)
    
    
    rod_positions=[80, 230, 530, 830]
    if active_regions is None:
        active_regions = {i: {"active": False, "rewarded": False} for i in range(len(rod_positions))}

    # Define the regions along the rods where collisions are checked
    collision_regions = [(rod_x - 80, rod_x + 80) for rod_x in rod_positions]

    # Check if the ball is within any of the collision regions
    for i, region in enumerate(collision_regions):
        if region[0] <= ball_x <= region[1]:
            # Detect if there is an increase in the ball's speed
            if abs(abs(current_ball_vx) - abs(prev_ball_vx)) > 0.05:
                if not active_regions[i]["active"]:
                    # Ball entered the region
                    active_regions[i]["active"] = True
                    active_regions[i]["rewarded"] = False

                if not active_regions[i]["rewarded"]:
                 #   print("Collision detected in region:", region)
                    # Determine the direction of the speed increase
                    if current_ball_vx > prev_ball_vx:
                        # Positive increase in speed (towards opponent's goal)
                        active_regions[i]["rewarded"] = True
                        return 5.0, active_regions  # Positive reward
                    else:
                        # Negative increase in speed (towards own goal)
                        active_regions[i]["rewarded"] = True
                        return -2.0, active_regions  # Negative reward

    # Reset active regions if the ball is not in proximity
    for i, region in enumerate(collision_regions):
        if not (region[0] <= ball_x <= region[1]):
            active_regions[i]["active"] = False

    # No collision detected
    return 0.0, active_regions


# SAM ZA PREVERIT, ČE JE DELAY 2 - POTEM SE GA LAHKO ZAKOMENTIRA
def detect_y_axis_changes(current_y, prev_vy, current_vy, threshold=0.05):

    print("-------------")
    #print(f"Previous y position: {prev_y}")
    print(f"Current y position: {current_y}")
    print("......")
    print(f"Previous y speed: {prev_vy}")
    print(f"Current y speed: {current_vy}")
   
    change_detected = False
    #reward = 0.0

    """ # Check for significant changes in y-axis position
    if abs(current_y - prev_y) > threshold:
        change_detected = True
        reward += 2.0  # Positive reward for significant position change"""

    # Check for significant changes in y-axis velocity
    if abs(current_vy - prev_vy) > threshold:
        change_detected = True
        #reward += 2.0  # Positive reward for significant velocity change

    if (prev_vy > 0 and current_vy < 0) or (prev_vy < 0 and current_vy > 0):
            print("Direction of y-axis velocity changed!")
           # reward += 3.0  # Additional reward for change in direction





class ActorCriticNet(nn.Module):
    """
    A simple Actor-Critic network.
    It outputs both action_mean (the policy) and value (the critic).
    """
    def __init__(self, obs_dim, act_dim, hidden_size=512):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.25),  # Add dropout
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.25),  # Add dropout
            nn.Linear(hidden_size, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.25),  # Add dropout
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
    def __init__(self, obs_dim, act_dim, size, gamma=0.8, lam=0.95):
        # večja lambda pomeni večjo varianco in bolj dolgotrajen trening
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
        #assert self.ptr < self.max_size, "Buffer overflow!"
        idx = self.ptr % self.max_size  # Ensure index wraps around when full
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.val_buf[idx] = val
        self.logp_buf[idx] = logp
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
        # assert self.ptr == self.max_size, "Buffer has to be full before you get()"
        # self.ptr, self.path_start_idx = 0, 0

        # adv_mean = np.mean(self.adv_buf)
        # adv_std  = np.std(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)

        indices = np.arange(min(self.ptr, self.max_size))  # Only take latest
        adv_mean = np.mean(self.adv_buf[indices])
        adv_std  = np.std(self.adv_buf[indices])
        self.adv_buf[indices] = (self.adv_buf[indices] - adv_mean) / (adv_std + 1e-8)

        data = dict(obs=self.obs_buf[indices],
                    act=self.act_buf[indices],
                    ret=self.ret_buf[indices],
                    adv=self.adv_buf[indices],
                    logp=self.logp_buf[indices]
                    )
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
                 hidden_size=1024,
                 steps_per_env=2048, # how many steps per iteration
                 gamma=0.99,
                 lam=0.95,
                 clip_ratio=0.2,
                 lr=3e-4,
                 train_iters=10,
                 target_kl=0.01,
                 delay_step=2,
                 save_model_every=100,   # save every N episodes
                 model_save_path="ppo_foos.pth",
                 l2_lambda=5e-4):
                        # L2 regularization strength
        
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.save_model_every = save_model_every
        self.model_save_path = model_save_path
        self.l2_lambda = l2_lambda  # L2 regularization strength

        self.prev_ball = 0
        self.prev_vy = 0
        self.active_regions = None  # Initialize active_regions
        self.MAX_EPISODE_STEPS = 100
        self.episode_steps = 0

        self.delay_steps = delay_step
        self.obs_buffer = [None] * self.delay_steps
        self.action_buffer = [None] * self.delay_steps
        self.reward_buffer = [None] * self.delay_steps
        self.value_buffer = [None] * self.delay_steps
        self.logp_buffer = [None] * self.delay_steps


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

        for i in range(self.train_iters):
            mean, value = self.ac(obs)
            log_std = self.log_std.expand_as(mean)
            std = torch.exp(log_std)

            # Compute log probability for the new actions
            logp_pi = mlp_gaussian_likelihood(act, mean, log_std)

            # Ratio for surrogate loss
            ratio = torch.exp(logp_pi - logp_old)

            # Clipped surrogate objective
            clip_adv = torch.where(
                ratio > (1 + self.clip_ratio),
                adv,
                torch.where(ratio < (1 - self.clip_ratio), adv, adv)
            )

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

            # Approximate KL divergence
            kl = torch.mean(logp_old - logp_pi).item()
            if kl > 1.5 * self.target_kl:
                print(f"[PPO] Early stopping at iter={i} due to reaching max kl.")
                break

    # The rest of your class remains unchanged...


    # ------------------------------------------------------------------
    # The main interface: "process_data(camera)" for each step in the sim
    # ------------------------------------------------------------------
    def process_data(self, camera):
        # Extract the current observation
        obs, bxy, vxy = self.extract_observation(camera)

        # Store the current observation in the buffer
        self.obs_buffer.append(obs)
        self.obs_buffer.pop(0)

        if not self.training_enabled:
            # If not training, just run the policy forward pass
            action, _, _ = self.compute_action(self.obs_buffer[-1])
        else:
            # Training: collect the step
            if self.last_obs is not None:
                # We have a previous observation; store the reward for that step
                reward = 10.0  # or update from your environment’s collision/score trackers

                # Store the current action, value, and logp in the buffer
                self.action_buffer.append(self.last_action)
                self.action_buffer.pop(0)
                self.value_buffer.append(self.last_val)
                self.value_buffer.pop(0)
                self.logp_buffer.append(self.last_logp)
                self.logp_buffer.pop(0)

                # Store the reward in the buffer
                self.reward_buffer.append(reward)
                self.reward_buffer.pop(0)

                # Use the delayed reward for training
                delayed_reward = self.reward_buffer[-1]

                # Store the transition in the buffer
                self.buf.store(self.obs_buffer[-2], self.action_buffer[-2], delayed_reward, self.value_buffer[-2], self.logp_buffer[-2])

            # Compute the action for the current step
            action, value, logp = self.compute_action(self.obs_buffer[-1])

            if self.episode_steps >= self.MAX_EPISODE_STEPS:
                print("Treniram")
                self.finish_episode()
                self.episode_steps = 0
            else:
                self.episode_steps += 1

            # Store the current action, value, and logp for the next step
            self.last_action = action
            self.last_val = value
            self.last_logp = logp

            # Keep track for next step
            self.last_obs = self.obs_buffer[-1]
            self.current_step += 1
            self.ep_reward += 0.0  # add reward from this step if you have it

        # Scale the raw action in [-1,1] to your motor commands
        commands = self.scale_to_motor_commands(action)

        # Return the motor commands so the simulator can drive the rods
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

        return obs, (bx, by), (bvx, bvy)

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
            rot_velocity = 0.5 * (rot_speed_raw+1)/8  # scale [-1,1]→[0,1], then multiply by max
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
        self.episode_count += 1
        # Let the buffer compute GAE, returns, etc.
        self.buf.finish_path(last_val=last_value)
        self.episode_rewards.append(self.ep_reward)

        # If we filled our buffer, we do a PPO update
        print("Buffer:", self.buf.ptr)
        if self.buf.ptr == self.buf.max_size:
            print("Buffer full!")
            self.train_on_buffer()

        # Housekeeping
        self.current_step = 0
        self.ep_reward = 0.0
        self.last_obs = None

        # Clear the buffers
        # self.obs_buffer = [None] * self.delay_steps
        # self.action_buffer = [None] * self.delay_steps
        # self.reward_buffer = [None] * self.delay_steps
        # self.value_buffer = [None] * self.delay_steps
        # self.logp_buffer = [None] * self.delay_steps

        # Save model every N episodes
        if self.episode_count % self.save_model_every == 0:
            self.save_model()


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
