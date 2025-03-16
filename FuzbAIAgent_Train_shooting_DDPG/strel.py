import requests
import json
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gc

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


##############################
# 2) Helper Functions (unchanged or minimal edits)
##############################

def ball_data(camera):

    CD0 = camera["camData"][0]
    if not CD0:
        # Fallback if the first camera is None
        CD0 = camera["camData"][1]

    vx = CD0["ball_vx"]
    vy = CD0["ball_vy"]
    bx = CD0["ball_x"]
    by = CD0["ball_y"]

    # Some threshold for 'flag' if ball is stationary
    flag = 1 if abs(vx) < 0.01 and abs(vy) < 0.01 else 0
    return bx, by, vx, vy, flag


def players_data(camera):

    CD0 = camera["camData"][0]
    CD1 = camera["camData"][1]

    if CD0 is not None:
        data = CD0
    else:
        data = CD1

    positions = []
    rotations = []
    for i in range(8):
        positions.append(data["rod_position_calib"][i])
        rotations.append(data["rod_angle"][i])
    return positions, rotations


def detect_collision(ball_pos, player_pos, ball_radius=0.017, player_radius=0.03):

    distance = math.hypot(ball_pos[0] - player_pos[0], ball_pos[1] - player_pos[1])
    return distance <= (ball_radius + player_radius)


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


##############################
# 3) Reward Function (unchanged)
##############################

def calculate_shooting_reward(bx, by, vx, vy, collision_detected, player_data):

    reward = 0
    goal_x_range = (1200, 1210)
    goal_y_range = (250, 450)

    # 1. Collision Reward
    if collision_detected:
        reward += 10
        print("boom")
    else:
        reward -= 5
        print("NO colision")

    # 2. Direction towards the goal
    goal_center = (1205, 350)
    ball_vector = np.array([vx, vy])
    direction_vector = np.array([goal_center[0] - bx, goal_center[1] - by])

    if np.linalg.norm(ball_vector) > 0:
        cosine_similarity = np.dot(ball_vector, direction_vector) / (
            np.linalg.norm(ball_vector) * np.linalg.norm(direction_vector)
        )
        if cosine_similarity > 0:
            directional_reward = cosine_similarity * 30
            reward += directional_reward
        else:
            reward -= 10
    else:
        reward -= 5

    # 3. Speed
    ball_speed = np.linalg.norm(ball_vector)
    reward += ball_speed * 5

    # 4. Check goal
    if goal_x_range[0] <= bx <= goal_x_range[1] and goal_y_range[0] <= by <= goal_y_range[1]:
        reward += 100
    elif bx < 1000 or bx > 1230:
        reward -= 50  # Own goal or out of bounds

    # 5. Slight penalty if ball is basically still
    if ball_speed < 0.01:
        reward -= 1

    # 6. Slight penalty if a player is oriented in the air
    for player in player_data:
        if player["team"] == "red":
            reward = 10 * (32 - abs(player["angle"]))
  

    return reward


##############################
# 4) DDPG‐Style Classes
##############################

class Actor(nn.Module):
    """
    Actor network: takes state -> outputs continuous actions in [-1, 1].
    """
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
        # Optional: Weight init to small, etc. Or just use default.

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Use tanh so that outputs are in [-1, 1]
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """
    Critic network: takes (state, action) -> outputs Q-value.
    """
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Critic, self).__init__()
        # One approach: concatenate state & action right at the input
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        # state: (batch, state_dim)
        # action: (batch, action_dim)
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """
    Simple replay buffer for storing transitions.
    """
    def __init__(self, max_size=50):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # batch = random.sample(self.buffer, batch_size)
        # states, actions, rewards, next_states, dones = zip(*batch)
        # return (
        #     np.array(states, dtype=np.float32),
        #     np.array(actions, dtype=np.float32),
        #     np.array(rewards, dtype=np.float32),
        #     np.array(next_states, dtype=np.float32),
        #     np.array(dones, dtype=np.float32)
        # )

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


##############################
# 5) The Continuous Agent (DDPG)
##############################

class ContinuousAgent:
    def __init__(self, state_dim=20, action_dim=16, gamma=0.99, lr_actor=0.0001, lr_critic=0.001,
                 tau=0.005, batch_size=64, max_memory=50):
        """
        :param state_dim: dimension of your input (e.g., ball + rods data)
        :param action_dim: dimension of your actions (4 rods × 4 continuous outputs each = 16)
        :param tau: soft-update parameter for target nets
        """
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Actor & Critic
        # self.actor = Actor(state_dim, action_dim)
        # self.critic = Critic(state_dim, action_dim)
        # self.target_actor = Actor(state_dim, action_dim)
        # self.target_critic = Critic(state_dim, action_dim)

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)

        # Copy weights initially
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.memory = ReplayBuffer(max_size=max_memory)

        # For exploration noise
        self.exploration_noise = 0.2  # Could be smaller. Tweak as needed.

        self.team_color="red"

        self.model = self.actor
        self.learn_step_counter = 0

        # Load geometry
        with open('geometry.json') as f:
            self.geometry = json.load(f)


    def learn_from_batch(self, states, actions, rewards, next_states, dones):
        # This is basically your 'learn()' steps, but taking arrays as arguments
        # instead of sampling from self.memory. For instance:

        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        # 1) Critic update
        current_Q = self.critic(states_t, actions_t)
        with torch.no_grad():
            next_actions = self.target_actor(next_states_t)
            next_Q = self.target_critic(next_states_t, next_actions)
            target_Q = rewards_t + (1.0 - dones_t) * self.gamma * next_Q
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2) Actor update
        actor_actions = self.actor(states_t)
        actor_loss = -self.critic(states_t, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3) Soft update targets
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)


    def choose_action(self, state):
        """
        Chooses an action using the actor plus some exploration noise.
        """
        #state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # shape (1, state_dim)
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        # Add noise for exploration
        noise = np.random.normal(0, self.exploration_noise, size=action.shape)
        action = action + noise
        
        # Clip to [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        return action

    def remember(self, state, action, reward, next_state, done):
        """
        Store the transition in replay buffer.
        """
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        """
        Sample from replay and update networks (actor & critic).
        """
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # states_t = torch.FloatTensor(states)
        # actions_t = torch.FloatTensor(actions)
        # rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        # next_states_t = torch.FloatTensor(next_states)
        # dones_t = torch.FloatTensor(dones).unsqueeze(1)

        states_t = torch.from_numpy(states).float().to(self.device).detach()
        actions_t = torch.from_numpy(actions).float().to(self.device).detach()
        rewards_t = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)

        # =====================
        # 1) Update Critic
        # =====================
        # Current Q
        current_Q = self.critic(states_t, actions_t)

        with torch.no_grad():
            # Next actions (from target actor)
            next_actions = self.target_actor(next_states_t).detach()
            next_Q = self.target_critic(next_states_t, next_actions).detach()

            # Target Q
            target_Q = rewards_t + (1.0 - dones_t) * self.gamma * next_Q

        # Critic Loss
        critic_loss = nn.MSELoss()(current_Q, target_Q.to(self.device))

        # Backprop Critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward(retain_graph=False) # Uprašljivp!
        self.critic_optimizer.step()

        # =====================
        # 2) Update Actor
        # =====================
        # Actor wants to maximize Q-value. So we do:
        actor_actions = self.actor(states_t)
        actor_loss = -self.critic(states_t, actor_actions).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward(retain_graph=False) # Uprašljivp!
        self.actor_optimizer.step()

        # =====================
        # 3) Soft Update Targets
        # =====================
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)

    def soft_update(self, target_net, source_net, tau):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )

    def save_model(self, actor_path="actor.pth", critic_path="critic.pth"):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path="actor.pth", critic_path="critic.pth"):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


    ################################################
    # 6) The main step for handling environment data
    ################################################
    def process_data(self, camera):
        """
        - Gathers state (ball + rod positions/angles)
        - Chooses an action
        - Computes reward
        - Remembers transition
        - Learns
        - Returns commands to send to environment
        """

        start = time.time()

        # 1) Build the state vector
        bx, by, vx, vy, _ = ball_data(camera)  # 4 values
        opp_pos, opp_rpt = players_data(camera)     # positions & angles -> 8 + 8 = 16
        # Combine into one vector (20 dims if your code is consistent)
        state = np.concatenate([[bx, by, vx, vy], opp_pos, opp_rpt])
        print("Time 1:", time.time() - start)

        # 2) Choose an action (continuous, shape=16)
        start = time.time()
        action = self.choose_action(state)
        print("Time 2:", time.time() - start)

        # 3) Collect new info to figure out "next_state" if needed
        #    (Often you'd do a second camera read, but here let's assume next_state is the same
        #     or you might call process_data again at next loop iteration. We'll keep it simple.)
        start = time.time()
        next_state = state  # or do more advanced logic
        print("Time 3:", time.time() - start)

        # 4) Collision detection
        start = time.time()
        player_data = calculate_player_positions_and_angles(camera, self.geometry)
        ball_position = (bx, by)
        ball_collision = False
        for player in player_data:
            if player["team"] == self.team_color:
                if detect_collision(ball_position, player["position"]):
                    ball_collision = True
                    break
        print("Time 4:", time.time() - start)

        # 5) Reward
        start = time.time()
        reward = calculate_shooting_reward(bx, by, vx, vy, ball_collision, player_data)
        print("Reward", reward)
        print("Time 5:", time.time() - start)

        # 6) Check if done
        #    We'll say 'done' if we scored a goal (reward=100),
        #    but that's up to your environment design.
        start = time.time()
        done = (reward >= 100)
        print("Time 6:", time.time() - start)

        # 7) Store in memory & learn
        start = time.time()
        self.remember(state, action, reward, next_state, done)
        self.learn()
        print("Time 7:", time.time() - start)


        # 8) Convert the 16‐dim action vector into your final commands
        #    Each rod has 4 fields: rotationTarget, rotationVelocity, translationTarget, translationVelocity
        #    We'll do a direct mapping from the continuous action vector to each rod’s 4 values.
        #    The user can tweak scaling as needed.
        start = time.time()
        rods_for_team = [rod for rod in self.geometry["rods"] if rod["team"] == self.team_color]
        commands = []
        for i, rod in enumerate(rods_for_team):
            base_idx = i * 4
            # Suppose each dimension is in [-1, 1]; let's rescale them:
            rotation_target      = action[base_idx + 0]  # stays in [-1, 1]
            rotation_velocity    = (action[base_idx + 1] + 1) * 1.0  # map [-1,1] -> [0,2]
            translation_target   = (action[base_idx + 2] + 1) * 0.5  # map [-1,1] -> [0,1]
            translation_velocity = (action[base_idx + 3] + 1) * 1.0  # map [-1,1] -> [0,2]

            cmd = {
                'driveID': i + 1,  # or i+1, depending on your environment
                'rotationTargetPosition': rotation_target,
                'rotationVelocity': rotation_velocity,
                'translationTargetPosition': translation_target,
                'translationVelocity': translation_velocity
            }
            
            commands.append(cmd)
        print("Time 8:", time.time() - start)

        torch.cuda.empty_cache()
        gc.collect()

        #print(commands)
        return commands


##############################
# 7) The Main Loop
##############################

if __name__ == "__main__":
    
    # Suppose our input has size=20 (ball + rods) and output has size=16 (4 rods × 4 each).
    agent = ContinuousAgent(state_dim=20, action_dim=16)
    episode = 0
    save_interval = 100

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

            # 4) Save model periodically
            if episode % save_interval == 0:
                agent.save_model("actor.pth", "critic.pth")

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        agent.save_model("actor.pth", "critic.pth")
