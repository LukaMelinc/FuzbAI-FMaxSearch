import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import time
import gc
import json
import math

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
# 2) Helper Functions (unchanged)
##############################

def ball_data(camera):
    CD0 = camera["camData"][0]
    if not CD0:
        CD0 = camera["camData"][1]
    vx = CD0["ball_vx"]
    vy = CD0["ball_vy"]
    bx = CD0["ball_x"]
    by = CD0["ball_y"]
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
        if isinstance(rod_position_calib, list):
            rod_position_calib = rod_position_calib[0]
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
    if collision_detected:
        reward += 10
    #    print("boom")
    else:
        reward -= 5
     #   print("NO colision")
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
        #    print(f"ball moving towards the goal, Reward: {directional_reward}")
        else:
            reward -= 10
    else:
        reward -= 5
    ball_speed = np.linalg.norm(ball_vector)
    reward += ball_speed * 5
   # print(f"Speed reward: {ball_speed}")
    if goal_x_range[0] <= bx <= goal_x_range[1] and goal_y_range[0] <= by <= goal_y_range[1]:
        reward += 100
    elif bx < 10 or bx > 1230:
        reward -= 50
   #     print(f"Goal received, Reward -50")
    if ball_speed < 0.01:
        reward -= 1
   #     print(f"Slow ball spet penalty: -1")
    for player in player_data:
        if player["team"] == "red":
            reward = 10 * (32 - abs(player["angle"]))
    print(reward)
   # print(bx)
   # print(by)
    print("----------------------------------------------------")
    return reward

##############################
# 4) PPO-Style Classes
##############################

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_dim)
        self.fc_logstd = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        logstd = self.fc_logstd(x)
        std = torch.exp(logstd)
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc_value(x)
        return value

class PPOAgent:
    def __init__(self, state_dim=20, action_dim=16, gamma=0.99, lr=0.001, epsilon=0.2, batch_size=32, max_memory=200):
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #    print("Using device:", self.device)

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

        self.memory = deque(maxlen=max_memory)

        self.team_color = "red"

        with open('geometry.json') as f:
            self.geometry = json.load(f)

    def choose_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = self.policy(state_t)
            action = torch.normal(mean, std)
        return action.cpu().numpy()[0]

    def remember(self, state, action, reward, next_state, done, prob):
        self.memory.append((state, action, reward, next_state, done, prob))

    def learn(self):
        if len(self.memory) < self.batch_size:
            print(len(self.memory))
            print(self.batch_size)
            print("Not enough data for learning")
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, old_probs = zip(*batch)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        old_probs_t = torch.FloatTensor(old_probs).to(self.device)

        # Compute value and advantage
        values = self.value(states_t)

        #print(values)
        next_values = self.value(next_states_t)
        td_errors = rewards_t + self.gamma * next_values * (1 - dones_t) - values
        advantages = td_errors.detach()

        # Compute new probabilities and policy loss
        mean, std = self.policy(states_t)
        print("Policy Network Outputs during Learning - Mean:", mean)
        print("Policy Network Outputs during Learning - Std:", std)


        dist = torch.distributions.Normal(mean, std)
        new_probs = dist.log_prob(actions_t).sum(dim=1, keepdim=True)
        ratio = torch.exp(new_probs - old_probs_t)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Compute value loss
        value_loss = nn.MSELoss()(values, rewards_t + self.gamma * next_values * (1 - dones_t))

        # Update networks
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def process_data(self, camera):


        bx, by, vx, vy, _ = ball_data(camera)
        opp_pos, opp_rpt = players_data(camera)
        state = np.concatenate([opp_pos, [bx, by, vx, vy],opp_rpt])



        action = self.choose_action(state)


 
        next_state = state
        


        player_data = calculate_player_positions_and_angles(camera, self.geometry)
        ball_position = (bx, by)
        ball_collision = False
        for player in player_data:
            if player["team"] == self.team_color:
                if detect_collision(ball_position, player["position"]):
                    ball_collision = True
                    break



        reward = calculate_shooting_reward(bx, by, vx, vy, ball_collision, player_data)
      #  print("Reward", reward)


        done = (reward >= 1000)


        mean, std = self.policy(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        dist = torch.distributions.Normal(mean, std)
        prob = dist.log_prob(torch.FloatTensor(action).to(self.device)).sum(dim=1, keepdim=True)
        self.remember(state, action, reward, next_state, done, prob.cpu().detach().numpy())
        self.learn()



        rods_for_team = [rod for rod in self.geometry["rods"] if rod["team"] == self.team_color]
        commands = []
        for i, rod in enumerate(rods_for_team):
            base_idx = i * 4
            rotation_target = action[base_idx + 0] * 32
            rotation_velocity = (action[base_idx + 1] + 1) * 0.5
            translation_target = (action[base_idx + 2] + 1) * 0.5
            translation_velocity = (action[base_idx + 3] + 1) * 0.5
            cmd = {
                'driveID': i + 1,
                'rotationTargetPosition': rotation_target,
                'rotationVelocity': rotation_velocity,
                'translationTargetPosition': translation_target,
                'translationVelocity': translation_velocity
            }
            commands.append(cmd)


        torch.cuda.empty_cache()
        gc.collect()

        return commands

    def save_model(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

##############################
# 7) The Main Loop
##############################

if __name__ == "__main__":
    agent = PPOAgent(state_dim=20, action_dim=16)
    episode = 0
    save_interval = 100

    try:
        while True:
            episode += 1
            time.sleep(0.02)
            cam_data = get_camera_state()
            motor_cmds = agent.process_data(cam_data)
            send_motor_commands({'commands': motor_cmds})
            if episode % save_interval == 0:
                torch.save(agent.policy.state_dict(), "policy.pth")
                torch.save(agent.value.state_dict(), "value.pth")
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(agent.policy.state_dict(), "policy.pth")
        torch.save(agent.value.state_dict(), "value.pth")
