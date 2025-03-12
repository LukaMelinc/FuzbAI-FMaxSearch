import requests
import json
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

HOST_ADDRESS = '127.0.0.1:23336'

# DQN Hyperparameters
ALPHA = 0.001                   # Learning rate
GAMMA = 0.99                    # Discount factor for future rewards
EPSILON = 0.9                   # Initial exploration rate
EPSILON_DECAY = 0.995           # Rate at which exploration decays
MIN_EPSILON = 0.01              # Minimum exploration rate
BATCH_SIZE = 32                  # Batch size for training
MEMORY_SIZE = 10000             # Replay memory size

POSITION_BINS = 10
VELOCITY_BINS = 10

def get_camera_state():
    """Fetch the current camera state from the simulator."""
    cam_url = f"http://{HOST_ADDRESS}/Camera/State"
    response = requests.get(cam_url)
    return response.json()

def send_motor_commands(cmds):
    """Send motor commands to the simulator."""
    motors_url = f"http://{HOST_ADDRESS}/Motors/SendCommand?blue=False"
    requests.post(motors_url, json=cmds)

class DQN(nn.Module):
    """Deep Q-Network model."""
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        """Forward pass to predict Q-values."""
        return self.fc(x)

class RLAgent:
    """Reinforcement Learning agent using DQN for multiple players."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Code will run on: ", self.device)
        self.actions = ['move_left', 'move_right', 'kick']
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.models = {i: DQN(4, len(self.actions)).to(self.device) for i in range(4)}
        self.optimizers = {i: optim.Adam(self.models[i].parameters(), lr=ALPHA) for i in range(4)}
        self.criterion = nn.MSELoss()
        self.epsilon = EPSILON

    def get_state(self, cam_data, rod_idx):
        """Extract and normalize state for each rod from camera data."""
        bx = cam_data['ball_x'] / 1210
        by = cam_data['ball_y'] / 700
        vx = (cam_data['ball_vx'] + 1) / 2
        vy = (cam_data['ball_vy'] + 1) / 2
        return np.array([bx, by, vx, vy], dtype=np.float32)

    def choose_action(self, state, rod_idx):
        """Choose an action using epsilon-greedy policy for each rod."""
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.models[rod_idx](state_tensor)).item()

    def remember(self, rod_idx, state, action, reward, next_state):
        """Store experience in replay memory."""
        print(reward)
        self.memory.append((rod_idx, state, action, reward, next_state))

    def learn(self):
        """Train the model using a batch of experiences for each rod."""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        for rod_idx in range(8):
            rod_batch = [b for b in batch if b[0] == rod_idx]
            if not rod_batch:
                continue

            _, states, actions, rewards, next_states = zip(*rod_batch)
            states = torch.from_numpy(np.array(states)).float().to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)

            current_q = self.models[rod_idx](states).gather(1, actions).squeeze()
            max_next_q = self.models[rod_idx](next_states).max(1)[0]
            expected_q = rewards + GAMMA * max_next_q

            loss = self.criterion(current_q, expected_q.detach())
            self.optimizers[rod_idx].zero_grad()
            loss.backward()
            self.optimizers[rod_idx].step()

        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

    def process_data(self, camera):
        """Process camera data and return motor commands for each rod."""
        commands = []

        for rod_idx in range(4):
            state = self.get_state(camera['camData'][0], rod_idx)
            action_idx = self.choose_action(state, rod_idx)

            # Ensure action is always updated
            next_state = self.get_state(camera['camData'][0], rod_idx)
            self.remember(rod_idx, state, action_idx, 0, next_state)  # Store experience


            # Reward for moving closer to the opponent's side
            reward = state[0] * 10 - state[1] * 5

            # Extra reward for kicking the ball
            if action == 'kick':
                reward += 5

            # Penalty if the ball is close to own goal
            if state[0] < 0.2:
                reward -= 10

            movement_threshold = 0.05
            if abs(next_state[1] - state[1]) > movement_threshold:
                reward += 1

            # reward = 0
            # if state[0] > 0.8:
            #     reward = 10
            # elif state[0] < 0.2:
            #     reward = -10


            action = self.actions[action_idx]
            cmd = {
                'driveID': rod_idx + 1,
                'rotationTargetPosition': 0.5 if action == 'kick' else 0,
                'rotationVelocity': 1,
                'translationTargetPosition': 0.4 if action == 'move_left' else 0.6,
                'translationVelocity': 1.0
            }
            commands.append(cmd)

        self.learn()
        return commands

if __name__ == "__main__":
    agent = RLAgent()
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
                agent.save_model("./") #TODO


    except KeyboardInterrupt:
        print("Training interrupted.")
