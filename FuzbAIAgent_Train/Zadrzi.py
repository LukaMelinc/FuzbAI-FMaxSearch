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
from functools import partial

HOST_ADDRESS = '127.0.0.1:23336'

def get_camera_state():
    cam_url = f"http://{HOST_ADDRESS}/Camera/State"
    response = requests.get(cam_url)    
    return response.json()

def send_motor_commands(cmds):
    motors_url = f"http://{HOST_ADDRESS}/Motors/SendCommand?blue=False"
    response = requests.post(motors_url, json=cmds)

# Neural Network for DQL
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class BallControlAgent:
    def __init__(self, state_size=4, action_size=4, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, lr=0.001, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size

        self.learn_step_counter = 0

        # Experience Replay Memory
        self.memory = deque(maxlen=2000)

        # Initialize networks
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.actions = ['kick', 'move_left', 'move_right', 'idle']

        # Use directly bound methods
        self.rod_reward_functions = {
            0: self.reward_rod_0,
            1: self.reward_rod_1,
            2: self.reward_rod_2,
            3: self.reward_rod_3
        }



    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples
        
        self.learn_step_counter += 1
        if self.learn_step_counter % 10 != 0:
            return  # Only learn every 10 steps

        # Sample mini-batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q-values
        q_values = self.model(states).gather(1, actions)

        # Max Q-values from target model
        next_q_values = self.target_model(next_states).max(1)[0].detach().unsqueeze(1)

        # Target Q-values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.criterion(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()

    def data_process(self, camera):
        CD0 = camera["camData"][0]
        CD1 = camera["camData"][1]

        vx = CD0["ball_vx"]
        vy = CD0["ball_vy"]

        bx = CD0["ball_x"]
        by = CD0["ball_y"]
        #print("Ball x, y: ", bx, by, "Ball vel:", vx, vy)

        flag = 1 if vx < 1 and vy < 1 else 0

        return bx, by, vx, vy, flag




    def reward_rod_0(self, data, state, next_state, action):
        """
        Reward function for rod 0 (goalie).
        Rewards the agent if the ball stops within the target area for rod 0.
        """

        _, _, vx, vy, flag = self.data_process(data)


        if self.is_ball_in_target_area(next_state, rod_idx=0) and flag:
            return 100

        else:
            return -25
        #return -1

    def reward_rod_1(self, data, state, next_state, action):
        """
        Reward function for rod 1 (defense).
        Rewards the agent if the ball stops within the target area for rod 1.
        """

        _, _, vx, vy, flag = self.data_process(data)

        if self.is_ball_in_target_area(next_state, rod_idx=1) and flag:
            return 100

        else:
            return -25
        #return -1

    def reward_rod_2(self, data, state, next_state, action):
        """
        Reward function for rod 2 (midfield).
        Rewards the agent if the ball stops within the target area for rod 2.
        """

        _, _, vx, vy, flag = self.data_process(data)

        if self.is_ball_in_target_area(next_state, rod_idx=2) and flag:
            return 100
        else:
            return -25
        #return -1

    def reward_rod_3(self, data, state, next_state, action):
        """
        Reward function for rod 3 (attack).
        Rewards the agent if the ball stops within the target area for rod 3.
        """

        _, _, vx, vy, flag = self.data_process(data)

        if self.is_ball_in_target_area(next_state, rod_idx=3) and flag:
            return 100
        else:
            return -25
        #return -1

    def is_ball_in_target_area(self, state, rod_idx):
        """
        Check if the ball is in the target area for the given rod.
        Each rod has a predefined target area where the ball should stop.
        """

        # Reward belt is +-40mm off the rod position
        # Reach of each rod is +-50mm
        # Red rods are at postions 80, 230, 530, 830 
        target_areas = {
            0: (40, 120),  # Example target area for rod 0
            1: (190, 270 ),  # Example target area for rod 1
            2: (490, 570),  # Example target area for rod 2
            3: (790, 870),  # Example target area for rod 3
        }
        lower_bound, upper_bound = target_areas[rod_idx]
        return lower_bound <= state[0] <= upper_bound

    def calculate_reward(self, data, state, next_state, action, rod_idx):
        """
        Calculate reward using the specific reward function for the rod.
        """
        reward_function = self.rod_reward_functions[rod_idx]
        return reward_function(data, state, next_state, action)

    def process_data(self, camera):
        """
        Process data and return dynamically decided commands.
        """
        commands = []
        rod_idx = random.choice([0, 1, 2, 3])  # Randomly select a rod to train

        # Get state from camera
        bx, by, vx, vy, _ = self.data_process(camera)
        state = np.array([bx, by, vx, vy])

        # RL decides the action
        action_idx = self.choose_action(state)

        # Next state (for now, assume it remains the same)
        next_state = state

        # ✅ Pass camera data to reward calculation
        reward = self.calculate_reward(camera, state, next_state, action_idx, rod_idx)

        # Episode completion condition
        done = reward == 100

        # Store experience
        self.remember(state, action_idx, reward, next_state, done)

        # Define dynamic actions
        action_map = {
            0: {"rotationTargetPosition": 0.5, "translationTargetPosition": 0.5},  # kick
            1: {"rotationTargetPosition": 0.0, "translationTargetPosition": max(0.0, bx / 1000)},  # move_left
            2: {"rotationTargetPosition": 0.0, "translationTargetPosition": min(1.0, bx / 1000)},  # move_right
            3: {"rotationTargetPosition": 0.0, "translationTargetPosition": 0.5},  # idle in the middle
        }

        # Get the mapped action parameters
        action_params = action_map.get(action_idx, {"rotationTargetPosition": 0, "translationTargetPosition": 0.5})

        # Dynamic motor command
        cmd = {
            'driveID': rod_idx + 1,
            'rotationTargetPosition': action_params["rotationTargetPosition"],
            'rotationVelocity': 1,
            'translationTargetPosition': action_params["translationTargetPosition"],
            'translationVelocity': 1.0
        }

        commands.append(cmd)

        # Train the model
        self.learn()

        return commands



if __name__ == "__main__":
    
    agent = BallControlAgent()
    episode = 0
    save_interval = 100

    try:
        while True:
            episode += 1
            time.sleep(0.02)

            # Get the current state of the camera
            cam_data = get_camera_state()

            # Process the camera data and get motor commands
            motor_cmds = agent.process_data(cam_data)

            # Send the motor commands to the simulator
            send_motor_commands({'commands': motor_cmds})

            # Save the model at regular intervals
            if episode % save_interval == 0:
                agent.save_model("ball_control_model.pth")

    except KeyboardInterrupt:
        print("Training interrupted. Saving last model...")
        agent.save_model("ball_control_model.pth")
