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

class PassingAgent:
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

        self.actions = ['kick', 'move_left', 'move_right', 'idle', 'pass']
        
        

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

        flag = 1 if vx < 1 and vy < 1 else 0        # TODO uredit enote hitrosti da bo meja ok

        return bx, by, vx, vy, flag




    def reward_pass(self, state, next_state, action, rod_idx):
        """
        Reward function for passing the ball between two
        consequtive rods.
        """

        # X is the long axis of the board

        camera = get_camera_state()
        _, _, vx, vy, flag = self.data_process(camera)

        if action == "pass" and self.is_ball_in_target_area(next_state, rod_idx + 1) and flag:
            return 100

        elif action == "pass" and not self.is_ball_in_target_area(next_state, rod_idx+1):
            return -100
        
        elif action == "pass" and self.is_ball_in_target_area(next_state, rod_idx + 1) and (vx < 1 and vy < 3):
            return 55

        else:
            return -1


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

    def calculate_reward(self, state, next_state, action, rod_idx):
        """
        Calculate reward using the specific reward function for the rod.
        This function delegates the reward calculation to the rod-specific reward function.
        """
        return self.rod_reward_functions[rod_idx](state, next_state, action)

    def process_data(self, camera):
        """
        Process data and return commands for the rod being trained.
        This function handles the main logic for interacting with the environment.
        """
        commands = []
        rod_idx = random.choice([0, 1, 2, 3])

        # Get state
        bx, by, vx, vy, _ = self.data_process(camera)
        state = np.array([bx, by, vx, vy])

        # Choose action
        action_idx = self.choose_action(state)

        # Define dummy next state for simplicity
        next_state = state  # You can refine this based on your environment

        # Calculate reward
        reward = self.calculate_reward(state, next_state, action_idx, rod_idx)

        # Check if the episode is done (define your own condition)
        done = reward == 100

        # Store experience
        self.remember(state, action_idx, reward, next_state, done)

        # Define action
        action = self.actions[action_idx]

        cmd = {
            'driveID': rod_idx + 1,
            'rotationTargetPosition': 0.5 if action == 'kick' else 0,
            'rotationVelocity': 1,
            'translationTargetPosition': 0.4 if action == 'move_left' else 0.6,
            'translationVelocity': 1.0
        }
        commands.append(cmd)
        

        # Train the model using the collected experiences
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
