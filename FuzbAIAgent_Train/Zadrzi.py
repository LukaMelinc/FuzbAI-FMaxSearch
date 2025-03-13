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
    def __init__(self, input_dim, output_dim = 4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Ensure output is between -1 and 1

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
        self.last_reward = None
        self.total_reward = 0

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
        state = torch.FloatTensor(state).unsqueeze(0)
        if np.random.rand() <= self.epsilon:
            # Random continuous actions for exploration in range [-1, 1]
            return np.random.uniform(-1, 1, 4)
        else:
            # NN decides continuous actions
            with torch.no_grad():
                return self.model(state).squeeze(0).numpy()


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
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q-values
        q_values = self.model(states)

        # Predicted Q-values for the next state
        next_q_values = self.target_model(next_states).detach()

        # Target Q-values
        target_q_values = rewards + (self.gamma * next_q_values.max(1)[0].unsqueeze(1) * (1 - dones))

        # Compute loss using MSE
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
            0: (0.040, 0.120),  # Example target area for rod 0
            1: (0.190, 0.270),  # Example target area for rod 1
            2: (0.490, 0.570),  # Example target area for rod 2
            3: (0.790, 0.870),  # Example target area for rod 3
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

        # Predict continuous action values in range [-1, 1]
        action_values = self.choose_action(state)

        # Scale actions appropriately (no longer constrained to [0, 1])
        rotation_target = action_values[0]  # Already in [-1, 1]
        rotation_velocity = (action_values[1] + 1) * 1.0  # Convert [-1, 1] to [0, 2]
        translation_target = (action_values[2] + 1) * 0.5  # Convert [-1, 1] to [0, 1]
        translation_velocity = (action_values[3] + 1) * 1.0  # Convert [-1, 1] to [0, 2]

        # Next state (for now, assume it remains the same)
        next_state = state

        # Calculate reward
        reward = self.calculate_reward(camera, state, next_state, action_values, rod_idx)
        self.last_reward = reward
        self.total_reward += reward

        print(f"Earned reward: {reward}, Total accumulated reward: {self.total_reward}")

        # Check if the episode is done
        done = reward == 100

        # Store experience
        self.remember(state, action_values, reward, next_state, done)

        # Dynamic motor command
        cmd = {
            'driveID': rod_idx + 1,
            'rotationTargetPosition': rotation_target,
            'rotationVelocity': rotation_velocity,
            'translationTargetPosition': translation_target,
            'translationVelocity': translation_velocity
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
                print(f"Model Saved!")

    except KeyboardInterrupt:
        print("Training interrupted. Saving last model...")
        agent.save_model("ball_control_model.pth")
