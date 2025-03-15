import time
import random
import torch
from Agents_Training.podaja import DQN
import requests
import json
import time
import math
import random
import torch

HOST_ADDRESS = '127.0.0.1:23336'

def get_camera_state():
    cam_url = f"http://{HOST_ADDRESS}/Camera/State"
    response = requests.get(cam_url)    
    return response.json()

def send_motor_commands(cmds):
    motors_url = f"http://{HOST_ADDRESS}/Motors/SendCommand?blue=False"
    response = requests.post(motors_url, json=cmds)

# Assuming you have already defined your agents like PassingAgent, ShootingAgent, DefendingAgent, etc.
class PassingAgent:
    def __init__(self, model_path):
        self.model = DQN(state_size=4, action_size=4)
        self.model.load_state_dict(torch.load(model_path))          # Loading the weights from the trained model
        self.model.eval()                                           # set to inference mode
        self.actions = ['kick', 'move_left', 'move_right', 'idle', 'pass']

    def process_data(self, camera_data):
        state = self.data_process(camera_data)      # extracting the usefull data from camera
        action_idx = self.choose_action(state)      # determening the best action
        action = self.actions[action_idx]           # selects the action
        cmd = self.create_command(action)           # creates a motor comand for the selected state
        return [cmd]

    def data_process(self, camera):
        # processes the data
        CD0 = camera["camData"][0]
        bx = CD0["ball_x"]
        by = CD0["ball_y"]
        vx = CD0["ball_vx"]
        vy = CD0["ball_vy"]
        return [bx, by, vx, vy]

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)

        # inference: performs a forward pass to get Q-values
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def create_command(self, action):
        return {
            'driveID': random.choice([1, 2, 3, 4]),
            'rotationTargetPosition': 0.5 if action == 'kick' else 0,
            'rotationVelocity': 1,
            'translationTargetPosition': 0.4 if action == 'move_left' else 0.6,
            'translationVelocity': 1.0
        }

class ShootingAgent:
    def __init__(self, model_path):
        self.model = DQN(state_size=4, action_size=4)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.actions = ['kick', 'move_left', 'move_right', 'idle', 'shoot']

    def process_data(self, camera_data):
        # Implement the logic to process data and return motor commands
        state = self.data_process(camera_data)
        action_idx = self.choose_action(state)
        action = self.actions[action_idx]
        cmd = self.create_command(action)
        return [cmd]

    def data_process(self, camera):
        CD0 = camera["camData"][0]
        bx = CD0["ball_x"]
        by = CD0["ball_y"]
        vx = CD0["ball_vx"]
        vy = CD0["ball_vy"]
        return [bx, by, vx, vy]

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def create_command(self, action):
        return {
            'driveID': random.choice([1, 2, 3, 4]),
            'rotationTargetPosition': 0.5 if action == 'kick' else 0,
            'rotationVelocity': 1,
            'translationTargetPosition': 0.4 if action == 'move_left' else 0.6,
            'translationVelocity': 1.0
        }

class DefendingAgent:
    def __init__(self, model_path):
        self.model = DQN(state_size=4, action_size=4)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.actions = ['kick', 'move_left', 'move_right', 'idle', 'defend']

    def process_data(self, camera_data):
        state = self.data_process(camera_data)
        action_idx = self.choose_action(state)
        action = self.actions[action_idx]
        cmd = self.create_command(action)
        return [cmd]

    def data_process(self, camera):
        CD0 = camera["camData"][0]
        bx = CD0["ball_x"]
        by = CD0["ball_y"]
        vx = CD0["ball_vx"]
        vy = CD0["ball_vy"]
        return [bx, by, vx, vy]

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def create_command(self, action):
        return {
            'driveID': random.choice([1, 2, 3, 4]),
            'rotationTargetPosition': 0.5 if action == 'kick' else 0,
            'rotationVelocity': 1,
            'translationTargetPosition': 0.4 if action == 'move_left' else 0.6,
            'translationVelocity': 1.0
        }
    

class ZadrziAgent:
    def __init__(self, model_path):
        self.model = DQN(state_size=4, action_size=4)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.actions = ['kick', 'move_left', 'move_right', 'idle', 'defend']

    def process_data(self, camera_data):
        state = self.data_process(camera_data)
        action_idx = self.choose_action(state)
        action = self.actions[action_idx]
        cmd = self.create_command(action)
        return [cmd]

    def data_process(self, camera):
        CD0 = camera["camData"][0]
        bx = CD0["ball_x"]
        by = CD0["ball_y"]
        vx = CD0["ball_vx"]
        vy = CD0["ball_vy"]
        return [bx, by, vx, vy]

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def create_command(self, action):
        return {
            'driveID': random.choice([1, 2, 3, 4]),
            'rotationTargetPosition': 0.5 if action == 'kick' else 0,
            'rotationVelocity': 1,
            'translationTargetPosition': 0.4 if action == 'move_left' else 0.6,
            'translationVelocity': 1.0
        }

class GameStateMachine:
    def __init__(self):
        self.state = 'IDLE'
        self.agents = {
            'PASSING': PassingAgent(model_path='passing_model.pth'),
            'SHOOTING': ShootingAgent(model_path='shooting_model.pth'),
            'DEFENDING': DefendingAgent(model_path='defending_model.pth'),
            'ZADRZI': ZadrziAgent(model_path=('zadrzi_model.pth'))
        }

    def transition_to_state(self, new_state):
        print(f"Transitioning from {self.state} to {new_state}")
        self.state = new_state

    def determine_state(self, camera_data):
        # Implement your logic to determine the state based on camera_data
        # For example, you can check the position of the ball, the positions of the players, etc.
        if self.is_ball_in_passing_position(camera_data):
            self.transition_to_state('PASSING')
        elif self.is_ball_in_shooting_position(camera_data):
            self.transition_to_state('SHOOTING')
        elif self.is_ball_in_defending_position(camera_data):
            self.transition_to_state('DEFENDING')
        else:
            self.transition_to_state('IDLE')

    def is_ball_in_passing_position(self, camera_data):
        # Implement your logic to check if the ball is in a passing position
        return True  # Placeholder

    def is_ball_in_shooting_position(self, camera_data):
        # Implement your logic to check if the ball is in a shooting position
        return False  # Placeholder

    def is_ball_in_defending_position(self, camera_data):
        # Implement your logic to check if the ball is in a defending position
        return False  # Placeholder
    
    def is_ball_in_zadrzi_position(self, camera_data):
        return False

    def process_data(self, camera_data):
        self.determine_state(camera_data)
        if self.state in self.agents:
            return self.agents[self.state].process_data(camera_data)
        else:
            return []

def get_camera_state():
    # Placeholder function to get the current state of the camera
    return {
        "camData": [
            {"ball_x": random.uniform(0, 1000), "ball_y": random.uniform(0, 1000), "ball_vx": random.uniform(0, 10), "ball_vy": random.uniform(0, 10)}
        ]
    }

def send_motor_commands(cmds):
    # Placeholder function to send motor commands
    print(f"Sending motor commands: {cmds}")

if __name__ == "__main__":
    state_machine = GameStateMachine()

    try:
        while True:
            time.sleep(0.02)

            # Get the current state of the camera
            cam_data = get_camera_state()

            # Process the camera data and get motor commands
            motor_cmds = state_machine.process_data(cam_data)

            # Send the motor commands to the simulator
            send_motor_commands({'commands': motor_cmds})

    except KeyboardInterrupt:
        print("Inference interrupted.")
