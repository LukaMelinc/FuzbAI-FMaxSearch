import random

HOST_ADDRESS = '127.0.0.1:23336'

def get_camera_state():
    cam_url = f"http://{HOST_ADDRESS}/Camera/State"
    response = requests.get(cam_url)    
    return response.json()

def send_motor_commands(cmds):
    motors_url = f"http://{HOST_ADDRESS}/Motors/SendCommand?blue=False"
    response = requests.post(motors_url, json=cmds)

class BallControlAgent(RLAgent):
    """
    RL Agent for training ball control for each rod.
    This agent specializes in stopping the ball within a designated area for each rod.
    """
    def __init__(self):
        super().__init__()
        # Define separate reward functions for each rod
        self.rod_reward_functions = {
            0: self.reward_rod_0,
            1: self.reward_rod_1,
            2: self.reward_rod_2,
            3: self.reward_rod_3,
        }

    def data_process(self, camera):
        CD0 = camera["camData"][0]
        CD1 = camera["camData"][1]

        vx = CD0["ball_vx"]
        vy = CD0["ball_vy"]

        bx = CD0["ball_x"]
        by = CD0["ball_y"]

        flag = 1 if vx < 1 and vy < 1 else 0

        return bx, by, vx, vy, flag




    def reward_rod_0(self, state, next_state, action):
        """
        Reward function for rod 0 (goalie).
        Rewards the agent if the ball stops within the target area for rod 0.
        """

        camera = get_camera_state()
        _, _, vx, vy, flag = data_process(camera)

        

        if self.is_ball_in_target_area(next_state, rod_idx=0) and flag:
            return 100

        else:
            return -25
        #return -1

    def reward_rod_1(self, state, next_state, action):
        """
        Reward function for rod 1 (defense).
        Rewards the agent if the ball stops within the target area for rod 1.
        """

        camera = get_camera_state()
        _, _, vx, vy, flag = data_process(camera)

        if self.is_ball_in_target_area(next_state, rod_idx=1) and flag:
            return 100

        else:
            return -25
        #return -1

    def reward_rod_2(self, state, next_state, action):
        """
        Reward function for rod 2 (midfield).
        Rewards the agent if the ball stops within the target area for rod 2.
        """

        camera = get_camera_state()
        _, _, vx, vy, flag = data_process(camera)

        if self.is_ball_in_target_area(next_state, rod_idx=2) and flag:
            return 100
        else:
            return -25
        #return -1

    def reward_rod_3(self, state, next_state, action):
        """
        Reward function for rod 3 (attack).
        Rewards the agent if the ball stops within the target area for rod 3.
        """


        camera = get_camera_state()
        _, _, vx, vy, flag = data_process(camera)

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
        rod_idx = random.choice([0, 1, 2, 3])  # Randomly select a rod to train

        # Get the current state for the selected rod
        state = self.get_state(camera['camData'][0], rod_idx)

        # Choose an action based on the current state
        action_idx = self.choose_action(state, rod_idx)

        # Get the next state after taking the action
        next_state = self.get_state(camera['camData'][0], rod_idx)

        # Calculate the reward for the action taken
        reward = self.calculate_reward(state, next_state, action_idx, rod_idx)

        # Store the experience in the replay memory
        self.remember(rod_idx, state, action_idx, reward, next_state)

        # Map the action index to the actual action
        action = self.actions[action_idx]

        # Create a command for the motor based on the chosen action
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
