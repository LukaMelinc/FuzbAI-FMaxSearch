import random

class BallControlAgent(RLAgent):
    """RL Agent for training ball control for each rod."""
    def __init__(self):
        super().__init__()
        self.rod_reward_functions = {
            0: self.reward_rod_0,
            1: self.reward_rod_1,
            2: self.reward_rod_2,
            3: self.reward_rod_3,
        }

    def reward_rod_0(self, state, next_state, action):
        """Reward function for rod 0 (goalie)."""
        if self.is_ball_in_target_area(next_state, rod_idx=0):
            return 100
        return -1

    def reward_rod_1(self, state, next_state, action):
        """Reward function for rod 1 (defense)."""
        if self.is_ball_in_target_area(next_state, rod_idx=1):
            return 100
        return -1

    def reward_rod_2(self, state, next_state, action):
        """Reward function for rod 2 (midfield)."""
        if self.is_ball_in_target_area(next_state, rod_idx=2):
            return 100
        return -1

    def reward_rod_3(self, state, next_state, action):
        """Reward function for rod 3 (attack)."""
        if self.is_ball_in_target_area(next_state, rod_idx=3):
            return 100
        return -1

    def is_ball_in_target_area(self, state, rod_idx):
        """Check if the ball is in the target area for the given rod."""
        target_areas = {
            0: (0.45, 0.55),  # Example target area for rod 0
            1: (0.35, 0.45),  # Example target area for rod 1
            2: (0.25, 0.35),  # Example target area for rod 2
            3: (0.15, 0.25),  # Example target area for rod 3
        }
        lower_bound, upper_bound = target_areas[rod_idx]
        return lower_bound <= state[0] <= upper_bound

    def calculate_reward(self, state, next_state, action, rod_idx):
        """Calculate reward using the specific reward function for the rod."""
        return self.rod_reward_functions[rod_idx](state, next_state, action)

    def process_data(self, camera):
        """Process data and return commands for the rod being trained."""
        commands = []
        rod_idx = random.choice([0, 1, 2, 3])  # Randomly select a rod to train

        state = self.get_state(camera['camData'][0], rod_idx)
        action_idx = self.choose_action(state, rod_idx)
        next_state = self.get_state(camera['camData'][0], rod_idx)
        reward = self.calculate_reward(state, next_state, action_idx, rod_idx)

        self.remember(rod_idx, state, action_idx, reward, next_state)
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
    agent = BallControlAgent()
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
                agent.save_model("ball_control_model.pth")  # Save the model

    except KeyboardInterrupt:
        print("Training interrupted. Saving last model...")
        agent.save_model("ball_control_model.pth")
