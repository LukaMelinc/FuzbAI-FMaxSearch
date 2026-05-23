def simple_reward(*, goal_scored: bool, ball_kicked: bool, terminated_by_x_threshold: bool, rod_angle: float):
    """Event-based reward used for shooting PPO experiment.

    Reward spec (stack additively):
    - +10 if a goal is scored by the learning agent's team
    - +2 if a ball kick is detected (contact-based)
    - -1.5 if the episode terminated due to ball_x below threshold
    """
    reward_breakdown = {
        #"goal_scored": 1.0 if goal_scored else 0.0,
        #"ball_kick": 0.25 if ball_kicked else 0.0,
        #"x_threshold_termination": -0.15 if terminated_by_x_threshold else 0.0,
        "rod_angle": rod_angle
    }
    reward = float(sum(reward_breakdown.values()))
    return reward, reward_breakdown
