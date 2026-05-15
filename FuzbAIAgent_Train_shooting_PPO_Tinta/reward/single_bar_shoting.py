def simple_reward(*, goal_scored: bool, ball_kicked: bool, terminated_by_x_threshold: bool):
    """Event-based reward used for shooting PPO experiment.

    Reward spec (stack additively):
    - +10 if a goal is scored by the learning agent's team
    - +2 if a ball kick is detected (contact-based)
    - -1.5 if the episode terminated due to ball_x below threshold
    """
    reward_breakdown = {
        "goal_scored": 10.0 if goal_scored else 0.0,
        "ball_kick": 2.0 if ball_kicked else 0.0,
        "x_threshold_termination": -1.5 if terminated_by_x_threshold else 0.0,
    }
    reward = float(sum(reward_breakdown.values()))
    return reward, reward_breakdown
