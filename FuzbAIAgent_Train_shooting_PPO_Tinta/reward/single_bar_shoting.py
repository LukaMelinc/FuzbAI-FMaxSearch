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


def kicking_reward(
    *,
    ball_kicked: bool,
    forward_ball_vx: float,
    episode_timeout: bool = False,
    time_penalty: float = -0.001,
):
    """Reward for teaching the rod to kick the ball forward.

    Arguments use the agent's forward direction. If the raw simulator/camera x
    axis points backward for the controlled rod, multiply vx/dx by -1 before
    calling this function.

    Reward spec:
    - small negative reward every step, so faster kicks are preferred
    - contact reward when the watched rod touches the ball
    - positive reward for forward ball velocity after contact
    - penalty for backward ball velocity after contact
    - penalty for timeout without a useful kick
    """
    forward_velocity = max(0.0, float(forward_ball_vx))
    backward_velocity = max(0.0, -float(forward_ball_vx))

    reward_breakdown = {
        "time_penalty": float(time_penalty),
        "ball_kick": 0.2 if ball_kicked else 0.0,
        "forward_velocity": 0.5 * forward_velocity,
        "backward_velocity": -0.3 * backward_velocity,
        "timeout": -0.3 if episode_timeout else 0.0,
    }
    reward = float(sum(reward_breakdown.values()))
    return reward, reward_breakdown
