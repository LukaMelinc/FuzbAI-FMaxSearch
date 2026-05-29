import math


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
    rod_alignment_reward: float = 0.0,
):
    """Reward for teaching the rod to kick the ball forward.

    Arguments use the agent's forward direction. If the raw simulator/camera x
    axis points backward for the controlled rod, multiply vx/dx by -1 before
    calling this function.

    Reward spec:
    - small negative reward every step, so faster kicks are preferred
    - small dense reward when a controlled-rod player is aligned with the ball
    - contact reward when the watched rod touches the ball
    - positive reward for forward ball velocity after contact
    - penalty for backward ball velocity after contact
    - penalty for timeout without a useful kick
    """
    forward_velocity = max(0.0, float(forward_ball_vx)) if ball_kicked else 0.0
    backward_velocity = max(0.0, -float(forward_ball_vx)) if ball_kicked else 0.0

    reward_breakdown = {
        "time_penalty": float(time_penalty),
        "ball_kick": 0.2 if ball_kicked else 0.0,
        "forward_velocity": 0.5 * forward_velocity,
        "backward_velocity": -0.3 * backward_velocity,
        "rod_alignment": float(rod_alignment_reward),
        "timeout": -0.3 if episode_timeout else 0.0,
    }
    reward = float(sum(reward_breakdown.values()))
    return reward, reward_breakdown


def closest_player_alignment_reward(
    *,
    ball_x: float,
    ball_y: float,
    rod_pos_calib: float,
    rod_info: dict,
    reward_scale: float = 0.03,
    y_sigma: float = 45.0,
    x_sigma: float = 120.0,
):
    """Dense reward for positioning one controlled-rod player behind the ball.

    The closest player is determined by checking every player on the rod. The
    x gate prevents the agent from receiving much reward when the ball is far
    away from this rod's kicking lane.
    """
    rod_y_base = float(rod_pos_calib) * float(rod_info["travel"])
    closest_dist_y = min(
        abs(float(ball_y) - (rod_y_base + float(rod_info["first_offset"]) + i * float(rod_info["spacing"])))
        for i in range(int(rod_info["players"]))
    )
    dist_x = abs(float(ball_x) - float(rod_info["position"]))

    y_alignment = math.exp(-((closest_dist_y / y_sigma) ** 2))
    x_gate = math.exp(-((dist_x / x_sigma) ** 2))

    return float(reward_scale * x_gate * y_alignment)
