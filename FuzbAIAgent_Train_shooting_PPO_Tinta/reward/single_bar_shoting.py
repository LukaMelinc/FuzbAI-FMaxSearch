import math


def player_alignment_target(*, ball_y: float, rod_info: dict):
    """Return the player and rod position that can best cover ``ball_y``.

    Selecting the player from the ball position, rather than from the rod's
    current position, gives the policy one continuous target instead of three
    competing local reward maxima.
    """
    travel = float(rod_info["travel"])
    first_offset = float(rod_info["first_offset"])
    spacing = float(rod_info["spacing"])

    candidates = []
    for player_index in range(int(rod_info["players"])):
        player_offset = first_offset + player_index * spacing
        target_rod_pos = min(1.0, max(0.0, (float(ball_y) - player_offset) / travel))
        reachable_player_y = target_rod_pos * travel + player_offset
        unavoidable_dist = abs(float(ball_y) - reachable_player_y)
        candidates.append((unavoidable_dist, player_index, target_rod_pos))

    unavoidable_dist, player_index, target_rod_pos = min(candidates)
    return int(player_index), float(target_rod_pos), float(unavoidable_dist)


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
    rod_angle: float = 0.0,
    target_rod_angle: float = 0.40,
    rod_angle_reward_scale: float = 0.006,
    rod_angle_sigma: float = 0.5,
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
    # --- Ball velocity rewards for kicking
    forward_velocity = max(0.0, float(forward_ball_vx)) if ball_kicked else 0.0
    backward_velocity = max(0.0, -float(forward_ball_vx)) if ball_kicked else 0.0

    # --- Rod angle rewards for rotating the row for better kicking
    angle_error = float(rod_angle) - float(target_rod_angle)
    angle_sigma = max(float(rod_angle_sigma), 1e-6)
    rod_angle_reward = float(rod_angle_reward_scale) * math.exp(-((angle_error / angle_sigma) ** 2))

    reward_breakdown = {
        "time_penalty": float(time_penalty),
        "ball_kick": 0.2 if ball_kicked else 0.0,
        "forward_velocity": 0.5 * forward_velocity,
        "backward_velocity": -0.3 * backward_velocity,
        "rod_alignment": 0.7 *float(rod_alignment_reward),
        #"rod_angle": rod_angle_reward,
        "timeout": -0.3 if episode_timeout else 0.0,
    }
    reward = float(sum(reward_breakdown.values()))
    #print(f"Reward: {reward:.4f}, target angle rod: {target_rod_angle:.4f}, rod angle: {rod_angle:.4f}, angle error: {angle_error:.4f}, angle reward: {rod_angle_reward:.4f}")
    return reward, reward_breakdown


def closest_player_alignment_reward(
    *,
    ball_x: float,
    ball_y: float,
    rod_pos_calib: float,
    rod_info: dict,
    reward_scale: float = 0.03,
    x_sigma: float = 120.0,
):
    """Dense reward for moving the correct controlled-rod player behind the ball."""
    rod_y_base = float(rod_pos_calib) * float(rod_info["travel"])
    player_index, _, _ = player_alignment_target(ball_y=ball_y, rod_info=rod_info)
    player_offset = float(rod_info["first_offset"]) + player_index * float(rod_info["spacing"])
    alignment_dist_y = abs(float(ball_y) - (rod_y_base + player_offset))
    dist_x = abs(float(ball_x) - float(rod_info["position"]))

    # Unlike a narrow Gaussian, this keeps a useful gradient even when the rod
    # starts near the wrong player branch.
    y_alignment = 1.0 - alignment_dist_y / float(rod_info["travel"])
    x_gate = math.exp(-((dist_x / x_sigma) ** 2))

    return float(reward_scale * x_gate * y_alignment)
