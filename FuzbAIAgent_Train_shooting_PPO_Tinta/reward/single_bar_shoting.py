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


def shot_heading_goal_reward(
    *,
    ball_kicked: bool,
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    goal_x: float = 1210.0,
    goal_y_center: float = 350.0,
    goal_width: float = 200.0,
    min_forward_vx: float = 0.05,
    reward_scale: float = 0.3,
    y_sigma: float = 120.0,
):
    """Reward a kicked ball whose projected path crosses near the goal mouth."""
    if not ball_kicked or float(ball_vx) <= float(min_forward_vx):
        return 0.0

    vx_mm_s = float(ball_vx) * 1000.0
    vy_mm_s = float(ball_vy) * 1000.0
    time_to_goal = (float(goal_x) - float(ball_x)) / vx_mm_s
    if time_to_goal <= 0.0:
        return 0.0

    predicted_goal_y = float(ball_y) + vy_mm_s * time_to_goal
    goal_y_min = float(goal_y_center) - float(goal_width) / 2.0
    goal_y_max = float(goal_y_center) + float(goal_width) / 2.0
    dist_to_goal_mouth = max(
        goal_y_min - predicted_goal_y,
        predicted_goal_y - goal_y_max,
        0.0,
    )
    sigma = max(float(y_sigma), 1e-6)
    return float(reward_scale) * math.exp(-((dist_to_goal_mouth / sigma) ** 2))


def kicking_reward(
    *,
    goal_scored: bool,
    ball_kicked: bool,
    ball_x: float,
    ball_y: float,
    forward_ball_vx: float,
    ball_vy: float,
    missed_kick: bool = False,
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
    - dense reward if the kicked ball trajectory is aimed at the goal mouth
    - penalty for backward ball velocity after contact
    - penalty if a kicked ball ends the episode without scoring
    - penalty for timeout without a useful kick
    """
    # --- Ball velocity rewards for kicking
    forward_velocity = max(0.0, float(forward_ball_vx)) if ball_kicked else 0.0
    backward_velocity = max(0.0, -float(forward_ball_vx)) if ball_kicked else 0.0
    shot_heading_goal = shot_heading_goal_reward(
        ball_kicked=ball_kicked,
        ball_x=ball_x,
        ball_y=ball_y,
        ball_vx=forward_ball_vx,
        ball_vy=ball_vy,
    )

    # --- Rod angle rewards for rotating the row for better kicking
    angle_error = float(rod_angle) - float(target_rod_angle)
    angle_sigma = max(float(rod_angle_sigma), 1e-6)
    rod_angle_reward = float(rod_angle_reward_scale) * math.exp(-((angle_error / angle_sigma) ** 2))

    reward_breakdown = {
        "goal_scored": 10.0 if goal_scored else 0.0,
        "time_penalty": float(time_penalty),
        #"ball_kick": 0.25 if ball_kicked else 0.0,
        #"forward_velocity": 0.3 * forward_velocity,
        #"backward_velocity": -0.2 * backward_velocity,
        "shot_heading_goal": shot_heading_goal,
        "missed_kick": -1.0 if missed_kick else 0.0,
        "rod_alignment": 0.5 *float(rod_alignment_reward),
        #"rod_angle": rod_angle_reward,
        "timeout": -0.2 if episode_timeout else 0.0,
    }
    reward = float(sum(reward_breakdown.values()))
    #print(f"Reward: {reward:.4f}, target angle rod: {target_rod_angle:.4f}, rod angle: {rod_angle:.4f}, angle error: {angle_error:.4f}, angle reward: {rod_angle_reward:.4f}")
    return reward, reward_breakdown

def maintaining_ball(
    *,
    ball_x: float,
    ball_vx: float,
    ball_vz: float,
    rod_x_pos: float,
    threshold_crossed: bool = False,
    ball_behind_rod: bool = False,
    prev_ball_vx: float | None = None,
    prev_ball_vz: float | None = None,
    episode_timeout: bool = False,
    time_penalty: float = -0.001,
    x_zone_radius: float = 89.0,
    x_sigma: float = 90.0,
    speed_sigma: float = 0.20,
    stop_speed: float = 0.08,
    threshold_penalty: float = -10.0,
    behind_rod_penalty: float = -0.25,
    rod_angle_reward: float = 0.0
):
    """Reward for receiving and keeping the ball near a rod.

    Position units are millimeters and velocity units are meters/second.
    ``ball_z``/``rod_z_pos`` are the lateral table coordinate; pass ball_y here
    if the caller uses x/y naming. Speed reduction is rewarded only inside the
    control zone, so slowing the ball elsewhere is not useful to the policy.
    ``ball_behind_rod`` should mean the ball crossed the protected side of the
    rod, but has not yet hit the episode termination threshold.
    """
    dx = abs(float(ball_x) - float(rod_x_pos))
    speed = math.sqrt(float(ball_vx) ** 2 + float(ball_vz) ** 2)

    x_control = math.exp(-((dx / max(float(x_sigma), 1e-6)) ** 2))
    zone_control = x_control

    in_control_zone = dx <= float(x_zone_radius) 
    low_speed = math.exp(-((speed / max(float(speed_sigma), 1e-6)) ** 2))
    speed_reduction = 0.0
    if prev_ball_vx is not None and prev_ball_vz is not None:
        prev_speed = math.sqrt(float(prev_ball_vx) ** 2 + float(prev_ball_vz) ** 2)
        speed_reduction = max(0.0, prev_speed - speed)

    stopped = in_control_zone and speed <= float(stop_speed)

    reward_breakdown = {
        "time_penalty": float(time_penalty),
        "threshold_failure": float(threshold_penalty) if threshold_crossed else 0.0,
        "behind_rod": float(behind_rod_penalty) if ball_behind_rod else 0.0,
        "zone_position": 0.03 * zone_control,
        "kept_in_front": 0.08 * zone_control if not ball_behind_rod else 0.0,
        "speed_reduction": 0.7 * speed_reduction * zone_control if in_control_zone else 0.0,
        "controlled_ball": 0.60 * zone_control * low_speed if in_control_zone else 0.0,
        "stopped_ball": 12.5 * zone_control if stopped else 0.0,
        "timeout": -0.2 if episode_timeout else 0.0,
        "rod_angle": 0.25 * rod_angle_reward
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
):
    """Dense reward for moving the correct controlled-rod player behind the ball."""
    rod_y_base = float(rod_pos_calib) * float(rod_info["travel"])
    player_index, _, _ = player_alignment_target(ball_y=ball_y, rod_info=rod_info)
    player_offset = float(rod_info["first_offset"]) + player_index * float(rod_info["spacing"])
    alignment_dist_y = abs(float(ball_y) - (rod_y_base + player_offset))

    # Unlike a narrow Gaussian, this keeps a useful gradient even when the rod
    # starts near the wrong player branch.
    y_alignment = 1.0 - alignment_dist_y / float(rod_info["travel"])

    return float(reward_scale * y_alignment)

def calculate_rod_angle_reward(
    *,
    rod_angle: float,
    target_rod_angle: float = 0.40,
    reward_scale: float = 0.6,
    angle_sigma: float = 0.5,
):
    """Reward for rotating the rod to a specific angle."""
    angle_error = float(rod_angle) - float(target_rod_angle)
    angle_reward = float(reward_scale) * math.exp(-((angle_error / max(float(angle_sigma), 1e-6)) ** 2))
    return angle_reward
