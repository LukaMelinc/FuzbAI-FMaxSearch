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

def kick_force_reward(
    *,
    ball_kicked: bool,
    kick_normal_force: float | None,
    desired_kick_force: float = 5.0,
    force_sigma: float = 8.0,
    reward_scale: float = 0.35,
):
    """Reward a kick based on how close the measured contact force is to target."""
    if not ball_kicked or kick_normal_force is None:
        return 0.0

    force = max(0.0, float(kick_normal_force))
    force_error = force - float(desired_kick_force)
    sigma = max(float(force_sigma), 1e-6)
    return float(reward_scale) * math.exp(-((force_error / sigma) ** 2))

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

def forward_backward_kick_reward(
    *,
    ball_kicked: bool,
    forward_ball_vx: float,
    forward_velocity_scale: float = 0.25,
    backward_velocity_scale: float = 0.25,
):
    """Simple reward for teaching the agent to kick the ball forward.

    This helper only looks at the ball's x velocity after a confirmed kick:
    - positive x velocity gets a positive reward
    - negative x velocity gets a negative penalty
    """
    # No kick means no direction reward.
    if not ball_kicked:
        return 0.0, {"forward_velocity": 0.0, "backward_velocity": 0.0}

    # Split the x velocity into a forward part and a backward part.
    forward_velocity = max(0.0, float(forward_ball_vx))
    backward_velocity = max(0.0, -float(forward_ball_vx))

    reward_breakdown = {
        "forward_velocity": float(forward_velocity_scale) * forward_velocity,
        "backward_velocity": -float(backward_velocity_scale) * backward_velocity,
    }
    reward = float(sum(reward_breakdown.values()))
    return reward, reward_breakdown

def kicking_reward(
    *,
    #goal_scored: bool,
    ball_kicked: bool,
    #kick_normal_force: float | None = None,
    #ball_x: float,
    #ball_y: float,
    forward_ball_vx: float,
    #ball_vy: float,
    #missed_kick: bool = False,
    #episode_timeout: bool = False,
    #time_penalty: float = -0.001,
    #rod_alignment_reward: float = 0.0,
    rod_angle: float = 0.0,
    target_rod_angle: float = 0.40,
    rod_angle_reward_scale: float = 0.006,
    rod_angle_sigma: float = 0.5,
    desired_kick_force: float = 5.0,
    kick_force_sigma: float = 8.0,
    kick_force_reward_scale: float = 0.35,
    forward_velocity_scale: float = 0.25,
    backward_velocity_scale: float = 0.25,
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
    """shot_heading_goal = shot_heading_goal_reward(
        ball_kicked=ball_kicked,
        ball_x=ball_x,
        ball_y=ball_y,
        ball_vx=forward_ball_vx,
        ball_vy=ball_vy,
    )"""
    """force_reward = kick_force_reward(
        ball_kicked=ball_kicked,
        kick_normal_force=kick_normal_force,
        desired_kick_force=desired_kick_force,
        force_sigma=kick_force_sigma,
        reward_scale=kick_force_reward_scale,
    )"""

    #if ball_kicked:
    #    print(f"Ball kicked, forward vx: {forward_velocity:.4f}, backward vx: {backward_velocity:.4f}")

    # --- Rod angle rewards for rotating the row for better kicking
    #angle_error = float(rod_angle) - float(target_rod_angle)
    #angle_sigma = max(float(rod_angle_sigma), 1e-6)
    #rod_angle_reward = float(rod_angle_reward_scale) * math.exp(-((angle_error / angle_sigma) ** 2))

    reward_breakdown = {
        #"goal_scored": 10.0 if goal_scored else 0.0,
        #"time_penalty": float(time_penalty),
        #"ball_kick": 0.25 if ball_kicked else 0.0,
        "forward_velocity": float(forward_velocity_scale) * forward_velocity,
        "backward_velocity": -float(backward_velocity_scale) * backward_velocity,
        #"shot_heading_goal": shot_heading_goal,
        #"kick_force": force_reward,
        #"missed_kick": -1.0 if missed_kick else 0.0,
        #"rod_alignment": 0.5 *float(rod_alignment_reward),
        #"rod_angle": rod_angle_reward,
        #"timeout": -0.2 if episode_timeout else 0.0,
    }
    reward = float(sum(reward_breakdown.values()))
    #print(f"Reward: {reward:.4f}, target angle rod: {target_rod_angle:.4f}, rod angle: {rod_angle:.4f}, angle error: {angle_error:.4f}, angle reward: {rod_angle_reward:.4f}")
    return reward, reward_breakdown

def controllable_kick_reward(
    *,
    ball_kicked: bool,
    kick_normal_force: float | None = None,
    ball_vx: float,
    ball_vy: float,
    rod_alignment_reward: float = 0.0,
    episode_timeout: bool = False,
    time_penalty: float = -0.002,
    min_forward_speed: float = 0.35,
    target_forward_speed: float = 1.00,
    max_good_angle_deg: float = 45.0,
    desired_kick_force: float = 5.0,
    kick_force_sigma: float = 8.0,
    kick_force_reward_scale: float = 0.35,
):
    """Reward one clean, controllable kick toward the receiving rod.

    Velocities are in m/s. In the current setup, positive x velocity is the
    desired forward kick direction.

    The large terms are evaluated *only on a confirmed player-ball contact*.
    Thus an already-moving spawned ball, or a ball that later bounces forward,
    cannot earn the kick reward.  A good kick is forward and reasonably
    straight: the angle reward is highest when vx dominates and vy is close to
    zero.
    """
    forward_speed = float(ball_vx)
    lateral_speed = abs(float(ball_vy))
    ball_speed = math.hypot(forward_speed, lateral_speed)

    # A speed below min_forward_speed is only a touch.  The quality rises to
    # one at target_forward_speed and then saturates: PPO has no incentive to
    # learn increasingly violent shots.
    target_span = max(float(target_forward_speed) - float(min_forward_speed), 1e-6)

    if ball_kicked:
        kicked_forward = forward_speed > 0.0
        kicked_backward = forward_speed < 0.0

        if ball_speed > 1e-6 and kicked_forward:
            travel_angle = math.atan2(lateral_speed, forward_speed)
            max_good_angle = math.radians(max(float(max_good_angle_deg), 1e-6))
            angle_quality = max(0.0, 1.0 - travel_angle / max_good_angle)
        else:
            angle_quality = 0.0

        speed_quality = max(
            0.0,
            min(1.0, (forward_speed - float(min_forward_speed)) / target_span),
        )

        # Reward only contacts that send the ball in the desired direction.
        # ``angle_reward`` is the main term: high vx and low |vy|.
        useful_contact = 0.20 if forward_speed >= float(min_forward_speed) else 0.0
        angle_reward = 1.00 * speed_quality * angle_quality
        forward_velocity = 0.25 * max(0.0, forward_speed)
        lateral_penalty = -0.25 * lateral_speed if kicked_forward else 0.0
        backward_kick_penalty = -1.25 if kicked_backward else 0.0
    else:
        kicked_forward = False
        kicked_backward = False
        useful_contact = 0.0
        angle_reward = 0.0
        forward_velocity = 0.0
        lateral_penalty = 0.0
        backward_kick_penalty = 0.0

    force_reward = kick_force_reward(
        ball_kicked=ball_kicked,
        kick_normal_force=kick_normal_force,
        desired_kick_force=desired_kick_force,
        force_sigma=kick_force_sigma,
        reward_scale=kick_force_reward_scale,
    )

    reward_breakdown = {
        "time_penalty": float(time_penalty),
        #"alignment": float(rod_alignment_reward),
        "useful_contact": useful_contact,
        "kick_force": force_reward,
        #"angle_reward": angle_reward,
        "forward_velocity": forward_velocity,
        #"lateral_penalty": lateral_penalty,
        "backward_kick": backward_kick_penalty,
        "kicked_forward": 0.05 if kicked_forward else 0.0,
        "timeout": -0.5 if episode_timeout else 0.0,
    }
    return float(sum(reward_breakdown.values())), reward_breakdown

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
    rod_angle_reward: float = 0.0,
    player_alignment_reward: float = 0.0,
):
    """Reward for receiving and keeping the ball near a rod.

    Position units are millimeters and velocity units are meters/second.
    ``ball_z``/``rod_z_pos`` are the lateral table coordinate; pass ball_y here
    if the caller uses x/y naming. Speed reduction is rewarded only inside the
    control zone, so slowing the ball elsewhere is not useful to the policy.
    ``player_alignment_reward`` provides light dense shaping toward the
    closest reachable player-ball alignment.
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
        "rod_angle": 0.25 * rod_angle_reward,
        # This dense term tells the policy which player on the rod should
        # meet the ball.  Keep it small: the control/stop outcomes remain the
        # main objective rather than merely centering a player under the ball.
        "player_alignment": float(player_alignment_reward),
    }
    reward = float(sum(reward_breakdown.values()))

    return reward, reward_breakdown
    
def closest_player_alignment_reward(
    *,
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

def predictive_player_alignment_reward(
    *,
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    rod_x: float,
    rod_pos_calib: float,
    rod_info: dict,
    vx_threshold: float = 0.02,
    t_max: float = 1.0,
    reward_scale: float = 0.03,
):
    """Simple predictive alignment reward.

    - Uses a linear extrapolation of the ball to predict its y when it crosses
      the rod x position. Units: `ball_x`/`rod_x` are millimetres, `ball_vx`/
      `ball_vy` are metres/second (consistent with other helpers here).
    - If lateral (x) speed is small or predicted crossing time is negative or
      beyond `t_max`, falls back to `closest_player_alignment_reward` using
      the current `ball_y`.
    - Keeps the same reward scale semantics as `closest_player_alignment_reward`.
    """
    try:
        vx_mm_s = float(ball_vx) * 1000.0
    except Exception:
        return closest_player_alignment_reward(
            ball_y=ball_y, rod_pos_calib=rod_pos_calib, rod_info=rod_info, reward_scale=reward_scale
        )

    # If lateral velocity is too small, use the instantaneous alignment reward
    if abs(vx_mm_s) < float(vx_threshold) * 1000.0:
        return closest_player_alignment_reward(
            ball_y=ball_y, rod_pos_calib=rod_pos_calib, rod_info=rod_info, reward_scale=reward_scale
        )

    # Predict time to cross rod x (mm / (mm/s) = s)
    t_cross = (float(rod_x) - float(ball_x)) / vx_mm_s

    # If crossing is in the past or too far in the future, fall back
    if t_cross <= 0.0 or abs(t_cross) > float(t_max):
        return closest_player_alignment_reward(
            ball_y=ball_y, rod_pos_calib=rod_pos_calib, rod_info=rod_info, reward_scale=reward_scale
        )

    # Predict crossing y in mm
    vy_mm_s = float(ball_vy) * 1000.0
    predicted_y = float(ball_y) + vy_mm_s * t_cross

    # Reuse the existing closest-player calculation on the predicted y
    return closest_player_alignment_reward(
        ball_y=predicted_y, rod_pos_calib=rod_pos_calib, rod_info=rod_info, reward_scale=reward_scale
    )

def calculate_rod_angle_reward(
    *,
    rod_angle: float,
    target_rod_angle: float = 0.40,
    reward_scale: float = 0.6,
    rotation_buffer_def: float = 5, 
    angle_sigma: float = 9.0,
):
    """Reward for rotating the rod to a specific angle."""
    
    if abs(float(rod_angle)) <= float(rotation_buffer_def):
        angle_error = 0
    elif abs(float(rod_angle)) > float(rotation_buffer_def):
        angle_error = float(rod_angle) - float(target_rod_angle)


    angle_reward = float(reward_scale) * math.exp(-((angle_error / max(float(angle_sigma), 1e-6)) ** 2))
    return angle_reward
