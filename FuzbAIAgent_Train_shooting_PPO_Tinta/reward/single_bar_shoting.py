import math


def passer_calm_reward(*, control_state, sample_time, rod_angle_radians,
                       rotation_target, ball_x, ball_vx, passer_x,
                       ball_kicked, release_distance_mm=50.0,
                       release_grace_seconds=0.1, motion_scale=0.05,
                       command_change_scale=0.01):
    """Penalize post-release motion and changes of the executed rotation target.

    Angles are unwrapped physical radians; rotation_target is the policy's
    [-1, 1] action (mapped to [-pi, pi] by the motor controller). The mutable
    state belongs to one episode. Integrate squared rates over elapsed time.
    """
    components = {"passer_rotation_motion": 0.0, "passer_rotation_command_change": 0.0}
    previous = control_state.get("previous")
    control_state["previous"] = (sample_time, rod_angle_radians, rotation_target)
    if ball_kicked and ball_vx > 0.08:
        control_state["forward_contact"] = True
    if (control_state.get("forward_contact") and "active_after" not in control_state
            and ball_x >= passer_x + release_distance_mm and ball_vx > 0.08):
        control_state["active_after"] = sample_time + release_grace_seconds
    if previous is None or "active_after" not in control_state:
        return components
    old_time, old_angle, old_target = previous
    dt = sample_time - old_time
    active_dt = max(0.0, sample_time - max(old_time, control_state["active_after"]))
    if dt <= 0 or active_dt <= 0:
        return components
    angular_rate = (rod_angle_radians - old_angle) / dt / math.pi
    components["passer_rotation_motion"] = -motion_scale * angular_rate ** 2 * active_dt
    if rotation_target is not None and old_target is not None:
        target_rate = (rotation_target - old_target) / dt
        components["passer_rotation_command_change"] = -command_change_scale * target_rate ** 2 * active_dt
    return components


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
    force_sigma: float = 5.0,
    reward_scale: float = 0.35,
):
    """Reward a kick based on how close the measured contact force is to target."""
    if not ball_kicked or kick_normal_force is None:
        return 0.0

    force = max(0.0, float(kick_normal_force))
    force_error = force - float(desired_kick_force)
    sigma = max(float(force_sigma), 1e-6)
    return float(reward_scale) * math.exp(-((force_error / sigma) ** 2))

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
    control_state: dict | None = None,
    sample_time: float | None = None,
    hold_duration: float = 0.3,
    hold_reward: float = 0.2,
    acceleration_scale: float = 1.0,
    rebound_scale: float = 2.0,
):
    """Reward for receiving and keeping the ball near a rod.

    Position units are millimeters and velocity units are meters/second.
    ``ball_vz`` is the lateral velocity; pass ball_vy here if appropriate.
    Speed reduction is rewarded only inside the
    control zone, so slowing the ball elsewhere is not useful to the policy.
    ``player_alignment_reward`` provides light dense shaping toward the
    closest reachable player-ball alignment.
    ``ball_behind_rod`` should mean the ball crossed the protected side of the
    rod, but has not yet hit the episode termination threshold.
    Pass a persistent ``control_state`` dictionary and simulation ``sample_time``
    in seconds to enable sustained holding and rebound-exit detection. Clear
    the dictionary on episode reset. Without timing state no hold reward is paid.
    Holding is allowed on either side of the rod. A rebound means returning
    through the entry side of the zone faster than ``stop_speed``.
    """

    dx = abs(float(ball_x) - float(rod_x_pos))
    speed = math.sqrt(float(ball_vx) ** 2 + float(ball_vz) ** 2)

    x_control = math.exp(-((dx / max(float(x_sigma), 1e-6)) ** 2))
    zone_control = x_control

    in_control_zone = dx <= float(x_zone_radius) 
    low_speed = math.exp(-((speed / max(float(speed_sigma), 1e-6)) ** 2))
    speed_reduction = 0.0
    speed_increase = 0.0
    if prev_ball_vx is not None and prev_ball_vz is not None:
        prev_speed = math.sqrt(float(prev_ball_vx) ** 2 + float(prev_ball_vz) ** 2)
        speed_reduction = max(0.0, prev_speed - speed)
        speed_increase = max(0.0, speed - prev_speed)

    stopped = in_control_zone and speed <= float(stop_speed) and not threshold_crossed
    held = False
    rebound = 0.0
    previously_in_zone = False
    if control_state is not None:
        previously_in_zone = control_state.get("in_zone", False)
        offset = float(ball_x) - float(rod_x_pos)
        if in_control_zone and not previously_in_zone:
            # Incoming vx identifies the entry side even if one sample crosses
            # the rod. Fall back to position when there is no incoming motion.
            incoming_vx = prev_ball_vx if prev_ball_vx is not None else ball_vx
            control_state["entry_side"] = (
                -math.copysign(1.0, incoming_vx) if abs(incoming_vx) > stop_speed
                else math.copysign(1.0, offset)
            )
        if previously_in_zone and not in_control_zone:
            side = control_state.get("entry_side", 0.0)
            if offset * side > 0 and float(ball_vx) * side > stop_speed:
                rebound = -float(rebound_scale) * abs(float(ball_vx))
        if stopped and sample_time is not None:
            now = float(sample_time)
            if now < control_state.get("last_time", now):
                control_state.pop("hold_start", None)
            start = control_state.setdefault("hold_start", now)
            held = now - start >= float(hold_duration) - 1e-9
            control_state["last_time"] = now
        else:
            control_state.pop("hold_start", None)
        control_state["in_zone"] = in_control_zone

    reward_breakdown = {
        "time_penalty": float(time_penalty),
        "threshold_failure": float(threshold_penalty) if threshold_crossed else 0.0,
        "behind_rod": float(behind_rod_penalty) if ball_behind_rod else 0.0,
        "zone_position": 0.03 * zone_control,
        "kept_in_front": 0.08 * zone_control if not ball_behind_rod else 0.0,
        "speed_reduction": 0.7 * speed_reduction * zone_control if in_control_zone else 0.0,
        "acceleration": -float(acceleration_scale) * speed_increase if (in_control_zone or previously_in_zone) else 0.0,
        "rebound": rebound,
        "controlled_ball": 0.60 * zone_control * low_speed if in_control_zone else 0.0,
        "stopped_ball": float(hold_reward) * zone_control if held else 0.0,
        "timeout": -0.2 if episode_timeout else 0.0,
        #"rod_angle": 0.25 * rod_angle_reward,
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

def predictive_alignment_target_y(
    *,
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    rod_x: float,
    vx_threshold: float = 0.02,
    t_max: float = 3.0,
    projection_state: dict | None = None,
    episode_sample: int = 0,
    episode_samples: int | None = None,
):
    """Return the crossing target shared by predictive observations and reward.

    - Uses a linear extrapolation of the ball to predict its y when it crosses
      the rod x position. Units: `ball_x`/`rod_x` are millimetres, `ball_vx`/
      `ball_vy` are metres/second (consistent with other helpers here).
    - If lateral (x) speed is small or predicted crossing time is negative or
      beyond `t_max`, falls back to the current `ball_y`.
    - Optional projection_state records the latest result for observations and
      diagnostics; it never holds a target across samples. Episode arguments
      remain accepted for compatibility with existing callers.
    """
    target_y = float(ball_y)
    vx = float(ball_vx)
    if math.isfinite(vx) and abs(vx) >= float(vx_threshold) and abs(vx) > 1e-12:
        t_cross = (float(rod_x) - float(ball_x)) / (vx * 1000.0)
        if 0.0 < t_cross <= float(t_max):
            candidate = float(ball_y) + float(ball_vy) * 1000.0 * t_cross
            if math.isfinite(candidate):
                target_y = candidate
    if projection_state is not None:
        projection_state.pop("refreshed", None)
        projection_state.update(target_y=target_y, last_sample=episode_sample)
    return target_y


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
    t_max: float = 3.0,
    reward_scale: float = 0.03,
    projection_state: dict | None = None,
    episode_sample: int = 0,
    episode_samples: int | None = None,
    debug: bool = False,
):
    """Reward player alignment with the crossing Y recalculated from the current ball state."""
    target_y = predictive_alignment_target_y(
        ball_x=ball_x, ball_y=ball_y, ball_vx=ball_vx, ball_vy=ball_vy,
        rod_x=rod_x, vx_threshold=vx_threshold, t_max=t_max,
        projection_state=projection_state, episode_sample=episode_sample,
        episode_samples=episode_samples,
    )
    reward = closest_player_alignment_reward(
        ball_y=target_y, rod_pos_calib=rod_pos_calib, rod_info=rod_info, reward_scale=reward_scale
    )
    if debug:
        # Inspect the exact target used above without updating held state again.
        travel = float(rod_info["travel"])
        selected, target_pos, unavoidable = player_alignment_target(
            ball_y=target_y, rod_info=rod_info,
        )
        positions = [float(rod_pos_calib) * travel + float(rod_info["first_offset"])
                     + i * float(rod_info["spacing"])
                     for i in range(int(rod_info["players"]))]
        errors = [abs(target_y - y) for y in positions]
        crossing_time = ((float(rod_x) - float(ball_x)) / (float(ball_vx) * 1000.0)
                         if abs(float(ball_vx)) > 1e-12 else None)
        live_y = (float(ball_y) + float(ball_vy) * 1000.0 * crossing_time
                  if crossing_time is not None and crossing_time > 0 else None)
        def fmt(value):
            return "none" if value is None else f"{value:.3f}"
        players = "; ".join(
            f"P{i + 1}:y={y:.2f},error={errors[i]:.2f}mm"
            for i, y in enumerate(positions))
        print(
            f"[alignment-debug]  "
            f"ball=({ball_x:.2f},{ball_y:.2f})mm "
            f"velocity=({ball_vx:.3f},{ball_vy:.3f})m/s "
            f"crossing_time={fmt(crossing_time)}s live_y={fmt(live_y)}mm "
            f"reward_target_y={target_y:.3f}mm held={projection_state is not None} "
            f"rod_pos={rod_pos_calib:.5f} target_pos={target_pos:.5f} "
            f"selected=P{selected + 1} unavoidable={unavoidable:.2f}mm "
            f"[{players}] reward={reward:.6f} "
        )
    return reward

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
    
    #if abs(float(rod_angle)) == float(rotation_buffer_def):
    #        angle_error = 0
    
    elif abs(float(rod_angle)) > float(rotation_buffer_def):
        angle_error = float(rod_angle) - float(target_rod_angle)

    #elif (float(rod_angle)) > float(rotation_buffer_def) or (float(rod_angle)) < float(rotation_buffer_def):
    #        angle_error = float(rod_angle) - float(target_rod_angle)

    angle_reward = float(reward_scale) * math.exp(-((angle_error / max(float(angle_sigma), 1e-6)) ** 2))
    return angle_reward


def receiving_reward(
    *, ball_x, ball_y, ball_vx, ball_vy, rod_info, rod_pos_calib,
    receiver_contact, control_state, sample_time,
    threshold_crossed=False, episode_timeout=False,
    stop_speed=0.08, hold_duration=0.3,
):
    """Contact-gated reception: mm, m/s, seconds. Reset state each episode.

    Dense terms are rates per second; speed changes and hold are event rewards.
    """
    now = float(sample_time)
    dt = max(0.0, now - control_state.get("last_time", now))
    speed = math.hypot(ball_vx, ball_vy)
    player_y = [rod_pos_calib * rod_info["travel"] + rod_info["first_offset"]
                + i * rod_info["spacing"] for i in range(rod_info["players"])]
    reachable = (abs(ball_x - rod_info["position"]) <= 89.0
                 and min(abs(ball_y - y) for y in player_y) <= 35.0)
    # Remember the approach direction before impact, not the rebound direction.
    if not control_state.get("contacted") and not receiver_contact and abs(ball_vx) > stop_speed:
        control_state["incoming_sign"] = math.copysign(1.0, ball_vx)
    if receiver_contact:
        control_state["contacted"] = True
        control_state.setdefault("incoming_sign", math.copysign(1.0, rod_info["position"] - ball_x))
    contacted = control_state.get("contacted", False)
    active = contacted and reachable and not threshold_crossed
    # Signed reduction makes speeding up costly as well as rewarding slowing down.
    deceleration = control_state.get("speed", speed) - speed if active else 0.0
    backwards = max(0.0, -ball_vx * control_state.get("incoming_sign", 0.0) - stop_speed)
    held = False
    if active and speed <= stop_speed:
        start = control_state.setdefault("hold_start", now)
        held = now - start >= hold_duration - 1e-9
    else:
        control_state.pop("hold_start", None)
    first_hold = held and not control_state.get("hold_paid", False)
    if first_hold:
        control_state["hold_paid"] = True
    escaped = contacted and not reachable and control_state.get("reachable", False)
    terms = {
        "time": -0.01 * dt,
        "deceleration": deceleration,
        "rebound": -4.0 * backwards * dt if contacted else 0.0,
        "controlled": 0.5 * math.exp(-((speed / 0.20) ** 2)) * dt if active else 0.0,
        "hold": 5.0 if first_hold else 0.0,
        "escape": -2.0 if escaped and not control_state.get("escape_paid", False) else 0.0,
        "failure": -5.0 if threshold_crossed else 0.0,
        "timeout": -0.5 if episode_timeout and not control_state.get("hold_paid", False) else 0.0,
    }
    if escaped:
        control_state["escape_paid"] = True
    control_state.update(last_time=now, speed=speed, reachable=reachable)
    return float(sum(terms.values())), terms
