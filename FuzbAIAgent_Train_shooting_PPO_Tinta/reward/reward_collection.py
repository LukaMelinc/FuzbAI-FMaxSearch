# 23D, 23D-2
def compute_pass_reward(
        self,
        
        bxy,
        vxy,
        receiver,
        passer,
        *,
        ball_kicked,
        threshold_failure,
        episode_timeout,
    ):
 
        ball_x = float(bxy[0])
        ball_y = float(bxy[1])
        ball_vx, ball_vy = (float(vxy[0]), float(vxy[1]))
        receiver_x = float(receiver["info"]["position"])


        alignment_quality_receiver = closest_player_alignment_reward(
            ball_y=ball_y,
            rod_pos_calib=float(receiver["pos_calib"]),
            rod_info=receiver["info"],
            reward_scale=1.0
        )

        allignment_quality_passer = closest_player_alignment_reward(
            ball_y=ball_y,
            rod_pos_calib=float(passer["pos_calib"]),
            rod_info=passer["info"],
            reward_scale=1.0,
        )

        rod_angle_reward = calculate_rod_angle_reward(
            rod_angle=float(receiver["angle"]),
            target_rod_angle=0.0
        )


        # The rod's effective receiving area is close to its fixed x position.
        x_error_mm = abs(ball_x - receiver_x)
        x_zone_radius_mm = 70.0
        in_receive_zone = x_error_mm <= x_zone_radius_mm
        zone_quality = math.exp(-((x_error_mm / 55.0) ** 2))

        abs_vx, abs_vy = abs(ball_vx), abs(ball_vy)
        speed = math.hypot(ball_vx, ball_vy)
        vx_reduction = 0.0
        vy_reduction = 0.0
        if self.prev_ball_vxy is not None:
            prev_vx, prev_vy = map(float, self.prev_ball_vxy)
            vx_reduction = max(0.0, abs(prev_vx) - abs_vx)
            vy_reduction = max(0.0, abs(prev_vy) - abs_vy)

        # A low-speed reward is continuous rather than a one-off event.  That
        # makes "keep control" valuable until the normal episode timeout.
        controlled_speed_quality = math.exp(-((speed / 0.12) ** 2))
        stopped = speed <= 0.08

        vicinity_reward = 0.05 * zone_quality
        speed_reduction_reward = 0.0
        if in_receive_zone:
            speed_reduction_reward = 0.9 * (vx_reduction + 0.7 * vy_reduction) * zone_quality

        controlled_ball_reward = 0.0
        if in_receive_zone:
            controlled_ball_reward = 0.25 * controlled_speed_quality * zone_quality

        stopped_ball_reward = 0.0
        if in_receive_zone and stopped:
            stopped_ball_reward = 8.0 * zone_quality

        threshold_failure_penalty = -1.5 if threshold_failure else 0.0
        timeout_penalty = -0.2 if episode_timeout else 0.0

        

        reward_breakdown = {
            "rod_angle_reward": rod_angle_reward * 0.25,
            #"rod_alignment_reward_passer": allignment_quality_passer,
            "rod_alignment_reward_receiver": alignment_quality_receiver,
            "vicinity_reward": vicinity_reward,
            "speed_reduction_reward": speed_reduction_reward,
            "controlled_ball_reward": controlled_ball_reward,
            "stopped_ball_reward": stopped_ball_reward,
            "threshold_failure": threshold_failure_penalty,
            "episode_timeout": timeout_penalty,
        }


        return float(sum(reward_breakdown.values())), reward_breakdown

