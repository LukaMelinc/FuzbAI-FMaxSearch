import logging
logger = logging.getLogger(__name__)


def simple_reward(ball_loc, ball_x_speed_now, ball_x_speed_before):
    """Minimal reward to verify training works"""
    reward = 0
    reward_breakdown = {
        'velocity_change': 0,
        'goal_scored': 0,
        'own_goal': 0,
        'speed_boost': 0
    }
    

    if ball_x_speed_before == None:
        reward += 0

    elif ball_x_speed_before != None:
        # If the agent kicks the ball towards its own goal and it did not happen by bouncing off the right end of the pitch

        # Negative reward for kicking the ball in our goal
        if (ball_x_speed_now < ball_x_speed_before) and (ball_x_speed_now < 0):
            if (ball_x_speed_before < 0) and (ball_x_speed_now < 0):
                if (ball_x_speed_now < (ball_x_speed_before * 5)) and (750 <= ball_loc[0] <= 950):
                    reward_breakdown['velocity_change'] = -5
                    reward -= 5
                    logger.error(f"Ball kicked in the wrong direction, ball speed: {ball_x_speed_before:.4f}->{ball_x_speed_now:.4f}")
                    #print(f"last_v: {ball_x_speed_before}, current_v: {ball_x_speed_now}")

            elif (ball_x_speed_before > 0) and (ball_x_speed_now < 0) and (750 <= ball_loc[0] <= 950):
                absolute_before = abs(ball_x_speed_before)
                absolute_now = abs(ball_x_speed_now)
                if absolute_now > (absolute_before * 5):
                    reward_breakdown['velocity_change'] = -5
                    reward -= 5
                    logger.error(f"Ball kicked in the wrong direction from the correct direction, speed: {ball_x_speed_before:.4f}->{ball_x_speed_now:.4f}")
                    #print(f"last_v: {ball_x_speed_before}, current_v: {ball_x_speed_now}")

        

        # Positive reward for kicking the ball in opponents goal
        if (ball_x_speed_now > ball_x_speed_before) and (ball_x_speed_now > 0):
            if (ball_x_speed_before > 0) and (ball_x_speed_now > 0):
                if (ball_x_speed_now > (ball_x_speed_before * 5)) and (750 <= ball_loc[0] <= 950):
                    reward_breakdown['velocity_change'] = 5
                    reward += 5
                    logger.info(f"Ball kicked in the correct direction, speed: {ball_x_speed_before:.4f}->{ball_x_speed_now:.4f}")
                    #print(f"last_v: {ball_x_speed_before}, current_v: {ball_x_speed_now}")

            elif (ball_x_speed_before < 0) and (ball_x_speed_now > 0) and (750 <= ball_loc[0] <= 950):
                absolute_before = abs(ball_x_speed_before)
                absolute_now = abs(ball_x_speed_now)
                if ball_x_speed_now > (absolute_now * 5):
                    reward_breakdown['velocity_change'] = 5
                    reward += 5
                    logger.info(f"Ball kicked in the correct direction from the incorrect direction, speed: {ball_x_speed_before:.4f}->{ball_x_speed_now:.4f}")
                    #print(f"last_v: {ball_x_speed_before}, current_v: {ball_x_speed_now}")
                    

    
    # 2. Goal scored
    if 1195 <= ball_loc[0] <= 1210 and 250 <= ball_loc[1] <= 450:
        reward_breakdown['goal_scored'] = 25
        reward += 25
        print("Goal scored: positive reward +25")
    
    # 3. Own goal
    elif 0 <= ball_loc[0] <= 15 and 250 <= ball_loc[1] <= 450:
        reward_breakdown['own_goal'] = -25
        reward -= 25
        print("Own goal scored: negative reward -25")
    

    return reward, reward_breakdown