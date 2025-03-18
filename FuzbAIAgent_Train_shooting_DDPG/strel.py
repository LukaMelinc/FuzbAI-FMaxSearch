import requests
import json
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gc

HOST_ADDRESS = '127.0.0.1:23336'  # IP or Host for your environment


##############################
# 1) Online Functions (unchanged)
##############################

def get_camera_state():
    cam_url = f"http://{HOST_ADDRESS}/Camera/State"
    response = requests.get(cam_url)
    return response.json()


def send_motor_commands(cmds):
    motors_url = f"http://{HOST_ADDRESS}/Motors/SendCommand?blue=False"
    requests.post(motors_url, json=cmds)


##############################
# 2) Helper Functions (unchanged or minimal edits)
##############################

def ball_data(camera):

    CD0 = camera["camData"][0]
    if not CD0:
        # Fallback if the first camera is None
        CD0 = camera["camData"][1]

    vx = CD0["ball_vx"]
    vy = CD0["ball_vy"]
    bx = CD0["ball_x"]
    by = CD0["ball_y"]

    # Some threshold for 'flag' if ball is stationary
    flag = 1 if abs(vx) < 0.01 and abs(vy) < 0.01 else 0
    return bx, by, vx, vy, flag


def players_data(camera):

    CD0 = camera["camData"][0]
    CD1 = camera["camData"][1]

    if CD0 is not None:
        data = CD0
    else:
        data = CD1

    positions = []
    rotations = []
    for i in range(8):
        positions.append(data["rod_position_calib"][i])
        rotations.append(data["rod_angle"][i])
    return positions, rotations


def detect_collision(ball_pos, player_pos, ball_radius=0.017, player_radius=0.03):

    distance = math.hypot(ball_pos[0] - player_pos[0], ball_pos[1] - player_pos[1])
    return distance <= (ball_radius + player_radius)


def calculate_player_positions_and_angles(camera, geometry):

    field = geometry["field"]
    rods = geometry["rods"]
    player_positions = []

    # We'll read from the camera the same way:
    cam_data = camera["camData"][0]
    if cam_data is None:
        cam_data = camera["camData"][1]

    for rod in rods:
        rod_id = rod["id"]
        team = rod["team"]
        rod_x = rod["position"]
        travel_range = rod["travel"]
        num_players = rod["players"]
        first_offset = rod["first_offset"]
        spacing = rod["spacing"]

        rod_position_calib = cam_data["rod_position_calib"][rod_id - 1]
        rod_angle = cam_data["rod_angle"][rod_id - 1]

        # If the calibration is stored as a list, just grab the first element
        if isinstance(rod_position_calib, list):
            rod_position_calib = rod_position_calib[0]

        # Convert the normalized rod_position_calib to actual table coordinates
        rod_y_base = rod_position_calib * travel_range

        for i in range(num_players):
            player_y = rod_y_base + first_offset + i * spacing
            player_positions.append({
                "rod_id": rod_id,
                "team": team,
                "position": (rod_x, player_y),
                "angle": rod_angle
            })

    return player_positions


####################################### ^^^^^^^^^^^^^^^^ DELA ^^^^^^^^^^^^^^^^ #######################################










##############################
# 3) Reward Function (unchanged)
##############################

def calculate_shooting_reward(bx, by, vx, vy, collision_detected, player_data):

    reward = 0
    goal_x_range = (1200, 1210)
    goal_y_range = (250, 450)

    # 1. Collision Reward
    if collision_detected:
        reward += 100
        print("boom!!!!!!!")
    else:
        reward -= 5
        #print("NO colision")

    # 2. Direction towards the goal
    goal_center = (1205, 350)
    ball_vector = np.array([vx, vy])
    direction_vector = np.array([goal_center[0] - bx, goal_center[1] - by])

    if np.linalg.norm(ball_vector) > 0:
        cosine_similarity = np.dot(ball_vector, direction_vector) / (
            np.linalg.norm(ball_vector) * np.linalg.norm(direction_vector)
        )
        if cosine_similarity > 0:
            directional_reward = cosine_similarity * 30
            reward += directional_reward
            #print(f"ball moving towards the goal, Reward: {directional_reward}")
        else:
            reward -= 10
    else:
        reward -= 5

    # 3. Speed
    ball_speed = np.linalg.norm(ball_vector)
    reward += ball_speed * 5
    #print(f"Speed reward: {ball_speed}")

    # 4. Check goal
    if goal_x_range[0] <= bx <= goal_x_range[1] and goal_y_range[0] <= by <= goal_y_range[1]:
        reward += 100
    elif bx < 1000 or bx > 1230:
        reward -= 50  # Own goal or out of bounds
        #print(f"Goal received, Reward -50")

    # 5. Slight penalty if ball is basically still
    if ball_speed < 0.01:
        reward -= 1
        #print(f"Slow ball spet penalty: -1")

    # 6. Slight penalty if a player is oriented in the air
    for player in player_data:
        if player["team"] == "red":
            rwd = 10 * (abs(player["angle"]) * -1)
            if rwd > -100:
                reward += 50 
            else:
                reward += rwd
  
    return reward




    ################################################
    # 6) The main step for handling environment data
    ################################################
    def process_data(self, camera):
       








       
        commands = []
        for i in range(4):
            base_idx = i
            # Suppose each dimension is in [-1, 1]; let's rescale them:
            rotation_target      = action[base_idx + 0] * 32
            rotation_velocity    = (action[base_idx + 1] + 1) * 0.5
            translation_target   = (action[base_idx + 2] + 1) * 0.5
            translation_velocity = (action[base_idx + 3] + 1) * 0.5

            cmd = {
                'driveID': i + 1,  # or i+1, depending on your environment
                'rotationTargetPosition': rotation_target,
                'rotationVelocity': rotation_velocity,
                'translationTargetPosition': translation_target,
                'translationVelocity': translation_velocity
            }
            
            commands.append(cmd)
        #print("Time 8:", time.time() - start)

        # torch.cuda.empty_cache()
        # gc.collect()

        for command in commands:
            values = [v.item() if isinstance(v, np.generic) else v for v in command.values()]
            print(values)

        return commands


##############################
# 7) The Main Loop
##############################

if __name__ == "__main__":
    
    # Suppose our input has size=20 (ball + rods) and output has size=16 (4 rods × 4 each).
    agent = ContinuousAgent(state_dim=20, action_dim=16)
    episode = 0
    save_interval = 100

    try:
        while True:
            episode += 1
            time.sleep(0.02)

            # 1) Get the current camera state
            cam_data = get_camera_state()

            # 2) Process data and get motor commands
            motor_cmds = agent.process_data(cam_data)

            # 3) Send commands to environment
            send_motor_commands({'commands': motor_cmds})

            # 4) Save model periodically
            if episode % save_interval == 0:
                agent.save_model("actor.pth", "critic.pth")

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        agent.save_model("actor.pth", "critic.pth")
