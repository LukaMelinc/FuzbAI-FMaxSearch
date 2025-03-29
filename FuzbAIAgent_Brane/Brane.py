import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import requests
import math
import json


#HOST_ADDRESS = '127.0.0.1:23336'  # IP or Host for your environment
HOST_ADDRESS = '192.168.222.23:22000' # Simulator

##############################
# Online Functions (unchanged)
##############################

def get_camera_state():
    cam_url = f"http://{HOST_ADDRESS}/Camera/State"
    response = requests.get(cam_url)
    return response.json()


# def send_motor_commands(cmds):
#     motors_url = f"http://{HOST_ADDRESS}/Motors/SendCommand?blue=False"
#     response = requests.post(motors_url, json=cmds)

def send_motor_commands(cmds):
    # Pretvori vse np.float32 v navadne float vrednosti (rekurzivno, če je potrebno)
    def convert_floats(obj):
        if isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_floats(i) for i in obj]
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        else:
            return obj

    motors_url = f"http://{HOST_ADDRESS}/Motors/SendCommand?blue=False"
    clean_cmds = convert_floats(cmds)
    response = requests.post(motors_url, json=clean_cmds)


##############################
#   Playground observation
##############################
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


class AgentBrane:

    def __init__(self):
        self.rod_positions=[80, 230, 530, 830]  # Koordinate palic
        self.possible_kick = 0                  # Flag za indikacijo možnega udarca
        self.team_color = "red"                 # Barva ekipe s katero igramo
        
        
        # Odpri file z geometrijo mize
        with open('geometry.json') as f:
            self.geometry = json.load(f)

        field = self.geometry["field"]
        self.field_x = field["dimension_x"]
        self.field_y = field["dimension_y"]



    ##############################
    ###      Main logika       ###
    ##############################
    def process_data(self, camera):

        commands = [] # List za ukaze
        playerMapping = [1, 2, -1, 3, -1, 4, -1, -1]  # List z možniki igralci (lahko se ne rabi)
        
        # Iz podatkov z kamere izlušči podatke
        player_positions, bxy, vxy = self.extract_observation(camera)

        # Preveri ali je mogoče izvesti akcijo na žogi
        self.possible_kick = self.can_kick(bxy[0])

        # Info o premikanju:
        # Vse hitrosti imajo trapezen profil [pospeševanje -> željena hitrost -> zaviranje]
        # 0,0 je spodaj 
        # + Rotacija
        #   - Kot: -0.5 do 0.5 (0 je navzdol, - je v desno, + je v levo)
        #   - Hitrost: 0 do 1
        # + Translacija:
        #   - Pozicija: 0 do 1 (0.5 je center)
        #   - Hitrost: 0 do 1

        tranPosBall, rodIdBall, tranVelBall = self.move_towards_the_ball(player_positions, bxy)

        for i in range(8):

            # The opponent
            if playerMapping[i] < 0:
                continue

            if playerMapping[i] == rodIdBall:
                cmd = {
                    "driveID": rodIdBall,
                    "rotationTargetPosition": 0.0,      
                    "rotationVelocity": 0.0,            
                    "translationTargetPosition": tranPosBall,
                    "translationVelocity": tranVelBall 
                    }        
                commands.append(cmd)
            else:
                cmd = {
                    "driveID": playerMapping[i],
                    "rotationTargetPosition": 0.0,      
                    "rotationVelocity": 0.0,            
                    "translationTargetPosition": 0.5,
                    "translationVelocity": 1.0 
                    }        
                commands.append(cmd)



        # Return ukaz za motorje
        return commands


    # Pridobivanje podatkov o žogi in vsakemu igralcu posebej (x, y, kot, ekipa)
    def extract_observation(self, camera):

        field = self.geometry["field"]
        rods = self.geometry["rods"]
        player_positions = []

        # Beri kamero 0, čene vzemi podatke z kamere 1
        CD0 = camera["camData"][0]
        if CD0 is None:
            # Fallback 
            CD0 = camera["camData"][1]

        # Ball
        bx = CD0["ball_x"]
        by = CD0["ball_y"]
        bvx = CD0["ball_vx"]
        bvy = CD0["ball_vy"]

        for rod in rods:
            rod_id = rod["id"]
            team = rod["team"]
            rod_x = rod["position"]
            travel_range = rod["travel"]
            num_players = rod["players"]
            first_offset = rod["first_offset"]
            spacing = rod["spacing"]

            rod_position_calib = CD0["rod_position_calib"][rod_id - 1]
            rod_angle = CD0["rod_angle"][rod_id - 1]

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

        return player_positions, (bx, by), (bvx, bvy)
    

    # Can we kick the ball
    def can_kick(self, ball_x):

        collision_regions = [(rod_x - 50, rod_x + 50) for rod_x in self.rod_positions]

        for i, region in enumerate(collision_regions):
            if region[0] <= ball_x <= region[1]:
                return 1 # Kick je mogoč / naša žoga
            else:
                return 0 # Kick ni mogoč / nasprotnikova žoga
            

    # Iščemo po y igralca (naše ekipe) ki je najbližje žogi
    def move_towards_the_ball(self, player_data, ball_xy):
        min_distance = float("inf")
        closest_player = None
        rod_id = 0

        for player in player_data:
            if player["team"] == self.team_color:
                dx = player["position"][0] - ball_xy[0]
                dy = player["position"][1] - ball_xy[1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_distance:
                    min_distance = dist
                    closest_player = player
                    rod_id = player["rod_id"]

        if closest_player is None:
            return 0.5, rod_id  # Default pozicija

        player_y = closest_player["position"][1]
        ball_y = ball_xy[1]
        travel_range = self.geometry["rods"][rod_id - 1]["travel"]

        # Normaliziranje
        translationTargetPosition = min(max((ball_y / travel_range), 0), 1)

        # hitrost
        distance = abs(player_y - ball_y)
        translationVelocity = min(1.0, distance / travel_range)

        return translationTargetPosition, rod_id, translationVelocity

            

    def scale_to_motor_commands(self, action):

        # Reshape to (4 rods, 4 dims)
        act_rod = action.reshape((4,4))

        commands = []
        # The rods we control (index in geometry): 0,1,3,5 => drive IDs = 1,2,3,4
        rod_map = [0,1,3,5]
        driveID = [1,2,3,4]
        for rod_i in range(4):
            # Raw from policy
            rot_target_raw  = act_rod[rod_i, 0]  # in [-1,1]
            rot_speed_raw   = act_rod[rod_i, 1]  # in [-1,1]
            trans_target_raw= act_rod[rod_i, 2]  # in [-1,1]
            trans_speed_raw = act_rod[rod_i, 3]  # in [-1,1]

            # print("rot_target_raw",rot_target_raw) 
            # print("rot_speed_raw",rot_speed_raw) 
            # print("trans_target_raw",trans_target_raw)
            # print("trans_speed_raw",trans_speed_raw) 

            # Example scaling:
            rot_target   = 0.3 * rot_target_raw     # we only want to rotate between -0.8..+0.8 0.8
            rot_velocity = 0.2 * (rot_speed_raw+1)/8  # scale [-1,1]→[0,1], then multiply by max 0.5
            trans_target = 0.2 * ((trans_target_raw+1)/2)  # scale [-1,1]→[0,1], you might want full 0..1 0.5
            trans_velocity = 1.0 * (trans_speed_raw+1)/2   # scale [-1,1]→[0,1]

            cmd = {
                "driveID": driveID[rod_i],
                "rotationTargetPosition": rot_target,
                "rotationVelocity": rot_velocity,
                "translationTargetPosition": trans_target,
                "translationVelocity": trans_velocity
            }
            commands.append(cmd)

        #print(commands)
        return commands



####################################################################################################################################
############################################################    MAIN    ############################################################
####################################################################################################################################

if __name__ == "__main__":
    # Create the PPO agent
    agent = AgentBrane()

    try:
        while True:
            time.sleep(0.02)

            # 1) Pridobi podatke s kamere
            cam_data = get_camera_state()

            # 2) Sprocesiraj podatke in pridobi ukaze
            motor_cmds = agent.process_data(cam_data)

            # 3) Pošlji ukaze na mizo
            send_motor_commands({'commands': motor_cmds}) #{'commands': motor_cmds}

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
