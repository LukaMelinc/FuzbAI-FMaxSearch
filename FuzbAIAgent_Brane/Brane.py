import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import requests
import math
import json
import time
from collections import deque


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


class AgentBrane:

    def __init__(self):
        self.rod_positions=[80, 230, 530, 830]  # Koordinate palic
        self.possible_kick = 0                  # Flag za indikacijo možnega udarca
        self.ball_front_back = 0                # Kje je žoga (spredej, zadej)
        self.team_color = "red"                 # Barva ekipe s katero igramo
        self.delay_queue = deque()              # Queue for delayed data
        self.delay_duration = 0.03              # 30 ms delay (in seconds)

        self.kick_states = [0,1,2,3]            # Koraki za brcnit žogo
        self.kick_stage = 0                     # Števec v katerem koraku brce je igralec
        self.kick_block = 10                    # Število iteracij ki prepreči ponoven brc
        self.kick_block_cnt = 0                 # Števec iteracij za blokado brca
        self.kick_block_en = 0                  # Enable flag za brcanje
        
        
        # Odpri file z geometrijo mize
        with open('geometry.json') as f:
            self.geometry = json.load(f)

        field = self.geometry["field"]
        self.field_x = field["dimension_x"]
        self.field_y = field["dimension_y"]

        # Reklama
        print(r"""

  ___                   _    ______                      
 / _ \                 | |   | ___ \                     
/ /_\ \ __ _ _ __   ___| |_  | |_/ /_ __ __ _ _ __   ___ 
|  _  |/ _` | '_ \ / _ \ __| | ___ \ '__/ _` | '_ \ / _ \
| | | | (_| | | | |  __/ |_  | |_/ / | | (_| | | | |  __/
\_| |_/\__, |_| |_|\___|\__| \____/|_|  \__,_|_| |_|\___|
        __/ |                                            
       |___/                                             


        """)



    ##############################
    ###      Main logika       ###
    ##############################
    def process_data(self, camera):

        commands = [] # List za ukaze
        playerMapping = [1, 2, -1, 3, -1, 4, -1, -1]  # List z možniki igralci (lahko se ne rabi)
        
        if self.kick_block_en == 1: self.kick_block_cnt += 1
        if self.kick_block_cnt > self.kick_block: 
            self.kick_block_cnt = 0
            self.kick_block_en = 0


        # Zaščiti pred slabimi podatki s slike #
        CD0 = None
        CD1 = None
        cam_data = None

        if camera is not None and isinstance(camera, dict) and "camData" in camera:
            camData = camera.get("camData", None)
            
            if isinstance(camData, list):
                if len(camData) > 0 and camData[0] is not None:
                    CD0 = camData[0]
                if len(camData) > 1 and camData[1] is not None:
                    CD1 = camData[1]

        # Prefer CD0, but fallback to CD1 if needed
        camera = CD0 if CD0 is not None else CD1

        if camera is None:
            print("ERROR! Bad camera data")
            for i in range(8):
                # The opponent
                if playerMapping[i] < 0:
                    continue
                cmd = {
                    "driveID": playerMapping[i],
                    "rotationTargetPosition": 0.0,      
                    "rotationVelocity": 0.0,            
                    "translationTargetPosition": 0.5,
                    "translationVelocity": 1.0 
                    }        
                commands.append(cmd)
            return commands
        ######################################################

        # # Vnesen 30 ms delay (se ga lahko tuna na vrhu kode)
        # current_time = time.time()
        # self.delay_queue.append((current_time, camera))

        # # Remove old entries from the queue
        # while len(self.delay_queue) > 0 and current_time - self.delay_queue[0][0] > 0.5:
        #     self.delay_queue.popleft()

        # # Find the first entry that is older than 30 ms
        # delayed_camera = None
        # for timestamp, cam_data in self.delay_queue:
        #     if current_time - timestamp >= self.delay_duration:
        #         delayed_camera = cam_data
        #         break

        # # If no valid delayed data is found, use the most recent one
        # if delayed_camera is None:
        #     delayed_camera = self.delay_queue[-1][1]
        
        #########################################################
    
        
        # Iz podatkov z kamere izlušči podatke
        player_positions, bxy, vxy = self.extract_observation(camera)

        tranPosBall, rodIdBall, tranVelBall = self.move_towards_the_ball(player_positions, bxy)

        # Info o premikanju:
        # Vse hitrosti imajo trapezen profil [pospeševanje -> željena hitrost -> zaviranje]
        # 0,0 je spodaj 
        # + Rotacija
        #   - Kot: -0.5 do 0.5 (0 je navzdol, - je v desno, + je v levo)
        #   - Hitrost: 0 do 1
        # + Translacija:
        #   - Pozicija: 0 do 1 (0.5 je center)
        #   - Hitrost: 0 do 1

        # Preveri ali je mogoče izvesti akcijo na žogi
        self.possible_kick, self.ball_front_back = self.can_kick(bxy[0], vxy)

        for i in range(8):

            # The opponent
            if playerMapping[i] < 0:
                continue

            if playerMapping[i] == rodIdBall:
                cmd = {
                    "driveID": rodIdBall,
                    "rotationTargetPosition": 0.0,      
                    "rotationVelocity": 0.0,            
                    "translationTargetPosition": tranPosBall, # 
                    "translationVelocity": tranVelBall  # 
                    }        
                
                if self.possible_kick == 1 and self.kick_block_en == 0:
                    cmd = self.kick_routine(rodIdBall)
                    self.kick_block_en = 1

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

        #print(camera)

        field = self.geometry["field"]
        rods = self.geometry["rods"]
        player_positions = []

        # Beri kamero 0, čene vzemi podatke z kamere 1
        bx, by, bvx, bvy = 0.0, 0.0, 0.0, 0.0

        # Ball
        bx = camera["ball_x"]
        by = camera["ball_y"]
        bvx = camera["ball_vx"]
        bvy = camera["ball_vy"]

        for rod in rods:
            rod_id = rod["id"]
            team = rod["team"]
            rod_x = rod["position"]
            travel_range = rod["travel"]
            num_players = rod["players"]
            first_offset = rod["first_offset"]
            spacing = rod["spacing"]

            rod_position_calib = camera["rod_position_calib"][rod_id - 1]
            rod_angle = camera["rod_angle"][rod_id - 1]

            # If the calibration is stored as a list, just grab the first element
            if isinstance(rod_position_calib, list):
                rod_position_calib = rod_position_calib[0]

            # Convert the normalized rod_position_calib to actual table coordinates
            rod_y_base = rod_position_calib * travel_range

            for i in range(num_players):
                player_y = rod_y_base + first_offset + i * spacing
                player_positions.append({
                    "rod_id": rod_id,               # zaporedna št. palice
                    "team": team,                   # Ekipa (rdeča/modra)
                    "player_pos": i + 1,            # pozicija igralca na palici
                    "position": (rod_x, player_y),  # koordinate igralca
                    "angle": rod_angle,             # kot igralca / naklon
                    "rod_position": rod_y_base      # pozicija palice - enkoder
                })

        return player_positions, (bx, by), (bvx, bvy)
    

    # Can we kick the ball
    def can_kick(self, ball_x, ball_vel):

        offset = 50
        collision_regions = [(rod_x - offset, rod_x + offset) for rod_x in self.rod_positions]

        #print(" --------------- Preveri meje ---------------")
        for region in collision_regions:
            # print("meja 1:", region[0])
            # print("zoga:", ball_x)
            # print("meja 2:", region[1])
            if region[0] <= ball_x <= region[1]:

                if ball_x <= region[0] + offset:
                    if ball_vel[0] < 0.1 and ball_vel[1] < 0.3:
                        return 1, 0 # Kick je mogoč / naša žoga, žoga za nami (malo kick)
                    
                if ball_x >= region[1] - offset:
                    if ball_vel[0] < 0.3 and ball_vel[1] < 0.3:
                        return 1, 1 # Kick je mogoč / naša žoga, žoga pred nami (kick na hard)
            
        return 0, 0 # Kick ni mogoč / nasprotnikova žoga
    

    # Rutina za brcanje žoge
    def kick_routine(self, rodId):

        if self.ball_front_back == 1:   # Žoga pred igralcem
            kick_power = 0.8
        else:                           # Žoga za igralcem
            kick_power = 0.1

        match self.kick_stage:
            case 0:
                cmd = {
                    "driveID": rodId,
                    "rotationTargetPosition": 0.0,      
                    "rotationVelocity": 0.8,            
                    "translationTargetPosition": 0.0,
                    "translationVelocity": 0.0 
                    }  
            case 1:
                cmd = {
                    "driveID": rodId,
                    "rotationTargetPosition": 0.25,      
                    "rotationVelocity": kick_power,            
                    "translationTargetPosition": 0.0,
                    "translationVelocity": 0.0 
                    }  
            case 2:
                cmd = {
                    "driveID": rodId,
                    "rotationTargetPosition": -0.25,      
                    "rotationVelocity": kick_power,            
                    "translationTargetPosition": 0.0,
                    "translationVelocity": 0.0 
                    }  
            case 3:
                cmd = {
                    "driveID": rodId,
                    "rotationTargetPosition": 0.0,      
                    "rotationVelocity": kick_power,            
                    "translationTargetPosition": 0.0,
                    "translationVelocity": 0.0 
                    }  
            case _:
                cmd = {
                    "driveID": rodId,
                    "rotationTargetPosition": 0.0,      
                    "rotationVelocity": 0.0,            
                    "translationTargetPosition": 0.0,
                    "translationVelocity": 0.0 
                    } 
                
        self.kick_stage += 1
        if self.kick_stage > 3: self.kick_stage = 0

        return cmd 
            

    # Iščemo po y igralca (naše ekipe) ki je najbližje žogi
    def move_towards_the_ball(self, player_data, ball_xy):
        # TODO: računaj razdalje na osnovi osnovnih položajev igralcev kot
        # da se teli ne premikajo -> tako vsak dobi žogo če ta gre po širini
        # igrišča in vsak izpolnjuje svoj max range premikanja.
        # Tako določiš kdo se bo ukvarjal z žogo in ko gre čez njegov range 
        # to določi naslednjemu igralcu 

        min_distance = float('inf')
        closest_player = None
        rod_id = 0
        closest_rod = None
        target_player = None

        # Na osnovnih pozicijah igralcev najdi rod najbližji žogi
        for rod in self.geometry['rods']:
            if rod['team'] == self.team_color:
                rod_x = rod['position']
                distance = abs(rod_x - ball_xy[0])

                if distance < min_distance:
                    min_distance = distance
                    closest_rod = rod

        # Če slušajno ne najde roda (nima lih smisla)
        if closest_rod is None:
            return 0.5, 0, 0.0  

        rod_id = closest_rod["id"]
        rod_players = [p for p in player_data if p["rod_id"] == rod_id]
        travel_range = closest_rod['travel']
        first_offset = closest_rod['first_offset']
        spacing = closest_rod['spacing']
        num_players = closest_rod['players']

        # Default pozicije vseh igralcev
        default_positions = [first_offset + i * spacing for i in range(num_players)]

        # po y osi najdi najbližjega igralca žogi
        min_y_distance = float('inf')
        chosen_local_index = 0
        for i, default_y in enumerate(default_positions):
            dist_y = abs(default_y - ball_xy[1])
            if dist_y < min_y_distance:
                min_y_distance = dist_y
                chosen_local_index = i

        print(chosen_local_index)
        #print("Rod:", rod_id, "Igralec:", target_player)
        # Če slušajno ne najde igralca (nima lih smisla)
        # if target_player is None:
        #     return 0.5, rod_id, 0.0  

        # Definiraj range premikanja igralca
        # player_default_y = default_positions[target_player]
        # rod_position = player_data[target_player]['rod_position']
        chosen_player_data = rod_players[chosen_local_index]
        rod_position = chosen_player_data["rod_position"]
        min_y = rod_position - (travel_range / 2)
        max_y = rod_position + (travel_range / 2)

        # Razdalja premika
        ball_y = ball_xy[1]
        translationTargetPosition = (ball_y - min_y) / (max_y - min_y)
        translationTargetPosition = min(max(translationTargetPosition, 0), 1)

        # Hitrost premika
        distance = abs(min_y_distance - ball_y)
        translationVelocity = min(1.0, abs(distance) / travel_range) + 0.4

        print("translacija:", translationTargetPosition, "hitrost:", translationVelocity)

        # Adjust rod ID for compatibility with motor commands
        if rod_id == 4: 
            rod_id = 3
        elif rod_id == 6: 
            rod_id = 4

        return translationTargetPosition, rod_id, translationVelocity




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
