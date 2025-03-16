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


HOST_ADDRESS = '127.0.0.1:23336'    # IP za povezavo

### Pridobi trenutne podatke o igri
def get_camera_state():
    cam_url = f"http://{HOST_ADDRESS}/Camera/State"
    response = requests.get(cam_url)    
    return response.json()

### Pošlji ukaze na mizo
def send_motor_commands(cmds):
    motors_url = f"http://{HOST_ADDRESS}/Motors/SendCommand?blue=False"
    response = requests.post(motors_url, json=cmds)


### Nevronska mreža - DQL
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Ensure output is between -1 and 1

### Class agenta 007
class ShootingAgent:
    def __init__(self, state_size=20, action_size=4, gamma=0.99, epsilon=0.50, epsilon_min=0.1, epsilon_decay=0.995, lr=0.001, batch_size=64):
        # Podatki za NN
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.last_reward = None
        self.total_reward = 0
        self.team_color = "red"

        # Naloži json s podatki o mizi
        with open('geometry.json') as f:
            self.geometry = json.load(f)

        # Količina zapomnjenih iteracij
        self.memory = deque(maxlen=2000)

        # Število rdečih palic
        self.red_rods = [rod for rod in self.geometry["rods"] if rod["team"] == self.team_color]
        self.action_size = len(self.red_rods) * 4

        # Init mreže
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()

        # Init optimizatorja
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()


    ### Sinhronizacija uteži ciljne mreže z main mreže (DQN zahteva, ker ima dve mreži, Main in Target) 
    """
    1) Main (or Online) Network (self.model):
        This network is actively trained and used to choose actions.
    
    2) Target Network (self.target_model):
        This network provides stable Q-value targets during training.
        It is not updated at every training step, which helps stabilize the learning process.
    """
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    ### Shranjuje eksperimente v replay spominu za kontekst v prihodnosti kaj je vse že poskušal
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    ### Povezano z epsilon-greedy strategy: ali bo eksperimentiralo ali uporabilo mrežo
    def choose_action(self, state):

        state = torch.FloatTensor(state).unsqueeze(0)

        # Random se odloči ali bo uporabilo NN ali bo raziskoval
        if np.random.rand() <= self.epsilon:
            # Random continuous actions for exploration in range [-1, 1]
            return np.random.uniform(-1, 1, self.action_size)
        else:
            # NN decides continuous actions
            with torch.no_grad():
                return self.model(state).squeeze(0).numpy()


    ### Učenje mreže
    def learn(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples

        self.learn_step_counter += 1
        if self.learn_step_counter % 10 != 0:
            return  # Only learn every 10 steps

        # Sample mini-batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        #actions = torch.LongTensor(actions).unsqueeze(1)
        actions = torch.LongTensor(actions).argmax(dim=1).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        if actions.ndim == 1:
            actions = actions.unsqueeze(1)

        # Current Q-values
        q_values = self.model(states).gather(1, actions)

        # Max Q-values from target model
        next_q_values = self.target_model(next_states).max(1)[0].detach().unsqueeze(1)

        # Target Q-values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.criterion(q_values, target_q_values) # dela

        # Ensure target_q_values matches q_values shape
        # target_q_values = target_q_values.expand_as(q_values)
        # loss = self.criterion(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    ### Funkcija namenjena paralelnemu učenju    
    def learn_from_batch(self, states, actions, rewards, next_states, dones):

        self.learn_step_counter += 1
        if self.learn_step_counter % 10 != 0:
            return  # Only learn every 10 steps

        # Optimize tensor conversion
        states = torch.from_numpy(np.array(states)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1)
        dones = torch.from_numpy(np.array(dones)).float().unsqueeze(1)

        # Predicted Q-values for current states
        q_values = self.model(states).gather(1, actions)


        # Target Q-values using target model for next states
        next_q_values = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Ensure target size matches
        target_q_values = target_q_values.expand_as(q_values)
        loss = self.criterion(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    ### Shrani model
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)


    ### Naloži model
    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()


    ### Procesiraj podatke s kamere: lokacija in hitrost premikanja žoge
    def data_process(self, camera):
        CD0 = camera["camData"][0]
        CD1 = camera["camData"][1]

        vx = CD0["ball_vx"]
        vy = CD0["ball_vy"]

        bx = CD0["ball_x"]
        by = CD0["ball_y"]

        flag = 1 if vx < 0.01 and vy < 0.01 else 0        # TODO uredit enote hitrosti da bo meja ok

        return bx, by, vx, vy, flag


    ### Procesiraj podatke s kamere: pozicije in kot igralcev
    def field_data(self, camera):

        # TODO: Če ostane čas mogoče kalmana za predikcijo žogice ;) ?

        CD0 = camera["camData"][0]
        CD1 = camera["camData"][1]

        positions = []
        rotations = []

        # Preveri ali kamera 0 vidi žigico -> če ne uporabi drugo kamero
        if CD0 is not None:

            for i in range(8):
                positions.append(CD0["rod_position_calib"][i])
                rotations.append(CD0["rod_angle"][i])

            # VZEL VSE POZICIJE IN ROTACIJA SKUPAJ, NAKONCU NAJ BI MODEL SAM UGOTOVIL???
            #opp_pos = [positions[i] for i in [2, 4, 6, 7]]
            #opp_rot = [rotations[i] for i in [2, 4, 6, 7]]

        else:

            for i in range(8):
                positions.append(CD1["rod_position_calib"][i])
                rotations.append(CD1["rod_angle"][i])

            # VZEL VSE POZICIJE IN ROTACIJA SKUPAJ, NAKONCU NAJ BI MODEL SAM UGOTOVIL???
            #opp_pos = [positions[i] for i in [2, 4, 6, 7]]
            #opp_rot = [rotations[i] for i in [2, 4, 6, 7]]

        return positions, rotations


    ### Zaznaj če se igralec dotika žoge
    def detect_collision(self, ball_pos, player_pos, ball_radius, player_radius):
        distance = math.hypot(ball_pos[0] - player_pos[0], ball_pos[1] - player_pos[1])         # Kva je tle player_pos, a position iz naslednje funkcije?
        return distance <= (ball_radius + player_radius)


    ### Preračunaj koordinate in kote igralcev na mizi - obe ekipi
    def calculate_player_positions_and_angles(self, camera):

        field = self.geometry["field"]
        rods = self.geometry["rods"]
        
        player_positions = []
        
        # Process each rod
        for rod in rods:
            rod_id = rod["id"]
            team = rod["team"]
            rod_x = rod["position"]
            travel_range = rod["travel"]
            num_players = rod["players"]
            first_offset = rod["first_offset"]
            spacing = rod["spacing"]

            # Get corresponding camera data for the rod
            cam_data = camera["camData"][0] 
            if cam_data is None:
                cam_data = camera["camData"][1] 

            rod_position_calib = cam_data["rod_position_calib"][rod_id - 1]
            rod_angle = cam_data["rod_angle"][rod_id - 1]


            if isinstance(rod_position_calib, list):
                rod_position_calib = rod_position_calib[0] 
            # Calculate the actual y position based on calibration
            rod_y_base = (rod_position_calib * travel_range)

            # Calculate positions for each player on the rod
            for i in range(num_players):
                player_y = rod_y_base + first_offset + i * spacing
                player_positions.append({
                    "rod_id": rod_id,
                    "team": team,
                    "position": (rod_x, player_y),
                    "angle": rod_angle
                })
        
        return player_positions


    ### Funkcija za reward-e
    def calculate_shooting_reward(self, bx, by, vx, vy, collision_detected):

        reward = 0

        # Parameters
        goal_x_range = (1200, 1210)
        goal_y_range = (250, 450)

        # 1. Collision Reward (boolean)
        if collision_detected:
            reward += 10
        else:
            reward -= 5  # Penalty for missing the ball

        # 2. Žoga gre v smer nasprotnikovega gola
        goal_center = (1205, 350)
        ball_vector = np.array([vx, vy])
        direction_vector = np.array([goal_center[0] - bx, goal_center[1] - by])

        if np.linalg.norm(ball_vector) > 0:
            cosine_similarity = np.dot(ball_vector, direction_vector) / (np.linalg.norm(ball_vector) * np.linalg.norm(direction_vector))

            # Reward for direction towards the goal
            if cosine_similarity > 0:
                directional_reward = cosine_similarity * 30
                reward += directional_reward
            else:
                # Penalty for moving away from the goal
                reward -= 10  # Adjust the penalty value as needed
        else:
            reward -= 5  # Penalty for stationary ball

        # 3. Hitrejša žoga je boljša
        ball_speed = np.linalg.norm(ball_vector)
        reward += ball_speed * 5  # Scale the speed reward

        # 4. Zadel je gol
        if goal_x_range[0] <= bx <= goal_x_range[1] and goal_y_range[0] <= by <= goal_y_range[1]:
            reward += 100
        elif bx < 1000 or bx > 1230:
            reward -= 50  # Own goal or out of bounds

        # 5. MAnjša kazen če ne dela nič
        if ball_speed < 0.01:
            reward -= 1

        return reward


    ### Glavno procesiranje vseh podatkov 
    def process_data(self, camera):

        commands = []

        # Podatki s kamere
        bx, by, vx, vy, _ = self.data_process(camera)           # Pozicija in hitrost zogice (4)
        opp_pos, opp_rpt = self.field_data(camera)              # Pozicije in rotacije vseh rodov (8 + 8)

        state = np.concatenate([[bx, by, vx, vy], opp_pos, opp_rpt])    # vse skupaj 20 - Vhodni podatki za NN

        # Predikcija ukazov - Območje vrednosti: [-1, 1], število izhodnih vrednosti: 16
        action_values = self.choose_action(state)
        #print(action_values)

        # Skaliranje vrednosti
        rotation_target = action_values[0]                      # Already in [-1, 1]
        rotation_velocity = (action_values[1] + 1) * 1.0        # Convert [-1, 1] to [0, 2]
        translation_target = (action_values[2] + 1) * 0.5       # Convert [-1, 1] to [0, 1]
        translation_velocity = (action_values[3] + 1) * 1.0     # Convert [-1, 1] to [0, 2]

        # Next state (for now, assume it remains the same)
        next_state = state

        player_data = self.calculate_player_positions_and_angles(camera)    # Računaj pozicije za vse igralce
        ball_position = (bx, by)    # Pozicija žoge

        # Zaznavanje kolizije z žogo
        ball_collision = False
        for player in player_data:
            if self.team_color == player["team"]:   # preverjaj ali ima žogo le za rdečo ekipo
                player_pos = player["position"]
                if self.detect_collision(ball_position, player_pos, 0.017, 0.03):      # ball radius je 34mm -> pou tega je 0.017m, player radius sem dal na 3cm, v navodilih je napisan 4 cm
                    ball_collision = True
                    print(f"ball collision calculated")
                    break

        # Računanje nagrade
        reward = self.calculate_shooting_reward(bx, by, vx, vy, ball_collision)

        self.last_reward = reward
        self.total_reward += reward

        print(f"Earned reward: {reward}, Total accumulated reward: {self.total_reward}")

        # Check if the episode is done
        done = reward == 100 

        # Shrani iteracije
        self.remember(state, action_values, reward, next_state, done)

        # Prodobi število ročk ene ekipe, itak vemo da je 4 ma..... ja
        red_rods = [rod for rod in self.geometry["rods"] if rod["team"] == self.team_color]
        #num_rods = len(self.red_rods)

        # Razporedi 16 ukazov po palicah (4 ukazi na palico)
        for idx, rod in enumerate(self.red_rods):
            base_idx = idx * 4
            rod_actions = action_values[base_idx:base_idx + 4]

            # Select the maximum action index for each rod
            max_action_idx = np.argmax(rod_actions)

            # Map max action index to specific command
            rotation_target = rod_actions[max_action_idx] if max_action_idx == 0 else 0
            rotation_velocity = (rod_actions[max_action_idx] + 1) * 1.0 if max_action_idx == 1 else 0
            translation_target = (rod_actions[max_action_idx] + 1) * 0.5 if max_action_idx == 2 else 0
            translation_velocity = (rod_actions[max_action_idx] + 1) * 1.0 if max_action_idx == 3 else 0

            # Construct the command
            cmd = {
                'driveID': idx + 1,
                'rotationTargetPosition': rotation_target,
                'rotationVelocity': rotation_velocity,
                'translationTargetPosition': translation_target,
                'translationVelocity': translation_velocity
            }

            #print(f"Rod ID: {rod['id']} - Command: {cmd}")  # Debugging

            commands.append(cmd)

        # Učenje modela
        self.learn()

        return commands
    
    
######################################## MAIN ########################################
if __name__ == "__main__":
    agent = ShootingAgent()
    episode = 0
    save_interval = 100

    try:
        while True:
            episode += 1
            time.sleep(0.02)

            # Get the current state of the camera
            cam_data = get_camera_state()

            # Process the camera data and get motor commands
            motor_cmds = agent.process_data(cam_data)

            # Send the motor commands to the simulator
            send_motor_commands({'commands': motor_cmds})

            # Save the model at regular intervals
            if episode % save_interval == 0:
                agent.save_model("ball_control_model.pth")

    except KeyboardInterrupt:
        print("Training interrupted. Saving last model...")
        agent.save_model("ball_control_model.pth")
