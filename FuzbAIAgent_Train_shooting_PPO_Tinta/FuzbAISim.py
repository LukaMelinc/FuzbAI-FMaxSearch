import pybullet as p
import pybullet_data
import time, datetime
from pprint import pprint
import threading
import math
from FuzbAIAgent_Example import PlayerAgent
from strel_PPO import PPOAgent
from pass_auxiliary_backbone import PassAuxiliaryBackboneAgent
import random
import traceback

from log_utils import setup_logging

class FuzbAISim:
    def __init__(self, episode_end_ball_x_threshold_mm: float = 600.0, kick_observed_rod_id: int = 6):
        print(" ______         _             _____ ")
        print("|  ____|       | |      /\   |_   _|")
        print("| |__ _   _ ___| |__   /  \    | |  ")
        print("|  __| | | |_  / '_ \ / /\ \   | |  ")
        print("| |  | |_| |/ /| |_) / ____ \ _| |_ ")
        print("|_|   \__,_/___|_.__/_/    \_\_____|")                                     
        print("")
        print("LAK FuzbAI simulator v1 - 2025")

        self.ballPos = None
        self.ballVel = None

        self.scoreDisp = None # Text used for score display
        self.roundDisp = None # Text used for round count display

        # Text used for player status display
        self.playerStatusDisp1 = None
        self.playerStatusDisp2 = None

        self.rodPositions = [0]*8
        self.rodAngles = [0]*8
        
        self.score = [0,0]

        self.ballPosNoise = 5
        self.ballVelNoise = 0.01

        self.defaultBallPos = [0.718,0.71,0.3]

        # Episode termination: end the episode if camera/geometry ball_x (mm) is below this threshold.
        # This is computed with the same mapping as in getCameraDict(): ball_x_mm = 1000*ballPos[0] - 115.
        self.episode_end_ball_x_threshold_mm = float(episode_end_ball_x_threshold_mm)
        self.episode_end_ball_other_x_threshold_mm = 1190.0

        # Kick observation: detect ball contact on the specified rod's player links.
        # The PPO shooting agent currently emits driveID=4. Player 1 maps that
        # through driveMap=[0, 1, 3, 5], so the physical axis is 5, i.e. rod 6.
        # Watching rod 4 here missed real kicks because it was checking the wrong
        # player links.
        self.kick_observed_rod_id = int(kick_observed_rod_id)
        self._kick_player_links = set()
        self._link_names_by_index = {}
        self._ball_kicked_latch = False
        self._kick_terminated_latch = False
        self._x_threshold_terminated_latch = False
        # Keep the kick latch for reward/observation, but do not end the episode
        # when a kick is detected. Goal scoring and x-threshold resets should
        # drive episode boundaries during shooting training.
        self.terminate_episode_on_kick = False
        # Episode boundary latch: set True when the ball gets reset (agent uses it to call finish_episode)
        self.end_episode_latch = False
        # Don't end an episode on the initial ball placement at startup
        self._end_episode_armed = False

        # Threshold of num of steps to end the iteration
        self.max_num_steps = 40
        self.current_step = 0

        # Control loop period (seconds). One "step" for the agent completes when this time has elapsed.
        # Increase this to make each step take longer (e.g. 0.05 for ~20 Hz, 0.1 for ~10 Hz).
        self.control_dt = 0.05
        self.debug_print_red_kicks = True
        self.debug_red_kick_force_threshold = 0.0
        self.debug_red_kick_cooldown_s = 0.05
        self._last_debug_red_kick_t = -1e9
        self.debug_print_any_ball_contact = False
        self._warned_empty_kick_links = False
        self.physics_timestep = 1.0 / 240.0
        self.physics_steps_per_loop = 4
        self.gui_sleep_s = 0.0

        self.stepDisp = None

        # Player object indices in the URDF tree model
        self.redPlayers = [ 2, 5, 6, 14, 15, 16, 17, 18, 28, 29, 30 ]
        self.bluePlayers = [ 9, 10, 11, 21, 22, 23, 24, 25, 33, 34, 37 ]

        # Joint indices in the URDF tree model
        self.revJoints = [0, 3, 7, 12, 19, 26, 31, 35]
        self.slideJoints = [1, 4, 8, 13, 20, 27, 32, 36]

        self.travels = [190, 356, 180, 116, 116, 180, 356, 190]
        self.redIndices = [0, 1, 3, 5]

        # Switch this to "pass_auxiliary_backbone" for phase-0 supervised
        # pretraining of two red rods: rod 4 passer + rod 6 receiver.
        #self.agent1_mode = "single_rod_ppo"
        self.agent1_mode = "pass_auxiliary_backbone"
        if self.agent1_mode == "pass_auxiliary_backbone":
            self.p1 = PassAuxiliaryBackboneAgent(
                passer_rod_id=4,
                receiver_rod_id=6,
                opponent_rod_id=5,
                model_save_path="/home/tinta/Desktop/FuzbAI-FMaxSearch/FuzbAIAgent_Train_shooting_PPO_Tinta/trained_models/#20.pth",
                load_model=True,
                inference=True,
                training_enabled=False,
            )
        else:
            self.p1 = PPOAgent(
                model_save_path="/home/tinta/Desktop/FuzbAI-FMaxSearch/FuzbAIAgent_Train_shooting_PPO_Tinta/trained_models/#18.pth",
                load_model=True,
                inference=False,
                training_enabeled=True,
                #action_std_override=(0.35, 0.35, 0.25, 0.30),
            )
        self.p2 = PlayerAgent()

        # Camera delay settings
        #self.simulatedDelay = 0.030
        self.simulatedDelay = 0.00  # NOTE: Temporarely set the delay to 0, implement delay tracker into the agent
        self.delayedMemory = []
        self.maxMemory = 0.5 # Maximum delay time
        

        # Deadband settings
        self.prevRefPositions = [0]*8
        self.motionDirection = [1]*8
        self.motorDeadband = 0.005

        self.loadSimulator(False)

        self.isRunning = False
        self.simThread = None

        self.t = 0

        self.status_player1 = 0
        self.status_player2 = 0

        self.motorCommandsExternal1 = []
        self.motorCommandsExternal2 = []

        self.round = 0 # Štetje rund učenja
        self.save_interval = 100

        # --- Auto-curriculum: gradually widen ball spawn range (Stage 1.5 -> Stage 2) ---
        # `self.round` increments on episode resets/terminations, so we use it as the
        # curriculum progress counter.
        # Stage 1.5 (easy): y_range=(0.35, 0.45)
        # Stage 2 (harder): y_range=(0.15, 0.65)
        # Entire y range: 0.091 - 0.67
        self.curriculum_enabled = True
        self.curriculum_y_start = (0.45, 0.50)
        self.curriculum_y_end = (0.091, 0.67)
        self.curriculum_warmup_rounds = 100
        self.curriculum_ramp_rounds = 10000 #100000
        self.curriculum_print_every_rounds = 500
        self._last_curriculum_print_round = -1

        # Ball-control finetuning spawn setup for rod 6.
        # PyBullet x maps to camera x as: camera_x_mm = 1000 * x - 115.
        # Rod 6 is around camera_x=830 mm, so x ~= 0.945 m.
        self.controlled_rod_x_m = 0.945
        self.ball_spawn_areas = {
            "behind": (0.74, 0.88),
            "ahead": (1.02, 1.16),
        }
        self.ball_spawn_speed_range = (0.20, 0.50)

    def _ball_x_mm_camera(self) -> float:
        # Must match getCameraDict mapping
        return 1000.0 * float(self.ballPos[0]) - 115.0

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return float(a) + (float(b) - float(a)) * float(t)

    def _get_curriculum_y_range(self):
        """Return current (y_min, y_max) for ball spawn based on curriculum progress."""
        y0_min, y0_max = map(float, self.curriculum_y_start)
        y1_min, y1_max = map(float, self.curriculum_y_end)

        if not self.curriculum_enabled:
            y_min, y_max = y0_min, y0_max
        else:
            warmup = max(0, int(self.curriculum_warmup_rounds))
            ramp = max(1, int(self.curriculum_ramp_rounds))
            progress_rounds = max(0, int(self.round) - warmup)
            t = min(1.0, progress_rounds / float(ramp))
            y_min = self._lerp(y0_min, y1_min, t)
            y_max = self._lerp(y0_max, y1_max, t)

        if y_min > y_max:
            y_min, y_max = y_max, y_min
        return (y_min, y_max)

    def _init_kick_player_links(self):
        """Resolve which PyBullet link indices correspond to the observed rod's players."""
        self._kick_player_links = set()
        self._link_names_by_index = {}
        try:
            joints_num = p.getNumJoints(self.mizaId)
            rod_prefix = f"rod{self.kick_observed_rod_id}_rigid"
            for ji in range(joints_num):
                jinfo = p.getJointInfo(self.mizaId, ji)
                # jointName is bytes at index 1, linkName is bytes at index 12.
                jname = None
                link_name = None
                try:
                    jname = jinfo[1].decode("utf-8")
                except Exception:
                    jname = str(jinfo[1])
                try:
                    link_name = jinfo[12].decode("utf-8")
                except Exception:
                    link_name = str(jinfo[12])
                self._link_names_by_index[ji] = link_name
                if jname.lower().startswith(rod_prefix):
                    # In PyBullet, the joint index corresponds to the child link index.
                    self._kick_player_links.add(ji)
            if self.debug_print_red_kicks:
                print(
                    f"Kick detection watches rod {self.kick_observed_rod_id} "
                    f"player links: {sorted(self._kick_player_links)}"
                )
        except Exception:
            # Fallback: keep empty set; kick flag will remain False.
            self._kick_player_links = set()

    def _update_kick_latch(self):
        """Latch True if the ball contacts the observed rod's player links in this interval."""

        # Cooldown to avoid spamming while in continuous contact
        #print(f"current time:{self.t}, last red kick time:{self._last_debug_red_kick_t}, cooldown: {self.debug_red_kick_cooldown_s}")
        if (self.t - self._last_debug_red_kick_t) < self.debug_red_kick_cooldown_s:
            return

        if not self._kick_player_links:
            if self.debug_print_red_kicks and not self._warned_empty_kick_links:
                print(
                    "Kick detection has no player links to watch. "
                    "Check kick_observed_rod_id and URDF joint names."
                )
                self._warned_empty_kick_links = True
            return

        # Query only contacts between the ball and the table URDF. The players are
        # links on self.mizaId, while the separate concave table mesh is
        # self.mizaCollisionId. This avoids counting wall/table contacts as kicks.
        cps = p.getContactPoints(bodyA=self.ball, bodyB=self.mizaId)
        if not cps:
            return

        for cp in cps:
            # Because bodyA is the ball and bodyB is self.mizaId, cp[4] is the
            # contacted link index on the table URDF. We only count contacts with
            # the observed rod's player links, not rods, bearings, walls, or base.
            table_link = cp[4]
            normal_force = cp[9]
            is_observed_player = table_link in self._kick_player_links
            is_solid_contact = normal_force >= self.debug_red_kick_force_threshold

            if not (is_observed_player and is_solid_contact):
                continue

            self._ball_kicked_latch = True
            if self.terminate_episode_on_kick:
                self._kick_terminated_latch = True
                self.end_episode_latch = True
            self._last_debug_red_kick_t = self.t

            # Debug printing is intentionally separate from latch logic. Turning
            # logs off should never change the reward/event behavior.
            if self.debug_print_red_kicks:
                link_name = self._link_names_by_index.get(table_link, "unknown")
                #print(f"Ball kicked by link {table_link} ({link_name}), force={normal_force:.3f}")

            return

        if self.debug_print_any_ball_contact:
            strongest = max(cps, key=lambda cp: cp[9])
            link_name = self._link_names_by_index.get(strongest[4], "unknown")
            print(
                "Ball touched table URDF but not watched player link "
                f"(link={strongest[4]} ({link_name}), force={strongest[9]:.3f})"
            )

    def _update_x_threshold_termination(self):
        """If ball_x is below threshold, latch termination and reset the ball (start new episode)."""
        if self._x_threshold_terminated_latch:
            return

        try:
            ball_x_mm = self._ball_x_mm_camera()
        except Exception:
            return

        #print(f"Ball x pos: {ball_x_mm:.1f} mm")

        # NOTE: Point of termination due to x-threshold crossing
        # Added another threshold for detecting when the ball failed to control the ball coming from behind
        #if ball_x_mm < self.episode_end_ball_x_threshold_mm:
        if ball_x_mm < self.episode_end_ball_x_threshold_mm or ball_x_mm > self.episode_end_ball_other_x_threshold_mm:
            self._x_threshold_terminated_latch = True
            self.ResetBallToLocation()
            self.round += 1
            self.reset_step_counter()

    def getCameraDict(self, player = 1): 
        ball_x, ball_y = 1000*self.ballPos[0] - 115, 730 - 1000*self.ballPos[1]
        ball_vx, ball_vy = self.ballVel[0][0], -self.ballVel[0][1]

        # Simple camera model
        camPos = [ [ 100, 350 ],  [ 1100, 350 ]] # Camera position
        z0 = 0.191 # z-position of the ball on the table

        camCorr = []
        ballSize = []

        for ci in range(2):
            dPos = [ ball_x - camPos[ci][0], ball_y - camPos[ci][1] ]
            d = math.sqrt(dPos[0]**2 + dPos[1]**2)    
            
            # Simulate shift in ball position based on ball's z-axis position
            camCorr.append([dPos[0] / d * (self.ballPos[2] - z0) * 100,   dPos[1] / d * (self.ballPos[2] - z0) * 100 ])
            ballSize.append(35 * 1e3/d)

        rp = self.rodPositions
        ra = self.rodAngles

        if player == 2:
            # Reverse the field
            ball_x = 1210 - ball_x
            ball_y = 700 - ball_y        

            ball_vx = -ball_vx
            ball_vy = -ball_vy
            
            rp = [1-rp[7-i] for i in range(8)]   
            ra = [-ra[7-i] for i in range(8)]     
        
        cam1 = { "cameraID": 0, 
                "ball_x": ball_x + camCorr[0][0] + (random.random() - 0.5) * self.ballPosNoise, "ball_y": ball_y + camCorr[0][1] + (random.random() - 0.5) * self.ballPosNoise, 
                "ball_vx": ball_vx, "ball_vy": ball_vy, "ball_size": ballSize[0], 
                "rod_position_calib": rp, "rod_angle": ra }

        cam2 = { "cameraID": 1, 
                "ball_x": ball_x + camCorr[1][0] + (random.random() - 0.5) * self.ballPosNoise, "ball_y": ball_y + camCorr[1][1] + (random.random() - 0.5) * self.ballPosNoise, 
                "ball_vx": ball_vx, "ball_vy": ball_vy, "ball_size": ballSize[1], 
                "rod_position_calib": rp, "rod_angle": ra }

        # Game score (like 14:13 or 24:29), not reward score
        if player == 1:
            score = self.score.copy()  # Copy otherwise all sampled data will contain the same reference to an array which will update itself.
        else:
            score = self.score[::-1]

   
        return {
            "camData": [cam1, cam2],
            "camDataOK": [True, True],
            "score": score,
            "curriculum_round": int(self.round),
            "curriculum_y_min": float(self._get_curriculum_y_range()[0]),
            "curriculum_y_max": float(self._get_curriculum_y_range()[1]),
            # New observation/event flags
            "ball_kicked": bool(self._ball_kicked_latch),
            "terminated_by_kick": bool(self._kick_terminated_latch),
            "terminated_by_x_threshold": bool(self._x_threshold_terminated_latch),
            "end_episode": bool(self.end_episode_latch),
        }

    def reset_step_counter(self):
        self.current_step = 0
        self.showCurrentStep()

    def showScore(self):
        if self.scoreDisp is not None:
            p.removeUserDebugItem(self.scoreDisp)

        self.scoreDisp = p.addUserDebugText(f"Score {self.score[0]}:{self.score[1]}", [-0.1, -0.5, 0.1],
                                            textColorRGB=[0, 0, 0],
                                            textSize=2,
                                            parentObjectUniqueId=self.mizaId)
    
    def showRound(self):    
        if self.roundDisp is not None:
            p.removeUserDebugItem(self.roundDisp)

        self.roundDisp = p.addUserDebugText(f"Round #{self.round+1}", [-0.1, -0.55, 0.1],
                                            textColorRGB=[0, 0, 0],
                                            textSize=2,
                                            parentObjectUniqueId=self.mizaId)

    def showPlayerStatus(self):
        if self.playerStatusDisp1 is not None:
            p.removeUserDebugItem(self.playerStatusDisp1)

        self.playerStatusDisp1 = p.addUserDebugText(f"Player 1: {'Demo' if self.status_player1 == 0 else 'External'}", [-0.8, -0.5, 0.1],
                                            textColorRGB=[1, 0, 0],
                                            textSize=1.5,
                                            parentObjectUniqueId=self.mizaId)

        if self.playerStatusDisp2 is not None:
            p.removeUserDebugItem(self.playerStatusDisp2)

        self.playerStatusDisp2 = p.addUserDebugText(f"Player 2: {'Demo' if self.status_player2 == 0 else 'External'}", [0.4, -0.5, 0.1],
                                            textColorRGB=[0, 0, 1],
                                            textSize=1.5,
                                            parentObjectUniqueId=self.mizaId)

    def showCurrentStep(self):
        replace_id = self.stepDisp if self.stepDisp is not None else -1
        self.stepDisp = p.addUserDebugText(
            f"Step {self.current_step}/{self.max_num_steps}",
            [-0.1, -0.60, 0.1],
            textColorRGB=[0, 0, 0],
            textSize=1.5,
            parentObjectUniqueId=self.mizaId,
            replaceItemUniqueId=replace_id,
        )

    def sampleCameras(self, t):
        self.delayedMemory.append((t, self.getCameraDict(1), self.getCameraDict(2)))
        #print(self.delayedMemory[-1])
        #print(f"Len of delayed memory: {len(self.delayedMemory)}")
        #pprint(self.delayedMemory)

        while len(self.delayedMemory) > 0 and t - self.delayedMemory[0][0] > self.maxMemory:
            self.delayedMemory.pop(0)

    def getDelayedCamera(self, player, t):
        if len(self.delayedMemory) == 0:
            return None

        for i in range(len(self.delayedMemory)):
            if self.delayedMemory[i][0] >= t:
                return self.delayedMemory[i][player]
    
        # Return first (the oldest) by default
        return self.delayedMemory[0][player]


    ### --- Function for spawning ball at specified location --- ###

    def ResetBallToLocation(self, mark_episode_end=True):
        # Randomize the drop position within specified ranges

        # IMPORTANT: Tilted groudn from 0.0 - 0.9 and from 0.67 on

        # y_range - the width of the field (shorter side)
        # zone - the length of the field (logner side)
        # --- KICKING TRAINING ---
        

        
        # Stage 1.5 -> Stage 2 curriculum: widen the y_range gradually.
        # If training is enabeled -> Run curriculum learniing ball spawn 
        #y_range = self._get_curriculum_y_range()
        y_range = (0.091, 0.67)
        #y_range = (0.20, 0.55)
        #y_range = (0.30, 0.55)
        spawn_side, x_range = random.choice(list(self.ball_spawn_areas.items()))

        custom_x = random.uniform(*x_range)
        custom_y = random.uniform(*y_range)
        speed = random.uniform(*self.ball_spawn_speed_range)
        custom_z = 0.2  # Ensure it's above the table to avoid collision

        custom_ball_pos = [custom_x, custom_y, custom_z]

        # Reset the ball to the randomized safe location
        p.resetBasePositionAndOrientation(self.ball, custom_ball_pos, p.getQuaternionFromEuler([0, 0, 0]))

        # Send the ball toward rod 6 from either side.
        x_sign = 1.0 if spawn_side == "behind" else -1.0
        rnd_vector_x = random.uniform(0.35, 1.0)
        rnd_vector_y = random.uniform(-0.35, 0.35)
        velocity_vector = [x_sign * rnd_vector_x, rnd_vector_y, 0.0]
        norm = (velocity_vector[0]**2 + velocity_vector[1]**2) ** 0.5
        velocity = [v / norm * speed for v in velocity_vector]

        # Apply velocity to the ball
        p.resetBaseVelocity(self.ball, linearVelocity=velocity, angularVelocity=[0, 0, 0])

        # Mark episode end (but not on initial startup placement)
        if mark_episode_end and self._end_episode_armed:
            self.end_episode_latch = True

        self.showRound()

    def placeBall(self, position, velocity=[0, 0, 0]):
        """
        Place the ball at a specific position with a specified velocity.

        :param position: List or tuple of (x, y, z) coordinates for the ball's position.
        :param velocity: List or tuple of (vx, vy, vz) for the ball's velocity. Default is [0, 0, 0].
        """
        # Set the ball's position
        p.resetBasePositionAndOrientation(self.ball, position, p.getQuaternionFromEuler([0, 0, 0]))

        # Set the ball's velocity
        p.resetBaseVelocity(self.ball, linearVelocity=velocity, angularVelocity=[0, 0, 0])

    def applyMotorDeadband(self, i, newPos):    
        motionDiff = newPos - self.prevRefPositions[i]

        if motionDiff > self.motorDeadband and self.motionDirection[i] <= 0:
            # Changed direction - forward
            self.motionDirection[i] = 1
        elif motionDiff < -self.motorDeadband and self.motionDirection[i] >= 0:
            # Changed direction - backward
            self.motionDirection[i] = -1
        elif (self.motionDirection[i] >= 0 and motionDiff > 0) or (self.motionDirection[i] <= 0 and motionDiff < 0):
            # Change nothing - motion in the same direction
            pass    
        else:
            # Inside the deadband - ignore motion
            return self.prevRefPositions[i]

        self.prevRefPositions[i] = newPos
        return newPos

    def loadSimulator(self, printJointInfo = False):
        print("Loading simulator...")
        physicsClient = p.connect(p.GUI)    # graphical version
        #physicsClient = p.connect(p.DIRECT) # non-graphical version

        #p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,0)
        #p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        #p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS,1)
        #p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,1)
        #p.setPhysicsEngineParameter(enableFileCaching=0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0,cameraPitch=-80, cameraTargetPosition=[0.72,0.375,0])

        p.setGravity(0,0,-9.8)

        # Load basic plane
        planeId = p.loadURDF("plane.urdf")

        shift = [0,0,0]
        meshScale = [0.001, 0.001, 0.001]

        mizaStartPos = [0,0,0]
        mizaStartOrientation = p.getQuaternionFromEuler([0,0,0])

        # Import collision model - miza
        print("Loading FuzbAI table model...")
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName="meshes/Miza.obj",
                                            rgbaColor=[1, 1, 1, 1],
                                            specularColor=[0.4, .4, 0],
                                            visualFramePosition=[0,0,0],
                                            meshScale=[1e-5, 1e-5, 1e-5]) # Very small visual model
                                        
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                            fileName="meshes/Miza.obj",
                                            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                                            collisionFramePosition=shift,
                                            meshScale=meshScale)

        mizaCollisionId = p.createMultiBody(baseMass=0,
                                            baseInertialFramePosition=[0, 0, 0],
                                            baseOrientation=p.getQuaternionFromEuler([math.pi/2,0,math.pi/2]),
                                            baseCollisionShapeIndex=collisionShapeId,
                                            baseVisualShapeIndex=visualShapeId,
                                            basePosition=[0,0,0],
                                            useMaximalCoordinates=True)

        self.mizaCollisionId = mizaCollisionId
        
        # Import main URDF model
        self.mizaId = p.loadURDF("urdf/miza_garlando.urdf",mizaStartPos, mizaStartOrientation, useFixedBase=1)

        # Resolve link indices for kick detection on the controlled rod
        self._init_kick_player_links()

        if printJointInfo:
            jointsNum = p.getNumJoints(self.mizaId)
            for i in range(jointsNum):
                jInfo = p.getJointInfo(self.mizaId, i)
                print(jInfo)

        # Load the ball
        print("Loading ball model...")
        self.ball = p.loadURDF("sphere_small.urdf", self.defaultBallPos, p.getQuaternionFromEuler([0,0,0]), globalScaling = 0.035 / (2*0.03)) # sphere_small has radius of 3 cm
        p.changeVisualShape(self.ball, -1, rgbaColor=[1,1,0,1])

        # Adjust dynamics
        p.changeDynamics(self.ball, -1, mass=0.0287, lateralFriction=0.2, rollingFriction=0.00005, spinningFriction=0.01, restitution=0.7, linearDamping=0)
        p.changeDynamics(mizaCollisionId, -1, restitution=0.8)

        # Set player colors
        for rp in self.redPlayers:
            p.changeVisualShape(self.mizaId, rp, rgbaColor=[1,0,0,1])

        for bp in self.bluePlayers:
            p.changeVisualShape(self.mizaId, bp, rgbaColor=[0,0,1,1])

        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0,cameraPitch=-80, cameraTargetPosition=[0.72,0.375,0])

        # Use explicit stepping instead of real-time simulation. This makes contact
        # events easier to catch because _update_kick_latch() runs immediately
        # after each p.stepSimulation() call in the main loop.
        p.setRealTimeSimulation(0)
        p.setTimeStep(self.physics_timestep)

    def run(self):
        self.isRunning = True
        self.simThread = threading.Thread(target=self.__run)
        self.simThread.start()

    def stop(self):
        self.isRunning = False

    def __run(self):
        self.t = 0.0

        refPos = 0
        prev_t = 0
        ball_moving = 0
            
        print(f'\n*********************************\nStarting main loop\n*********************************\n')

        self.showScore()
        self.showRound()
        self.ResetBallToLocation()
        self.showPlayerStatus()
        # Arm end_episode latch only after initial setup reset
        self.end_episode_latch = False
        self._end_episode_armed = True

        prev_key_t = 0

        PARAM_linVel = 1
        PARAM_force = 5                                         
        PARAM_positionGain = 0.2
        PARAM_velocityGain = 4.5

        try:   

            """ Code that checks, if a goal was scored in the step """ 
            while self.isRunning:    
                # Advance several small physics steps per Python loop. The contact
                # latch is still updated immediately after each physics step, which
                # preserves kick detection while avoiding one full Python control
                # pass for every 1/240 s tick.
                for _ in range(self.physics_steps_per_loop):
                    p.stepSimulation()
                    self.t += self.physics_timestep
                    self._update_kick_latch()

                self.ballPos, ballOrn = p.getBasePositionAndOrientation(self.ball)        
                self.ballVel = p.getBaseVelocity(self.ball)


                if self.ballPos[2] < 0.1:
                    goal_scored_this_step = False
                    #print(ballPos)
                    # Is the ball under the table?
                    if (self.ballPos[0] > 0 and self.ballPos[0] < 1.4 and self.ballPos[1] > 0 and self.ballPos[1] < 0.7):
                        goal_scored_this_step = True
                        # On which side?
                        if self.ballPos[0] < 0.72:
                            # Blue scored a goal
                            self.score[1] += 1
                            print(f'Blue scored goal ({self.score[0]}:{self.score[1]})')
                        else:
                            # Red scored a goal
                            self.score[0] += 1
                            #print(f'Red scored goal ({self.score[0]}:{self.score[1]})')

                        self.showScore()
                        self.showRound()

                    # Safe drop coordinates within table limits
                    self.ResetBallToLocation()
                    self.round += 1
                    self.reset_step_counter()
                
                # Update episode/reset latches after the latest sampled ball state.
                self._update_x_threshold_termination()

                angles = []
                rodPoses = []     
                
                for ji in range(8):        
                    angles.append(32*p.getJointState(self.mizaId, self.revJoints[ji])[0] / math.pi)

                # Linear...
                for ji in range(8):                         
                    rodPoses.append(1-1000*p.getJointState(self.mizaId, self.slideJoints[ji])[0] / self.travels[ji])
                
                self.rodPositions = rodPoses
                self.rodAngles = angles

                #linVel = 1.5910861528058136
                rotVel = 174.74649915501303

                # Process the agents...
                if self.t - prev_t > self.control_dt:

                    self.current_step += 1
                    self.showCurrentStep()


                    try:     
                        if self.status_player1 == 0:             
                            # Robust (no-delay) training: feed the current snapshot directly.
                            motors1 = self.p1.process_data(self.getCameraDict(1))
                        else:
                            # Use the external motor data...
                            motors1 = self.motorCommandsExternal1
                            self.motorCommandsExternal1 = []        

                        driveMap = [0, 1, 3, 5]
                        for m in motors1:
                            axisID = driveMap[m["driveID"]-1]
                            jId_rot = self.revJoints[axisID]
                            jId_lin = self.slideJoints[axisID]

                            refAngle = m["rotationTargetPosition"]*2*math.pi                
                            p.setJointMotorControl2(self.mizaId, jId_rot, controlMode=p.POSITION_CONTROL, targetPosition=refAngle, force=2.0943448919793832, maxVelocity=rotVel*m["rotationVelocity"], positionGain=2.817867199313025, velocityGain=7.574019729635704)

                            refPos = self.applyMotorDeadband(axisID, self.travels[axisID]*(1-m["translationTargetPosition"])/1000)
                            #p.setJointMotorControl2(self.mizaId, jId_lin, controlMode=p.POSITION_CONTROL, targetPosition=refPos, force=13.303989530423438, maxVelocity=linVel*m["translationVelocity"], positionGain=0.19343157707177333, velocityGain=3.9227062400839023)
                            p.setJointMotorControl2(self.mizaId, jId_lin, controlMode=p.POSITION_CONTROL, targetPosition=refPos, 
                                                     force=PARAM_force, 
                                                     maxVelocity=PARAM_linVel*m["translationVelocity"], 
                                                     positionGain=PARAM_positionGain, 
                                                     velocityGain=PARAM_velocityGain)
                    except:
                        print("Exception in agent 1")
                        traceback.print_exc()

                    try:                                               
                        if self.status_player2 == 0:             
                            # Robust (no-delay) training: feed the current snapshot directly.
                            motors2 = self.p2.process_data(self.getCameraDict(2))
                        else:
                            # Use the external motor data...
                            motors2 = self.motorCommandsExternal2
                            self.motorCommandsExternal2 = []        

                        driveMap = [7, 6, 4, 2]                        
                        for m in motors2:
                            axisID = driveMap[m["driveID"]-1]
                            jId_rot = self.revJoints[axisID]
                            jId_lin = self.slideJoints[axisID]

                            refAngle = -m["rotationTargetPosition"]*2*math.pi                
                            p.setJointMotorControl2(self.mizaId, jId_rot, controlMode=p.POSITION_CONTROL, targetPosition=refAngle, force=2.0943448919793832, maxVelocity=rotVel*m["rotationVelocity"], positionGain=2.817867199313025, velocityGain=7.574019729635704)

                            refPos = self.applyMotorDeadband(axisID, self.travels[axisID]*(m["translationTargetPosition"])/1000)
                            #p.setJointMotorControl2(self.mizaId, jId_lin, controlMode=p.POSITION_CONTROL, targetPosition=refPos, force=13.303989530423438, maxVelocity=linVel*m["translationVelocity"], positionGain=0.19343157707177333, velocityGain=3.9227062400839023)
                            p.setJointMotorControl2(self.mizaId, jId_lin, controlMode=p.POSITION_CONTROL, targetPosition=refPos, 
                                                     force=PARAM_force, 
                                                     maxVelocity=PARAM_linVel*m["translationVelocity"], 
                                                     positionGain=PARAM_positionGain, 
                                                     velocityGain=PARAM_velocityGain)
                    except:
                        print("Exception in agent 2")

                    prev_t = self.t

                    episode_ended_this_control_step = bool(
                        self._kick_terminated_latch
                        or self._x_threshold_terminated_latch
                        or self.end_episode_latch
                    )
                    reset_after_kick = bool(self._kick_terminated_latch)

                    # Clear per-step latches after agents have consumed the observation stream
                    self._ball_kicked_latch = False
                    self._kick_terminated_latch = False
                    self._x_threshold_terminated_latch = False
                    self.end_episode_latch = False

                    if reset_after_kick:
                        self.ResetBallToLocation(mark_episode_end=False)
                        self.round += 1
                        self.reset_step_counter()
                    elif (not episode_ended_this_control_step) and self.current_step >= self.max_num_steps:
                        self.ResetBallToLocation()
                        self.round += 1
                        self.reset_step_counter()     

                # Zajemanje podatkov (gol, konec, podatki iz kamer)
                self.sampleCameras(self.t)


                # Checks, if the goal was scored to end the iteration
                keys = p.getKeyboardEvents()
                if self.t - prev_key_t > 0.1:
                    for k, v in keys.items():        
                        if (k == 65309 and (v & p.KEY_WAS_TRIGGERED)): # 65309 == enter
                            # Move the ball over the table
                            #p.resetBasePositionAndOrientation(self.ball, self.defaultBallPos, p.getQuaternionFromEuler([0,0,0]))
                            self.ResetBallToLocation()
                        if (k == 32): # Esc
                            running = False 
                            break            

                        # Enable/disable player 1
                        if (k == 49): # 1
                            self.status_player1 = (self.status_player1 + 1) % 2
                            self.showPlayerStatus()
                            pass    

                        # Enable/disable player 2
                        if (k == 50): # 2
                            self.status_player2 = (self.status_player2 + 1) % 2
                            self.showPlayerStatus()
                            pass    

                if len(keys) > 0:
                    prev_key_t = self.t

                if self.gui_sleep_s > 0:
                    time.sleep(self.gui_sleep_s)

            print("Stopping simulation...")
            p.disconnect()
            print("Stopping server...")
            print("Done")

        except KeyboardInterrupt:
            print("Stopping simulation on keyboard interrupt...")
            p.disconnect()
            print("Stopping server...")

        except:
            pass

        print(f'Main loop stopped')

        self.isRunning = False

if __name__ == "__main__":
    print("Working with up-to-date code")
    setup_logging()
    sim = FuzbAISim()
    sim.run()

    try:
        while sim.isRunning:
            time.sleep(0.1)
    except KeyboardInterrupt:
        sim.stop()
        if sim.simThread is not None:
            sim.simThread.join()
