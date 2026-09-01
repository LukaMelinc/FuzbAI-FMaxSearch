import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import requests
import math
import json
import csv
from datetime import datetime
from pprint import pprint

from actor_critic.main import ActorCriticNet, TwoRodActorCriticNet, ThreeRodActorCriticNet
from export.main import Export
from memory.main import PPOBuffer
from reward.single_bar_shoting import (
    closest_player_alignment_reward,
    controllable_kick_reward,
    kick_force_reward,
    kicking_reward,
    player_alignment_target,
    simple_reward,
    maintaining_ball,
    calculate_rod_angle_reward,
    predictive_player_alignment_reward
)


HOST_ADDRESS = '127.0.0.1:23336'  # IP or Host for your environment

def get_camera_state():
    """Gets the data from the camera"""
    cam_url = f"http://{HOST_ADDRESS}/Camera/State"
    response = requests.get(cam_url)
    return response.json()

def send_motor_commands(cmds):
    """Sends REST communication to the HW"""
    motors_url = f"http://{HOST_ADDRESS}/Motors/SendCommand?blue=False"
    requests.post(motors_url, json=cmds)

def calculate_player_positions_and_angles(camera, geometry):
    "Function that calculates the state parameters of the rods in the enviroment"
    # NOTE: Not used for single row training, activate later
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

def mlp_gaussian_likelihood(action, mean, log_std):
    """
    Compute log likelihood of a Gaussian distribution with diagonal covariance
    (log_std is a vector).
    """
    # 1. Compute σ from log_std
    std = torch.exp(log_std)  # σ = e^(log_std)

    # 2. Standardized residual: (action - mean) / σ
    z = (action - mean) / std

    # 3. Compute the log-probability for each action dimension
    #    Formula: -0.5 * z² - log(σ) - 0.5·log(2π)
    pre_sum = -0.5 * (z**2 + 2*log_std + np.log(2*np.pi))
    #pre_sum = -0.5 * (((action - mean) / (torch.exp(log_std)))**2 + 2*log_std + np.log(2*np.pi))
    return torch.sum(pre_sum, axis=1)

def atanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def squashed_gaussian_likelihood(pre_tanh_action, squashed_action, mean, log_std):
    gaussian_logp = mlp_gaussian_likelihood(pre_tanh_action, mean, log_std)
    correction = torch.sum(torch.log(1 - squashed_action.pow(2) + 1e-6), axis=1)
    return gaussian_logp - correction


class PPOAgent:
    """
    PPO RL agent that:
      - Holds a policy network (actor-critic).
      - On each call to `process_data(camera)`, picks an action in [-1,1] for each rod.
      - Uses a buffer to accumulate experiences for PPO updates.
    """
    def __init__(self,
                 obs_dim=8, #ball(4) + controlled rod rot and trans(2) + 2 engineered features (TODO: Keep them or change to 6)    #10, # ball(4) + rod(5) + ball_kicked(1)
                 act_dim=4, #act_dim=4,          # 1 rod × 4 numbers each
                 hidden_size=512,
                 steps_per_env=512,#256,  # how many steps per iteration
                 gamma=0.99,
                 lam=0.95,
                 clip_ratio=0.2,
                 lr=1e-4,
                 train_iters=4,
                 target_kl=0.01,
                 delay_step=2,
                 save_model_every=1500,   # save every N episodes
                 model_save_path="./trained_models/STAGE_1_shooting_still_ball_a_bit_bigger_ball_spawn_area_MODEL#6.pth",   # Path for loading the model from
                 training_log_export_every=10,
                 l2_lambda=5e-4,
                 controlled_rod_id=4,
                 training_enabeled=True,
                 load_model=False,
                 pretrained_actor_path=None,
                 load_pretrained_log_std=False,
                 inference=False,
                 auto_train=True,
                 action_std_override=(1.5, 1.5, 0.4, 0.4)):      # std override - fist two rod rotation, last two rod translation

        self.controlled_rod_id = controlled_rod_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.save_model_every = save_model_every
        self.model_save_path = model_save_path      # Path from where a trained model is loaded
        self.model_name = "#18"
        self.training_log_export_every = training_log_export_every
        self.l2_lambda = l2_lambda  # L2 regularization strength
        self.inference = inference
        self.should_load_model = load_model
        self.pretrained_actor_path = pretrained_actor_path
        self.load_pretrained_log_std = bool(load_pretrained_log_std)
        self.training_enabled = training_enabeled # For training mode vs. inference mode
        self.action_std_override = action_std_override
        self.auto_train = bool(auto_train)
        self.pending_train = False
        
        if self.inference and self.training_enabled:
            print("[PPOAgent] Inference mode enabled, disabling training.")
            self.training_enabled = False

        self.prev_vel = None
        self.prev_ball_vxy = None
        self.prev_score = None
        self.episode_steps = 0

        

        with open('geometry.json') as f:
            self.geometry = json.load(f)

        self.rods_by_id = {int(rod["id"]): rod for rod in self.geometry["rods"]}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Actor-Critic network
        self.ac = ActorCriticNet(obs_dim, act_dim, hidden_size)
        self.ac.to(self.device)

        # Separate or shared log_std for continuous actions
        self.log_std = nn.Parameter(-1*torch.ones(act_dim, dtype=torch.float32, device=self.device), requires_grad=True)
        self.log_std = self.log_std.to(self.device)

        # Imitation pretraining and task PPO have different critics/rewards.  For
        # RL fine-tuning, transfer only the policy (and optionally its exploration
        # scale) while leaving this newly-created critic untouched.
        if self.pretrained_actor_path is not None:
            self.load_pretrained_actor(
                self.pretrained_actor_path,
                load_log_std=self.load_pretrained_log_std,
            )

        # Optimizer
        self.optimizer = optim.Adam(list(self.ac.parameters()) + [self.log_std], lr=lr)

        # PPO hyperparameters
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.target_kl = target_kl

        # Replay buffer
        self.steps_per_env = steps_per_env
        self.buf = PPOBuffer(obs_dim, act_dim, steps_per_env, gamma, lam)

        # Tracking
        self.episode_count = 0   # how many episodes finished so far
        self.current_step = 0    # how many steps in the current episode
        self.ep_reward = 0.0     # reward this episode
        self.last_obs = None     # store last observation
        self.episode_rewards = []
        self.reward = 0
        self.total_steps = 0
        self.training_count = 0
        self.training_export = Export()

        # Episode statistics
        self.episode_stats = []  # Store detailed stats for each episode
        self.current_episode_goals = 0
        self.current_episode_opponent_goals = 0
        self.current_episode_ball_kicks = 0
        self.current_episode_x_threshold_terminations = 0
        self.current_episode_step_rewards = []  # Track reward at each step
        self.episodes_with_kick = 0
        self.episodes_with_goal = 0
        self.kick_rate_threshold = 0.8
        self.latest_env_metrics = {}
        self.update_reward_breakdown_sums = {}
        self.update_samples_with_kick = 0

        if self.should_load_model:
            self.load_model()

    @staticmethod
    def _as_float(value):
        if isinstance(value, list):
            value = value[0]
        return float(value)

    def policy_log_std(self, mean):
        return torch.clamp(self.log_std, min=-5.0, max=2.0).unsqueeze(0).expand_as(mean)

    def action_learning_slice(self):
        return getattr(self, "learning_action_slice", slice(None))

    def action_log_prob(self, pre_tanh_action, action, mean, log_std):
        action_slice = self.action_learning_slice()
        return squashed_gaussian_likelihood(
            pre_tanh_action[..., action_slice],
            action[..., action_slice],
            mean[..., action_slice],
            log_std[..., action_slice],
        )

    def compute_ball_control_reward(
        self,
        bxy,
        vxy,
        active_rod,
        *,
        terminated_by_x_threshold,
        end_episode,
    ):
        """Additional reward for intercepting, slowing, and controlling the ball."""
        controlled_rod_x = float(active_rod["info"]["position"])
        prev_ball_vx = self.prev_ball_vxy[0] if self.prev_ball_vxy is not None else None
        prev_ball_vy = self.prev_ball_vxy[1] if self.prev_ball_vxy is not None else None

        # For the red controlled rods, lower camera x is the protected side.
        # Penalize once the ball slips behind the rod even before the simulator
        # threshold ends the episode.
        ball_behind_rod = float(bxy[0]) < (controlled_rod_x - 10.0)

        reward, reward_breakdown = maintaining_ball(
            ball_x=bxy[0],
            ball_vx=vxy[0],
            ball_vz=vxy[1],
            rod_x_pos=controlled_rod_x,
            threshold_crossed=terminated_by_x_threshold,
            ball_behind_rod=ball_behind_rod,
            prev_ball_vx=prev_ball_vx,
            prev_ball_vz=prev_ball_vy,
            episode_timeout=bool(end_episode and not terminated_by_x_threshold),
        )

        return reward, {
            f"ball_control_{key}": value
            for key, value in reward_breakdown.items()
        }

    def save_model(self, path=None):
        if path is None:
            save_dir = self.model_save_path
            if os.path.splitext(save_dir)[1]:
                save_dir = os.path.dirname(save_dir) or "."
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{self.model_name}_steps_{self.total_steps}.pth")
        checkpoint = {
            "actor_critic_state_dict": self.ac.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "training_count": self.training_count,
        }

        # The original ActorCriticNet exposes a single ``actor`` module, while
        # the multi-head agents expose encoders/action heads directly.  The
        # complete actor-critic state above is the canonical checkpoint for all
        # architectures; retain these component entries where they exist for
        # compatibility with imitation-pretraining checkpoints.
        if hasattr(self.ac, "actor"):
            checkpoint["actor_state_dict"] = self.ac.actor.state_dict()
        if hasattr(self.ac, "critic"):
            checkpoint["critic_state_dict"] = self.ac.critic.state_dict()

        torch.save(checkpoint, path)
        print(f"[PPOAgent] Model saved to {path}")

    def load_pretrained_actor(self, checkpoint_path, *, load_log_std=False):
        """Initialize the policy from imitation while keeping a fresh RL critic."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or "actor_state_dict" not in checkpoint:
            raise KeyError(
                f"Imitation checkpoint {checkpoint_path!r} has no actor_state_dict"
            )
        self.ac.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        if load_log_std and "log_std" in checkpoint:
            source_log_std = checkpoint["log_std"].to(self.device)
            if source_log_std.shape != self.log_std.shape:
                raise ValueError(
                    "Imitation log_std shape does not match the RL action space: "
                    f"{tuple(source_log_std.shape)} != {tuple(self.log_std.shape)}"
                )
            self.log_std.data.copy_(source_log_std)
        print(
            f"[PPOAgent] Loaded imitation-pretrained actor from {checkpoint_path}; "
            "RL critic remains freshly initialized."
        )

    def load_model(self, path=None):

        #if not self.inference:
        #    print("Agent in training mode, skipping model load.")
        #    return
        
        if path is None:
            path = self.model_save_path
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and "actor_critic_state_dict" in checkpoint:
                self.ac.load_state_dict(checkpoint["actor_critic_state_dict"])
                if "log_std" in checkpoint:
                    self.log_std.data.copy_(checkpoint["log_std"].to(self.device))
                self.training_count = int(checkpoint.get("training_count", self.training_count))
            else:
                self.ac.load_state_dict(checkpoint)
            print(f"[PPOAgent] Model loaded from {path}")
        else:
            print("[PPOAgent] No saved model found, skipping load.")

    def compute_action(self, obs, deterministic=False):
        """
        Given a single observation (numpy array),
        return an action in [-1,1], value estimate, and log probability.
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mean, value_t = self.ac(obs_t)

            log_std = self.policy_log_std(mean)
            if deterministic:
                pre_tanh_action = mean
            else:
                std = torch.exp(log_std)
                pre_tanh_action = mean + std * torch.randn_like(mean)

            action_t = torch.tanh(pre_tanh_action)

            # Computing logp of the action executed
            #logp = squashed_gaussian_likelihood(pre_tanh_action, action_t, mean, log_std)
            logp = self.action_log_prob(pre_tanh_action, action_t, mean, log_std)

        # Squeeze out the batch dimension
        action = action_t.detach().cpu().numpy()[0]
        value  = value_t.detach().cpu().numpy()[0,0]
        logp   = logp.detach().cpu().numpy()[0]

        return action, value, logp

    def train_on_buffer(self):
        """
        Run PPO update once we have a full buffer (N steps).
        """
        buffer_sample_count = int(self.buf.ptr)
        accumulated_reward = float(np.sum(self.buf.rew_buf[:buffer_sample_count]))

        data = self.buf.get()  # get everything as torch tensors

        # Add validation
        if data is None:
            print("[Warning] No data in buffer to train on")
            return

        self.training_count += 1
        # Collect PPO diagnostics for this update (averaged over train_iters)
        update_metrics = {
            #"train_iters": int(self.train_iters),
            "clip_ratio": float(self.clip_ratio),
            #"target_kl": float(self.target_kl),
            #"lr": float(self.optimizer.param_groups[0].get("lr", 0.0)),
            "log_std_mean": float(self.log_std.detach().mean().item()),
            "log_std_min": float(self.log_std.detach().min().item()),
            "log_std_max": float(self.log_std.detach().max().item()),
        }
    

        obs = data["obs"].to(self.device)
        act = data["act"].to(self.device)
        ret = data["ret"].to(self.device)
        adv = data["adv"].to(self.device)
        logp_old = data["logp"].to(self.device)

        if not torch.isfinite(logp_old).all():
            print("NaN in logp_old!")

        # Track per-iteration stats and average them at the end
        stats = {
            "loss_pi": [],
            "loss_v": [],
            "loss_total": [],
            "approx_kl": [],
            "clip_frac": [],
            "entropy_gauss": [],
            "ratio_mean": [],
            "ratio_std": [],
            "grad_norm": [],
        }

        iters_done = 0
        early_stop = 0

        for i in range(self.train_iters):

            # Forward pass
            mean, value = self.ac(obs)
            log_std = self.policy_log_std(mean)

            # Compute log probability for the new actions
            safe_act = torch.clamp(act, -1.0 + 1e-6, 1.0 - 1e-6)
            pre_tanh_act = atanh(safe_act)
            #logp_pi = squashed_gaussian_likelihood(pre_tanh_act, safe_act, mean, log_std)
            logp_pi = self.action_log_prob(pre_tanh_act, safe_act, mean, log_std)

            # Ratio for surrogate loss
            ratio = torch.exp(logp_pi - logp_old)

            # PPO clip fraction (how often the ratio goes outside the clip range)
            clip_frac = ((ratio > (1 + self.clip_ratio)) | (ratio < (1 - self.clip_ratio))).float().mean()

            obj = ratio * adv
            clipped_obj = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv

            loss_pi = -torch.mean(torch.min(obj, clipped_obj))
            loss_vf = torch.mean((ret - value.squeeze())**2)

            # Calculate L2 regularization
            l2_norm = sum(p.pow(2.0).sum() for p in self.ac.parameters())
            loss = loss_pi + 0.5 * loss_vf + self.l2_lambda * l2_norm

            self.optimizer.zero_grad()
            loss.backward()

            # Global grad norm (helps detect exploding/vanishing grads)
            with torch.no_grad():
                grad_squares = []
                for p in list(self.ac.parameters()) + [self.log_std]:
                    if p.grad is None:
                        continue
                    grad_squares.append(torch.sum(p.grad.detach() ** 2))
                if grad_squares:
                    grad_norm = torch.sqrt(torch.sum(torch.stack(grad_squares))).item()
                else:
                    grad_norm = float("nan")

            self.optimizer.step()
            print("Backprop complete!")

            # Approximate KL divergence
            kl = torch.mean(logp_old - logp_pi).item()

            # Gaussian entropy (pre-tanh) as a proxy for exploration
            # H(N(mu, sigma)) = 0.5 * sum_d [1 + log(2*pi) + 2*log_std]
            #entropy = (0.5 * (1.0 + np.log(2.0 * np.pi)) + log_std).sum(dim=1).mean().item()
            action_slice = self.action_learning_slice()
            entropy_log_std = log_std[..., action_slice]
            entropy = (0.5 * (1.0 + np.log(2.0 * np.pi)) + entropy_log_std).sum(dim=1).mean().item()

            # Ratio stats
            ratio_mean = ratio.mean().item()
            ratio_std = ratio.std(unbiased=False).item()

            # Record stats
            stats["loss_pi"].append(loss_pi.item())
            stats["loss_v"].append(loss_vf.item())
            stats["loss_total"].append(loss.item())
            stats["approx_kl"].append(float(kl))
            stats["clip_frac"].append(clip_frac.item())
            stats["entropy_gauss"].append(float(entropy))
            stats["ratio_mean"].append(float(ratio_mean))
            stats["ratio_std"].append(float(ratio_std))
            stats["grad_norm"].append(float(grad_norm))

            iters_done += 1
            if kl > 1.5 * self.target_kl:
                print(f"[PPO] Early stopping at iter={i} due to reaching max kl.")
                early_stop = 1
                break

        # Explained variance of the value function (1 is best, 0 means no better than predicting mean)
        with torch.no_grad():
            _, v_pred = self.ac(obs)
            v_pred = v_pred.squeeze()
            var_y = torch.var(ret)
            if var_y.item() > 1e-8:
                explained_var = (1.0 - torch.var(ret - v_pred) / var_y).item()
            else:
                explained_var = float("nan")

        # Aggregate metrics
        def _mean(xs):
            return float(np.mean(xs)) if xs else float("nan")

        update_metrics.update({
            #"early_stop": int(early_stop),
            "loss_pi": _mean(stats["loss_pi"]),
            "loss_v": _mean(stats["loss_v"]),
            "loss_total": _mean(stats["loss_total"]),
            "approx_kl": _mean(stats["approx_kl"]),
            "clip_frac": _mean(stats["clip_frac"]),
            "entropy_gauss": _mean(stats["entropy_gauss"]),
            "ratio_mean": _mean(stats["ratio_mean"]),
            "ratio_std": _mean(stats["ratio_std"]),
            "grad_norm": _mean(stats["grad_norm"]),
            "explained_variance": float(explained_var),
            "ret_mean": float(ret.mean().item()),
            "ret_std": float(ret.std(unbiased=False).item()),
            "samples_with_kick": int(self.update_samples_with_kick),
        })

        if buffer_sample_count > 0:
            for key, value in self.update_reward_breakdown_sums.items():
                update_metrics[f"reward_{key}_mean"] = float(value) / float(buffer_sample_count)

        update_metrics.update(self.latest_env_metrics)

        self.training_export.add_training_result(
            training_number=self.training_count,
            accumulated_reward=accumulated_reward,
            sample_count=buffer_sample_count,
            metrics=update_metrics,
        )

        reward_per_sample = (
            accumulated_reward / float(buffer_sample_count)
            if buffer_sample_count > 0
            else float("nan")
        )

        print(
            f"[Training] #{self.training_count}: accumulated reward "
            f"{accumulated_reward:.3f} over {buffer_sample_count} samples "
            f"(reward/sample={reward_per_sample:.6f})"
        )

        if self.training_count % self.training_log_export_every == 0:
            self.training_export.export_csv()
            #self.save_model()

        self.update_reward_breakdown_sums = {}
        self.update_samples_with_kick = 0
        self.pending_train = False

    def process_data(self, camera):

        self.total_steps += 1
        #if self.total_steps % 250 == 0:
        #    print(f"Processing step {self.total_steps} at episode {self.episode_count}, step in episode: {self.episode_steps}")

        # Extract the current observation
        obs, bxy, vxy, rod_angle, active_rod = self.extract_observation(camera)
        self.latest_env_metrics = {
            "curriculum_round": int(camera.get("curriculum_round", -1)),
            "curriculum_y_min": float(camera.get("curriculum_y_min", float("nan"))),
            "curriculum_y_max": float(camera.get("curriculum_y_max", float("nan"))),
        }

        # Events from environment - zajem podatkov o brci žoge in terminaciji zaradi premajhne x vrednosti
        ball_kicked = bool(camera.get("ball_kicked", False))
        kick_normal_force = float(camera.get("kick_normal_force", 0.0))
        terminated_by_kick = bool(camera.get("terminated_by_kick", False))
        terminated_by_x_threshold = bool(camera.get("terminated_by_x_threshold", False))
        end_episode = bool(camera.get("end_episode", False))

        #if end_episode:
            #print(f"Enviroment singals episode end.")
        # Goal detection via score delta (more reliable than ball position thresholds)
        # Za potrditev gola
        score = camera.get("score", None)
        goal_scored = False
        opponent_goal_scored = False
        if isinstance(score, (list, tuple)) and len(score) >= 2:
            if self.prev_score is not None:
                goal_scored = score[0] > self.prev_score[0]
                opponent_goal_scored = score[1] > self.prev_score[1]
            self.prev_score = list(score)

        # If not training, just run the policy forward pass
        # Not important during training, doesn't execute
        if not self.training_enabled:
            action, _, _ = self.compute_action(obs, deterministic=self.inference)
            commands = self.scale_to_motor_commands(action)
            return commands

    
        # Training: collect the step
        episode_finished_this_sample = False
        if self.last_obs is not None:
            controlled_rod_x = float(active_rod["info"]["position"])
            ball_past_rod = float(bxy[0]) < (controlled_rod_x - 10.0)

            allignment_quality_reward = closest_player_alignment_reward(
                ball_y=bxy[1],
                rod_pos_calib=float(active_rod["pos_calib"]),
                rod_info=active_rod["info"],
                reward_scale=1.0,
            )

            predictive_allignment_reward = predictive_player_alignment_reward(
                ball_x=bxy[0],
                ball_y=bxy[1],
                ball_vx=vxy[0],
                ball_vy=vxy[1],
                rod_x=float(active_rod["info"]["position"]),
                rod_pos_calib=float(active_rod["pos_calib"]),
                rod_info=active_rod["info"],
                vx_threshold=0.05,
                t_max=0.5,
                reward_scale=1.0,
            )

            rod_angle_reward = calculate_rod_angle_reward(
                rod_angle=float(active_rod["angle"]),
                reward_scale=1,
                target_rod_angle=0.0,
                rotation_buffer_def=3
                )

            #print(f"allignment: {allignment_quality_reward}, angle:{rod_angle_reward}" )
            
            reward_breakdown = {
                #"allignment": allignment_quality_reward,
                "predictive_allignment": predictive_allignment_reward*0.5,
                "rod_angle": rod_angle_reward * 0.15,
                "past_rod": -1.0 if ball_past_rod else 0.0,
            }

            reward = sum(reward_breakdown.values())

            for key, value in reward_breakdown.items():
                self.update_reward_breakdown_sums[key] = (
                    self.update_reward_breakdown_sums.get(key, 0.0) + float(value)
                )
            if ball_kicked:
                self.update_samples_with_kick += 1

            # NEW: Track step-level reward
            self.current_episode_step_rewards.append(reward)
            
            # NEW: Track specific events
            if goal_scored:
                self.current_episode_goals += 1
            if opponent_goal_scored:
                self.current_episode_opponent_goals += 1
            if ball_kicked:
                self.current_episode_ball_kicks += 1
            if terminated_by_x_threshold:
                self.current_episode_x_threshold_terminations += 1

            if ball_past_rod:
                self.current_episode_x_threshold_terminations += 1

            # Shranitev celotne tranzicije (s_t-1, a_t-1, r_t, V_t-1)
            stored = self.buf.store(self.last_obs, self.last_action, reward, self.last_val, self.last_logp)
            
            if not stored:
                # Buffer is full - must train immediately
                _, final_value, _ = self.compute_action(obs)
                self.buf.finish_path(last_val=final_value)
                print(f"[Training] Buffer full, training now...")
                if self.auto_train:
                    self.train_on_buffer()
                else:
                    self.pending_train = True


                # Reset episode tracking
                self.episode_count += 1
                self.current_step = 0
                self.episode_steps = 0
                self.ep_reward = 0.0
                self.last_obs = None
                self.last_action = None
                self.last_val = None
                self.last_logp = None
                episode_finished_this_sample = True
            else:
                self.ep_reward += reward

            # Early episode termination on kick/x-threshold/environment reset
            if (not episode_finished_this_sample) and (terminated_by_kick or terminated_by_x_threshold or end_episode):
                self.finish_episode(last_value=0)
                self.episode_steps = 0
                episode_finished_this_sample = True

            if (not episode_finished_this_sample) and ball_past_rod:
                self.finish_episode(last_value=0)
                self.episode_steps = 0
                episode_finished_this_sample = True


        else:
            print(f"=====First step, no reward yet =====")

        if not episode_finished_this_sample:
            self.episode_steps += 1

        # Izračun akcije za trenutno stanje
        action, value, logp = self.compute_action(obs)

        if not np.all(np.isfinite(action)):
            print("Nan or Inf detected in action:", action)
            action = np.zeros_like(action)  # fallback to safe value


        self.prev_vel = vxy[0]
        self.prev_ball_vxy = vxy
        # store for next iteration
        self.last_obs = obs
        self.last_action = action
        self.last_val = value
        self.last_logp = logp
        self.current_step += 1

        commands = self.scale_to_motor_commands(action)
        return commands

    def _camera_data(self, camera):
        return camera["camData"][0] if camera["camData"][0] is not None else camera["camData"][1]

    def _rod_state(self, cd, rod_id):
        rod_idx = int(rod_id) - 1
        rod_info = self.rods_by_id[int(rod_id)]
        rod_pos_calib = self._as_float(cd["rod_position_calib"][rod_idx])
        rod_angle = self._as_float(cd["rod_angle"][rod_idx])
        rod_angle_normalized = float(np.clip(rod_angle / 32.0, -1.0, 1.0))
        _, target_rod_pos, unavoidable_dist = player_alignment_target(
            ball_y=cd["ball_y"],
            rod_info=rod_info,
        )
        target_error = float(target_rod_pos) - rod_pos_calib
        return {
            "info": rod_info,
            "pos_calib": rod_pos_calib,
            "angle": rod_angle,
            "angle_normalized": rod_angle_normalized,
            "target_pos": float(target_rod_pos),
            "target_error": target_error,
            "unavoidable_dist": float(unavoidable_dist),
        }

    def extract_observation(self, camera):
        """
        Convert camera dict into a flat numpy array (obs_dim).
        Fill in whatever you need: ball pos, ball vel, rod pos, rod angles, etc.
        NOTE: Currently training only 
        """

        cd = self._camera_data(camera)
        active_rod = self._rod_state(cd, self.controlled_rod_id)

        # Just a minimal example:
        CD0 = camera["camData"][0] if camera["camData"][0] is not None else camera["camData"][1]

        # Ball
        ball_x = (CD0["ball_x"] - 605) / 605  # Center around table middle [-1,1]
        ball_y = (CD0["ball_y"] - 350) / 350  # Center around table middle [-1,1]
        ball_vx = np.clip(CD0["ball_vx"] / 5.0, -2, 2)  # Velocity normalized
        ball_vy = np.clip(CD0["ball_vy"] / 5.0, -2, 2)


        # Find controlled rod in geometry
        controlled_rod_info = None
        for rod in self.geometry["rods"]:
            if rod["id"] == self.controlled_rod_id:
                controlled_rod_info = rod
                break

        
        if controlled_rod_info is None:
            raise ValueError(f"Rod {self.controlled_rod_id} not found in geometry")
        

        # Controlled rod state (5 values)
        """
        Blue rod idx: 
        2: Attack, 4: Middle, 6: Defender 7: Goalkeeper

        Red rod idx:
        0: Goalkeeper, 1: defender, 3: middle, 5: attack

        Id of the rod in geometry is 1-based so red goalkeeper has id=1
        """
        rod_idx = self.controlled_rod_id - 1  # Convert to 0-based index
        rod_pos_calib = CD0["rod_position_calib"][rod_idx]
        rod_angle = CD0["rod_angle"][rod_idx]
        
        
        if isinstance(rod_pos_calib, list):
            rod_pos_calib = rod_pos_calib[0]
        
        # Team encoding (0=red, 1=blue)
        team = 0.0 if controlled_rod_info["team"] == "red" else 1.0
        
        # Rod position (already normalized 0-1)
        rod_y_normalized = rod_pos_calib

        
        # Relative positioning - KEY FOR SINGLE ROD LEARNING
        rod_x_world = controlled_rod_info["position"]
        ball_x_world = CD0["ball_x"]
        ball_y_world = CD0["ball_y"] 
        
        # Give the policy the x distance and a direct error toward the player
        # branch selected for this ball region.
        ball_rod_dist_x = (ball_x_world - rod_x_world) / 605  # Normalized [-1,1]
        _, target_rod_pos, _ = player_alignment_target(
            ball_y=ball_y_world,
            rod_info=controlled_rod_info,
        )
        target_rod_error = target_rod_pos - rod_pos_calib

        
        # Rod angle normalized
        angle_normalized = np.clip(rod_angle / 32.0, -1, 1)  # Assuming ±45° range
        
        # Combine: Ball(4) + Rod(5) = 9 total
        ball_kicked = float(camera.get("ball_kicked", False))
        #print(F"ball rod dist x: {ball_rod_dist_x}, target rod error: {target_rod_error}")

        # Combine: Ball(4) + Rod(5) + ball_kicked(1) = 10 total
        obs = np.array([
            ball_x, ball_y, ball_vx, ball_vy,           # Ball state (4)
            #team,                                        # Rod team (1)
            ball_rod_dist_x,                # Distance of the ball to the observed rod in x distance
            target_rod_error,          # Relative error between the the ball and the closest player on the rod in y position
            rod_y_normalized,                            # Rod position (1) 
            angle_normalized,                            # Rod angle (1)
            #ball_kicked                                  # Ball contact flag (1)
        ], dtype=np.float32)

        # Verify size
        #assert len(obs) == 10, f"Expected obs_dim=10, got {len(obs)}"
        
        return obs, (CD0["ball_x"], CD0["ball_y"]), (CD0["ball_vx"], CD0["ball_vy"]), angle_normalized, active_rod

    def scale_to_motor_commands(self, action):
        """
        We have 4 values in [-1,1]: for the forward-most rod, each rod has 4 values:
        [rot_target, rot_speed, trans_target, trans_speed]
        We'll scale them appropriately into the JSON commands expected by the simulator.
        """

        # Example scaling:
        rot_target   = 0.5 * action[0]  # in [-1,1]
        rot_velocity = 0.5 * (action[1] + 1) / 4  # scale [-1,1]→[0,1], then multiply by max 0.5
        trans_target = ((action[2] + 1) / 2)  # scale [-1,1]→[0,1], you might want full 0..1 0.5
        #trans_target = 0.8 * ((action[2] + 1) / 2)  # scale [-1,1]→[0,1], you might want full 0..1 0.5
        trans_velocity = 1.0 * (action[3] + 1) / 2   # scale [-1,1]→[0,1]

        # driveID singals, which agent's rod is active. 1 for goalkeeper, 2 for the defender, 3 for midfield, 4 for attacker

        cmd = {
            "driveID": 3,
            "rotationTargetPosition": rot_target,
            "rotationVelocity": rot_velocity,
            "translationTargetPosition": trans_target,
            "translationVelocity": trans_velocity
        }

        # Idle commands for the other rods
        idle_commands = [
            {
                "driveID": 2,
                "rotationTargetPosition": 0.5,
                "rotationVelocity": 0.1,
                "translationTargetPosition": 0.5,
                "translationVelocity": 0.0
            },
            {
                "driveID": 4,
                "rotationTargetPosition": 0.5,
                "rotationVelocity": 0.0,
                "translationTargetPosition": 0.5,
                "translationVelocity": 0.0
            },
            {
                "driveID": 1,
                "rotationTargetPosition": 0.5,
                "rotationVelocity": 0.0,
                "translationTargetPosition": 0.5,
                "translationVelocity": 0.0
            }
        ]

        commands = [cmd] + idle_commands
        return commands

    def finish_episode(self, last_value=0):
        """Fixed episode finishing with proper buffer managment"""

        ended_with_kick = self.current_episode_ball_kicks > 0
        ended_with_goal = self.current_episode_goals > 0

        # Only finish path if we have data in the buffer
        if self.buf.ptr > self.buf.path_start_idx:
            self.buf.finish_path(last_val=last_value)
            self.episode_rewards.append(self.ep_reward)

            # Train when buffer is sufficiently full (≥80% capacity)
            buffer_fill = self.buf.ptr / self.buf.max_size
            if buffer_fill >= 0.8:
                print(f"[Training] Buffer {buffer_fill*100:.1f}% full ({self.buf.ptr}/{self.buf.max_size}), Episode {self.episode_count}")
                if self.auto_train:
                    self.train_on_buffer() # -> calls buf.get() which resets ptr internally
                else:
                    self.pending_train = True



        self.episode_count += 1
        if ended_with_kick:
            self.episodes_with_kick += 1
        if ended_with_goal:
            self.episodes_with_goal += 1

        kick_rate = self.episodes_with_kick / self.episode_count if self.episode_count > 0 else 0.0
        print(
            f"[Episode stats] Episode {self.episode_count}: "
            f"{'kick' if ended_with_kick else 'no kick'}, "
            f"kicks {self.episodes_with_kick}/{self.episode_count} "
            f"({100.0 * kick_rate:.1f}%), goals {self.episodes_with_goal}, "
            f"threshold {100.0 * self.kick_rate_threshold:.0f}% "
            f"{'reached' if kick_rate >= self.kick_rate_threshold else 'not reached'}."
        )

        self.current_step = 0
        self.ep_reward = 0.0
        self.last_obs = None
        self.prev_ball_vxy = None
        # Reset episode-specific variables
        self.last_action = None
        self.last_val = None
        self.last_logp = None

        self.current_episode_goals = 0
        self.current_episode_opponent_goals = 0
        self.current_episode_ball_kicks = 0
        self.current_episode_x_threshold_terminations = 0
        self.current_episode_step_rewards = []


        # Save model periodically
        if self.episode_count % self.save_model_every == 0:
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            print(f"Episode {self.episode_count}, Avg Reward (last 100): {avg_reward:.2f}")
            self.save_model()

class TwoRodPPOAgent(PPOAgent):
    """
    PPO agent for one controlled rod with one observed opponent rod.

    Observation layout, obs_dim=16:
      ball(4) + controlled_rod(5) + observed_opponent_rod(5)
      + ball_kicked(1) + opponent_active(1)

    Action layout, act_dim=4:
      [
        controlled_rotation_target,
        controlled_rotation_velocity,
        controlled_translation_target,
        controlled_translation_velocity,
      ]
    """

    def __init__(
        self,
        obs_dim=14,#16,
        act_dim=4,
        hidden_size=512,
        steps_per_env=512,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        lr=1e-4,
        train_iters=4,
        target_kl=0.01,
        save_model_every=500,
        model_save_path="./trained_models/two_rod_defend_shoot.pth",
        training_log_export_every=10,
        l2_lambda=5e-4,
        controlled_rod_id=6,
        observed_rod_id=5,
        training_task="defending",
        opponent_active=True,
        target_rod_angle=0.085,
        training_enabeled=True,
        load_model=False,
        inference=False,
        auto_train=True,
        action_std_override=(0.4, 0.4, 0.25, 0.25),
    ):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            steps_per_env=steps_per_env,
            gamma=gamma,
            lam=lam,
            clip_ratio=clip_ratio,
            lr=lr,
            train_iters=train_iters,
            target_kl=target_kl,
            save_model_every=save_model_every,
            model_save_path=model_save_path,
            training_log_export_every=training_log_export_every,
            l2_lambda=l2_lambda,
            controlled_rod_id=controlled_rod_id,
            training_enabeled=training_enabeled,
            load_model=False,
            inference=inference,
            auto_train=auto_train,
            action_std_override=action_std_override,
        )

        self.observed_rod_id = int(observed_rod_id)
        self.training_task = str(training_task).lower()
        self.opponent_active = bool(opponent_active)
        self.target_rod_angle = float(target_rod_angle)
        self.model_name = f"two_rod_{self.training_task}"
        self.rods_by_id = {int(rod["id"]): rod for rod in self.geometry["rods"]}
        self.field_x = float(self.geometry["field"]["dimension_x"])
        self.field_y = float(self.geometry["field"]["dimension_y"])

        self.ac = TwoRodActorCriticNet(obs_dim=self.obs_dim, hidden_size=hidden_size)
        self.ac.to(self.device)
        self.optimizer = optim.Adam(list(self.ac.parameters()) + [self.log_std], lr=lr)

        if load_model:
            self.load_model()
        #else:
        #    self.apply_action_std_override()

    @staticmethod
    def _as_float(value):
        if isinstance(value, list):
            value = value[0]
        return float(value)

    @staticmethod
    def freeze_layers(model, trainable_prefixes):
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)    

    def rebuild_optimizer(self, lr=None):
        if lr is None:
            lr = self.optimizer.param_groups[0]["lr"]
        trainable_params = [p for p in self.ac.parameters() if p.requires_grad]
        trainable_params.append(self.log_std)
        self.optimizer = optim.Adam(trainable_params, lr=lr)

    def _camera_data(self, camera):
        return camera["camData"][0] if camera["camData"][0] is not None else camera["camData"][1]

    def _rod_state(self, cd, rod_id):
        rod_idx = int(rod_id) - 1
        rod_info = self.rods_by_id[int(rod_id)]
        rod_pos_calib = self._as_float(cd["rod_position_calib"][rod_idx])
        rod_angle = self._as_float(cd["rod_angle"][rod_idx])
        rod_angle_normalized = float(np.clip(rod_angle / 32.0, -1.0, 1.0))
        _, target_rod_pos, unavoidable_dist = player_alignment_target(
            ball_y=cd["ball_y"],
            rod_info=rod_info,
        )
        target_error = float(target_rod_pos) - rod_pos_calib
        return {
            "info": rod_info,
            "pos_calib": rod_pos_calib,
            "angle": rod_angle,
            "angle_normalized": rod_angle_normalized,
            "target_pos": float(target_rod_pos),
            "target_error": target_error,
            "unavoidable_dist": float(unavoidable_dist),
        }

    def _rod_features(self, rod_state):
        rod_info = rod_state["info"]
        return [
            (float(rod_info["position"]) - self.field_x / 2.0) / (self.field_x / 2.0),
            rod_state["pos_calib"],
            rod_state["angle_normalized"],
            rod_state["target_error"],
            rod_state["unavoidable_dist"] / max(float(rod_info["travel"]), 1e-6),
        ]

    def extract_observation(self, camera):
        cd = self._camera_data(camera)
        controlled = self._rod_state(cd, self.controlled_rod_id)
        observed = self._rod_state(cd, self.observed_rod_id)

        ball_x = float(cd["ball_x"])
        ball_y = float(cd["ball_y"])
        ball_vx = float(cd["ball_vx"])
        ball_vy = float(cd["ball_vy"])
        opponent_active = float(camera.get("opponent_active", self.opponent_active))

        obs = np.array(
            [
                (ball_x - self.field_x / 2.0) / (self.field_x / 2.0),   # Ball normalized x direction
                (ball_y - self.field_y / 2.0) / (self.field_y / 2.0),   # Ball normalized y direction
                np.clip(ball_vx / 5.0, -2.0, 2.0),                      # Ball vx
                np.clip(ball_vy / 5.0, -2.0, 2.0),                      # Ball vy
                *self._rod_features(controlled),
                *self._rod_features(observed),
                #float(camera.get("ball_kicked", False)),
                #opponent_active,
            ],
            dtype=np.float32,
        )

        #assert len(obs) == self.obs_dim, f"Expected obs_dim={self.obs_dim}, got {len(obs)}"
        return obs, (ball_x, ball_y), (ball_vx, ball_vy), controlled, observed

    def calculate_reward(
        self,
        bxy,
        vxy,
        controlled,
        *,
        goal_scored,
        ball_kicked,
        kick_normal_force,
        terminated_by_x_threshold,
        end_episode,
    ):
        alignment_reward = closest_player_alignment_reward(
            ball_y=bxy[1],
            rod_pos_calib=float(controlled["pos_calib"]),
            rod_info=controlled["info"],
            reward_scale=1.0
        )
        rod_angle_reward = calculate_rod_angle_reward(
            rod_angle=float(controlled["angle"]),
            target_rod_angle=self.target_rod_angle,
        )


        reward_breakdown = {
            "allignment": alignment_reward 
        }

        return alignment_reward, reward_breakdown

        if self.training_task == "shooting":
            shot_attempted = self.current_episode_ball_kicks > 0 or ball_kicked
            missed_kick = bool((end_episode or terminated_by_x_threshold) and shot_attempted and not goal_scored)
            return kicking_reward(
                goal_scored=goal_scored,
                ball_kicked=ball_kicked,
                kick_normal_force=kick_normal_force,
                ball_x=bxy[0],
                ball_y=bxy[1],
                forward_ball_vx=vxy[0],
                ball_vy=vxy[1],
                missed_kick=missed_kick,
                episode_timeout=bool(end_episode and not (ball_kicked or terminated_by_x_threshold)),
                rod_alignment_reward=alignment_reward,
                rod_angle=float(controlled["angle"]),
                target_rod_angle=self.target_rod_angle,
            )

        controlled_rod_x = float(controlled["info"]["position"])
        prev_ball_vx = self.prev_ball_vxy[0] if self.prev_ball_vxy is not None else None
        prev_ball_vy = self.prev_ball_vxy[1] if self.prev_ball_vxy is not None else None
        ball_behind_rod = float(bxy[0]) < (controlled_rod_x - 10.0)
        return maintaining_ball(
            ball_x=bxy[0],
            ball_vx=vxy[0],
            ball_vz=vxy[1],
            rod_x_pos=controlled_rod_x,
            threshold_crossed=terminated_by_x_threshold,
            ball_behind_rod=ball_behind_rod,
            prev_ball_vx=prev_ball_vx,
            prev_ball_vz=prev_ball_vy,
            episode_timeout=bool(end_episode and not terminated_by_x_threshold),
            rod_angle_reward=rod_angle_reward,
            player_alignment_reward=alignment_reward,
        )

    def scale_to_motor_commands(self, action):
        red_rod_to_drive_id = {
            1: 1,
            2: 2,
            4: 3,
            6: 4,
        }
        if int(self.controlled_rod_id) not in red_rod_to_drive_id:
            raise ValueError(f"Rod {self.controlled_rod_id} is not mapped as a controllable red rod.")

        return [
            {
                "driveID": red_rod_to_drive_id[int(self.controlled_rod_id)],
                "rotationTargetPosition": 0.5 * float(action[0]),
                "rotationVelocity": 0.5 * (float(action[1]) + 1.0) / 4.0,
                "translationTargetPosition": (float(action[2]) + 1.0) / 2.0,
                "translationVelocity": (float(action[3]) + 1.0) / 2.0,
            }
        ]

    def process_data(self, camera):
        self.total_steps += 1
        if self.total_steps % 250 == 0:
            print(
                f"[TwoRodPPO] Processing step {self.total_steps} at episode "
                f"{self.episode_count}, step in episode: {self.episode_steps}"
            )

        obs, bxy, vxy, controlled, observed = self.extract_observation(camera)
        self.latest_env_metrics = {
            "curriculum_round": int(camera.get("curriculum_round", -1)),
            "curriculum_y_min": float(camera.get("curriculum_y_min", float("nan"))),
            "curriculum_y_max": float(camera.get("curriculum_y_max", float("nan"))),
            "opponent_active": float(camera.get("opponent_active", self.opponent_active)),
        }

        ball_kicked = bool(camera.get("ball_kicked", False))
        kick_normal_force = float(camera.get("kick_normal_force", 0.0))
        terminated_by_kick = bool(camera.get("terminated_by_kick", False))
        terminated_by_x_threshold = bool(camera.get("terminated_by_x_threshold", False))
        end_episode = bool(camera.get("end_episode", False))

        score = camera.get("score", None)
        goal_scored = False
        opponent_goal_scored = False
        if isinstance(score, (list, tuple)) and len(score) >= 2:
            if self.prev_score is not None:
                goal_scored = score[0] > self.prev_score[0]
                opponent_goal_scored = score[1] > self.prev_score[1]
            self.prev_score = list(score)

        if not self.training_enabled:
            action, _, _ = self.compute_action(obs, deterministic=self.inference)
            return self.scale_to_motor_commands(action)

        episode_finished_this_sample = False
        if self.last_obs is not None:
            """reward, reward_breakdown = self.calculate_reward(
                bxy,
                vxy,
                controlled,
                goal_scored=goal_scored,
                ball_kicked=ball_kicked,
                kick_normal_force=float(camera.get("kick_normal_force", 0.0)),
                terminated_by_x_threshold=terminated_by_x_threshold,
                end_episode=end_episode,
            )"""

            allignment_quality_reward = closest_player_alignment_reward(
                            ball_y=bxy[1],
                            rod_pos_calib=float(controlled["pos_calib"]),
                            rod_info=controlled["info"],
                            reward_scale=1.0,
                        )

            rod_angle_reward = calculate_rod_angle_reward(
                            rod_angle=float(controlled["angle"]),
                            reward_scale=1,
                            target_rod_angle=0.0,
                            rotation_buffer_def=3
                            )

            #print(f"angle: {controlled['angle']}, rod angle reward: {rod_angle_reward}")

            reward_breakdown = {
                "allignmment": allignment_quality_reward,
                "rod_angle": rod_angle_reward * 0.25,
            }
            reward = sum(reward_breakdown.values())


            for key, value in reward_breakdown.items():
                self.update_reward_breakdown_sums[key] = (
                    self.update_reward_breakdown_sums.get(key, 0.0) + float(value)
                )
            if ball_kicked:
                self.update_samples_with_kick += 1
                self.current_episode_ball_kicks += 1
            if goal_scored:
                self.current_episode_goals += 1
            if opponent_goal_scored:
                self.current_episode_opponent_goals += 1
            if terminated_by_x_threshold:
                self.current_episode_x_threshold_terminations += 1

            self.current_episode_step_rewards.append(reward)
            stored = self.buf.store(self.last_obs, self.last_action, reward, self.last_val, self.last_logp)
            if not stored:
                _, final_value, _ = self.compute_action(obs)
                self.buf.finish_path(last_val=final_value)
                print("[TwoRodPPO] Buffer full, training now...")
                if self.auto_train:
                    self.train_on_buffer()
                else:
                    self.pending_train = True
                self.episode_count += 1
                self.episode_steps = 0
                self.ep_reward = 0.0
                self.last_obs = None
                self.last_action = None
                self.last_val = None
                self.last_logp = None
                episode_finished_this_sample = True
            else:
                self.ep_reward += reward

            if (not episode_finished_this_sample) and (terminated_by_kick or terminated_by_x_threshold or end_episode):
                self.finish_episode(last_value=0)
                self.episode_steps = 0
                episode_finished_this_sample = True
        else:
            print("[TwoRodPPO] First step, no reward yet.")

        if not episode_finished_this_sample:
            self.episode_steps += 1

        action, value, logp = self.compute_action(obs, deterministic=self.inference)
        if not np.all(np.isfinite(action)):
            print("[TwoRodPPO] Nan or Inf detected in action:", action)
            action = np.zeros_like(action)

        self.prev_ball_vxy = vxy
        self.last_obs = obs
        self.last_action = action
        self.last_val = value
        self.last_logp = logp
        self.current_step += 1

        return self.scale_to_motor_commands(action)

class PassPPOAgent(PPOAgent):
    """
    PPO agent for coordinated passing with two controlled red rods and one
    observed opponent rod.

    Action layout:
      [
        passer_rotation_target,
        passer_rotation_velocity,
        passer_translation_target,
        passer_translation_velocity,
        receiver_rotation_target,
        receiver_rotation_velocity,
        receiver_translation_target,
        receiver_translation_velocity,
      ]

    Default task setup:
      - passer rod: red rod 4
      - receiver rod: red rod 6
      - opponent rod: blue rod 5
    """

    def __init__(
        self,
        obs_dim=19,#20,
        act_dim=8,
        hidden_size=512,
        steps_per_env=512,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        lr=1e-4,
        train_iters=4,
        target_kl=0.01,
        save_model_every=1000,
        model_save_path="./trained_models/pass_ppo_no_std_override.pth",
        backbone_model_path="./trained_models/pass_backbone_aux.pth",
        load_model=False,
        load_backbone=True,
        training_log_export_every=50,
        l2_lambda=5e-4,
        passer_rod_id=4,
        receiver_rod_id=6,
        opponent_rod_id=5,
        training_enabeled=True,
        inference=False,
        auto_train=True,
        action_std_override=(0.3, 0.3, 0.30, 0.30, 0.3, 0.3, 0.30, 0.30),
        target_rod_angle=0.085,
    ):
        self.passer_rod_id = int(passer_rod_id)
        self.receiver_rod_id = int(receiver_rod_id)
        self.opponent_rod_id = int(opponent_rod_id)
        self.backbone_model_path = backbone_model_path
        self.target_rod_angle = float(target_rod_angle)

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.save_model_every = save_model_every
        self.model_save_path = model_save_path
        self.model_name = "pass_ppo"
        self.training_log_export_every = training_log_export_every
        self.l2_lambda = l2_lambda
        self.inference = inference
        self.training_enabled = training_enabeled
        self.action_std_override = action_std_override
        self.auto_train = bool(auto_train)
        self.pending_train = False
        if self.inference and self.training_enabled:
            print("[PassPPO] Inference mode enabled, disabling training.")
            self.training_enabled = False

        self.prev_ball_x = None
        self.prev_ball_vxy = None
        self.prev_score = None
        self.kick_latch = False
        self.episode_steps = 0

        with open("geometry.json") as f:
            self.geometry = json.load(f)
        self.rods_by_id = {int(rod["id"]): rod for rod in self.geometry["rods"]}
        self.field_x = float(self.geometry["field"]["dimension_x"])
        self.field_y = float(self.geometry["field"]["dimension_y"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[PassPPO] Using device:", self.device)

        self.ac = ThreeRodActorCriticNet(obs_dim=self.obs_dim, hidden_size=hidden_size)
        self.ac.to(self.device)

        self.log_std = nn.Parameter(
            -1 * torch.ones(self.act_dim, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        self.optimizer = optim.Adam(list(self.ac.parameters()) + [self.log_std], lr=lr)

        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.steps_per_env = steps_per_env
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, steps_per_env, gamma, lam)

        self.episode_count = 0
        self.current_step = 0
        self.ep_reward = 0.0
        self.last_obs = None
        self.last_action = None
        self.last_val = None
        self.last_logp = None
        self.episode_rewards = []
        self.reward = 0.0
        self.total_steps = 0
        self.training_count = 0
        self.training_export = Export()

        self.episode_stats = []
        self.current_episode_goals = 0
        self.current_episode_opponent_goals = 0
        self.current_episode_ball_kicks = 0
        self.current_episode_x_threshold_terminations = 0
        self.current_episode_step_rewards = []
        self.episodes_with_kick = 0
        self.episodes_with_goal = 0
        self.kick_rate_threshold = 0.8
        self.latest_env_metrics = {}
        self.update_reward_breakdown_sums = {}
        self.update_samples_with_kick = 0

        if load_model:
            self.load_model()
        elif load_backbone:
            self.load_backbone_model()


    @staticmethod
    def freeze_layers(model, trainable_prefixes):
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)

    def rebuild_optimizer(self, lr=None):
        if lr is None:
            lr = self.optimizer.param_groups[0]["lr"]
        trainable_params = [p for p in self.ac.parameters() if p.requires_grad]
        trainable_params.append(self.log_std)
        self.optimizer = optim.Adam(trainable_params, lr=lr)

    @staticmethod
    def _as_float(value):
        if isinstance(value, list):
            value = value[0]
        return float(value)

    def load_backbone_model(self, path=None):
        if path is None:
            path = self.backbone_model_path
        if not path or not os.path.exists(path):
            print(f"[PassPPO] No auxiliary backbone found at {path}, starting from random weights.")
            return

        checkpoint = torch.load(path, map_location=self.device)
        state_dict = (
            checkpoint["actor_critic_state_dict"]
            if isinstance(checkpoint, dict) and "actor_critic_state_dict" in checkpoint
            else checkpoint
        )
        self.ac.load_state_dict(state_dict)
        print(f"[PassPPO] Auxiliary backbone loaded from {path}")

    def _camera_data(self, camera):
        return camera["camData"][0] if camera["camData"][0] is not None else camera["camData"][1]

    def _rod_state(self, cd, rod_id):
        rod_idx = int(rod_id) - 1
        rod_info = self.rods_by_id[int(rod_id)]
        rod_pos_calib = self._as_float(cd["rod_position_calib"][rod_idx])
        rod_angle = self._as_float(cd["rod_angle"][rod_idx])
        rod_angle_normalized = float(np.clip(rod_angle / 32.0, -1.0, 1.0))
        _, target_rod_pos, unavoidable_dist = player_alignment_target(
            ball_y=cd["ball_y"],
            rod_info=rod_info,
        )
        target_error = float(target_rod_pos) - rod_pos_calib
        return {
            "info": rod_info,
            "pos_calib": rod_pos_calib,
            "angle": rod_angle,
            "angle_normalized": rod_angle_normalized,
            "target_pos": float(target_rod_pos),
            "target_error": target_error,
            "unavoidable_dist": float(unavoidable_dist),
        }

    def extract_observation(self, camera):
        cd = self._camera_data(camera)
        passer = self._rod_state(cd, self.passer_rod_id)
        receiver = self._rod_state(cd, self.receiver_rod_id)
        opponent = self._rod_state(cd, self.opponent_rod_id)

        ball_x = float(cd["ball_x"])
        ball_y = float(cd["ball_y"])
        ball_vx = float(cd["ball_vx"])
        ball_vy = float(cd["ball_vy"])

        def rod_features(state):
            rod_info = state["info"]
            return [
                (float(rod_info["position"]) - self.field_x / 2.0) / (self.field_x / 2.0),
                state["pos_calib"],     # pos-calib is the translation(lateral) move
                state["angle_normalized"],
                state["target_error"],
                state["unavoidable_dist"] / max(float(rod_info["travel"]), 1e-6),
            ]

        obs = np.array(
            [
                (ball_x - self.field_x / 2.0) / (self.field_x / 2.0),
                (ball_y - self.field_y / 2.0) / (self.field_y / 2.0),
                np.clip(ball_vx / 5.0, -2.0, 2.0),
                np.clip(ball_vy / 5.0, -2.0, 2.0),
                *rod_features(passer),
                *rod_features(receiver),
                *rod_features(opponent),
                #float(camera.get("ball_kicked", False)),
            ],
            dtype=np.float32,
        )

        #assert len(obs) == self.obs_dim, f"Expected obs_dim={self.obs_dim}, got {len(obs)}"
        return obs, (ball_x, ball_y), (ball_vx, ball_vy), passer, receiver, opponent

    def compute_pass_reward(
        self,
        bxy,
        vxy,
        receiver,
        passer,
        *,
        ball_kicked,
        kick_normal_force,
        threshold_failure,
        episode_timeout,
    ):
        """
        Allignment reciever/passing palice z žogico:
            "rod_angle_reward": rod_angle_reward * 0.2,
            "rod_alignment_reward": alignment_quality,

        Passer kicking the ball:

        Reciever stopping the ball:


        End-to-end passing the ball:


        """
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

        kick_force = kick_force_reward(
            ball_kicked=ball_kicked,
            kick_normal_force=kick_normal_force,
            desired_kick_force=12.0,
            force_sigma=8.0,
            reward_scale=0.35,
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
            controlled_ball_reward = 0.75 * controlled_speed_quality * zone_quality

        stopped_ball_reward = 0.0
        if in_receive_zone and stopped:
            stopped_ball_reward = 8.0 * zone_quality

        threshold_failure_penalty = -1.5 if threshold_failure else 0.0
        timeout_penalty = -0.2 if episode_timeout else 0.0
        

        

        reward_breakdown = {
            #"rod_angle_reward": rod_angle_reward * 0.25,
            #"rod_alignment_reward_passer": allignment_quality_passer,
            #"rod_alignment_reward_receiver": alignment_quality_receiver,
            #"vicinity_reward": vicinity_reward,
            "speed_reduction_reward": speed_reduction_reward,
            "controlled_ball_reward": controlled_ball_reward,
            "stopped_ball_reward": stopped_ball_reward,
            "kick_force": kick_force,
            #"threshold_failure": threshold_failure_penalty,
            "episode_timeout": timeout_penalty,
        }


        return float(sum(reward_breakdown.values())), reward_breakdown

    def scale_to_motor_commands(self, action):
        return [
            self._command_for_rod(self.passer_rod_id, action[0:4]),
            self._command_for_rod(self.receiver_rod_id, action[4:8]),
        ]

    def _command_for_rod(self, rod_id, action_slice):
        red_rod_to_drive_id = {
            1: 1,
            2: 2,
            4: 3,
            6: 4,
        }
        if int(rod_id) not in red_rod_to_drive_id:
            raise ValueError(f"Rod {rod_id} is not mapped as a controllable red rod.")

        rot_target = 0.5 * float(action_slice[0])
        rot_velocity = 0.5 * (float(action_slice[1]) + 1.0) / 4.0
        trans_target = (float(action_slice[2]) + 1.0) / 2.0
        trans_velocity = (float(action_slice[3]) + 1.0) / 2.0

        return {
            "driveID": red_rod_to_drive_id[int(rod_id)],
            "rotationTargetPosition": rot_target,
            "rotationVelocity": rot_velocity,
            "translationTargetPosition": trans_target,
            "translationVelocity": trans_velocity,
        }

    def process_data(self, camera):
        self.total_steps += 1
        if self.total_steps % 250 == 0:
            print(
                f"[PassPPO] step {self.total_steps}, "
                f"episode {self.episode_count}, step in episode {self.episode_steps}"
            )

        obs, bxy, vxy, passer, receiver, opponent = self.extract_observation(camera)
        self.latest_env_metrics = {
            "curriculum_round": int(camera.get("curriculum_round", -1)),
            "curriculum_y_min": float(camera.get("curriculum_y_min", float("nan"))),
            "curriculum_y_max": float(camera.get("curriculum_y_max", float("nan"))),
        }

        end_episode = bool(camera.get("end_episode", False))
        terminated_by_kick = bool(camera.get("terminated_by_kick", False))
        terminated_by_x_threshold = bool(camera.get("terminated_by_x_threshold", False))
        ball_kicked = bool(camera.get("ball_kicked", False))
        kick_normal_force = float(camera.get("kick_normal_force", 0.0))

        

        episode_finished_this_sample = False
        if self.last_obs is not None and self.training_enabled:
            reward, reward_breakdown = self.compute_pass_reward(
                bxy,
                vxy,
                receiver,
                passer,
                ball_kicked=ball_kicked,
                kick_normal_force=float(camera.get("kick_normal_force", 0.0)),
                threshold_failure=terminated_by_x_threshold,
                episode_timeout=(
                    end_episode
                    and not terminated_by_kick
                    and not terminated_by_x_threshold
                    and not ball_kicked
                ),
            )
            for key, value in reward_breakdown.items():
                self.update_reward_breakdown_sums[key] = (
                    self.update_reward_breakdown_sums.get(key, 0.0) + float(value)
                )
            if ball_kicked:
                self.update_samples_with_kick += 1
                self.current_episode_ball_kicks += 1
                self.kick_latch = True
            if terminated_by_x_threshold:
                self.current_episode_x_threshold_terminations += 1

            self.current_episode_step_rewards.append(reward)
            stored = self.buf.store(self.last_obs, self.last_action, reward, self.last_val, self.last_logp)
            if not stored:
                _, final_value, _ = self.compute_action(obs)
                self.buf.finish_path(last_val=final_value)
                print("[PassPPO] Buffer full, training now...")
                if self.auto_train:
                    self.train_on_buffer()
                else:
                    self.pending_train = True
                self.episode_count += 1
                self.episode_steps = 0
                self.ep_reward = 0.0
                self.last_obs = None
                self.last_action = None
                self.last_val = None
                self.last_logp = None
                episode_finished_this_sample = True
            else:
                self.ep_reward += reward

            if (not episode_finished_this_sample) and (self.kick_latch or terminated_by_kick or terminated_by_x_threshold or end_episode):
                self.finish_episode(last_value=0)
                self.episode_steps = 0
                episode_finished_this_sample = True

        if not episode_finished_this_sample:
            self.episode_steps += 1

        action, value, logp = self.compute_action(obs, deterministic=self.inference)

        #if self.total_steps % 250 == 0:
        #    with torch.no_grad():
        #        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        #        mean, _ = self.ac(obs_t)
        #        mean_action = torch.tanh(mean).detach().cpu().numpy()[0]
        #
        #    print(
        #        "[receiver debug]",
        #        "ball_y=", round(bxy[1], 2),
        #        "pos=", round(receiver["pos_calib"], 3),
        #        "target=", round(receiver["target_pos"], 3),
        #        "error=", round(receiver["target_error"], 3),
        #        "mean_action[4:8]=", np.round(mean_action[4:8], 3),
        #        "sampled_action[4:8]=", np.round(action[4:8], 3),
        #        "mean_trans=", round((mean_action[6] + 1.0) / 2.0, 3),
        #    )


        if not np.all(np.isfinite(action)):
            print("[PassPPO] Nan or Inf detected in action:", action)
            action = np.zeros_like(action)

        self.prev_ball_x = bxy[0]
        self.prev_ball_vxy = vxy
        self.last_obs = obs
        self.last_action = action
        self.last_val = value
        self.last_logp = logp
        self.current_step += 1

        return self.scale_to_motor_commands(action)

    def finish_episode(self, last_value=0):
        if self.buf.ptr > self.buf.path_start_idx:
            self.buf.finish_path(last_val=last_value)
            self.episode_rewards.append(self.ep_reward)
            buffer_fill = self.buf.ptr / self.buf.max_size
            if buffer_fill >= 0.8:
                print(
                    f"[PassPPO] Buffer {buffer_fill*100:.1f}% full "
                    f"({self.buf.ptr}/{self.buf.max_size}), episode {self.episode_count}"
                )
                if self.auto_train:
                    self.train_on_buffer()
                else:
                    self.pending_train = True

        self.episode_count += 1
        #print(
        #    f"[PassPPO episode] {self.episode_count}: reward={self.ep_reward:.3f}, "
        #    f"kicks={self.current_episode_ball_kicks}, "
        #    f"x_terms={self.current_episode_x_threshold_terminations}"
        #)

        self.current_step = 0
        self.ep_reward = 0.0
        self.last_obs = None
        self.last_action = None
        self.last_val = None
        self.last_logp = None
        self.prev_ball_x = None
        self.prev_ball_vxy = None
        self.kick_latch = False
        self.current_episode_goals = 0
        self.current_episode_opponent_goals = 0
        self.current_episode_ball_kicks = 0
        self.current_episode_x_threshold_terminations = 0
        self.current_episode_step_rewards = []

        if self.episode_count % self.save_model_every == 0:
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            print(f"[PassPPO] Episode {self.episode_count}, avg reward last 100: {avg_reward:.3f}")
            self.save_model()

# --------------------------------------------
# If you want to run standalone:
if __name__ == "__main__":
    # Create the PPO agent
    agent = PPOAgent()

    # Possibly load an existing model
    agent.load_model()

    episode =0

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

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        agent.save_model("model.pth")
