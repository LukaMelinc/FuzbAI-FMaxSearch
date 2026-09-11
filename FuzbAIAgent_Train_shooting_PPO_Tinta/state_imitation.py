"""State-only imitation-from-observation experiment for the single-rod agent.

The expert recorder intentionally stores observations, not teacher actions.  The
imitation agent learns a discriminator over (state, next_state) transitions and
uses its output as the reward for an otherwise standard PPO policy.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

"""python3 FuzbAISim.py \
    --headless \
    --mode state_imitation \
    --expert-csv imitation_data/threshold_expert_8d.csv \
    --ball-x-threshold 605 \
    --steps-per-env 512 \
    --save-model-every 50"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from strel_PPO import PPOAgent
from reward.single_bar_shoting import (
    calculate_rod_angle_reward,
    closest_player_alignment_reward,
)


OBS_COLUMNS = (
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    #"team",
    "ball_rod_dist_x",
    "target_rod_error",
    "rod_position",
    "rod_angle",
    #"ball_kicked",
)


class ExpertTransitionDataset:
    """Loads adjacent state pairs without crossing episode boundaries."""

    def __init__(self, csv_path):
        states = []
        episode_ids = []
        with open(csv_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            missing = set(("episode_id", *OBS_COLUMNS)) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Expert CSV is missing columns: {sorted(missing)}")
            for row in reader:
                episode_ids.append(int(row["episode_id"]))
                states.append([float(row[name]) for name in OBS_COLUMNS])

        transitions = [
            np.concatenate((states[i], states[i + 1])).astype(np.float32)
            for i in range(len(states) - 1)
            if episode_ids[i] == episode_ids[i + 1]
        ]
        if not transitions:
            raise ValueError(f"No within-episode transitions found in {csv_path}")
        self.transitions = torch.as_tensor(np.stack(transitions), dtype=torch.float32)
        expected_width = 2 * len(OBS_COLUMNS)
        if self.transitions.shape[1] != expected_width:
            raise ValueError(
                f"Expert transitions have width {self.transitions.shape[1]}, "
                f"expected {expected_width}"
            )
        print(
            f"[ExpertDataset] Loaded {len(self.transitions)} state transitions "
            f"from {csv_path}"
        )

    def sample(self, batch_size, device):
        indices = torch.randint(0, len(self.transitions), (int(batch_size),))
        return self.transitions[indices].to(device)

class TransitionDiscriminator(nn.Module):
    def __init__(self, obs_dim=10, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, transitions):
        return self.net(transitions).squeeze(-1)


class StateImitationPPOAgent(PPOAgent):
    """PPO whose reward is learned from state-only expert transitions."""

    def __init__(
        self,
        expert_csv,
        discriminator_lr=3e-4,
        discriminator_updates=8,
        discriminator_batch_size=128,
        imitation_reward_scale=1.0,
        **ppo_kwargs,
    ):
        should_load_model = bool(ppo_kwargs.pop("load_model", False))
        super().__init__(**ppo_kwargs)
        self.expert_dataset = ExpertTransitionDataset(expert_csv)
        self.discriminator = TransitionDiscriminator(self.obs_dim).to(self.device)
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=float(discriminator_lr)
        )
        self.discriminator_updates = int(discriminator_updates)
        self.discriminator_batch_size = int(discriminator_batch_size)
        self.imitation_reward_scale = float(imitation_reward_scale)
        self.agent_transitions = []
        self.discriminator_ready = False
        self.model_name = "state_imitation"
        if should_load_model:
            self.load_model()

    def imitation_reward(self, state, next_state):
        transition = torch.as_tensor(
            np.concatenate((state, next_state)),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            expert_probability = torch.sigmoid(self.discriminator(transition))
            reward = -torch.log(torch.clamp(1.0 - expert_probability, min=1e-6))
        return self.imitation_reward_scale * float(reward.item())

    def train_discriminator(self):
        if not self.agent_transitions:
            return
        agent_data = torch.as_tensor(
            np.stack(self.agent_transitions), dtype=torch.float32, device=self.device
        )
        losses = []
        expert_accuracies = []
        agent_accuracies = []
        for _ in range(self.discriminator_updates):
            batch_size = min(self.discriminator_batch_size, len(agent_data))
            expert = self.expert_dataset.sample(batch_size, self.device)
            indices = torch.randint(0, len(agent_data), (batch_size,), device=self.device)
            agent = agent_data[indices]

            expert_logits = self.discriminator(expert)
            agent_logits = self.discriminator(agent)
            loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    expert_logits, torch.ones_like(expert_logits)
                )
                + nn.functional.binary_cross_entropy_with_logits(
                    agent_logits, torch.zeros_like(agent_logits)
                )
            )
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.discriminator_optimizer.step()

            losses.append(float(loss.item()))
            expert_accuracies.append(float((expert_logits > 0).float().mean().item()))
            agent_accuracies.append(float((agent_logits < 0).float().mean().item()))

        print(
            f"[Discriminator] loss={np.mean(losses):.4f}, "
            f"expert_acc={np.mean(expert_accuracies):.3f}, "
            f"agent_acc={np.mean(agent_accuracies):.3f}"
        )
        self.agent_transitions.clear()

    def train_on_buffer(self):
        self.train_discriminator()
        if not self.discriminator_ready:
            # The first rollout exists only to provide negative examples. PPO
            # must not optimize rewards emitted by a random discriminator.
            self.buf.get()
            self.discriminator_ready = True
            self.pending_train = False
            print("[StateImitation] Discriminator bootstrapped; PPO starts next rollout.")
            return
        super().train_on_buffer()

    def process_data(self, camera):
        self.total_steps += 1
        obs, _, _, _, _ = self.extract_observation(camera)
        if not self.training_enabled:
            action, _, _ = self.compute_action(obs, deterministic=self.inference)
            return self.scale_to_motor_commands(action)
        terminated = bool(
            camera.get("terminated_by_kick", False)
            or camera.get("terminated_by_x_threshold", False)
            or camera.get("end_episode", False)
        )

        episode_finished = False
        if terminated and self.last_obs is not None:
            # The simulator reports termination together with the already-reset
            # observation.  Do not teach the discriminator that teleporting from
            # the old episode into the new spawn is an agent transition.
            self.finish_episode(last_value=0.0)
            episode_finished = True
        elif self.last_obs is not None:
            transition = np.concatenate((self.last_obs, obs)).astype(np.float32)
            self.agent_transitions.append(transition)
            reward = self.imitation_reward(self.last_obs, obs)
            self.current_episode_step_rewards.append(reward)
            self.ep_reward += reward
            stored = self.buf.store(
                self.last_obs, self.last_action, reward, self.last_val, self.last_logp
            )
            if not stored:
                _, final_value, _ = self.compute_action(obs)
                self.buf.finish_path(last_val=final_value)
                self.train_on_buffer()
                self.last_obs = None
                episode_finished = True

        action, value, logp = self.compute_action(obs)
        if not np.all(np.isfinite(action)):
            action = np.zeros_like(action)

        if not episode_finished or terminated:
            self.last_obs = obs
            self.last_action = action
            self.last_val = value
            self.last_logp = logp
            self.episode_steps += 1

        return self.scale_to_motor_commands(action)

    def save_model(self, path=None):
        if path is None:
            save_dir = self.model_save_path
            if os.path.splitext(save_dir)[1]:
                save_dir = os.path.dirname(save_dir) or "."
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{self.model_name}_steps_{self.total_steps}.pth")
        torch.save({
            "actor_critic_state_dict": self.ac.state_dict(),
            "actor_state_dict": self.ac.actor.state_dict(),
            "critic_state_dict": self.ac.critic.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "training_count": self.training_count,
        }, path)
        print(f"[StateImitation] Model saved to {path}")

    def load_model(self, path=None):
        path = path or self.model_save_path
        if not os.path.isfile(path):
            print(f"[StateImitation] No checkpoint found at {path}")
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.ac.load_state_dict(checkpoint["actor_critic_state_dict"])
        if "log_std" in checkpoint:
            self.log_std.data.copy_(checkpoint["log_std"].to(self.device))
        if "discriminator_state_dict" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            self.discriminator_ready = True
        self.training_count = int(checkpoint.get("training_count", 0))
        print(f"[StateImitation] Model loaded from {path}")


class HybridImitationPPOAgent(StateImitationPPOAgent):
    """Normal task PPO augmented with a learned state-imitation reward.

    The first rollout uses only the environment reward while it gathers negative
    examples for the discriminator.  Later rollouts use
    ``environment_reward + imitation_weight * imitation_reward``.
    """

    def __init__(
        self,
        expert_csv,
        imitation_weight=0.2,
        final_imitation_weight=0.0,
        imitation_anneal_steps=0,
        pretrained_discriminator_path=None,
        **ppo_kwargs,
    ):
        super().__init__(expert_csv=expert_csv, **ppo_kwargs)
        self.initial_imitation_weight = float(imitation_weight)
        self.final_imitation_weight = float(final_imitation_weight)
        self.imitation_anneal_steps = int(imitation_anneal_steps)
        self.model_name = "hybrid_imitation_ppo"

        if pretrained_discriminator_path is not None:
            checkpoint = torch.load(
                pretrained_discriminator_path,
                map_location=self.device,
                weights_only=True,
            )
            if "discriminator_state_dict" not in checkpoint:
                raise KeyError(
                    f"Checkpoint {pretrained_discriminator_path!r} has no "
                    "discriminator_state_dict"
                )
            self.discriminator.load_state_dict(
                checkpoint["discriminator_state_dict"], strict=True
            )
            self.discriminator_ready = True
            print(
                "[HybridImitationPPO] Loaded pretrained discriminator from "
                f"{pretrained_discriminator_path}"
            )

    def current_imitation_weight(self):
        if self.imitation_anneal_steps <= 0:
            return self.initial_imitation_weight
        fraction = min(max(self.total_steps / self.imitation_anneal_steps, 0.0), 1.0)
        return (
            self.initial_imitation_weight
            + fraction
            * (self.final_imitation_weight - self.initial_imitation_weight)
        )

    def train_on_buffer(self):
        # Unlike imitation-only training, the bootstrap rollout has valid task
        # rewards, so it must still be used for a PPO update.
        self.train_discriminator()
        if not self.discriminator_ready:
            self.discriminator_ready = True
            print(
                "[HybridImitationPPO] Discriminator bootstrapped; imitation "
                "reward starts on the next rollout."
            )
        PPOAgent.train_on_buffer(self)

    def process_data(self, camera):
        self.total_steps += 1
        obs, bxy, vxy, _, active_rod = self.extract_observation(camera)

        if not self.training_enabled:
            action, _, _ = self.compute_action(obs, deterministic=self.inference)
            return self.scale_to_motor_commands(action)

        ball_kicked = bool(camera.get("ball_kicked", False))
        terminated_by_kick = bool(camera.get("terminated_by_kick", False))
        terminated_by_x_threshold = bool(
            camera.get("terminated_by_x_threshold", False)
        )
        end_episode = bool(camera.get("end_episode", False))
        terminated = bool(
            terminated_by_kick or terminated_by_x_threshold or end_episode
        )

        episode_finished = False
        if self.last_obs is not None:
            alignment_reward = closest_player_alignment_reward(
                ball_y=bxy[1],
                rod_pos_calib=float(active_rod["pos_calib"]),
                rod_info=active_rod["info"],
                reward_scale=1.0,
            )
            angle_reward = calculate_rod_angle_reward(
                rod_angle=float(active_rod["angle"]),
                reward_scale=1.0,
                target_rod_angle=0.0,
                rotation_buffer_def=3,
            )
            reward_breakdown = {
                "allignment": float(alignment_reward),
                "rod_angle": float(angle_reward) * 0.33,
            }
            _, ball_control_breakdown = self.compute_ball_control_reward(
                bxy,
                vxy,
                active_rod,
                terminated_by_x_threshold=terminated_by_x_threshold,
                end_episode=end_episode,
            )
            reward_breakdown.update(ball_control_breakdown)
            environment_reward = float(sum(reward_breakdown.values()))

            imitation_reward = 0.0
            imitation_weight = self.current_imitation_weight()
            # Do not include reset teleports as discriminator examples.
            if not terminated:
                transition = np.concatenate((self.last_obs, obs)).astype(np.float32)
                self.agent_transitions.append(transition)
                if self.discriminator_ready:
                    imitation_reward = self.imitation_reward(self.last_obs, obs)

            weighted_imitation_reward = imitation_weight * imitation_reward
            reward = environment_reward + weighted_imitation_reward
            reward_breakdown.update({
                "environment_total": environment_reward,
                "imitation_raw": imitation_reward,
                "imitation_weighted": weighted_imitation_reward,
                "imitation_weight": imitation_weight,
            })
            for key, value in reward_breakdown.items():
                self.update_reward_breakdown_sums[key] = (
                    self.update_reward_breakdown_sums.get(key, 0.0) + float(value)
                )

            if ball_kicked:
                self.current_episode_ball_kicks += 1
                self.update_samples_with_kick += 1
            self.current_episode_step_rewards.append(reward)
            self.ep_reward += reward

            stored = self.buf.store(
                self.last_obs,
                self.last_action,
                reward,
                self.last_val,
                self.last_logp,
            )
            if not stored:
                _, final_value, _ = self.compute_action(obs)
                self.buf.finish_path(last_val=final_value)
                self.train_on_buffer()
                self.last_obs = None
                episode_finished = True

            if not episode_finished and terminated:
                self.finish_episode(last_value=0.0)
                self.episode_steps = 0
                episode_finished = True

        action, value, logp = self.compute_action(obs)
        if not np.all(np.isfinite(action)):
            action = np.zeros_like(action)

        if not episode_finished:
            self.last_obs = obs
            self.last_action = action
            self.last_val = value
            self.last_logp = logp
            self.episode_steps += 1

        self.prev_ball_vxy = vxy
        return self.scale_to_motor_commands(action)
