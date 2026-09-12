"""State-only imitation-from-observation experiment for the single-rod agent.

Real table recordings supply calibrated observation transitions. The
imitation agent learns a discriminator over (state, next_state) transitions and
uses its output as the reward for an otherwise standard PPO policy.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from strel_PPO import PPOAgent


from recorded_expert import OBS_COLUMNS, ExpertTransitionDataset, RecordingConfig


class TransitionDiscriminator(nn.Module):
    def __init__(self, obs_dim=len(OBS_COLUMNS), hidden_size=128):
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
        expert_data=None,
        recording_config=None,
        discriminator_lr=3e-4,
        discriminator_updates=8,
        discriminator_batch_size=128,
        imitation_reward_scale=1.0,
        **ppo_kwargs,
    ):
        should_load_model = bool(ppo_kwargs.pop("load_model", False))
        super().__init__(**ppo_kwargs)
        self.recording_config = recording_config or RecordingConfig()
        self.expert_dataset = (
            ExpertTransitionDataset(expert_data, self.geometry, self.recording_config)
            if expert_data is not None else None
        )
        if self.training_enabled and self.expert_dataset is None:
            raise ValueError("Training requires --expert-data")
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

    def transition_reward(self, camera, obs):
        return self.imitation_reward(self.last_obs, obs) if self.discriminator_ready else 0.0

    def process_data(self, camera):
        """Consume a physical terminal state before the simulator resets."""
        self.total_steps += 1
        obs, c, _, _, _ = self.extract_observation(camera)

        print(c)
        if not self.training_enabled:
            action, _, _ = self.compute_action(obs, deterministic=self.inference)
            return self.scale_to_motor_commands(action)
        terminated = any(bool(camera.get(key, False)) for key in (
            "end_episode", "terminated_by_kick", "terminated_by_x_threshold", "terminated_by_timeout"))
        if self.last_obs is not None:
            self.agent_transitions.append(np.concatenate((self.last_obs, obs)).astype(np.float32))
            reward = self.transition_reward(camera, obs)
            self.current_episode_step_rewards.append(reward)
            self.ep_reward += reward
            if camera.get("ball_kicked", False):
                self.current_episode_ball_kicks += 1
                self.update_samples_with_kick += 1
            if not self.buf.store(self.last_obs, self.last_action, reward, self.last_val, self.last_logp):
                raise RuntimeError("PPO buffer was not drained before the next transition")
        if terminated:
            self.finish_episode(last_value=0.0)
            self.episode_steps = 0
            return []
        if self.buf.ptr == self.buf.max_size:
            _, value, _ = self.compute_action(obs)
            self.buf.finish_path(last_val=value)
            self.train_on_buffer()
        action, value, logp = self.compute_action(obs)
        if not np.all(np.isfinite(action)):
            raise RuntimeError("Policy produced a non-finite action")
        self.last_obs, self.last_action = obs, action
        self.last_val, self.last_logp = value, logp
        self.episode_steps += 1
        return self.scale_to_motor_commands(action)

    def scale_to_motor_commands(self, action):
        # Other rods are initialized from the recording and held by the simulator.
        return super().scale_to_motor_commands(action)[:1]

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
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
            "discriminator_ready": self.discriminator_ready,
            "recording_config": vars(self.recording_config),
            "episode_starts": self.expert_dataset.starts if self.expert_dataset is not None else self.episode_starts,
            "observation_columns": list(OBS_COLUMNS),
        }, path)
        print(f"[StateImitation] Model saved to {path}")

    def load_model(self, path=None):
        path = path or self.model_save_path
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("observation_columns") != list(OBS_COLUMNS):
            raise ValueError("Checkpoint lacks the real-recording observation schema; start a new run")
        if vars(RecordingConfig(**checkpoint.get("recording_config", {}))) != vars(self.recording_config):
            raise ValueError("Checkpoint and recording calibration differ")
        self.episode_starts = checkpoint["episode_starts"]
        self.ac.load_state_dict(checkpoint["actor_critic_state_dict"])
        if "log_std" in checkpoint:
            self.log_std.data.copy_(checkpoint["log_std"].to(self.device))
        if "discriminator_state_dict" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            self.discriminator_ready = True
        self.training_count = int(checkpoint.get("training_count", 0))
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.episode_count = int(checkpoint.get("episode_count", 0))
        self.discriminator_ready = bool(checkpoint.get("discriminator_ready", False))
        if self.training_enabled:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
        print(f"[StateImitation] Model loaded from {path}")


class HybridImitationPPOAgent(StateImitationPPOAgent):
    """Normal task PPO augmented with a learned state-imitation reward.

    The first rollout uses only the environment reward while it gathers negative
    examples for the discriminator.  Later rollouts use
    ``environment_reward + imitation_weight * imitation_reward``.
    """

    def __init__(
        self,
        expert_data=None,
        imitation_weight=0.2,
        final_imitation_weight=0.0,
        imitation_anneal_steps=0,
        pretrained_discriminator_path=None,
        **ppo_kwargs,
    ):
        super().__init__(expert_data=expert_data, **ppo_kwargs)
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

    def transition_reward(self, camera, obs):
        _, bxy, vxy, _, active_rod = self.extract_observation(camera)
        # Forward kick objective: avoid a resting-angle reward that can oppose kicking.
        from reward.single_bar_shoting import forward_backward_kick_reward
        task_reward, _ = forward_backward_kick_reward(
            ball_kicked=bool(camera.get("ball_kicked", False)), forward_ball_vx=vxy[0],
        )
        imitation = super().transition_reward(camera, obs)
        return task_reward + self.current_imitation_weight() * imitation
