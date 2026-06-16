import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from actor_critic.main import ThreeRodActorCriticNet
from reward.single_bar_shoting import player_alignment_target


class PassAuxiliaryBackboneAgent:
    """
    Auxiliary trainer for the first two-rod passing backbone.

    This is not PPO. It trains the policy with supervised MSE targets:
      - passer translation aligns one passer player with the ball y position
      - receiver translation aligns one receiver player with the ball y position
      - passer rotation points toward a fixed "feet down" angle
      - receiver rotation points toward a fixed "feet down" angle

    Rod IDs are geometry IDs. For the first red-side passing setup:
      - passer_rod_id=4   red midfield rod
      - receiver_rod_id=6 red attack rod
      - opponent_rod_id=5 blue midfield rod between them
    """

    def __init__(
        self,
        obs_dim=20,
        hidden_size=512,
        head_hidden_size=128,
        lr=1e-4,
        geometry_path="geometry.json",
        model_save_path="./trained_models/pass_backbone_aux.pth",
        load_model=False,
        training_enabled=True,
        inference=False,
        passer_rod_id=4,
        receiver_rod_id=6,
        opponent_rod_id=5,
        target_angle_normalized=0.085,
        rotation_velocity_target=0.15,
        translation_velocity_min=0.15,
        translation_velocity_max=1.0,
        print_every_steps=250,
        save_every_steps=5000,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = 8
        self.model_save_path = model_save_path
        self.training_enabled = bool(training_enabled)
        self.inference = bool(inference)
        if self.inference and self.training_enabled:
            print("[PassAux] Inference mode enabled, disabling training.")
            self.training_enabled = False

        self.passer_rod_id = int(passer_rod_id)
        self.receiver_rod_id = int(receiver_rod_id)
        self.opponent_rod_id = int(opponent_rod_id)
        self.target_angle_normalized = float(target_angle_normalized)
        self.rotation_velocity_target = float(rotation_velocity_target)
        self.translation_velocity_min = float(translation_velocity_min)
        self.translation_velocity_max = float(translation_velocity_max)
        self.print_every_steps = int(print_every_steps)
        self.save_every_steps = int(save_every_steps)

        self.geometry_path = self._resolve_geometry_path(geometry_path)
        with open(self.geometry_path) as f:
            self.geometry = json.load(f)

        self.rods_by_id = {int(rod["id"]): rod for rod in self.geometry["rods"]}
        self.field_x = float(self.geometry["field"]["dimension_x"])
        self.field_y = float(self.geometry["field"]["dimension_y"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[PassAux] Using device:", self.device)

        self.ac = ThreeRodActorCriticNet(
            obs_dim=self.obs_dim,
            hidden_size=hidden_size,
            head_hidden_size=head_hidden_size,
        ).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.mse = nn.MSELoss()

        self.total_steps = 0
        self.training_steps = 0
        self.loss_sums = {
            "total": 0.0,
            "passer_rotation": 0.0,
            "passer_translation": 0.0,
            "receiver_rotation": 0.0,
            "receiver_translation": 0.0,
        }

        if load_model:
            self.load_model()

    @staticmethod
    def _resolve_geometry_path(geometry_path):
        if os.path.exists(geometry_path):
            return geometry_path

        local_path = os.path.join(os.path.dirname(__file__), geometry_path)
        if os.path.exists(local_path):
            return local_path

        return geometry_path

    @staticmethod
    def _as_float(value):
        if isinstance(value, list):
            value = value[0]
        return float(value)

    @staticmethod
    def _clip_action(value):
        return float(np.clip(value, -1.0, 1.0))

    def save_model(self, path=None):
        if path is None:
            path = self.model_save_path
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "actor_critic_state_dict": self.ac.state_dict(),
                "training_steps": self.training_steps,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "passer_rod_id": self.passer_rod_id,
                "receiver_rod_id": self.receiver_rod_id,
                "opponent_rod_id": self.opponent_rod_id,
            },
            path,
        )
        print(f"[PassAux] Model saved to {path}")

    def load_model(self, path=None):
        if path is None:
            path = self.model_save_path
        if not os.path.exists(path):
            print(f"[PassAux] No saved model found at {path}, skipping load.")
            return

        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "actor_critic_state_dict" in checkpoint:
            self.ac.load_state_dict(checkpoint["actor_critic_state_dict"])
            self.training_steps = int(checkpoint.get("training_steps", self.training_steps))
        else:
            self.ac.load_state_dict(checkpoint)
        print(f"[PassAux] Model loaded from {path}")

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

        def rod_features(rod_id, state):
            rod_info = state["info"]
            return [
                (float(rod_info["position"]) - self.field_x / 2.0) / (self.field_x / 2.0),
                state["pos_calib"],
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
                *rod_features(self.passer_rod_id, passer),
                *rod_features(self.receiver_rod_id, receiver),
                *rod_features(self.opponent_rod_id, opponent),
                float(camera.get("ball_kicked", False)),
            ],
            dtype=np.float32,
        )

        assert len(obs) == self.obs_dim, f"Expected obs_dim={self.obs_dim}, got {len(obs)}"
        return obs, passer, receiver, opponent

    def _target_translation_action(self, rod_state):
        target_pos = float(np.clip(rod_state["target_pos"], 0.0, 1.0))
        error_mag = min(1.0, abs(float(rod_state["target_error"])) * 4.0)
        velocity = self.translation_velocity_min + (
            self.translation_velocity_max - self.translation_velocity_min
        ) * error_mag
        velocity = float(np.clip(velocity, 0.0, 1.0))

        return [
            self._clip_action(2.0 * target_pos - 1.0),
            self._clip_action(2.0 * velocity - 1.0),
        ]

    def _target_rotation_action(self):
        # Existing motor scaling uses rot_target = 0.5 * action[0], while
        # normalized rod angle is approximately 2 * rot_target. Therefore the
        # action target is the desired normalized rod angle.
        rot_target = self._clip_action(self.target_angle_normalized)
        rot_velocity = self._clip_action(8.0 * self.rotation_velocity_target - 1.0)
        return [rot_target, rot_velocity]

    def build_auxiliary_targets(self, passer, receiver):
        target = np.array(
            [
                *self._target_rotation_action(),
                *self._target_translation_action(passer),
                *self._target_rotation_action(),
                *self._target_translation_action(receiver),
            ],
            dtype=np.float32,
        )
        assert len(target) == self.act_dim
        return target

    def train_auxiliary_step(self, obs, target):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        target_t = torch.as_tensor(target, dtype=torch.float32, device=self.device).unsqueeze(0)

        _, _, heads = self.ac(obs_t, return_heads=True)
        pred = {
            key: torch.tanh(value)
            for key, value in heads.items()
        }

        target_heads = {
            "passer_rotation": target_t[:, 0:2],
            "passer_translation": target_t[:, 2:4],
            "receiver_rotation": target_t[:, 4:6],
            "receiver_translation": target_t[:, 6:8],
        }

        losses = {
            key: self.mse(pred[key], target_heads[key])
            for key in target_heads
        }
        loss = sum(losses.values())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1
        self.loss_sums["total"] += float(loss.detach().cpu().item())
        for key, value in losses.items():
            self.loss_sums[key] += float(value.detach().cpu().item())

        return float(loss.detach().cpu().item())

    def compute_action(self, obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mean, _ = self.ac(obs_t)
            action = torch.tanh(mean)
        return action.detach().cpu().numpy()[0]

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

    def scale_to_motor_commands(self, action):
        return [
            self._command_for_rod(self.passer_rod_id, action[0:4]),
            self._command_for_rod(self.receiver_rod_id, action[4:8]),
        ]

    def _print_training_stats(self):
        if self.training_steps <= 0:
            return

        n = float(max(1, min(self.print_every_steps, self.training_steps)))
        print(
            "[PassAux] "
            f"step={self.total_steps}, train_step={self.training_steps}, "
            f"loss={self.loss_sums['total'] / n:.6f}, "
            f"passer_rot={self.loss_sums['passer_rotation'] / n:.6f}, "
            f"passer_trans={self.loss_sums['passer_translation'] / n:.6f}, "
            f"receiver_rot={self.loss_sums['receiver_rotation'] / n:.6f}, "
            f"receiver_trans={self.loss_sums['receiver_translation'] / n:.6f}"
        )
        for key in self.loss_sums:
            self.loss_sums[key] = 0.0

    def process_data(self, camera):
        self.total_steps += 1

        obs, passer, receiver, _ = self.extract_observation(camera)
        target = self.build_auxiliary_targets(passer, receiver)

        if self.training_enabled:
            self.train_auxiliary_step(obs, target)
            if self.print_every_steps > 0 and self.total_steps % self.print_every_steps == 0:
                self._print_training_stats()
            if self.save_every_steps > 0 and self.total_steps % self.save_every_steps == 0:
                self.save_model()

        action = self.compute_action(obs)
        return self.scale_to_motor_commands(action)
