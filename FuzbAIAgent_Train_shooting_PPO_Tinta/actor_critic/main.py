import torch
import torch.nn as nn

class ActorCriticNet(nn.Module):
    """
    A simple Actor-Critic network.
    It outputs both action_mean (the policy) and value (the critic).
    """
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),    # 9 -> 128
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), # 128 -> 128
            nn.ReLU(), 
            nn.Linear(hidden_size, act_dim),     # 128 -> 4
            #nn.Tanh()   # NOTE: Temporarely, later change the approach
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),    # 9 -> 128
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), # 128 -> 128
            nn.ReLU(),
            nn.Linear(hidden_size, 1)           # 128 -> 1
        )

    def forward(self, x):
        "Takes in the actin and calculates the log-probabilities of actions "
        "and estimated value of the action"
        action_mean = self.actor(x)
        value = self.critic(x)
        return action_mean, value


class ThreeRodActorCriticNet(nn.Module):
    """
    Actor-Critic network for a coordinated three-rod (2 own and 1 away rod) tasks.

    Expected action layout:
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

    The actor uses separate role encoders and four control heads:
      - passer encoder shared by the passer rotation/translation heads
      - receiver encoder shared by the receiver rotation/translation heads
      - passer rotation head: 2 values
      - passer translation head: 2 values
      - receiver rotation head: 2 values
      - receiver translation head: 2 values

    The network does not prescribe the observation layout, but it is intended
    for observations containing ball state, passer rod state, receiver rod
    state, and the intermediate opponent rod state.
    """

    ACTION_LAYOUT = (
        "passer_rotation_target",
        "passer_rotation_velocity",
        "passer_translation_target",
        "passer_translation_velocity",
        "receiver_rotation_target",
        "receiver_rotation_velocity",
        "receiver_translation_target",
        "receiver_translation_velocity",
    )

    HEAD_SLICES = {
        "passer_rotation": slice(0, 2),
        "passer_translation": slice(2, 4),
        "receiver_rotation": slice(4, 6),
        "receiver_translation": slice(6, 8),
    }

    def __init__(self, obs_dim, hidden_size=512, head_hidden_size=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = 8

        self.passer_encoder = self._make_encoder(obs_dim, hidden_size)
        self.receiver_encoder = self._make_encoder(obs_dim, hidden_size)

        self.passer_rotation_head = self._make_action_head(hidden_size, head_hidden_size)
        self.passer_translation_head = self._make_action_head(hidden_size, head_hidden_size)
        self.receiver_rotation_head = self._make_action_head(hidden_size, head_hidden_size)
        self.receiver_translation_head = self._make_action_head(hidden_size, head_hidden_size)

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    @staticmethod
    def _make_encoder(obs_dim, hidden_size):
        return nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    @staticmethod
    def _make_action_head(hidden_size, head_hidden_size):
        return nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Linear(head_hidden_size, 2),
        )

    def forward(self, x, return_heads=False):
        passer_features = self.passer_encoder(x)
        receiver_features = self.receiver_encoder(x)

        passer_rotation = self.passer_rotation_head(passer_features)
        passer_translation = self.passer_translation_head(passer_features)
        receiver_rotation = self.receiver_rotation_head(receiver_features)
        receiver_translation = self.receiver_translation_head(receiver_features)

        action_mean = torch.cat(
            [
                passer_rotation,
                passer_translation,
                receiver_rotation,
                receiver_translation,
            ],
            dim=-1,
        )
        value = self.critic(x)

        if return_heads:
            heads = {
                "passer_rotation": passer_rotation,
                "passer_translation": passer_translation,
                "receiver_rotation": receiver_rotation,
                "receiver_translation": receiver_translation,
            }
            return action_mean, value, heads

        return action_mean, value

    @staticmethod
    def _upgrade_legacy_state_dict(state_dict):
        """Copy old shared-encoder checkpoints into both role encoders."""
        has_legacy_encoder = any(key.startswith("encoder.") for key in state_dict)
        has_role_encoder = any(
            key.startswith("passer_encoder.") or key.startswith("receiver_encoder.")
            for key in state_dict
        )
        if not has_legacy_encoder or has_role_encoder:
            return state_dict

        upgraded = dict(state_dict)
        for key, value in state_dict.items():
            if not key.startswith("encoder."):
                continue
            suffix = key[len("encoder."):]
            upgraded[f"passer_encoder.{suffix}"] = value
            upgraded[f"receiver_encoder.{suffix}"] = value
            del upgraded[key]
        return upgraded

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict = self._upgrade_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, *args, **kwargs)


# Backwards-compatible alias used by earlier pass-agent drafts.
TwoRodPassActorCriticNet = ThreeRodActorCriticNet
