from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass
class SelfPlayParticipant:
    """Bind one agent to one simulator side."""

    name: str
    agent: Any
    camera_player_id: int
    trainable: bool = True
    deterministic: bool = False


class SelfPlayManager:
    """Run two agents in lockstep and synchronize their PPO updates.

    The simulator still owns physics and motor application. This manager owns
    the training barrier: both agents keep collecting experience until both are
    ready, and then backpropagation runs for both in the same control cycle.
    """

    def __init__(self, participants: Sequence[SelfPlayParticipant]):
        if len(participants) != 2:
            raise ValueError("SelfPlayManager expects exactly two participants.")

        self.participants = list(participants)
        self.training_cycle = 0

    def _all_trainable_agents_ready(self) -> bool:
        ready = []
        for participant in self.participants:
            if not participant.trainable:
                continue
            ready.append(bool(getattr(participant.agent, "pending_train", False)))
        return bool(ready) and all(ready)

    def _clear_pending_flags(self) -> None:
        for participant in self.participants:
            if hasattr(participant.agent, "pending_train"):
                participant.agent.pending_train = False

    def step(self, camera_player_1: dict, camera_player_2: dict):
        """Advance both agents one control step and return both command lists."""
        camera_by_player = {1: camera_player_1, 2: camera_player_2}
        command_lists = {}

        for participant in self.participants:
            camera = camera_by_player[int(participant.camera_player_id)]
            command_lists[participant.camera_player_id] = participant.agent.process_data(camera)

        if self._all_trainable_agents_ready():
            for participant in self.participants:
                if participant.trainable and hasattr(participant.agent, "train_on_buffer"):
                    participant.agent.train_on_buffer()
            self._clear_pending_flags()
            self.training_cycle += 1

        return command_lists[1], command_lists[2]

    def all_agents(self) -> Iterable[Any]:
        return (participant.agent for participant in self.participants)