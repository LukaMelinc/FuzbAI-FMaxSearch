from __future__ import annotations

from typing import Any, Callable

from FuzbAIAgent_Example import PlayerAgent
from strel_PPO import PPOAgent, PassPPOAgent, TwoRodPPOAgent
from self_play_manager import SelfPlayManager, SelfPlayParticipant


_AGENT_REGISTRY = {
    "demo": PlayerAgent,
    "player": PlayerAgent,
    "ppo": PPOAgent,
    "two_rod_ppo": TwoRodPPOAgent,
    "pass_ppo": PassPPOAgent,
}


def resolve_agent_factory(agent_spec: Any) -> Callable[..., Any]:
    """Resolve a user-provided agent spec into a callable factory.

    Supported specs:
    - an already-instantiated agent object
    - a callable/class
    - a string key in the local registry
    - a dict with keys: type, kwargs
    """
    if hasattr(agent_spec, "process_data"):
        return lambda **_: agent_spec

    if isinstance(agent_spec, dict):
        agent_type = agent_spec.get("type")
        if agent_type is None:
            raise ValueError("Agent spec dict must contain a 'type' key.")
        factory = resolve_agent_factory(agent_type)
        kwargs = dict(agent_spec.get("kwargs", {}))
        return lambda **extra_kwargs: factory(**{**kwargs, **extra_kwargs})

    if isinstance(agent_spec, str):
        try:
            return _AGENT_REGISTRY[agent_spec.lower()]
        except KeyError as exc:
            raise ValueError(
                f"Unknown agent type '{agent_spec}'. Supported types: {sorted(_AGENT_REGISTRY)}"
            ) from exc

    if callable(agent_spec):
        return agent_spec

    raise TypeError(
        "Agent spec must be an instantiated agent, a callable, a registry string, or a dict."
    )


def create_agent(agent_spec: Any = None, **kwargs) -> Any:
    """Build an agent instance from a spec or return an existing agent."""
    if agent_spec is None:
        return PlayerAgent(**kwargs) if kwargs else PlayerAgent()

    factory = resolve_agent_factory(agent_spec)
    return factory(**kwargs)


def create_self_play_manager(
    participant_1_spec: Any,
    participant_2_spec: Any,
    participant_1_kwargs: dict | None = None,
    participant_2_kwargs: dict | None = None,
    participant_1_player_id: int = 1,
    participant_2_player_id: int = 2,
) -> SelfPlayManager:
    """Build a synchronized two-agent self-play manager from agent specs."""
    participant_1_kwargs = dict(participant_1_kwargs or {})
    participant_2_kwargs = dict(participant_2_kwargs or {})
    participant_1_kwargs.setdefault("auto_train", False)
    participant_2_kwargs.setdefault("auto_train", False)

    participant_1 = SelfPlayParticipant(
        name="player_1",
        agent=create_agent(participant_1_spec, **participant_1_kwargs),
        camera_player_id=int(participant_1_player_id),
    )
    participant_2 = SelfPlayParticipant(
        name="player_2",
        agent=create_agent(participant_2_spec, **participant_2_kwargs),
        camera_player_id=int(participant_2_player_id),
    )
    return SelfPlayManager([participant_1, participant_2])
