from __future__ import annotations

from typing import Any, Callable

from FuzbAIAgent_Example import PlayerAgent
from pass_auxiliary_backbone import PassAuxiliaryBackboneAgent
from strel_PPO import PPOAgent, PassPPOAgent, TwoRodPPOAgent


_AGENT_REGISTRY = {
    "demo": PlayerAgent,
    "player": PlayerAgent,
    "ppo": PPOAgent,
    "two_rod_ppo": TwoRodPPOAgent,
    "pass_ppo": PassPPOAgent,
    "pass_auxiliary_backbone": PassAuxiliaryBackboneAgent,
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
