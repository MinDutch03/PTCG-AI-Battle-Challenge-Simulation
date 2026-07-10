"""Heuristic-only baseline: the ptcg_agent rollout policy with no search."""

import random

from cg.api import to_observation_class
from ptcg_agent.heuristics import choose, fallback

from .random_agent import read_deck

_rng = random.Random(7)


def agent(obs: dict) -> list[int]:
    if obs["select"] is None:
        return read_deck()
    try:
        return choose(to_observation_class(obs), _rng)
    except Exception:
        return fallback(obs)
