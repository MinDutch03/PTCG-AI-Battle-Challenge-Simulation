"""PTCG AI Battle Challenge — submission entry point.

Deck selection returns deck.csv; every in-game selection goes through a
determinized Monte-Carlo search (ptcg_agent.search) with layered fallbacks:
search -> heuristic policy -> first-legal-option. The agent never raises.
"""

import os
import random
import sys
import time

# __file__ can be undefined inside Kaggle's agent sandbox (main.py is exec'd).
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = "/kaggle_simulations/agent" if os.path.isdir(
        "/kaggle_simulations/agent") else os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cg.api import SelectType, to_observation_class  # noqa: E402
from ptcg_agent import search  # noqa: E402
from ptcg_agent.heuristics import choose, fallback  # noqa: E402

def read_deck_csv() -> list[int]:
    for path in ("deck.csv",
                 os.path.join(_HERE, "deck.csv"),
                 "/kaggle_simulations/agent/deck.csv"):
        if os.path.exists(path):
            with open(path) as f:
                return [int(x) for x in f.read().split() if x.strip()][:60]
    raise FileNotFoundError("deck.csv not found")


def _budget(obs_dict: dict, obs) -> float:
    """Seconds to spend on this decision."""
    remaining = obs_dict.get("remainingOverageTime", 600) or 0
    if remaining > 400:
        base = 0.9
    elif remaining > 200:
        base = 0.5
    elif remaining > 60:
        base = 0.2
    else:
        return 0.0  # heuristics only, preserve the clock
    if obs.select.type in (SelectType.MAIN, SelectType.ATTACK):
        return base
    return base * 0.6


def make_agent(deck: list[int] | None = None, seed: int = 20260711):
    """Build an agent closure bound to a specific deck (for local harnesses)."""
    rng = random.Random(seed)
    state = {"deck": deck}

    def _agent(obs_dict: dict) -> list[int]:
        try:
            if state["deck"] is None:
                state["deck"] = read_deck_csv()
            if obs_dict["select"] is None:
                return state["deck"]
            obs = to_observation_class(obs_dict)
            deadline = time.time() + _budget(obs_dict, obs)
            return search.decide(obs, state["deck"], deadline, rng)
        except Exception:
            try:
                return choose(to_observation_class(obs_dict), random)
            except Exception:
                return fallback(obs_dict)

    return _agent


agent = make_agent()
