"""PTCG AI Battle Challenge — submission entry point.

Deck selection returns deck.csv; every in-game selection goes through a
determinized Monte-Carlo search (ptcg_agent.search) with layered fallbacks:
search -> heuristic policy -> first-legal-option. The agent never raises.
"""

import inspect
import os
import random
import sys
import time


def _agent_dir() -> str:
    """Directory holding this file and deck.csv, under any loading scheme.

    Kaggle exec()s main.py without __file__, so fall back to the compiled
    code object's filename, then to sys.path entries that contain deck.csv.
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    try:
        fn = inspect.currentframe().f_back.f_code.co_filename
        d = os.path.dirname(os.path.abspath(fn))
        if os.path.exists(os.path.join(d, "deck.csv")):
            return d
    except Exception:
        pass
    for d in ["/kaggle_simulations/agent", os.getcwd()] + list(sys.path):
        if d and os.path.exists(os.path.join(d, "deck.csv")):
            return d
    return os.getcwd()


_HERE = _agent_dir()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# If anything here fails (engine binary missing/incompatible on the host),
# the module still loads and the agent plays on raw-dict fallback logic.
try:
    from cg.api import SelectType, to_observation_class  # noqa: E402
    from ptcg_agent import search  # noqa: E402
    from ptcg_agent.heuristics import choose, fallback  # noqa: E402
    _IMPORTS_OK = True
except Exception:  # pragma: no cover - defensive
    _IMPORTS_OK = False

    def fallback(obs_dict: dict) -> list[int]:
        sel = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        if not sel:
            return []
        need = sel["minCount"] if sel["minCount"] > 0 else min(1, sel["maxCount"])
        return list(range(need))

def read_deck_csv() -> list[int]:
    for path in ("deck.csv",
                 os.path.join(_HERE, "deck.csv"),
                 "/kaggle_simulations/agent/deck.csv"):
        if os.path.exists(path):
            with open(path) as f:
                return [int(x) for x in f.read().split() if x.strip()][:60]
    raise FileNotFoundError("deck.csv not found")


_TIME_SCALE = float(os.environ.get("PTCG_TIME_SCALE", "1"))


def _budget(obs_dict: dict, obs) -> float:
    """Seconds to spend on this decision.

    The overage pool is 600s per episode and a game runs ~100-200 decisions;
    an average near 2.5s/decision uses the clock instead of donating it.
    Throttles as the pool drains, heuristics-only when it is nearly gone.
    PTCG_TIME_SCALE (<1) speeds up local evaluation runs.
    """
    remaining = obs_dict.get("remainingOverageTime", 600) or 0
    if remaining > 450:
        base = 4.5
    elif remaining > 300:
        base = 2.5
    elif remaining > 150:
        base = 1.2
    elif remaining > 60:
        base = 0.4
    else:
        return 0.0  # heuristics only, preserve the clock
    if obs.select.type not in (SelectType.MAIN, SelectType.ATTACK):
        base *= 0.6
    # Late-game decisions decide close prize races AND cost more per search
    # sample (search_begin replays the whole history) — scale up with turn.
    turn = obs.current.turn if obs.current else 0
    base *= 1.0 + min(turn, 20) / 15.0
    return base * _TIME_SCALE


def make_agent(deck: list[int] | None = None, seed: int = 20260711,
               safety: bool = True):
    """Build an agent closure bound to a specific deck (for local harnesses)."""
    rng = random.Random(seed)
    state = {"deck": deck}

    def _agent(obs_dict: dict) -> list[int]:
        try:
            if state["deck"] is None:
                state["deck"] = read_deck_csv()
            if obs_dict["select"] is None:
                return state["deck"]
            if not _IMPORTS_OK:
                return fallback(obs_dict)
            obs = to_observation_class(obs_dict)
            state["n"] = state.get("n", 0) + 1
            if state["n"] % 25 == 1 and "remainingOverageTime" in obs_dict:
                # Search-health heartbeat; shows up in Kaggle agent logs.
                print(f"[ptcg] decision {state['n']}: turn "
                      f"{obs.current.turn if obs.current else '?'} "
                      f"search ok={search._search_successes} "
                      f"fail={search._search_failures} "
                      f"overage={obs_dict.get('remainingOverageTime', '?')}",
                      file=sys.stderr, flush=True)
            budget = _budget(obs_dict, obs)
            if budget <= 0:
                return choose(obs, rng)
            deadline = time.time() + budget
            return search.decide(obs, state["deck"], deadline, rng,
                                 safety=safety)
        except Exception:
            try:
                if _IMPORTS_OK and obs_dict.get("select") is not None:
                    return choose(to_observation_class(obs_dict), random)
            except Exception:
                pass
            try:
                return fallback(obs_dict)
            except Exception:
                return []

    return _agent


agent = make_agent()
