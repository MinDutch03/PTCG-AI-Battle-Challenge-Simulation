"""Decision making: flat Monte-Carlo search over determinized games.

For each legal root action we run heuristic rollouts to the end of the
current turn inside the engine's search sandbox (search_begin/search_step),
score the resulting position, and pick the action with the best average.
Falls back to the pure heuristic policy on any search trouble.
"""

import itertools
import random
import time

from cg.api import (
    Observation,
    SelectType,
    search_begin,
    search_end,
    search_step,
)

from . import determinize
from .evaluate import evaluate
from .heuristics import choose

MAX_DETERMINIZATIONS = 6
MAX_ROLLOUT_STEPS = 80
MAX_CANDIDATES = 16

_search_failures = 0
_SEARCH_DISABLED_AFTER = 5


def _candidates(obs: Observation, rng: random.Random) -> list[list[int]]:
    """Candidate selections to compare at the root."""
    sel = obs.select
    n = len(sel.option)
    lo, hi = sel.minCount, sel.maxCount

    if hi == 1:
        singles = [[i] for i in range(n)]
        if lo == 0:
            singles.append([])
        if len(singles) > MAX_CANDIDATES:
            # Keep the heuristic pick, trim the rest arbitrarily but stably.
            keep = {tuple(choose(obs, rng))}
            for i in range(n):
                if len(keep) >= MAX_CANDIDATES:
                    break
                keep.add((i,))
            singles = [list(t) for t in keep]
        return singles

    # Multi-selection: the heuristic subset plus a few size variants and
    # random subsets. Exhaustive enumeration only when it is small.
    cands: list[tuple[int, ...]] = []
    if n <= 5 and hi >= n and lo <= 2:
        for size in range(lo, min(hi, n) + 1):
            cands.extend(itertools.combinations(range(n), size))
    else:
        base = choose(obs, rng)
        cands.append(tuple(base))
        order = list(range(n))
        rng.shuffle(order)
        for size in {lo, min(hi, n)}:
            cands.append(tuple(sorted(order[:size])))
        for _ in range(4):
            size = rng.randint(lo, min(hi, n))
            cands.append(tuple(sorted(rng.sample(range(n), size))))
    uniq = list(dict.fromkeys(cands))
    return [list(c) for c in uniq[:MAX_CANDIDATES]]


def _stop_turn(state, me: int) -> int:
    """Roll until this turn number is exceeded: through the end of our own
    turn AND the opponent's reply turn, so the evaluation sees the punch-back.
    If we are deciding during the opponent's turn (forced switch etc.), their
    current turn already is the reply."""
    t = state.turn
    if t <= 0:
        return 1
    my_turn = (t % 2 == 1) == (me == state.firstPlayer)
    return t + 1 if my_turn else t


def _rollout(state, stop_turn: int, rng: random.Random):
    """Play heuristically inside the sandbox until stop_turn is over."""
    for _ in range(MAX_ROLLOUT_STEPS):
        cur = state.observation.current
        if cur.result >= 0 or cur.turn > stop_turn:
            break
        state = search_step(state.searchId, choose(state.observation, rng))
    return state


def decide(obs: Observation, my_deck_list: list[int], deadline: float,
           rng: random.Random) -> list[int]:
    """Best selection for this observation, within the time budget."""
    global _search_failures

    sel = obs.select
    n = len(sel.option)

    # Forced or trivial selections need no thought.
    if n == 0 or (sel.minCount >= n and sel.maxCount >= n):
        return list(range(n))
    if n == 1 and sel.minCount >= 1:
        return [0]

    if _search_failures >= _SEARCH_DISABLED_AFTER or obs.search_begin_input is None:
        return choose(obs, rng)

    cands = _candidates(obs, rng)
    if len(cands) == 1:
        return cands[0]

    me = obs.current.yourIndex
    stop_turn = _stop_turn(obs.current, me)
    totals = [0.0] * len(cands)
    counts = [0] * len(cands)

    try:
        for _ in range(MAX_DETERMINIZATIONS):
            if time.time() > deadline:
                break
            det = determinize.sample(obs, my_deck_list, rng)
            root = search_begin(obs, *det)
            try:
                for ci, cand in enumerate(cands):
                    if time.time() > deadline and counts[0] > 0:
                        break
                    st = search_step(root.searchId, cand)
                    st = _rollout(st, stop_turn, rng)
                    totals[ci] += evaluate(st.observation.current, me)
                    counts[ci] += 1
            finally:
                search_end()
    except Exception:
        _search_failures += 1
        return choose(obs, rng)

    scored = [(totals[i] / counts[i], i) for i in range(len(cands)) if counts[i] > 0]
    if not scored:
        return choose(obs, rng)
    best = max(scored)[1]
    return cands[best]
