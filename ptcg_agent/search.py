"""Decision making: flat Monte-Carlo search over determinized games.

For each legal root action we run heuristic rollouts to the end of the
current turn inside the engine's search sandbox (search_begin/search_step),
score the resulting position, and pick the action with the best average.
Falls back to the pure heuristic policy on any search trouble.
"""

import itertools
import os
import random
import time

from cg.api import (
    Observation,
    OptionType,
    SelectContext,
    SelectType,
    search_begin,
    search_end,
    search_step,
)

from . import determinize
from .evaluate import evaluate
from .heuristics import choose

# PTCG_MAX_DETS caps determinizations (used by the tuner for fast games).
MAX_DETERMINIZATIONS = int(os.environ.get("PTCG_MAX_DETS", 24))
MAX_ROLLOUT_STEPS = 120
MAX_CANDIDATES = 16

# Search health: failures fall back per-decision; only a clearly systemic
# failure count disables search for the game. Stats are printed to stderr by
# main.py so Kaggle agent logs show whether search was alive.
_search_failures = 0
_search_successes = 0
_SEARCH_DISABLED_AFTER = 25


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
           rng: random.Random, safety: bool = True,
           weights: dict | None = None) -> list[int]:
    """Best selection for this observation, within the time budget."""
    global _search_failures, _search_successes

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

    # Evaluate the heuristic pick first: if the clock cuts the screening pass
    # short, the sampled scores still include the policy move.
    base = choose(obs, rng)
    if base in cands:
        cands.remove(base)
    cands.insert(0, base)

    me = obs.current.yourIndex
    stop_turn = _stop_turn(obs.current, me)
    totals = [0.0] * len(cands)
    counts = [0] * len(cands)
    wins = [0] * len(cands)
    losses = [0] * len(cands)

    active = list(range(len(cands)))
    try:
        for det_i in range(MAX_DETERMINIZATIONS):
            if time.time() > deadline and det_i > 0:
                break
            det = determinize.sample(obs, my_deck_list, rng)
            root = search_begin(obs, *det)
            try:
                for ci in active:
                    if time.time() > deadline and any(counts):
                        break
                    st = search_step(root.searchId, cands[ci])
                    st = _rollout(st, stop_turn, rng)
                    end = st.observation.current
                    totals[ci] += evaluate(end, me, weights)
                    counts[ci] += 1
                    if end.result == me:
                        wins[ci] += 1
                    elif end.result == 1 - me:
                        losses[ci] += 1
            finally:
                search_end()
            if det_i == 0 and len(active) > 6:
                # Screening pass done: keep only the promising candidates.
                scored_now = [i for i in active if counts[i] > 0]
                scored_now.sort(key=lambda i: totals[i] / counts[i], reverse=True)
                active = scored_now[:6]
    except Exception:
        _search_failures += 1
        if _search_failures <= 3 or _search_failures % 10 == 0:
            import sys
            import traceback
            print(f"[ptcg] search failure #{_search_failures}:",
                  file=sys.stderr)
            traceback.print_exc()
        return choose(obs, rng)

    _search_successes += 1

    # Only survivors of the screening pass compete: a pruned candidate's
    # single-sample score must not beat a multi-sample average on a fluke.
    pool = [i for i in active if counts[i] > 0]
    if not pool:
        pool = [i for i in range(len(cands)) if counts[i] > 0]
    if not pool:
        return choose(obs, rng)

    if safety:
        # Lethal: a candidate that won in every sampled world is taken, period.
        # (No loss-veto beyond this: the terminal +/-1M inside the averages
        # already prices loss risk; a hard veto turns late-game turns passive.)
        sure = [i for i in pool if wins[i] == counts[i] and counts[i] >= 2]
        if sure:
            return cands[max(sure, key=lambda i: totals[i] / counts[i])]

    avg = lambda i: totals[i] / counts[i]  # noqa: E731
    best = max(pool, key=avg)

    # Dominance floors: late-game determinizations are few (search_begin
    # replays the whole history), so noisy averages must not override
    # obviously-dominant lines (audited ladder losses did exactly this).
    def _opt_type(i):
        c = cands[i]
        return sel.option[c[0]].type if len(c) == 1 else None

    if sel.type == SelectType.MAIN and _opt_type(best) == OptionType.END:
        # Passing must beat attacking by a clear margin (~1 prize).
        attacks = [i for i in pool if _opt_type(i) == OptionType.ATTACK]
        if attacks:
            best_atk = max(attacks, key=avg)
            if avg(best) - avg(best_atk) < 1000.0:
                best = best_atk
    if sel.type == SelectType.YES_NO and sel.context == SelectContext.ACTIVATE:
        # Own-card effects are near-universally beneficial: declining needs
        # a clear margin (Punk Up declines threw audited games).
        yes = [i for i in pool if _opt_type(i) == OptionType.YES]
        if yes and _opt_type(best) == OptionType.NO \
                and avg(best) - avg(yes[0]) < 400.0:
            best = yes[0]

    return cands[best]
