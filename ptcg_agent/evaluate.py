"""Board evaluation from one player's perspective.

Operates on the dataclass `State` returned by the search API. Scale guide:
one prize card ~ 1000, so a win (+inf-ish) > prizes > damage > development.

All weights live in DEFAULT_WEIGHTS and can be overridden per agent (the
self-play tuner in tools/tune_eval.py optimizes them; tuned values ship in
ptcg_agent/weights.json and are loaded automatically).
"""

import json
import os
from functools import lru_cache

from cg.api import Pokemon, State

from .cards import attack_db, card_db, prize_value, stage

WIN = 1_000_000

DEFAULT_WEIGHTS = {
    "prize": 1000.0,        # per prize-count difference
    "dmg_frac": 0.55,       # damage as fraction of prize equity (quadratic)
    "body": 40.0,           # having a Pokemon in play
    "hp": 0.35,             # per point of printed HP
    "stage": 35.0,          # per evolution stage
    "energy": 28.0,         # per useful attached energy
    "energy_excess": 4.0,   # per energy beyond the priciest attack cost
    "tool": 10.0,           # per attached tool
    "naked_ex": 55.0,       # penalty per extra prize on a 0-energy body
    "no_bench": 250.0,      # empty bench (one KO from losing)
    "hand": 9.0,            # per card in own hand
    "opp_hand": 6.0,        # per card in opponent hand
    "deckout": 300.0,       # per card under 3 left in deck
    "cond_hard": 60.0,      # paralysis/sleep on the opposing active
    "cond_soft": 35.0,      # poison/burn/confusion
}


def _load_tuned() -> dict:
    w = dict(DEFAULT_WEIGHTS)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "weights.json")
    try:
        with open(path) as f:
            w.update({k: float(v) for k, v in json.load(f).items()
                      if k in DEFAULT_WEIGHTS})
    except (OSError, ValueError):
        pass
    return w


WEIGHTS = _load_tuned()


@lru_cache(maxsize=2048)
def _useful_energy_cap(card_id: int) -> int:
    """Energy beyond the priciest attack cost is idle risk, not value:
    a KO takes every attached card with it (audited losses stacked 6 on one
    attacker and lost the game to energy bankruptcy)."""
    c = card_db().get(card_id)
    if c is None or not c.attacks:
        return 1
    adb = attack_db()
    return max((len(adb[a].energies) for a in c.attacks if a in adb), default=1)


def _pokemon_score(p: Pokemon, w: dict) -> float:
    """Value of having this Pokemon in play, ignoring damage."""
    c = card_db().get(p.id)
    s = w["body"]
    if c is not None:
        s += c.hp * w["hp"]
        s += stage(p.id) * w["stage"]
    cap = _useful_energy_cap(p.id)
    n = len(p.energies)
    s += min(n, cap) * w["energy"] + max(n - cap, 0) * w["energy_excess"]
    s += len(p.tools) * w["tool"]
    # An undeveloped multi-prize body is a gift target for gust effects.
    if n == 0:
        s -= w["naked_ex"] * (prize_value(p.id) - 1)
    return s


def _damage_score(p: Pokemon, w: dict) -> float:
    """How much of this Pokemon's prize value the damage represents."""
    if p.maxHp <= 0:
        return 0.0
    frac = (p.maxHp - p.hp) / p.maxHp
    # Damage is worth a growing fraction of the KO prize value; nearly-dead
    # Pokemon are almost-taken prizes.
    return frac * frac * prize_value(p.id) * w["prize"] * w["dmg_frac"]


def evaluate(state: State, me: int, w: dict | None = None) -> float:
    if state.result >= 0:
        if state.result == 2:
            return 0.0
        return WIN if state.result == me else -WIN

    w = w or WEIGHTS
    opp = 1 - me
    mine = state.players[me]
    theirs = state.players[opp]

    s = 0.0
    # Prizes: my pile shrinking means I am winning.
    s += w["prize"] * (len(theirs.prize) - len(mine.prize))

    # Damage in play (worth a fraction of prizes).
    for p in [x for x in theirs.active if x] + list(theirs.bench):
        s += _damage_score(p, w)
    for p in [x for x in mine.active if x] + list(mine.bench):
        s -= _damage_score(p, w)

    # Board development.
    for p in [x for x in mine.active if x] + list(mine.bench):
        s += _pokemon_score(p, w)
    for p in [x for x in theirs.active if x] + list(theirs.bench):
        s -= _pokemon_score(p, w)

    # Having no bench is very risky (single KO loses the game).
    if len(mine.bench) == 0:
        s -= w["no_bench"]
    if len(theirs.bench) == 0:
        s += w["no_bench"]

    # Hand and deck resources.
    s += w["hand"] * mine.handCount - w["opp_hand"] * theirs.handCount
    if mine.deckCount <= 2:
        s -= (3 - mine.deckCount) * w["deckout"]
    if theirs.deckCount <= 2:
        s += (3 - theirs.deckCount) * w["deckout"]

    # Special conditions on the actives.
    s += w["cond_hard"] * (theirs.paralyzed + theirs.asleep) \
        + w["cond_soft"] * (theirs.poisoned + theirs.burned + theirs.confused)
    s -= w["cond_hard"] * (mine.paralyzed + mine.asleep) \
        + w["cond_soft"] * (mine.poisoned + mine.burned + mine.confused)

    return s
