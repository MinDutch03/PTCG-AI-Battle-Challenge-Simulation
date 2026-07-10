"""Sampling of hidden information for the engine's search API.

We know our own 60-card list, so our deck + face-down prizes are the list
minus every card of ours we can see. The opponent's hidden zones are filled
with inert placeholders (basic energy + one basic Pokemon where required):
we only roll out to the end of the current turn, so the opponent's exact
cards rarely matter — their board state (which we can see) does.
"""

import random
from collections import Counter

from cg.api import Observation, State

FILLER_UNKNOWN = 1  # Basic Grass Energy — inert placeholder for unknown cards
FILLER_BASIC = 1072  # Snorlax — community-standard basic Pokemon placeholder
_PAD = 64  # native SearchBegin reads array lengths from game state; padding
           # with valid IDs guards against short-array reads (segfault risk)


def _my_visible_ids(state: State, me: int) -> list[int]:
    ps = state.players[me]
    seen: list[int] = []
    if ps.hand:
        seen += [c.id for c in ps.hand]
    seen += [c.id for c in ps.discard]
    for p in list(ps.active) + list(ps.bench):
        if p is None:
            continue
        seen.append(p.id)
        seen += [c.id for c in p.energyCards]
        seen += [c.id for c in p.tools]
        seen += [c.id for c in p.preEvolution]
    for c in ps.prize:
        if c is not None:
            seen.append(c.id)
    for c in state.stadium:
        if c.playerIndex == me:
            seen.append(c.id)
    if state.looking:
        for c in state.looking:
            if c is not None and c.playerIndex == me:
                seen.append(c.id)
    return seen


def sample(obs: Observation, my_deck_list: list[int], rng: random.Random):
    """Build the seven search_begin prediction arguments."""
    state = obs.current
    me = state.yourIndex
    opp = 1 - me
    my_ps = state.players[me]
    opp_ps = state.players[opp]

    pool = Counter(my_deck_list)
    pool.subtract(Counter(_my_visible_ids(state, me)))
    unknown = [cid for cid, k in pool.items() for _ in range(max(k, 0))]
    rng.shuffle(unknown)

    my_prize: list[int] = []
    for c in my_ps.prize:
        if c is not None:
            my_prize.append(c.id)
        else:
            my_prize.append(unknown.pop() if unknown else FILLER_UNKNOWN)

    my_deck = unknown  # whatever is left is the deck (search checks len >= deckCount)
    while len(my_deck) < max(my_ps.deckCount, _PAD):
        my_deck.append(FILLER_UNKNOWN)

    opp_hand = [FILLER_UNKNOWN] * max(opp_ps.handCount, _PAD)
    if state.turn <= 0 or not opp_ps.active:
        opp_hand[0] = FILLER_BASIC  # setup: opponent must be able to place an active
    opp_prize = [c.id if c is not None else FILLER_UNKNOWN for c in opp_ps.prize]
    opp_prize += [FILLER_UNKNOWN] * (_PAD - len(opp_prize))
    opp_deck = [FILLER_UNKNOWN] * max(opp_ps.deckCount, _PAD)
    opp_deck[0] = FILLER_BASIC  # setup requires a basic Pokemon in the deck

    opp_active = [FILLER_BASIC] * 4
    my_prize_padded = my_prize + [FILLER_UNKNOWN] * (_PAD - len(my_prize))

    return my_deck, my_prize_padded, opp_deck, opp_prize, opp_hand, opp_active
