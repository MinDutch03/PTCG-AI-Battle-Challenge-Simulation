"""Board evaluation from one player's perspective.

Operates on the dataclass `State` returned by the search API. Scale guide:
one prize card ~ 1000, so a win (+inf-ish) > prizes > damage > development.
"""

from cg.api import Pokemon, State

from .cards import card_db, prize_value, stage

WIN = 1_000_000
PRIZE = 1000.0


def _pokemon_score(p: Pokemon, on_active: bool) -> float:
    """Value of having this Pokemon in play, ignoring damage."""
    c = card_db().get(p.id)
    s = 40.0  # a body on the board
    if c is not None:
        s += c.hp * 0.35
        s += stage(p.id) * 35.0
    s += min(len(p.energies), 4) * 28.0
    s += len(p.tools) * 10.0
    return s


def _damage_score(p: Pokemon) -> float:
    """How much of this Pokemon's prize value the damage represents."""
    if p.maxHp <= 0:
        return 0.0
    frac = (p.maxHp - p.hp) / p.maxHp
    # Damage is worth a growing fraction of the KO prize value; nearly-dead
    # Pokemon are almost-taken prizes.
    return frac * frac * prize_value(p.id) * PRIZE * 0.55


def evaluate(state: State, me: int) -> float:
    if state.result >= 0:
        if state.result == 2:
            return 0.0
        return WIN if state.result == me else -WIN

    opp = 1 - me
    mine = state.players[me]
    theirs = state.players[opp]

    s = 0.0
    # Prizes: my pile shrinking means I am winning.
    s += PRIZE * (len(theirs.prize) - len(mine.prize))

    # Damage in play (worth a fraction of prizes).
    for p in [x for x in theirs.active if x] + list(theirs.bench):
        s += _damage_score(p)
    for p in [x for x in mine.active if x] + list(mine.bench):
        s -= _damage_score(p)

    # Board development.
    for p in [x for x in mine.active if x]:
        s += _pokemon_score(p, True)
    for p in mine.bench:
        s += _pokemon_score(p, False)
    for p in [x for x in theirs.active if x]:
        s -= _pokemon_score(p, True)
    for p in theirs.bench:
        s -= _pokemon_score(p, False)

    # Having no bench is very risky (single KO loses the game).
    if len(mine.bench) == 0:
        s -= 250.0
    if len(theirs.bench) == 0:
        s += 250.0

    # Hand and deck resources.
    s += 9.0 * mine.handCount - 6.0 * theirs.handCount
    if mine.deckCount <= 2:
        s -= (3 - mine.deckCount) * 300.0  # deck-out looms
    if theirs.deckCount <= 2:
        s += (3 - theirs.deckCount) * 300.0

    # Special conditions on the actives.
    s += 60.0 * (theirs.paralyzed + theirs.asleep) + 35.0 * (
        theirs.poisoned + theirs.burned + theirs.confused
    )
    s -= 60.0 * (mine.paralyzed + mine.asleep) + 35.0 * (
        mine.poisoned + mine.burned + mine.confused
    )

    return s
