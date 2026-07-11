"""Opponent deck inference for determinization.

We bundle the known meta deck lists and match the opponent's visible cards
(board, discard, attachments, stadium) against each list. If one list
explains what we have seen, the opponent's hidden zones are dealt from its
remainder, which makes simulated reply turns far more realistic than energy
filler (supporters get played, energy gets attached, gust effects happen).
"""

import os
from collections import Counter

from cg.api import State

_HERE = os.path.dirname(os.path.abspath(__file__))

# (name, {card_id: copies}) for every bundled meta list.
_META: list[tuple[str, Counter]] | None = None


def _load_meta() -> list[tuple[str, Counter]]:
    global _META
    if _META is None:
        _META = []
        deck_dir = os.path.join(_HERE, "meta_decks")
        if os.path.isdir(deck_dir):
            for fn in sorted(os.listdir(deck_dir)):
                # skip hidden/AppleDouble junk; never let one bad file
                # empty the whole library
                if not fn.endswith(".csv") or fn.startswith("."):
                    continue
                try:
                    with open(os.path.join(deck_dir, fn)) as f:
                        ids = [int(x) for x in f.read().split() if x.strip()]
                except (OSError, ValueError, UnicodeDecodeError):
                    continue
                if len(ids) == 60:
                    _META.append((fn[:-4], Counter(ids)))
    return _META


def visible_ids(state: State, player: int) -> Counter:
    """Every opponent card whose identity we know."""
    ps = state.players[player]
    seen: list[int] = []
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
        if c.playerIndex == player:
            seen.append(c.id)
    if state.looking:
        for c in state.looking:
            if c is not None and c.playerIndex == player:
                seen.append(c.id)
    return Counter(seen)


def infer_deck(state: State, opponent: int) -> list[int] | None:
    """Best-matching meta list's unseen remainder, or None if nothing fits.

    Returns the multiset of the opponent's *hidden* cards (60 minus visible)
    when a bundled list covers at least 70% of what we have seen.
    """
    seen = visible_ids(state, opponent)
    total_seen = sum(seen.values())
    if total_seen == 0:
        return None

    best_name, best_cover, best_rest = None, 0.0, None
    for name, deck in _load_meta():
        covered = sum(min(k, deck.get(cid, 0)) for cid, k in seen.items())
        cover = covered / total_seen
        if cover > best_cover:
            rest = deck.copy()
            rest.subtract(seen)
            best_name, best_cover = name, cover
            best_rest = [cid for cid, k in rest.items() for _ in range(max(k, 0))]
    if best_cover >= 0.6 and best_rest:
        return best_rest
    return None
