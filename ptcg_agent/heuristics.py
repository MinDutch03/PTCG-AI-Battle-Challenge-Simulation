"""Rule-based policy: used as the search rollout policy and as the no-search
fallback. Must return a legal selection for every SelectType/SelectContext."""

import random

from cg.api import (
    AreaType,
    CardType,
    Observation,
    Option,
    OptionType,
    Pokemon,
    SelectContext,
    SelectType,
    SpecialConditionType,
    State,
)

from .cards import attack_db, card_db, stage


def _pokemon_at(state: State, player: int, area, index) -> Pokemon | None:
    try:
        ps = state.players[player]
        if area == AreaType.ACTIVE:
            return ps.active[index] if ps.active else None
        if area == AreaType.BENCH:
            return ps.bench[index]
    except (IndexError, TypeError):
        pass
    return None


def _card_value(card_id: int) -> float:
    """Rough desirability of holding/fetching a card."""
    c = card_db().get(card_id)
    if c is None:
        return 0.0
    if c.cardType == CardType.POKEMON:
        return 30.0 + c.hp * 0.1 + stage(card_id) * 15.0
    if c.cardType == CardType.SUPPORTER:
        return 30.0
    if c.cardType == CardType.ITEM:
        return 25.0
    if c.cardType == CardType.TOOL:
        return 18.0
    if c.cardType == CardType.STADIUM:
        return 15.0
    return 12.0  # energy


def _attack_damage(attack_id: int) -> int:
    a = attack_db().get(attack_id)
    return a.damage if a else 0


def _battle_value(p: Pokemon | None) -> float:
    """How good this Pokemon is as the fighter (for TO_ACTIVE/SWITCH picks)."""
    if p is None:
        return 0.0
    best = max((_attack_damage(a) for a in card_db()[p.id].attacks), default=0) \
        if p.id in card_db() else 0
    return len(p.energies) * 30.0 + p.hp * 0.3 + best * 0.4 + stage(p.id) * 10.0


def _option_pokemon_value(state: State, opt: Option) -> float:
    p = _pokemon_at(state, opt.playerIndex, opt.area, opt.index)
    return _battle_value(p)


def _remaining_hp(state: State, opt: Option) -> float:
    p = _pokemon_at(state, opt.playerIndex, opt.area, opt.index)
    return p.hp if p else 9999.0


def _main_priority(state: State, me: int, opt: Option) -> float:
    """Higher = do earlier. Attack near the end, END last."""
    t = opt.type
    if t == OptionType.EVOLVE:
        return 90.0
    if t == OptionType.PLAY:
        # A card in our hand.
        try:
            card = state.players[me].hand[opt.index]
            c = card_db().get(card.id)
        except (IndexError, TypeError):
            c = None
        if c is not None and c.cardType == CardType.POKEMON:
            return 85.0
        if c is not None and c.cardType == CardType.SUPPORTER:
            return 70.0
        if c is not None and c.cardType == CardType.ITEM:
            return 65.0
        if c is not None and c.cardType == CardType.STADIUM:
            return 40.0
        return 50.0
    if t == OptionType.ABILITY:
        return 75.0
    if t == OptionType.ATTACH:
        # Prefer attaching to the active.
        return 60.0 + (5.0 if opt.inPlayArea == AreaType.ACTIVE else 0.0)
    if t == OptionType.ATTACK:
        return 20.0 + _attack_damage(opt.attackId) * 0.01
    if t == OptionType.RETREAT:
        return 1.0
    if t == OptionType.DISCARD:
        return 2.0
    if t == OptionType.END:
        return 0.0
    return 30.0


def choose(obs: Observation, rng: random.Random | None = None) -> list[int]:
    """Pick option indices for any selection. Always legal."""
    rng = rng or random
    sel = obs.select
    state = obs.current
    me = state.yourIndex
    opts = sel.option
    n = len(opts)
    lo, hi = sel.minCount, sel.maxCount

    def top(scored: list[float], count: int, reverse: bool = True) -> list[int]:
        order = sorted(range(n), key=lambda i: scored[i], reverse=reverse)
        return order[:count]

    if sel.type == SelectType.MAIN:
        # Anti-loop guard: some abilities (energy shuffling etc.) stay legal
        # forever; after many actions this turn, wrap up with attack/end.
        if state.turnActionCount > 20:
            wrap = [i for i, o in enumerate(opts)
                    if o.type in (OptionType.ATTACK, OptionType.END)]
            if wrap:
                atk = [i for i in wrap if opts[i].type == OptionType.ATTACK]
                if atk:
                    return [max(atk, key=lambda i: _attack_damage(opts[i].attackId or 0))]
                return [wrap[0]]
        scores = [_main_priority(state, me, o) for o in opts]
        return [max(range(n), key=lambda i: scores[i])]

    if sel.type == SelectType.ATTACK:
        if sel.context == SelectContext.DISABLE_ATTACK:
            # Disable the opponent's biggest attack.
            return top([_attack_damage(o.attackId or 0) for o in opts], max(lo, 1))
        return top([_attack_damage(o.attackId or 0) for o in opts], max(lo, 1))

    if sel.type == SelectType.YES_NO:
        yes = next((i for i, o in enumerate(opts) if o.type == OptionType.YES), 0)
        no = next((i for i, o in enumerate(opts) if o.type == OptionType.NO), 0)
        if sel.context == SelectContext.MORE_DEVOLVE:
            return [no]
        return [yes]  # activate effects, go first, choose heads, draw on mulligan

    if sel.type == SelectType.COUNT:
        # Draw as much as possible; place as many damage counters as possible.
        nums = [o.number or 0 for o in opts]
        return top(nums, max(lo, 1))

    if sel.type == SelectType.SPECIAL_CONDITION:
        pref = {
            SpecialConditionType.PARALYZE: 5,
            SpecialConditionType.SLEEP: 4,
            SpecialConditionType.CONFUSE: 3,
            SpecialConditionType.POISON: 2,
            SpecialConditionType.BURN: 1,
        }
        return top([pref.get(o.specialConditionType, 0) for o in opts], max(lo, 1))

    if sel.type == SelectType.EVOLVE:
        # Prefer evolving the active.
        return top(
            [10.0 if o.inPlayArea == AreaType.ACTIVE else 1.0 for o in opts],
            max(lo, 1),
        )

    ctx = sel.context
    count = max(lo, 1) if hi >= 1 else lo

    # Selections that target Pokemon in play.
    if ctx in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
               SelectContext.DAMAGE_COUNTER_ANY):
        # Hurt the opponent's Pokemon closest to a KO.
        return top([-_remaining_hp(state, o) if o.playerIndex != me else -99999
                    for o in opts], count)
    if ctx in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
        def healing(o):
            p = _pokemon_at(state, o.playerIndex, o.area, o.index)
            return (p.maxHp - p.hp) if p else 0
        return top([healing(o) for o in opts], count)
    if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE,
               SelectContext.SWITCH):
        if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
            # Options are cards in hand; judge by card quality.
            def setup_value(o):
                try:
                    card = state.players[me].hand[o.index]
                except (IndexError, TypeError):
                    return 0.0
                c = card_db().get(card.id)
                if c is None:
                    return 0.0
                best = max((_attack_damage(a) for a in c.attacks), default=0)
                return c.hp * 0.4 + best * 0.5
            return top([setup_value(o) for o in opts], count)

        def promote_value(o):
            # Choosing the OPPONENT's new active (Boss's Orders etc.): drag in
            # their weakest, most nearly-KO'd piece — the opposite of picking
            # our own best fighter.
            from .cards import prize_value

            if o.playerIndex is not None and o.playerIndex != me:
                p = _pokemon_at(state, o.playerIndex, o.area, o.index)
                if p is None:
                    return 0.0
                return 500.0 - p.hp + 40.0 * prize_value(p.id) \
                    - 10.0 * len(p.energies)
            v = _option_pokemon_value(state, o)
            # Never feed a multi-prize body that cannot fight back: an
            # energy-less ex promoted as a shield is 2-3 free prizes.
            p = _pokemon_at(state, o.playerIndex if o.playerIndex is not None
                            else me, o.area, o.index)
            if p is not None and not p.energies:
                v -= 90.0 * prize_value(p.id)
            return v

        return top([promote_value(o) for o in opts], count)
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        return list(range(min(hi, n)))  # bench everything
    if ctx in (SelectContext.DISCARD, SelectContext.TO_DECK,
               SelectContext.TO_DECK_BOTTOM, SelectContext.TO_PRIZE,
               SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
        # Give up the least valuable cards.
        def discard_value(o):
            cid = o.cardId
            if cid is None and o.area == AreaType.HAND and o.index is not None:
                try:
                    cid = state.players[o.playerIndex or me].hand[o.index].id
                except (IndexError, TypeError):
                    cid = None
            return _card_value(cid) if cid is not None else 20.0
        return top([discard_value(o) for o in opts], max(lo, min(count, hi)),
                   reverse=False)
    if ctx in (SelectContext.TO_HAND, SelectContext.TO_FIELD,
               SelectContext.TO_BENCH, SelectContext.LOOK,
               SelectContext.EFFECT_TARGET, SelectContext.NOT_MOVE,
               SelectContext.ATTACH_TO):
        def fetch_value(o):
            cid = o.cardId
            if cid is None and sel.deck is not None and o.area == AreaType.DECK \
                    and o.index is not None:
                try:
                    cid = sel.deck[o.index].id
                except (IndexError, TypeError):
                    cid = None
            if cid is None and o.area == AreaType.HAND and o.index is not None:
                try:
                    cid = state.players[o.playerIndex or me].hand[o.index].id
                except (IndexError, TypeError):
                    cid = None
            return _card_value(cid) if cid is not None else 20.0
        # Fetch/attach effects are free value: take the maximum allowed
        # (a Punk Up that attaches 2 of 5 energies threw an audited game).
        want = hi if ctx in (SelectContext.TO_HAND, SelectContext.TO_FIELD,
                             SelectContext.TO_BENCH, SelectContext.ATTACH_TO,
                             SelectContext.EFFECT_TARGET) else max(lo, 1)
        return top([fetch_value(o) for o in opts], max(lo, min(want, hi)))
    if ctx in (SelectContext.ATTACH_FROM, SelectContext.DETACH_FROM):
        return top([_option_pokemon_value(state, o) for o in opts], count)

    # Anything else (energy cost payment, skill order, coin choices...):
    # the first minCount options (or one if a choice is required).
    need = lo if lo > 0 else min(1, hi)
    return list(range(need))


def fallback(obs_dict: dict) -> list[int]:
    """Last-resort legal move from the raw dict, no card knowledge needed."""
    sel = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if not sel:
        return []
    need = sel["minCount"] if sel["minCount"] > 0 else min(1, sel["maxCount"])
    return list(range(need))
