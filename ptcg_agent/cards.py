"""Card metadata helpers built on the engine's card database."""

from functools import lru_cache

from cg.api import CardData, CardType, all_card_data


@lru_cache(maxsize=1)
def card_db() -> dict[int, CardData]:
    return {c.cardId: c for c in all_card_data()}


@lru_cache(maxsize=1)
def attack_db() -> dict[int, "object"]:
    from cg.api import all_attack

    return {a.attackId: a for a in all_attack()}


def is_pokemon(card_id: int) -> bool:
    c = card_db().get(card_id)
    return c is not None and c.cardType == CardType.POKEMON


def is_basic_pokemon(card_id: int) -> bool:
    c = card_db().get(card_id)
    return c is not None and c.cardType == CardType.POKEMON and c.basic


def is_energy(card_id: int) -> bool:
    c = card_db().get(card_id)
    return c is not None and c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)


def prize_value(card_id: int) -> int:
    """Prizes the opponent takes when this Pokemon is knocked out."""
    c = card_db().get(card_id)
    if c is None or c.cardType != CardType.POKEMON:
        return 0
    if c.megaEx:
        return 3
    if c.ex:
        return 2
    return 1


@lru_cache(maxsize=1)
def _evolvable_names() -> frozenset:
    return frozenset(c.evolvesFrom for c in card_db().values() if c.evolvesFrom)


def has_evolution(card_id: int) -> bool:
    c = card_db().get(card_id)
    return c is not None and c.name in _evolvable_names()


def stage(card_id: int) -> int:
    c = card_db().get(card_id)
    if c is None:
        return 0
    if c.stage2:
        return 2
    if c.stage1:
        return 1
    return 0
