"""Uniform-random baseline: picks maxCount random legal options."""

import os
import random


def read_deck() -> list[int]:
    path = os.path.join(os.path.dirname(__file__), "..", "deck.csv")
    with open(path) as f:
        return [int(line) for line in f.read().split("\n") if line.strip()][:60]


def agent(obs: dict) -> list[int]:
    if obs["select"] is None:
        return read_deck()
    return random.sample(range(len(obs["select"]["option"])), obs["select"]["maxCount"])
