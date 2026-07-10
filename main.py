"""PTCG AI Battle Challenge agent entry point (placeholder — random policy)."""

import os
import random


def read_deck_csv() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        return [int(line) for line in f.read().split("\n") if line.strip()][:60]


def agent(obs_dict: dict) -> list[int]:
    if obs_dict["select"] is None:
        return read_deck_csv()
    sel = obs_dict["select"]
    return random.sample(range(len(sel["option"])), sel["maxCount"])
