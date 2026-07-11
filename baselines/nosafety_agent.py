"""A/B baseline: the full search pilot with the lethal/loss-veto layer off."""

import main as _main


def make_agent(deck=None, seed=0):
    return _main.make_agent(deck, seed=seed, safety=False)


agent = make_agent()
