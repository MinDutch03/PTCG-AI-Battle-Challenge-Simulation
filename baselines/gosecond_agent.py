"""A/B baseline: full search pilot, but chooses to go SECOND at IS_FIRST."""

import main as _main

IS_FIRST = 41  # SelectContext.IS_FIRST
NO = 2  # OptionType.NO


def make_agent(deck=None, seed=0):
    inner = _main.make_agent(deck, seed=seed)

    def agent(obs_dict: dict) -> list[int]:
        sel = obs_dict.get("select")
        if sel and sel.get("context") == IS_FIRST:
            for i, o in enumerate(sel["option"]):
                if o.get("type") == NO:
                    return [i]
        return inner(obs_dict)

    return agent


agent = make_agent()
