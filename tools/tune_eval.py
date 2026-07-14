"""Self-play tuning of the evaluation weights ((1+1)-ES).

Each iteration perturbs the incumbent weight vector, plays a batch of games
(candidate pilot vs incumbent pilot) across a mix of ladder-relevant
matchups, and adopts the candidate only if it clearly wins. The champion is
persisted to ptcg_agent/weights.json, which evaluate.py loads by default —
so a long overnight run directly improves the shipped agent.

Run inside the linux container:
    docker run -d --rm --platform linux/amd64 -e PTCG_TIME_SCALE=0.06 \
        -v "$PWD":/work -w /work --name tuner cabt-dev \
        python tools/tune_eval.py --hours 8 --games 30
Progress: tail tune_log.jsonl
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptcg_agent.evaluate import DEFAULT_WEIGHTS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(ROOT, "ptcg_agent", "weights.json")
LOG_PATH = os.path.join(ROOT, "tune_log.jsonl")

# (our deck, opponent deck, share of games) — the observed ladder mix.
MATCHUPS = [
    ("decks/grimmsnarl.csv", "decks/grimmsnarl.csv", 0.30),           # mirror
    ("decks/grimmsnarl.csv", "decks/mega_lucario.csv", 0.25),
    ("decks/grimmsnarl.csv", "ptcg_agent/meta_decks/trevenant_v4.csv", 0.25),  # alakazam
    ("decks/grimmsnarl.csv", "decks/abomasnow.csv", 0.10),
    ("decks/grimmsnarl.csv", "decks/dragapult.csv", 0.10),
]


def read_deck(path):
    with open(os.path.join(ROOT, path)) as f:
        return [int(x) for x in f.read().split() if x.strip()][:60]


def load_incumbent():
    w = dict(DEFAULT_WEIGHTS)
    try:
        with open(WEIGHTS_PATH) as f:
            w.update({k: float(v) for k, v in json.load(f).items()
                      if k in DEFAULT_WEIGHTS})
    except (OSError, ValueError):
        pass
    return w


def perturb(w, rng, sigma):
    """Multiplicative log-normal noise on a random subset of weights."""
    out = dict(w)
    keys = rng.sample(sorted(w), k=max(2, len(w) // 3))
    for k in keys:
        out[k] = round(w[k] * (2.0 ** rng.gauss(0.0, sigma)), 4)
    return out


def play_batch(cand, inc, games, rng):
    """Candidate-weights pilot vs incumbent-weights pilot. Returns wins."""
    import main as M
    from tools.run_match import play_game

    wins = 0
    plan = []
    for my_deck, opp_deck, share in MATCHUPS:
        plan += [(my_deck, opp_deck)] * max(1, round(games * share))
    plan = plan[:games]
    for g, (my_deck, opp_deck) in enumerate(plan):
        a = M.make_agent(read_deck(my_deck), seed=rng.randrange(1 << 30),
                         weights=cand)
        b = M.make_agent(read_deck(opp_deck), seed=rng.randrange(1 << 30),
                         weights=inc)
        swap = g % 2 == 1
        p0, p1 = (b, a) if swap else (a, b)
        result, _, _ = play_game(p0, p1)
        if result in (0, 1) and ((result == 1) if swap else (result == 0)):
            wins += 1  # the candidate-weights seat won
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--games", type=int, default=30, help="games per iteration")
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--adopt", type=float, default=0.60,
                    help="candidate win rate needed for adoption")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed or int(time.time()))
    incumbent = load_incumbent()
    deadline = time.time() + args.hours * 3600
    it = adopted = 0

    while time.time() < deadline:
        it += 1
        cand = perturb(incumbent, rng, args.sigma)
        t0 = time.time()
        wins = play_batch(cand, incumbent, args.games, rng)
        rate = wins / args.games
        take = rate >= args.adopt
        if take:
            adopted += 1
            incumbent = cand
            with open(WEIGHTS_PATH, "w") as f:
                json.dump(incumbent, f, indent=1, sort_keys=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({
                "iter": it, "winrate": rate, "adopted": take,
                "secs": round(time.time() - t0),
                "weights": cand if take else None,
            }) + "\n")
        print(f"iter {it}: cand {wins}/{args.games} "
              f"({'ADOPTED' if take else 'rejected'}) "
              f"[{time.time() - t0:.0f}s]", flush=True)

    print(f"done: {it} iterations, {adopted} adoptions")


if __name__ == "__main__":
    main()
