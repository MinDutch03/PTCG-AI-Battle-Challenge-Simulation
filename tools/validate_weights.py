"""Full-depth gate for tuned evaluation weights: champion (weights.json)
vs DEFAULT_WEIGHTS at real search depth across the ladder matchup mix.

    docker run --rm --platform linux/amd64 -e PTCG_TIME_SCALE=0.1 \
        -v "$PWD":/work -w /work cabt-dev python tools/validate_weights.py -n 48
"""

import argparse
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptcg_agent.evaluate import DEFAULT_WEIGHTS  # noqa: E402
from tools.tune_eval import load_incumbent, play_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=48)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    champion = load_incumbent()
    if champion == dict(DEFAULT_WEIGHTS):
        sys.exit("no tuned weights.json found — nothing to validate")

    pool = mp.get_context("fork").Pool(args.workers)
    rng = random.Random(123)
    wins = play_batch(pool, champion, dict(DEFAULT_WEIGHTS), args.games, rng)
    rate = wins / args.games
    print(f"champion vs default at full depth: {wins}/{args.games} "
          f"({rate * 100:.1f}%)")
    print("GATE PASSED" if rate >= 0.55 else "GATE FAILED — do not ship")


if __name__ == "__main__":
    main()
