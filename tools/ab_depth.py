"""A/B: does raising the determinization ceiling win games?

Deep agent (max_dets=64) vs current agent (max_dets=24), same deck mix as
the tuner, parallel workers.

    docker run --rm --platform linux/amd64 -e PTCG_TIME_SCALE=0.3 \
        -v "$PWD":/work -w /work cabt-dev python tools/ab_depth.py -n 24
"""

import argparse
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.tune_eval import MATCHUPS, read_deck  # noqa: E402


def _play_one(spec):
    my_deck, opp_deck, swap, seed, deep_dets, base_dets = spec
    import main as M
    from tools.run_match import play_game

    a = M.make_agent(my_deck, seed=seed, max_dets=deep_dets)
    b = M.make_agent(opp_deck, seed=seed + 1, max_dets=base_dets)
    p0, p1 = (b, a) if swap else (a, b)
    result, _, _ = play_game(p0, p1)
    return result in (0, 1) and ((result == 1) if swap else (result == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=24)
    ap.add_argument("--deep", type=int, default=64)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rng = random.Random(7)
    plan = []
    for my_deck, opp_deck, share in MATCHUPS:
        plan += [(my_deck, opp_deck)] * max(1, round(args.games * share))
    plan = plan[:args.games]
    specs = [(read_deck(md), read_deck(od), g % 2 == 1,
              rng.randrange(1 << 30), args.deep, args.base)
             for g, (md, od) in enumerate(plan)]
    pool = mp.get_context("fork").Pool(args.workers)
    wins = sum(pool.map(_play_one, specs))
    print(f"deep({args.deep}) vs base({args.base}): {wins}/{args.games} "
          f"({100 * wins / args.games:.1f}%)")


if __name__ == "__main__":
    main()
