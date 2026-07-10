"""Round-robin deck tournament under the same pilot (main.make_agent).

Usage: python tools/tournament.py -n 10 [--decks decks/a.csv decks/b.csv ...]
Prints a win-rate cross table and overall standings.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.run_match import play_game  # noqa: E402


def read_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(x) for x in f.read().split() if x.strip()][:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=10, help="games per pair")
    ap.add_argument("--decks", nargs="*", default=sorted(glob.glob("decks/*.csv")))
    ap.add_argument("--pilot", default="main")
    args = ap.parse_args()

    import importlib

    pilot = importlib.import_module(args.pilot)
    names = [os.path.splitext(os.path.basename(d))[0] for d in args.decks]
    decks = {n: read_deck(d) for n, d in zip(names, args.decks)}

    wins = {n: {m: 0 for m in names} for n in names}
    played = {n: {m: 0 for m in names} for n in names}

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for g in range(args.games):
                swap = g % 2 == 1
                p0, p1 = (b, a) if swap else (a, b)
                agents = (pilot.make_agent(decks[p0], seed=g),
                          pilot.make_agent(decks[p1], seed=1000 + g))
                result, turns, _ = play_game(*agents)
                played[a][b] += 1
                played[b][a] += 1
                if result in (0, 1):
                    winner = (p0, p1)[result]
                    if winner == a:
                        wins[a][b] += 1
                    else:
                        wins[b][a] += 1
            wa = wins[a][b]
            print(f"{a} vs {b}: {wa}-{played[a][b] - wa}", flush=True)

    col = max(len(n) for n in names) + 1
    print("\n" + " " * col + " ".join(f"{n[:7]:>7}" for n in names))
    for a in names:
        row = []
        for b in names:
            row.append("      -" if a == b else
                       f"{100.0 * wins[a][b] / max(played[a][b], 1):6.0f}%")
        print(f"{a:<{col}}" + " ".join(row))

    print("\nStandings (overall win rate):")
    total = {n: (sum(wins[n].values()), sum(played[n].values())) for n in names}
    for n, (w, p) in sorted(total.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        print(f"  {n:<12} {w}/{p} ({100.0 * w / max(p, 1):.0f}%)")


if __name__ == "__main__":
    main()
