"""Local match runner for the cabt engine.

Drives games directly through cg.game (battle_start / battle_select), mirroring
the kaggle-environments "cabt" interpreter but without its overhead. Runs N
games between two agent modules, alternating seats, and reports win rates.

Usage (inside the linux container, repo root on PYTHONPATH):
    python tools/run_match.py --p0 main --p1 baselines.random_agent -n 20
Agent modules must expose `agent(obs_dict) -> list[int]`.
"""

import argparse
import importlib
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_agent(spec: str):
    mod = importlib.import_module(spec)
    return getattr(mod, "agent")


def empty_obs() -> dict:
    return {"select": None, "logs": [], "current": None, "search_begin_input": None}


def play_game(agent0, agent1, verbose=False, log_file=None):
    """Play one game. Returns (result, turns, times) where result is
    0/1 winner index, 2 draw, or -0.5/-1.5 meaning agent 0/1 crashed."""
    from cg.game import battle_start, battle_select, battle_finish
    from cg.sim import Battle

    agents = [agent0, agent1]
    times = [0.0, 0.0]

    decks = []
    for i in (0, 1):
        t = time.time()
        decks.append(agents[i](empty_obs()))
        times[i] += time.time() - t

    obs, start_data = battle_start(decks[0], decks[1])
    if obs is None:
        raise RuntimeError(f"deck error, player {start_data.errorPlayer}")

    turns = 0
    try:
        while True:
            s = obs["current"]
            if s["result"] >= 0:
                return s["result"], s["turn"], times
            idx = s["yourIndex"]
            turns = s["turn"]
            t = time.time()
            try:
                action = agents[idx](obs)
            except Exception:
                traceback.print_exc()
                return -0.5 - idx, turns, times
            times[idx] += time.time() - t
            if log_file is not None:
                log_file.write(json.dumps({"obs": {k: obs[k] for k in ("select", "current")},
                                           "player": idx, "action": action}) + "\n")
            try:
                obs = battle_select(action)
            except Exception:
                traceback.print_exc()
                print(f"illegal action by player {idx}: {action}", file=sys.stderr)
                return -0.5 - idx, turns, times
    finally:
        battle_finish()
        Battle.battle_ptr = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", required=True, help="agent module for seat 0")
    ap.add_argument("--p1", required=True, help="agent module for seat 1")
    ap.add_argument("-n", "--games", type=int, default=10)
    ap.add_argument("--no-swap", action="store_true", help="do not alternate seats")
    ap.add_argument("--log", help="write per-move JSONL log of the first game")
    args = ap.parse_args()

    a = load_agent(args.p0)
    b = load_agent(args.p1)

    # score from the perspective of --p0
    wins = losses = draws = crashes = 0
    total_turns = 0
    t_start = time.time()
    times = [0.0, 0.0]
    for g in range(args.games):
        swap = (not args.no_swap) and (g % 2 == 1)
        p0, p1 = (b, a) if swap else (a, b)
        log_file = open(args.log, "w") if (args.log and g == 0) else None
        try:
            result, turns, game_times = play_game(p0, p1, log_file=log_file)
        finally:
            if log_file:
                log_file.close()
        total_turns += turns
        if swap:
            game_times.reverse()
        times[0] += game_times[0]
        times[1] += game_times[1]
        if result < 0:
            crashed_seat = int(-result - 0.5)
            crashed_a = (crashed_seat == 1) if swap else (crashed_seat == 0)
            crashes += 1
            if crashed_a:
                losses += 1
            else:
                wins += 1
            outcome = "CRASH"
        elif result == 2:
            draws += 1
            outcome = "draw"
        else:
            a_won = (result == 1) if swap else (result == 0)
            if a_won:
                wins += 1
                outcome = "win"
            else:
                losses += 1
                outcome = "loss"
        print(f"game {g + 1}/{args.games}: {outcome} (turns={turns})", flush=True)

    n = args.games
    elapsed = time.time() - t_start
    print(f"\n{args.p0} vs {args.p1}: {wins}W-{losses}L-{draws}D "
          f"({100.0 * wins / n:.1f}% win) crashes={crashes}")
    print(f"avg turns {total_turns / n:.1f}, wall {elapsed:.1f}s, "
          f"agent time p0={times[0]:.1f}s p1={times[1]:.1f}s "
          f"({times[0] / n:.2f}s/game vs {times[1] / n:.2f}s/game)")


if __name__ == "__main__":
    main()
