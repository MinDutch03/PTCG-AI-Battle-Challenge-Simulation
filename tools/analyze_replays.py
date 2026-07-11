"""Summarize downloaded Kaggle episode replays: how did each game end,
what deck did the opponent play, would our archetype inference have matched,
and how much clock did we use.

Usage: python tools/analyze_replays.py ladder/replay_*.json
"""

import glob
import json
import os
import sys
from collections import Counter

MY_TEAM = "Đức Nguyễn Minh"

RESULT_REASON = {1: "prizes", 2: "deck-out", 3: "no-active", 4: "card-effect"}


def load_meta():
    metas = {}
    for fn in sorted(glob.glob("ptcg_agent/meta_decks/*.csv")):
        ids = [int(x) for x in open(fn).read().split() if x.strip()]
        metas[os.path.basename(fn)[:-4]] = Counter(ids)
    return metas


def classify(deck: list[int], metas) -> tuple[str, float]:
    seen = Counter(deck)
    best, cover = "unknown", 0.0
    for name, m in metas.items():
        c = sum(min(k, m.get(cid, 0)) for cid, k in seen.items()) / 60.0
        if c > cover:
            best, cover = name, c
    return best, cover


def main():
    metas = load_meta()
    files = sys.argv[1:] or sorted(glob.glob("ladder/replay_*.json"))
    rows = []
    for f in files:
        r = json.load(open(f))
        teams = r["info"]["TeamNames"]
        me = teams.index(MY_TEAM) if MY_TEAM in teams else 0
        opp = 1 - me
        steps = r["steps"]
        decks = [steps[1][i].get("action") or [] for i in (0, 1)]
        arch, cover = classify(decks[opp], metas)
        my_arch, _ = classify(decks[me], metas)
        reward = r["rewards"][me]
        res = {1: "W", -1: "L", 0: "D", None: "?"}.get(reward, "?")
        statuses = r["statuses"]

        # final board state + result reason from the last meaningful obs
        last_obs = None
        for s in reversed(steps):
            for a in s:
                if a.get("observation", {}).get("current"):
                    last_obs = a["observation"]
                    break
            if last_obs:
                break
        turn = prizes_me = prizes_opp = None
        reason = ""
        if last_obs:
            cur = last_obs["current"]
            turn = cur["turn"]
            prizes_me = len(cur["players"][me]["prize"])
            prizes_opp = len(cur["players"][opp]["prize"])
            for lg in last_obs.get("logs", []):
                if lg.get("type") == 23:
                    reason = RESULT_REASON.get(lg.get("reason"), str(lg.get("reason")))

        # our remaining overage at the end (clock usage)
        overage = None
        for s in reversed(steps):
            o = s[me].get("observation", {})
            if o.get("remainingOverageTime") is not None:
                overage = o["remainingOverageTime"]
                break

        ep = os.path.basename(f).split("_")[1].split(".")[0]
        rows.append((res, ep, arch, round(cover, 2), reason, turn,
                     prizes_me, prizes_opp, round(600 - (overage or 600)),
                     statuses[me], my_arch))

    rows.sort()
    print(f"{'R':1} {'episode':9} {'opp deck':14} {'cov':4} {'end':11} "
          f"{'turn':4} {'myPz':4} {'opPz':4} {'used_s':6} {'status':8}")
    for row in rows:
        print(f"{row[0]:1} {row[1]:9} {row[2]:14} {row[3]:<4} {row[4]:11} "
              f"{str(row[5]):4} {str(row[6]):4} {str(row[7]):4} {row[8]:<6} {row[9]:8}")

    print("\nOpponent archetypes in LOSSES:")
    for arch, k in Counter(r[2] for r in rows if r[0] == "L").most_common():
        print(f"  {arch}: {k}")
    print("Loss end reasons:", dict(Counter(r[4] for r in rows if r[0] == "L")))


if __name__ == "__main__":
    main()
