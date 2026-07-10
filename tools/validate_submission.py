"""Simulate Kaggle's validation episode: extract submission.tar.gz to a temp
dir and run a self-play game through kaggle_environments' cabt env, importing
main.py the way the Kaggle runner does (agent dir on sys.path, cwd elsewhere).

Run inside the cabt-kaggle image:
    docker run --platform linux/amd64 --rm -v "$PWD":/work -w /work cabt-kaggle \
        python tools/validate_submission.py
"""

import os
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    bundle = os.path.join(ROOT, "submission.tar.gz")
    if not os.path.exists(bundle):
        sys.exit("submission.tar.gz not found — run tools/package.sh first")

    agent_dir = tempfile.mkdtemp(prefix="agent_")
    with tarfile.open(bundle) as tf:
        tf.extractall(agent_dir)
    for required in ("main.py", "deck.csv"):
        if not os.path.exists(os.path.join(agent_dir, required)):
            sys.exit(f"{required} missing from archive root")

    os.chdir(tempfile.mkdtemp(prefix="cwd_"))  # cwd is NOT the agent dir on Kaggle
    sys.path.insert(0, agent_dir)

    import importlib

    main_mod = importlib.import_module("main")
    agent = main_mod.agent

    from kaggle_environments import make

    env = make("cabt", debug=True)
    env.run([agent, agent])
    statuses = [s.status for s in env.state]
    rewards = [s.reward for s in env.state]
    print("statuses:", statuses, "rewards:", rewards)
    if any(s not in ("DONE", "ACTIVE", "INACTIVE") for s in statuses):
        sys.exit("validation episode FAILED")
    print("validation episode OK")


if __name__ == "__main__":
    main()
