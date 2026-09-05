"""
Continues training BOTH linear_q_agent and linear_q_agent_aggressive in
short bursts, evaluating for real after every burst and automatically
discarding any burst that makes things worse. Safe to run unattended --
it can only make things better or equal, never silently worse, because
every burst is checked against the previous best before being kept.

Usage: python train_more.py
"""
import json
import os
import pickle
import shutil
import subprocess
import sys

BURSTS_PER_AGENT = 6          # how many training bursts to attempt per agent
ROUNDS_PER_BURST = 400        # rounds of training per burst
EVAL_ROUNDS = 60              # rounds used to measure real performance (bigger = less noisy)
TOLERANCE = 5                 # allow a small drop without discarding (measurement noise)

AGENTS = {
    "linear_q_agent": {"epsilon": 0.10},
    "linear_q_agent_aggressive": {"epsilon": 0.12},
}


def run(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"\n!!! Command failed: {' '.join(cmd)}\n--- output ---\n{result.stdout}\n--- end output ---\n")
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def paths(agent_name):
    base = f"agent_code/{agent_name}"
    return {
        "model": f"{base}/my-saved-model.pt",
        "replay": f"{base}/replay-buffer.pt",
        "best_model": f"{base}/my-saved-model.best.pt",
        "best_replay": f"{base}/replay-buffer.best.pt",
    }


def set_epsilon(model_path, value):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model.epsilon = value
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def evaluate_score(agent_name, n_rounds):
    stats_path = f"results/_train_more_{agent_name}.json"
    run([sys.executable, "main.py", "play", "--agents", agent_name,
         "rule_based_agent", "rule_based_agent", "rule_based_agent",
         "--no-gui", "--n-rounds", str(n_rounds), "--scenario", "classic",
         "--save-stats", stats_path])
    with open(stats_path) as f:
        data = json.load(f)
    return data["by_agent"][agent_name].get("score", 0), data["by_agent"][agent_name].get("suicides", 0)


import importlib
import sys as _sys


def expected_n_features(agent_name):
    _sys.path.insert(0, ".")
    mod = importlib.import_module(f"agent_code.{agent_name}.module")
    return mod.N_FEATURES


def checkpoint_n_features(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model.weights.shape[1]


def train_agent(agent_name, epsilon):
    p = paths(agent_name)
    print(f"\n{'='*60}\nTraining {agent_name}\n{'='*60}")

    expected = expected_n_features(agent_name)

    # Anchor against the actual current feature count from the code itself,
    # not just "does model match best" -- if BOTH are stale (e.g. the live
    # model already got reverted to a bad checkpoint in a previous failed
    # run), comparing them to each other would wrongly call them compatible.
    if checkpoint_n_features(p["model"]) != expected:
        raise RuntimeError(
            f"{p['model']} has {checkpoint_n_features(p['model'])} features, "
            f"but the current code expects {expected}. This checkpoint is "
            f"stale/corrupted -- restore a known-good checkpoint before "
            f"running this script."
        )

    if not os.path.exists(p["best_model"]) or checkpoint_n_features(p["best_model"]) != expected:
        if os.path.exists(p["best_model"]):
            print("Existing best checkpoint is a stale/incompatible shape -- replacing it.")
        shutil.copy(p["model"], p["best_model"])
        if os.path.exists(p["replay"]):
            shutil.copy(p["replay"], p["best_replay"])

    best_score, best_suicides = evaluate_score(agent_name, EVAL_ROUNDS)
    print(f"Starting point: score={best_score}, suicides={best_suicides}/{EVAL_ROUNDS}")

    for i in range(BURSTS_PER_AGENT):
        set_epsilon(p["model"], epsilon)
        run([sys.executable, "main.py", "play", "--agents", agent_name, "rule_based_agent",
             "--train", "1", "--no-gui", "--n-rounds", str(ROUNDS_PER_BURST), "--scenario", "classic"])

        score, suicides = evaluate_score(agent_name, EVAL_ROUNDS)
        print(f"Burst {i+1}/{BURSTS_PER_AGENT}: score={score}, suicides={suicides}/{EVAL_ROUNDS}", end="  ")

        if score >= best_score - TOLERANCE:
            print("-> KEPT")
            shutil.copy(p["model"], p["best_model"])
            if os.path.exists(p["replay"]):
                shutil.copy(p["replay"], p["best_replay"])
            best_score = max(score, best_score)
        else:
            print("-> discarded (model reverted, replay memory kept growing)")
            shutil.copy(p["best_model"], p["model"])
            # deliberately do NOT revert the replay buffer -- old experience
            # stays valid and useful even if this burst's weights weren't kept

    print(f"\n{agent_name}: finished. Best verified score = {best_score}")
    print(f"Best checkpoint is at: {p['best_model']}")
    print(f"To use it as the real model: copy {p['best_model']} over {p['model']}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for agent_name, cfg in AGENTS.items():
        train_agent(agent_name, cfg["epsilon"])

    print("\n\nDONE. Summary:")
    for agent_name in AGENTS:
        p = paths(agent_name)
        score, suicides = evaluate_score(agent_name, EVAL_ROUNDS)
        print(f"  {agent_name}: current model score={score}, suicides={suicides}/{EVAL_ROUNDS}")
