"""
Introduces rule_based_agent into training, using the same safety-net
principle used throughout this whole project: short bursts, a checkpoint
backup before each one, a REAL evaluation (no training, actual tournament
score) after each one, and automatic discard of any burst that regresses.

Critically, this evaluates on SCORE (coins + 5*kills), not training
reward -- we've already seen once in this project that a shaped training
reward climbing does not necessarily mean real score is improving.

Do not run this until solo classic performance clears the threshold
noted alongside it: survive-and-bomb success >= 30% and death rate <= 25%,
confirmed on two separate 100-round checks. Introducing an active
adversary before solo skill is solid tends to destabilize training rather
than accelerate it.

Usage: python train_vs_opponent.py
"""
import json
import os
import shutil
import subprocess
import sys

AGENT = "ppo_agent_balanced"
BURSTS = 2
ROUNDS_PER_BURST = 300       # short bursts -- PPO training against an
                             # active adversary has proven fragile
                             # throughout this project; frequent checks
                             # catch a regression early rather than after
                             # hours of wasted training
EVAL_ROUNDS = 60            # larger sample for a less noisy score read
TOLERANCE = 2                # allow a small score drop without discarding
                             # (measurement noise)


def run(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"\n!!! Command failed: {' '.join(cmd)}\n--- output ---\n{result.stdout}\n--- end output ---\n")
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def paths():
    base = f"agent_code/{AGENT}"
    return {
        "model": f"{base}/ppo-model.pt",
        "best_model": f"{base}/ppo-model.best_vs_opponent.pt",
    }


def evaluate_score(n_rounds):
    """Real score, no training -- against rule_based_agent, the actual
    tournament-representative opponent, not just solo."""
    stats_path = f"results/_train_vs_opponent_eval.json"
    run([sys.executable, "main.py", "play", "--agents", AGENT,
         "rule_based_agent", "rule_based_agent", "rule_based_agent",
         "--no-gui", "--n-rounds", str(n_rounds), "--scenario", "classic",
         "--save-stats", stats_path])
    with open(stats_path) as f:
        data = json.load(f)
    d = data["by_agent"][AGENT]
    coins = d.get("coins", 0)
    kills = d.get("kills", 0)
    suicides = d.get("suicides", 0)
    score = coins + 5 * kills
    return score, coins, kills, suicides


def main():
    os.makedirs("results", exist_ok=True)
    p = paths()

    if not os.path.exists(p["model"]):
        print(f"No checkpoint found at {p['model']} -- train solo first.")
        return

    if not os.path.exists(p["best_model"]):
        shutil.copy(p["model"], p["best_model"])

    best_score, best_coins, best_kills, best_suicides = evaluate_score(EVAL_ROUNDS)
    print(f"Starting point vs rule_based_agent ({EVAL_ROUNDS} rounds): "
          f"score={best_score}, coins={best_coins}, kills={best_kills}, suicides={best_suicides}")

    for i in range(BURSTS):
        print(f"\n{'='*60}\nBurst {i+1}/{BURSTS}: training {ROUNDS_PER_BURST} rounds vs rule_based_agent\n{'='*60}")
        run([sys.executable, "main.py", "play", "--agents", AGENT, "rule_based_agent",
             "--train", "1", "--no-gui", "--n-rounds", str(ROUNDS_PER_BURST), "--scenario", "classic"])

        score, coins, kills, suicides = evaluate_score(EVAL_ROUNDS)
        print(f"Burst {i+1}: score={score} (coins={coins}, kills={kills}, suicides={suicides})", end="  ")

        if score >= best_score - TOLERANCE:
            print("-> KEPT")
            shutil.copy(p["model"], p["best_model"])
            best_score = max(score, best_score)
        else:
            print("-> DISCARDED (reverted to best checkpoint)")
            shutil.copy(p["best_model"], p["model"])

    print(f"\n\nDone. Best verified score vs rule_based_agent: {best_score}")
    print(f"Best checkpoint: {p['best_model']}")
    print(f"To promote it: cp {p['best_model']} {p['model']}")


if __name__ == "__main__":
    main()
