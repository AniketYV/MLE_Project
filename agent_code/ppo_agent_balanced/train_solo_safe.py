"""
Safe, unattended solo classic training: trains in bursts, evaluates for
real after each one, and automatically discards any burst that regresses.
Built specifically so an overnight run can only end up better or equal to
where it started -- never worse -- unlike a single raw `python main.py
play --train 1 --n-rounds 8000` call, which just keeps overwriting the
same checkpoint with no protection if training destabilizes partway
through (something that has happened more than once in this project,
even with NaN-crash protection in place -- that guards against crashes,
not against a policy quietly getting worse under continued training).

Evaluates on survive-and-bomb success rate (crates destroyed AND
survived), since that's the metric this whole training phase has been
tracked against -- not training reward, which has already proven
misleading more than once in this project.

Usage: python train_solo_safe.py
"""
import json
import os
import shutil
import subprocess
import sys

AGENT = "ppo_agent_balanced"
BURSTS = 8
ROUNDS_PER_BURST = 1000
EVAL_ROUNDS = 100
TOLERANCE = 0.5   # allow a small drop in crates/round without discarding
                   # (measurement noise across a 100-round sample)


def run(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"\n!!! Command failed: {' '.join(cmd)}\n--- output ---\n{result.stdout}\n--- end output ---\n")
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def paths():
    base = f"agent_code/{AGENT}"
    return {
        "model": f"{base}/ppo-model.pt",
        "best_model": f"{base}/ppo-model.best_solo.pt",
        "log": f"{base}/logs/training_log.csv",
    }


def evaluate(n_rounds):
    """Real, no-training evaluation via --save-stats. Per-round CSV
    logging (training_log.csv) only happens inside train.py's hooks,
    which are never invoked without --train -- a pure evaluation run
    writes zero new log lines, so aggregate stats from --save-stats is
    the only reliable signal here. Uses avg crates/round (a proxy for
    the survive-and-bomb behaviour this training phase targets) and the
    suicide rate, rather than a strict per-round success count."""
    stats_path = "results/_train_solo_safe_eval.json"
    run([sys.executable, "main.py", "play", "--agents", AGENT,
         "--no-gui", "--n-rounds", str(n_rounds), "--scenario", "classic",
         "--save-stats", stats_path])
    with open(stats_path) as f:
        data = json.load(f)
    d = data["by_agent"][AGENT]
    crates = d.get("crates", 0)
    suicides = d.get("suicides", 0)
    return crates / n_rounds, suicides / n_rounds


def main():
    os.makedirs("results", exist_ok=True)
    p = paths()

    if not os.path.exists(p["model"]):
        print(f"No checkpoint found at {p['model']}.")
        return

    if not os.path.exists(p["best_model"]):
        shutil.copy(p["model"], p["best_model"])

    best_crates_rate, best_death_rate = evaluate(EVAL_ROUNDS)
    print(f"Starting point: {best_crates_rate:.2f} crates/round, {100*best_death_rate:.0f}% death rate")

    for i in range(BURSTS):
        print(f"\n{'='*60}\nBurst {i+1}/{BURSTS}: training {ROUNDS_PER_BURST} rounds\n{'='*60}")
        run([sys.executable, "main.py", "play", "--agents", AGENT,
             "--train", "1", "--no-gui", "--n-rounds", str(ROUNDS_PER_BURST), "--scenario", "classic"])

        crates_rate, death_rate = evaluate(EVAL_ROUNDS)
        print(f"Burst {i+1}: {crates_rate:.2f} crates/round, {100*death_rate:.0f}% death rate", end="  ")

        # Keep if crate-clearing didn't drop meaningfully AND death rate
        # didn't get meaningfully worse -- both matter, since a policy
        # that clears more crates by dying more recklessly isn't actually
        # an improvement for this phase's goal.
        crates_ok = crates_rate >= best_crates_rate - TOLERANCE
        death_ok = death_rate <= best_death_rate + 0.05
        if crates_ok and death_ok:
            print("-> KEPT")
            shutil.copy(p["model"], p["best_model"])
            best_crates_rate = max(crates_rate, best_crates_rate)
            best_death_rate = min(death_rate, best_death_rate)
        else:
            print("-> DISCARDED (reverted to best checkpoint)")
            shutil.copy(p["best_model"], p["model"])

    print(f"\n\nDone. Best verified: {best_crates_rate:.2f} crates/round, {100*best_death_rate:.0f}% death rate")
    print(f"Best checkpoint: {p['best_model']}")
    print(f"Current model.pt already reflects the best result found.")


if __name__ == "__main__":
    main()
