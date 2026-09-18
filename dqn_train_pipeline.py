"""
Curriculum training pipeline for dqn_agent, structured around the four
official tasks from the project spec (section 4):

  Task 1: solo coin collection, no crates/opponents      -> coin-heaven
  Task 2: solo crate-bombing + escaping, no opponents     -> loot-crate / classic
  Task 3: hunt peaceful_agent (easy) + coin_collector_agent (hard)
  Task 4: hold your own against rule_based_agent (1v1, then 3v1)

Each stage trains a burst, then evaluates over multiple seeds (never trust
a single seed -- see the many false leads in this project's dev history)
and keeps the burst only if it didn't regress; otherwise the previous best
checkpoint is restored. This mirrors the evaluate-after-every-burst /
keep-only-if-not-worse pattern already used by train_pipeline.py and
train_more.py for linear_q_agent.

Usage:
    python dqn_train_pipeline.py            # resume from existing checkpoint
    python dqn_train_pipeline.py --fresh    # wipe everything, start over
"""
import json
import os
import shutil
import subprocess
import sys

AGENT_NAME = "dqn_agent"
AGENT_DIR = f"agent_code/{AGENT_NAME}"

MODEL_PATH = f"{AGENT_DIR}/dqn-model.pt"
STATE_PATH = f"{AGENT_DIR}/dqn-train-state.json"
EPSILON_OVERRIDE_PATH = f"{AGENT_DIR}/dqn-epsilon-override.txt"

BEST_MODEL_PATH = f"{AGENT_DIR}/dqn-model.best.pt"
BEST_STATE_PATH = f"{AGENT_DIR}/dqn-train-state.best.json"

EVAL_STATS = "results/_dqn_pipeline_eval.json"
EVAL_SEEDS = (1, 2, 3)          # solo stages: only board layout is random, well-controlled by --seed
EVAL_SEEDS_VS_OPPONENTS = (1, 2, 3, 4, 5)  # rule_based_agent/coin_collector_agent use
# Python's plain `random` internally, NOT the --seed-controlled generator --
# confirmed by checking agent_code/rule_based_agent -- so "same seed" only
# fixes the board, not opponent behavior. More seeds compensate with more
# averaging since the noise itself can't be removed.


def run(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"\n!!! Command failed: {' '.join(cmd)}\n--- output ---\n{result.stdout}\n--- end output ---\n")
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def set_epsilon(value):
    with open(EPSILON_OVERRIDE_PATH, "w") as f:
        f.write(str(value))


def evaluate_once(n_rounds, scenario, opponents, seed):
    agents = [AGENT_NAME] + list(opponents)
    run([sys.executable, "main.py", "play", "--agents", *agents,
         "--no-gui", "--n-rounds", str(n_rounds), "--scenario", scenario,
         "--seed", str(seed), "--save-stats", EVAL_STATS])
    with open(EVAL_STATS) as f:
        data = json.load(f)
    d = data["by_agent"][AGENT_NAME]
    return d.get("score", 0), d.get("suicides", 0), d.get("steps", 0), d.get("rounds", n_rounds)


def evaluate(n_rounds, scenario, opponents=(), seeds=EVAL_SEEDS):
    """Averages score (and reports suicide rate / avg survival) across
    multiple seeds. Returns just the mean score for the keep/discard
    decision, but prints the fuller picture so you have it for the report."""
    total_score, total_suicides, total_steps, total_rounds = 0, 0, 0, 0
    for seed in seeds:
        sc, su, st, r = evaluate_once(n_rounds, scenario, opponents, seed)
        total_score += sc
        total_suicides += su
        total_steps += st
        total_rounds += r
    mean_score = total_score / len(seeds)
    print(f"    avg score={mean_score:.1f}  "
          f"suicide_rate={total_suicides / total_rounds:.3f}/round  "
          f"avg_steps={total_steps / total_rounds:.1f}  "
          f"(over {len(seeds)} seeds x {n_rounds} rounds)")
    return mean_score


def backup_best():
    if not os.path.exists(MODEL_PATH):
        return  # nothing trained yet, nothing to back up
    shutil.copy(MODEL_PATH, BEST_MODEL_PATH)
    if os.path.exists(STATE_PATH):
        shutil.copy(STATE_PATH, BEST_STATE_PATH)


def restore_best():
    if not os.path.exists(BEST_MODEL_PATH):
        return  # no prior checkpoint to fall back to yet, keep what we have
    shutil.copy(BEST_MODEL_PATH, MODEL_PATH)
    if os.path.exists(BEST_STATE_PATH):
        shutil.copy(BEST_STATE_PATH, STATE_PATH)


def stage(name, train_rounds, epsilon, scenario, opponents=(), eval_rounds=30, best_scores=None):
    """best_scores: dict shared across all stage() calls in this run, keyed
    by (scenario, opponents), tracking the best score ever seen for that
    exact task configuration. Comparing against this (rather than against
    the immediately preceding stage's own score) prevents compounding drift:
    a 15% tolerance applied stage-to-stage, chained across ~10 stages, can
    lose the majority of real progress even though each individual step
    looked fine -- 0.85^10 ≈ 0.20, an 80% possible cumulative loss. Anchoring
    to the per-config best-ever closes that gap. Different scenarios/
    opponents aren't comparable in raw score, hence keying by config rather
    than tracking one single global best."""
    if best_scores is None:
        best_scores = {}
    key = (scenario, opponents)
    eval_seeds = EVAL_SEEDS_VS_OPPONENTS if opponents else EVAL_SEEDS

    print(f"\n=== {name} ===")
    print(f"    train: --agents {AGENT_NAME} {' '.join(opponents)} "
          f"--scenario {scenario} --n-rounds {train_rounds} (epsilon={epsilon})")

    if not os.path.exists(BEST_MODEL_PATH):
        backup_best()

    if key in best_scores:
        baseline = best_scores[key]
        print(f"  baseline (best-ever for this config): {baseline:.1f}")
    else:
        print("  baseline:")
        baseline = evaluate(eval_rounds, scenario, opponents, seeds=eval_seeds)
        best_scores[key] = baseline

    set_epsilon(epsilon)
    cmd = [sys.executable, "main.py", "play", "--agents", AGENT_NAME, *opponents,
           "--train", "1", "--no-gui", "--n-rounds", str(train_rounds), "--scenario", scenario]
    run(cmd)

    print("  after training:")
    new_score = evaluate(eval_rounds, scenario, opponents, seeds=eval_seeds)

    if new_score >= baseline * 0.85:  # small tolerance -- noisy evals (esp.
        # rule_based_agent's own randomness isn't covered by --seed) can
        # make a genuinely-fine burst look slightly worse by chance; a
        # strict >= caused 8 consecutive false-regression reverts in
        # testing, permanently stalling progress after Task 2.
        print("  -> kept (improved, held steady, or within noise tolerance)")
        backup_best()
        best_scores[key] = max(best_scores[key], new_score)
    else:
        print("  -> DISCARDED (regressed), reverted to previous best")
        restore_best()

    return best_scores


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    fresh_start = "--fresh" in sys.argv
    if fresh_start:
        print("--fresh passed: wiping model + training state, starting from scratch.")
        for p in [MODEL_PATH, STATE_PATH, BEST_MODEL_PATH, BEST_STATE_PATH, EPSILON_OVERRIDE_PATH]:
            if os.path.exists(p):
                os.remove(p)

    scores = {}  # shared across all stages: tracks best-ever score per (scenario, opponents) config

    # --- Task 1: solo coin collection, no crates, no opponents ---
    # "The agent should learn how to navigate the board efficiently."
    stage("Task 1: solo coin collection", 800, 1.0, "coin-heaven", best_scores=scores)
    stage("Task 1: solo coin collection (burst 2)", 800, 0.4, "coin-heaven", best_scores=scores)

    # --- Task 2: solo crate-bombing + escaping, no opponents ---
    # "It should learn how to use bombs without killing itself ... place
    # proper emphasis on this step." loot-crate has the same crate density
    # as classic but denser coins for a faster reward signal; classic-solo
    # (the spec's literal example command) is used as a periodic sanity
    # check since it's the actual tournament density/coin-count.
    stage("Task 2: solo bombing (loot-crate, burst 1)", 1000, 0.5, "loot-crate", best_scores=scores)
    stage("Task 2: solo bombing (loot-crate, burst 2)", 1000, 0.25, "loot-crate", best_scores=scores)
    stage("Task 2: solo bombing (classic sanity check)", 800, 0.15, "classic", best_scores=scores)

    # --- Task 3: hunt weak opponents ---
    # peaceful_agent never bombs (easy target); coin_collector_agent bombs
    # only for coins, no combat behavior (harder, but still not fighting back).
    stage("Task 3: hunt peaceful_agent", 1000, 0.3, "classic",
          opponents=("peaceful_agent",), best_scores=scores)
    stage("Task 3: hunt peaceful_agent + coin_collector_agent", 1200, 0.2, "classic",
          opponents=("peaceful_agent", "coin_collector_agent"), best_scores=scores)

    # --- Task 4: hold your own against real opposition ---
    # Spec's example is 1v1 first; ramp to the full 3-opponent tournament
    # configuration since "you must be able to beat the rule_based_agent
    # in order to have any chance of winning."
    stage("Task 4: 1v1 vs rule_based_agent", 1200, 0.3, "classic",
          opponents=("rule_based_agent",), best_scores=scores)
    stage("Task 4: 1v1 vs rule_based_agent (burst 2)", 1200, 0.15, "classic",
          opponents=("rule_based_agent",), best_scores=scores)
    for i in range(3):
        stage(f"Task 4: full tournament config (burst {i + 1})", 1500, 0.15, "classic",
              opponents=("rule_based_agent", "rule_based_agent", "rule_based_agent"), best_scores=scores)

    print("\n=== PIPELINE DONE. Final check: tournament configuration, 5 seeds x 100 rounds ===")
    final = evaluate(100, "classic", opponents=("rule_based_agent",) * 3, seeds=(1, 2, 3, 4, 5))
    print(f"final avg score vs 3x rule_based_agent: {final:.1f}")
