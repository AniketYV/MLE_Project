"""
Curriculum training pipeline for sarsa_agent, structured around the same
four official tasks as dqn_train_pipeline.py (see that file / the project
spec section 4 for the task rationale):

  Task 1: solo coin collection, no crates/opponents      -> coin-heaven
  Task 2: solo crate-bombing + escaping, no opponents     -> loot-crate / classic
  Task 3: hunt peaceful_agent (easy) + coin_collector_agent (hard)
  Task 4: hold your own against rule_based_agent (1v1, then 3v1)

Every lesson learned building dqn_train_pipeline.py is carried over:
  - Multi-seed evaluation (never trust a single seed -- rule_based_agent's
    own randomness isn't covered by --seed, confirmed directly in its
    source: it calls np.random.seed() with no argument in setup(), plus
    uses Python's separate `random.shuffle`).
  - Anchor keep/discard decisions to the best-EVER (score, suicide_rate)
    seen for each exact (scenario, opponents) config, not to the
    immediately preceding burst's own result -- a sequential comparison
    lets small tolerated regressions compound across many stages
    (0.85^10 ≈ 0.20, an 80% possible cumulative loss).
  - Gate on BOTH score and suicide rate. Score-only gating let a burst
    that spiked suicide rate while getting lucky on score slip through
    as "kept" during dqn_agent's development (score 41->92 while suicide
    rate went 0.0->0.867 in one real run).

Usage:
    python sarsa_train_pipeline.py            # resume from existing checkpoint
    python sarsa_train_pipeline.py --fresh    # wipe everything, start over
"""
import json
import os
import shutil
import subprocess
import sys

AGENT_NAME = "sarsa_agent"
AGENT_DIR = f"agent_code/{AGENT_NAME}"

MODEL_PATH = f"{AGENT_DIR}/sarsa-model.pt"
STATE_PATH = f"{AGENT_DIR}/sarsa-train-state.json"
EPSILON_OVERRIDE_PATH = f"{AGENT_DIR}/sarsa-epsilon-override.txt"

BEST_MODEL_PATH = f"{AGENT_DIR}/sarsa-model.best.pt"
BEST_STATE_PATH = f"{AGENT_DIR}/sarsa-train-state.best.json"

EVAL_STATS = "results/_sarsa_pipeline_eval.json"
EVAL_SEEDS = (1, 2, 3)                       # solo stages
EVAL_SEEDS_VS_OPPONENTS = (1, 2, 3, 4, 5)    # opponent stages, more seeds to
# average out rule_based_agent's/coin_collector_agent's own randomness


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
    total_score, total_suicides, total_steps, total_rounds = 0, 0, 0, 0
    for seed in seeds:
        sc, su, st, r = evaluate_once(n_rounds, scenario, opponents, seed)
        total_score += sc
        total_suicides += su
        total_steps += st
        total_rounds += r
    mean_score = total_score / len(seeds)
    suicide_rate = total_suicides / total_rounds
    print(f"    avg score={mean_score:.1f}  "
          f"suicide_rate={suicide_rate:.3f}/round  "
          f"avg_steps={total_steps / total_rounds:.1f}  "
          f"(over {len(seeds)} seeds x {n_rounds} rounds)")
    return mean_score, suicide_rate


def backup_best():
    if not os.path.exists(MODEL_PATH):
        return
    shutil.copy(MODEL_PATH, BEST_MODEL_PATH)
    if os.path.exists(STATE_PATH):
        shutil.copy(STATE_PATH, BEST_STATE_PATH)


def restore_best():
    if not os.path.exists(BEST_MODEL_PATH):
        return
    shutil.copy(BEST_MODEL_PATH, MODEL_PATH)
    if os.path.exists(BEST_STATE_PATH):
        shutil.copy(BEST_STATE_PATH, STATE_PATH)


def stage(name, train_rounds, epsilon, scenario, opponents=(), eval_rounds=30, best_scores=None):
    if best_scores is None:
        best_scores = {}
    key = (scenario, opponents)
    eval_seeds = EVAL_SEEDS_VS_OPPONENTS if opponents else EVAL_SEEDS
    SUICIDE_RATE_TOLERANCE = 0.15

    print(f"\n=== {name} ===")
    print(f"    train: --agents {AGENT_NAME} {' '.join(opponents)} "
          f"--scenario {scenario} --n-rounds {train_rounds} (epsilon={epsilon})")

    if not os.path.exists(BEST_MODEL_PATH):
        backup_best()

    if key in best_scores:
        baseline, baseline_suicide = best_scores[key]
        print(f"  baseline (best-ever for this config): score={baseline:.1f}  "
              f"suicide_rate={baseline_suicide:.3f}")
    else:
        print("  baseline:")
        baseline, baseline_suicide = evaluate(eval_rounds, scenario, opponents, seeds=eval_seeds)
        best_scores[key] = (baseline, baseline_suicide)

    set_epsilon(epsilon)
    cmd = [sys.executable, "main.py", "play", "--agents", AGENT_NAME, *opponents,
           "--train", "1", "--no-gui", "--n-rounds", str(train_rounds), "--scenario", scenario]
    run(cmd)

    print("  after training:")
    new_score, new_suicide = evaluate(eval_rounds, scenario, opponents, seeds=eval_seeds)

    score_ok = new_score >= baseline * 0.85
    suicide_ok = new_suicide <= baseline_suicide + SUICIDE_RATE_TOLERANCE

    if score_ok and suicide_ok:
        print("  -> kept (improved, held steady, or within noise tolerance)")
        backup_best()
        if new_score > best_scores[key][0]:
            best_scores[key] = (new_score, new_suicide)
    else:
        reason = []
        if not score_ok:
            reason.append(f"score dropped too much ({new_score:.1f} vs baseline {baseline:.1f})")
        if not suicide_ok:
            reason.append(f"suicide rate rose too much ({new_suicide:.3f} vs baseline "
                           f"{baseline_suicide:.3f}, allowed +{SUICIDE_RATE_TOLERANCE})")
        print(f"  -> DISCARDED ({'; '.join(reason)}), reverted to previous best")
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

    scores = {}

    # --- Task 1: solo coin collection ---
    scores = stage("Task 1: solo coin collection", 800, 1.0, "coin-heaven", best_scores=scores)
    scores = stage("Task 1: solo coin collection (burst 2)", 800, 0.4, "coin-heaven", best_scores=scores)

    # --- Task 2: solo crate-bombing + escaping ---
    scores = stage("Task 2: solo bombing (loot-crate, burst 1)", 1000, 0.5, "loot-crate", best_scores=scores)
    scores = stage("Task 2: solo bombing (loot-crate, burst 2)", 1000, 0.25, "loot-crate", best_scores=scores)
    scores = stage("Task 2: solo bombing (classic sanity check)", 800, 0.15, "classic", best_scores=scores)

    # --- Task 3: hunt weak opponents ---
    scores = stage("Task 3: hunt peaceful_agent", 1000, 0.3, "classic",
                   opponents=("peaceful_agent",), best_scores=scores)
    scores = stage("Task 3: hunt peaceful_agent + coin_collector_agent", 1200, 0.2, "classic",
                   opponents=("peaceful_agent", "coin_collector_agent"), best_scores=scores)

    # --- Task 4: hold your own against real opposition ---
    scores = stage("Task 4: 1v1 vs rule_based_agent", 1200, 0.3, "classic",
                   opponents=("rule_based_agent",), best_scores=scores)
    scores = stage("Task 4: 1v1 vs rule_based_agent (burst 2)", 1200, 0.15, "classic",
                   opponents=("rule_based_agent",), best_scores=scores)
    for i in range(3):
        scores = stage(f"Task 4: full tournament config (burst {i + 1})", 1500, 0.15, "classic",
                       opponents=("rule_based_agent", "rule_based_agent", "rule_based_agent"), best_scores=scores)

    print("\n=== PIPELINE DONE. Final check: tournament configuration, 5 seeds x 100 rounds ===")
    final_score, final_suicide = evaluate(100, "classic", opponents=("rule_based_agent",) * 3, seeds=(1, 2, 3, 4, 5))
    print(f"final avg score vs 3x rule_based_agent: {final_score:.1f}  (suicide_rate={final_suicide:.3f})")
