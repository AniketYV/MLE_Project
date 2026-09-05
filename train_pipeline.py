"""
Robust training pipeline: every training burst is followed by a REAL
evaluation (no exploration, same as tournament play). If the result isn't
at least as good as the current best, the burst is discarded and the best
checkpoint (model AND replay buffer) is restored automatically.

Usage: python train_pipeline.py
"""
import json
import os
import pickle
import shutil
import subprocess

AGENT_DIR = "agent_code/linear_q_agent"
MODEL_PATH = f"{AGENT_DIR}/my-saved-model.pt"
REPLAY_PATH = f"{AGENT_DIR}/replay-buffer.pt"
BEST_MODEL_PATH = f"{AGENT_DIR}/my-saved-model.best.pt"
BEST_REPLAY_PATH = f"{AGENT_DIR}/replay-buffer.best.pt"
EVAL_STATS = "results/_pipeline_eval.json"


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def set_epsilon(value):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    model.epsilon = value
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def evaluate_score(eval_agents, n_rounds, scenario):
    run(["python3", "main.py", "play", "--agents", *eval_agents,
         "--no-gui", "--n-rounds", str(n_rounds), "--scenario", scenario,
         "--save-stats", EVAL_STATS])
    with open(EVAL_STATS) as f:
        data = json.load(f)
    return data["by_agent"]["linear_q_agent"]["score"]


def backup_best():
    shutil.copy(MODEL_PATH, BEST_MODEL_PATH)
    if os.path.exists(REPLAY_PATH):
        shutil.copy(REPLAY_PATH, BEST_REPLAY_PATH)


def restore_best():
    shutil.copy(BEST_MODEL_PATH, MODEL_PATH)
    if os.path.exists(BEST_REPLAY_PATH):
        shutil.copy(BEST_REPLAY_PATH, REPLAY_PATH)


def stage(name, train_agents, train_rounds, epsilon, eval_agents, scenario, eval_rounds=40):
    print(f"\n=== {name} ===")

    if not os.path.exists(BEST_MODEL_PATH):
        backup_best()
    baseline = evaluate_score(eval_agents, eval_rounds, scenario)
    print(f"baseline score: {baseline}")

    set_epsilon(epsilon)
    run(["python3", "main.py", "play", "--agents", *train_agents, "--train", "1",
         "--no-gui", "--n-rounds", str(train_rounds), "--scenario", scenario])

    new_score = evaluate_score(eval_agents, eval_rounds, scenario)
    print(f"score after training: {new_score}")

    if new_score >= baseline:
        print("-> kept (improved or held steady)")
        backup_best()
    else:
        print("-> DISCARDED (regressed), reverted to previous best")
        restore_best()

    return max(new_score, baseline)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    for p in [MODEL_PATH, REPLAY_PATH, BEST_MODEL_PATH, BEST_REPLAY_PATH]:
        if os.path.exists(p):
            os.remove(p)

    # Stage 1: coin-heaven, fresh
    run(["python3", "main.py", "play", "--agents", "linear_q_agent", "--train", "1",
         "--no-gui", "--n-rounds", "10", "--scenario", "coin-heaven"])
    stage("coin-heaven", ["linear_q_agent"], 500, 1.0, ["linear_q_agent"], "coin-heaven")

    # Stage 2: classic solo
    stage("classic solo (burst 1)", ["linear_q_agent"], 400, 0.25, ["linear_q_agent"], "classic")
    stage("classic solo (burst 2)", ["linear_q_agent"], 400, 0.10, ["linear_q_agent"], "classic")

    # Stage 3: vs 1 rule_based_agent -- BIGGER bursts this time (400, not
    # 150), so the replay buffer actually has room to build up useful
    # experience before each evaluation checkpoint.
    for i in range(4):
        stage(f"vs rule_based_agent (burst {i+1})",
              ["linear_q_agent", "rule_based_agent"], 400, 0.12,
              ["linear_q_agent", "rule_based_agent"], "classic")

    print("\n=== PIPELINE DONE. Final vs 3x rule_based_agent check: ===")
    final = evaluate_score(
        ["linear_q_agent", "rule_based_agent", "rule_based_agent", "rule_based_agent"],
        50, "classic")
    print(f"final score vs 3x rule_based_agent: {final}")
