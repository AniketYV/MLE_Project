"""
Curriculum training driver for the PPO agent.

Runs training in stages of increasing difficulty, same philosophy that
worked for your DQN agent: coin-heaven (learn basic movement + coin
collection with no crates/threats) -> classic (crates hide coins, no
opponents) -> classic with rule_based_agent opponents.

This wraps `main.py play` as subprocess calls rather than re-implementing
the game loop — keeps this file simple and lets the framework's own
training loop (which we already validated works) do the real work.

Usage:
    python3 train_pipeline.py
"""

import subprocess
import sys
import os
import shutil
import re

STAGES = [
    {
        "name": "coin-heaven",
        "n_rounds": 2000,
        "extra_args": ["--scenario", "coin-heaven"],
    },
    {
        "name": "classic (solo)",
        "n_rounds": 3000,
        "extra_args": ["--scenario", "classic"],
    },
    {
        "name": "classic vs rule_based_agent",
        "n_rounds": 8000,
        "extra_args": [
            "--scenario", "classic",
            "--agents", "ppo_agent", "rule_based_agent", "rule_based_agent", "rule_based_agent",
        ],
    },
]


def _slug(name: str) -> str:
    """Turns a stage name like 'classic vs rule_based_agent' into a
    filesystem-safe slug like 'classic-vs-rule-based-agent'."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def archive_checkpoints(stage_name: str, project_root: str):
    """
    Copies the live checkpoint files to stage-tagged names so each
    stage's result is preserved, since the live files keep getting
    overwritten as training continues into the next stage.
    """
    checkpoint_dir = os.path.join(project_root, "agent_code", "ppo_agent", "checkpoints")
    slug = _slug(stage_name)

    for live_name in ["policy-network.pt", "value-network.pt"]:
        live_path = os.path.join(checkpoint_dir, live_name)
        if not os.path.exists(live_path):
            print(f"WARNING: expected checkpoint not found: {live_path}")
            continue
        base, ext = os.path.splitext(live_name)
        archived_path = os.path.join(checkpoint_dir, f"{base}.{slug}{ext}")
        shutil.copy2(live_path, archived_path)
        print(f"Archived checkpoint: {archived_path}")


def run_stage(stage: dict, project_root: str):
    print(f"\n{'=' * 60}")
    print(f"STAGE: {stage['name']}  ({stage['n_rounds']} rounds)")
    print(f"{'=' * 60}\n")

    cmd = [
        sys.executable, "main.py", "play",
        "--agents", "ppo_agent",
        "--train", "1",
        "--n-rounds", str(stage["n_rounds"]),
        "--no-gui",
    ]

    # The opponent stage overrides --agents entirely (needs 4 agents listed);
    # strip the default single-agent args if extra_args redefines --agents.
    if "--agents" in stage["extra_args"]:
        cmd = [
            sys.executable, "main.py", "play",
            "--train", "1",
            "--n-rounds", str(stage["n_rounds"]),
            "--no-gui",
        ] + stage["extra_args"]
    else:
        cmd += stage["extra_args"]

    print(f"Running: {' '.join(cmd)}\n")
    env = os.environ.copy()
    env["PPO_STAGE_NAME"] = stage["name"]
    result = subprocess.run(cmd, cwd=project_root, env=env)

    if result.returncode != 0:
        print(f"\nStage '{stage['name']}' exited with code {result.returncode} — stopping pipeline.")
        sys.exit(result.returncode)

    print(f"\nStage '{stage['name']}' complete.")
    archive_checkpoints(stage["name"], project_root)


if __name__ == "__main__":
    # Assumes this script is run from the bomberman_rl project root, or
    # adjust project_root below to an absolute path if needed.
    project_root = os.getcwd()

    print(f"Starting curriculum training. Project root: {project_root}")
    print(f"Stages: {[s['name'] for s in STAGES]}")

    for stage in STAGES:
        run_stage(stage, project_root)

    print("\nCurriculum complete. Check episode_log.csv and the checkpoint files.")
