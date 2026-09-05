"""
Reads a --save-stats JSON output and prints a clean comparison of our
agent against its opponents. Run main.py with --save-stats first, then
point this at the resulting file.

Usage:
    python main.py play --agents linear_q_agent rule_based_agent rule_based_agent rule_based_agent \\
        --no-gui --n-rounds 50 --scenario classic --save-stats results/eval.json
    python evaluate.py results/eval.json
"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/eval.json"

with open(path) as f:
    data = json.load(f)

by_agent = data["by_agent"]
by_round = data["by_round"]
n_rounds = len(by_round)

print(f"=== {path} ({n_rounds} rounds) ===\n")
rows = []
for name, stats in by_agent.items():
    rows.append((
        name,
        stats.get("score", 0),
        stats.get("coins", 0),
        stats.get("kills", 0),
        stats.get("suicides", 0),
        stats.get("rounds", 0),
    ))

rows.sort(key=lambda r: -r[1])  # sort by score, descending
print(f"{'agent':<20}{'score':>8}{'coins':>8}{'kills':>8}{'suicides':>10}{'rounds':>8}")
for name, score, coins, kills, suicides, rounds in rows:
    print(f"{name:<20}{score:>8}{coins:>8}{kills:>8}{suicides:>10}{rounds:>8}")

total_coins = sum(r["coins"] for r in by_round.values())
total_kills = sum(r["kills"] for r in by_round.values())
total_suicides = sum(r["suicides"] for r in by_round.values())
avg_steps = sum(r["steps"] for r in by_round.values()) / n_rounds
print(f"\nBoard totals: {total_coins} coins collected, {total_kills} kills, "
      f"{total_suicides} suicides, avg {avg_steps:.0f} steps/round")
