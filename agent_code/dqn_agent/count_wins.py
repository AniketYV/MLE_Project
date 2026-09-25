"""
Counts how many rounds each agent actually WON (highest score that round) --
not captured by --save-stats, game.log, or replay files, since "X has won
the round!" is only ever drawn on the GUI screen, never logged anywhere.

Usage:
    python count_wins.py --agents dqn_agent rule_based_agent rule_based_agent rule_based_agent \
        --scenario classic --n-rounds 500 --seed 1
"""
import argparse
from collections import Counter

import settings as s
from environment import BombeRLeWorld, WorldArgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", required=True)
    parser.add_argument("--scenario", default="classic", choices=s.SCENARIOS)
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    world_args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir="logs",
        save_stats=False, match_name=None, seed=args.seed,
        silence_errors=False, scenario=args.scenario,
    )
    agents = [(name, False) for name in args.agents]
    world = BombeRLeWorld(world_args, agents)

    wins = Counter()
    ties = 0

    for _ in range(args.n_rounds):
        world.new_round()
        while world.running:
            world.do_step(None)

        scores = {a.name: a.score for a in world.agents}
        best_score = max(scores.values())
        winners = [name for name, sc in scores.items() if sc == best_score]
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            ties += 1  # multiple agents tied for the highest score that round

    world.end()

    print(f"\nResults over {args.n_rounds} rounds ({args.scenario}, seed={args.seed}):")
    for name in args.agents:
        # args.agents may repeat a name (e.g. 3x rule_based_agent); world
        # renames duplicates to name_0, name_1, ... -- report both forms.
        pass
    for name, count in wins.most_common():
        print(f"  {name}: {count} wins ({count/args.n_rounds:.1%})")
    if ties:
        print(f"  (ties, no single winner): {ties} rounds ({ties/args.n_rounds:.1%})")


if __name__ == "__main__":
    main()
