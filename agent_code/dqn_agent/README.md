# dqn_agent

A small DQN agent for the bomberman_rl framework.

## ⚠️ Compliance with the course's ML-only rule -- read before submitting
The project spec states: *"Your solution must involve machine learning or
it will be rejected"* and explicitly disallows *"a feature that
deterministically returns the action which results in the best move."*
The safety patches in this agent (`_apply_safety_shield`,
`_apply_bomb_safety_check`) are hard-coded rules that can override the
network's chosen action in specific situations -- closer to published
"safe RL via action shielding" than to the disallowed pattern (the network
still has to learn everything else; the override only fires in a narrow,
safety-critical slice of states), but this is a judgment call, not an
obviously-safe one.

**Ask on Discord (#final-project-questions) before submitting.** Two
switches at the top of `callbacks.py` let you resolve this either way
without touching any other code:
```python
USE_SAFETY_PATCHES_AT_INFERENCE = True     # affects tournament/evaluation behavior
USE_SAFETY_PATCHES_DURING_TRAINING = True  # affects only what training data looks like
```
If told the patches aren't allowed at evaluation time, set the first to
`False` for your submission (training can still use them, since that only
shapes what experience the network learns from, not what it does at
decision time). Every decision the agent makes is counted in
`self.total_decisions` / `self.shield_interventions`, and the intervention
rate is logged every 1000 decisions -- grep `"intervention rate"` out of
`agent_code/dqn_agent/logs/dqn_agent.log` for a concrete number to cite in
your report's Methods/Experiments section either way.

## Known limitation: shielded training vs. genuine learning
Since the safety patches are active during training too
(`USE_SAFETY_PATCHES_DURING_TRAINING = True`), the network rarely
experiences the actual consequence of choosing `WAIT`/`BOMB` while in
danger -- the shield intercepts it first, and the *shielded* action is
what gets recorded and rewarded (`FLED_DANGER`, `STAYED_PUT_IN_DANGER`).
This means the network is being *guided* to flee, not necessarily
*learning* to flee through its own trial and error. Practically: if the
compliance answer requires disabling `USE_SAFETY_PATCHES_AT_INFERENCE`
for your tournament submission, don't assume the raw network has
independently learned good self-preservation -- it may not have, since
it was protected from experiencing the alternative throughout training.
Worth mentioning explicitly in your report's Methods or Conclusion
section as a design tradeoff, not something to silently paper over.

## Known limitation: rule_based_agent's randomness isn't seed-controlled
Confirmed directly in `agent_code/rule_based_agent/callbacks.py`: its
`setup()` calls `np.random.seed()` with no argument (deliberately
re-randomizing from OS entropy every process launch) and it uses
`from random import shuffle` elsewhere -- both independent of the
`--seed` flag, which only controls the *board's* random generator
(crate/coin layout) in `environment.py`. This means no evaluation
involving `rule_based_agent` (or `coin_collector_agent`, which shares
similar logic) is ever fully reproducible, seed or not -- a structural
property of the provided framework, not something fixable in your own
code. `dqn_train_pipeline.py` compensates by using more seeds
(`EVAL_SEEDS_VS_OPPONENTS`, 5 instead of 3) for any stage with opponents,
to average out more of this unavoidable noise -- it reduces the problem,
it doesn't eliminate it. Keep this in mind when interpreting any single
evaluation run, and mention it in your report if you discuss variance.

## Training: aligned with the spec's four tasks
`dqn_train_pipeline.py` is structured around the four tasks from section 4
of the project spec, each evaluated over multiple seeds (single-seed
comparisons on this agent were repeatedly misleading during development --
see the git history / conversation this came from):

| Task | Scenario | Opponents |
|---|---|---|
| 1: solo coin collection | `coin-heaven` | none |
| 2: solo bombing + escaping | `loot-crate`, then `classic` | none |
| 3: hunt weak opponents | `classic` | `peaceful_agent`, then + `coin_collector_agent` |
| 4: hold your own | `classic` | `rule_based_agent` (1v1, then 3v1) |

## What's here
- `callbacks.py` — the `QNet` model (19 -> 128 -> 128 -> 6 MLP), the
  `state_to_features` hand-crafted feature extractor, and `act()`.
- `train.py` — replay buffer, target network, Bellman update, reward
  shaping, and epsilon decay.

## Feature vector (19 floats)
1. Can I step UP/RIGHT/DOWN/LEFT? (4 floats, 1 = free)
2. Danger **urgency** of that tile: 0 = safe, ramping up toward 1 as a
   bomb affecting it gets closer to exploding, 1 = live explosion (4 floats)
3. Danger urgency of my current tile right now (1 float)
4. Can I drop a bomb? (1 float)
5. (dx, dy, reachable?) to the nearest coin, crate, opponent — found via
   BFS through walkable tiles, normalized by board size (17). (3x3 = 9 floats)

This is deliberately compact (not a CNN over the raw 17x17 grid) so the
network trains fast and doesn't need huge amounts of data. Danger is a
continuous countdown (based on each bomb's remaining timer), not just a
binary flag, so the network can learn "I have N steps to get clear" rather
than only "this tile is/isn't dangerous".

## Install
```
pip install torch numpy --break-system-packages   # or in a venv, drop the flag
```

## Train

### Recommended: curriculum pipeline
```
python dqn_train_pipeline.py          # resumes an existing model if present
python dqn_train_pipeline.py --fresh  # wipes everything, starts from scratch
```
This runs staged bursts (`empty` → `coin-heaven` → `loot-crate` → `classic`),
re-evaluating with a greedy (no-exploration) match after every burst and
automatically discarding any burst that made the agent worse — mirroring
the same pattern already used by `train_pipeline.py`/`train_more.py` for
`linear_q_agent`. Best checkpoints are kept in `dqn-model.best.pt` /
`dqn-train-state.best.json` inside this folder.

### Manual / single command
```
python main.py play --my-agent dqn_agent --train 1 --n-rounds 5000 --no-gui
```
Training state (epsilon, rounds played) is saved to `dqn-train-state.json`
after every round and reloaded on the next run, so repeated invocations of
this command correctly continue epsilon decay instead of resetting it —
important if you're doing your own multi-session training outside the
pipeline script above.

To force a specific exploration rate for the *next* run only (what the
pipeline uses internally to bump exploration when moving to a harder
scenario), drop a number into `dqn-epsilon-override.txt` next to
`dqn-model.pt` before running; it's consumed and deleted automatically.

The model is checkpointed to `dqn-model.pt` after every round, so training
can be interrupted and resumed (just rerun with `--train 1`; it'll load the
existing weights first).

## Play (no exploration, no training)
```
python main.py play --my-agent dqn_agent --n-rounds 10
```

## Tuning knobs (top of train.py)
- `GAMMA`, `LR`, `BATCH_SIZE`, `BUFFER_SIZE`
- `EPSILON_START` / `EPSILON_END` / `EPSILON_DECAY_ROUNDS`
- `TARGET_UPDATE_EVERY`
- The reward dict in `reward_from_events` — this is the biggest lever for
  shaping behavior (e.g. raise `CRATE_DESTROYED` once coin-collecting works
  well, to push the agent toward bombing).

## Note on the danger map
The danger map in `state_to_features` stops a bomb's blast line only at
walls (`field == -1`), continuing through crates — this matches the real
game rule in `items.py`'s `get_blast_coords` exactly (bombs blow through
crates, not through walls).

## Reward shaping beyond the built-in game events
On top of the standard events (coin collected, kills, deaths, etc.),
`train.py` adds:
- **Coin-seeking bonus**: small +/- reward for moving closer to / further
  from the nearest coin, so there's useful signal before the agent ever
  stumbles into one by chance.
- **Danger-fleeing shaping**: a real failure mode we saw in testing was
  the agent dropping a bomb, then repeatedly trying (and failing) to drop
  another one instead of moving away, until its own bomb killed it.
  Standing still (`WAIT` or an unusable `BOMB`) while on a dangerous tile
  now costs -1.0, while actually moving off a dangerous tile earns +0.3.
- **Useless-bomb-spam penalty**: choosing `BOMB` when it's invalid (no
  bomb available) gets an extra -0.3 on top of the normal invalid-action
  penalty, on top of the danger-fleeing penalty if applicable.

## Safety patches (callbacks.py) + Double DQN (train.py)
A 200-round evaluation against `rule_based_agent` showed `dqn_agent`
self-eliminating in ~94-95% of rounds. Several deterministic rules were
added on top of the learned policy to address this directly:
- **Reactive shield** (`_apply_safety_shield`): if standing on a
  dangerous tile, forces a step toward a genuinely escapable neighbor --
  it BFS-verifies that a candidate direction actually leads to safety
  within the remaining time (not just that it *looks* locally safer right
  now), reusing the same reachability logic as the proactive bomb check.
  A `game.log` trace of a real death showed the agent fleeing DOWN, DOWN,
  then reversing UP (back toward the bomb) before dying; a second trace
  after fixing that showed the same pattern recurring because the fix
  only checked the *immediate* neighbor's danger, not whether it led
  anywhere -- a locally-safer-looking tile turned out to dead-end a few
  steps later on the real, crate-filled board.
- **Proactive bomb check** (`_apply_bomb_safety_check`): before allowing
  a bomb drop, BFS-checks whether a tile safe from the new bomb *and* any
  already-ticking bombs is reachable within `BOMB_TIMER` steps. Its
  fallback (when blocking a bomb) is restricted to legal moves only, so
  it can't itself cause an `INVALID_ACTION`.

`train.py` also switched to **Double DQN** (online net selects the next
action, target net evaluates it) to reduce Q-value overestimation bias.

**Honest history of testing these** (worth reading before trusting any
single number from a run of this agent):
- An early unseeded A/B suggested a large (~20%) improvement from the
  shield. That mostly turned out to be noise from the other agents' own
  internal randomness, not a real effect -- confirmed once the comparison
  was redone with a fixed `--seed`, which showed almost no difference.
- A `game.log` trace of real deaths showed the dominant failure was
  *oscillation* (fleeing, then reversing), which the original shield
  didn't cover, and which drove the near-null result above.
- After strengthening the shield to catch this (checking direction
  quality, not just "is this a move"), a single-seed comparison looked
  *worse* (62 vs 56 suicides/100). Aggregating across 3 seeds (300 rounds
  total per config) flipped that to a real improvement (suicide rate
  0.653 -> 0.590/round).
- A *second* `game.log` trace, on the strengthened version, showed the
  same oscillation pattern still occurring: fleeing correctly, then
  reversing, dying anyway. Replaying the exact real sequence through the
  shield showed why -- it only checked which of the 4 *immediate*
  neighbor tiles looked locally safer, not whether that tile actually
  led anywhere. On an open board that's usually fine; on a real,
  crate-filled board the "locally safer" tile can dead-end a few steps
  later. The shield now does a proper BFS from each candidate tile to
  verify a real escape route exists (reusing the same machinery as the
  proactive bomb check), preferring a *provably* escapable direction over
  a merely-locally-safer one.
- A 2-seed, same-model comparison of BFS-aware vs. the old 1-step version
  showed a large, consistent improvement (suicide rate 0.985 -> 0.760,
  avg steps 31.2 -> 139.4, score 479 -> 1642) -- by far the biggest gain
  of any single change so far, and it directly closes the gap the log
  evidence pointed to. **Single-seed, or even small unseeded,
  comparisons on this agent are not reliable -- use `--seed` and average
  multiple runs before trusting a result.**

**Takeaway**: the pattern that worked throughout was: get a real
`game.log`, trace an actual death, find the specific mechanism, then fix
exactly that mechanism and verify with a seeded, controlled test --
not guessing at plausible-sounding fixes and hoping. Even with the
BFS-aware shield, suicide rate is a real, measured 0.76/round in this
test, not zero -- there's still headroom, and the same recipe (get a
fresh log, find what's still killing it, fix that specific thing) is the
way to keep closing the gap. Further large gains beyond that still need
the learned policy itself to improve: more training rounds, and/or
smoother reward shaping (e.g. a danger penalty that scales continuously
with urgency at every step rather than a hard threshold).
