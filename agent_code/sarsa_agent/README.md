# sarsa_agent

A second, genuinely distinct model from `dqn_agent` -- same features, same
network architecture, but trained with **on-policy SARSA** instead of
off-policy Double DQN. Built as a completely separate agent folder so it
carries zero risk to your validated `dqn_agent` submission.

## Why this comparison is actually meaningful (not just "a second model")
`dqn_agent`'s development hit a real problem: its safety shield intervened
before dangerous actions were ever executed, so the network's own Q-values
for "what to do near a bomb" never got directly punished or rewarded --
they were guided around instead of learned. Removing the shield for the
"pure RL" requirement revealed this directly: the raw network died almost
instantly without it.

SARSA is on-policy: its update target is `r + gamma * Q(s', a')` using the
action the policy **actually takes** next, not the best possible one like
DQN's `max_a' Q(s', a')`. There's no shield here at all -- the agent learns
the value of whatever it actually does, mistakes included, from the start.

## What we actually found (after ~2000 rounds total: 1000 coin-heaven + 1000 loot-crate)
- **coin-heaven**: 101 coins over 30 rounds, 1.08% invalid-action rate --
  learns clean, effective navigation quickly.
- **loot-crate**: **zero suicides**, but also **zero crates destroyed**.
  It converged toward "stay safe and collect coins" well before it
  discovered "bombing crates is worth the risk." This is a genuinely
  different learning trajectory than dqn_agent's -- DQN's replay buffer,
  drawing from a wide pool of past experience including plenty of
  bombing attempts, pushed it toward attempting (and initially dying
  badly at) bombing much sooner. SARSA's single-sample, on-policy updates
  are more conservative by construction.

This is legitimate, interesting material for a report: DQN reached higher
peak scores and more aggressive play, at the cost of a much harder,
longer struggle with self-destruction; SARSA reached safety faster, at
the cost of slower discovery of the crate-bombing reward. Neither is
"better" in the abstract -- they represent a real, textbook
exploration/risk tradeoff between off-policy and on-policy learning.

## Implementation note: the one-step-delayed update
SARSA's target needs `a_{t+1}` -- the action taken from the *next* state --
which isn't known until the next call into this module. `train.py` handles
this with a single pending-transition buffer (`self.pending`), completed
one call later once `a_{t+1}` is known (see the module docstring in
`train.py` for the exact call-by-call reasoning). No large replay buffer
is used deliberately: replaying old transitions from an earlier, different
policy would contradict SARSA's on-policy premise. Updates are single-
sample and online, not batched -- a genuine algorithmic difference from
`dqn_agent`, not an oversight.

## Train
```
python main.py play --my-agent sarsa_agent --train 1 --n-rounds 800 --no-gui --scenario coin-heaven
python main.py play --my-agent sarsa_agent --train 1 --n-rounds 1500 --no-gui --scenario loot-crate
python main.py play --my-agent sarsa_agent --train 1 --n-rounds 1500 --no-gui --scenario classic
```
Given the results above, if you want it to learn active bombing (not just
coin-collection), give it substantially more `loot-crate`/`classic` rounds
than we tested with here -- this was a quick, illustrative training run,
not a fully matured one.

## Play
```
python main.py play --my-agent sarsa_agent --n-rounds 10
```

## Pure RL
No hard-coded action-overriding rules anywhere in this agent -- every
decision is `argmax(Q)` (or epsilon-random during training), full stop.
