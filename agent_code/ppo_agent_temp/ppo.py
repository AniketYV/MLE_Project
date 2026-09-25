"""
Core PPO (Proximal Policy Optimization) algorithm.

This file is entirely game-agnostic -- it doesn't know anything about
Bomberman, only about training an actor-critic network from a buffer of
(state, action, reward, value, log_prob, done) transitions. That
separation mirrors the actual PPO paper (Schulman et al., 2017): the
algorithm is generic, only the observation encoding and reward function
(see observation.py and train.py) are specific to this game.

Three pieces:
  1. RolloutBuffer -- stores a batch of recent transitions.
  2. compute_gae -- Generalized Advantage Estimation (Schulman et al.,
     2016), a way of estimating "how much better was this action than
     average" that balances bias and variance better than either raw
     returns or a single-step TD error alone.
  3. ppo_update -- the clipped surrogate objective that gives PPO its
     name: it caps how far a single update is allowed to shift the
     policy, no matter how extreme the batch of experience that
     triggered it. This is the mechanism most directly aimed at the
     kind of training instability seen elsewhere in this project when
     updating live, online, near a dangerous opponent.
"""
import numpy as np
import torch
import torch.nn.functional as F

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.1   # tightened from 0.2 -- caps how far any single gradient
                 # step can shift the policy's probability ratio
VALUE_COEF = 0.5
ENTROPY_COEF = 0.05  # raised from 0.01 -- the lower value let the policy
                      # collapse into a fully deterministic "always WAIT"
                      # local optimum during classic training, at which
                      # point it stopped generating any exploratory
                      # experience needed to discover a better strategy.
                      # A stronger entropy bonus keeps enough randomness
                      # alive to escape traps like this.
PPO_EPOCHS = 2   # reduced from 4 -- an observed single "update" event (all
                 # epochs x minibatches together) flipped the policy's
                 # behaviour 100% of the time in one shot; fewer passes
                 # over the same rollout means less cumulative drift per
                 # update, even though each individual step is clipped
MINIBATCH_SIZE = 64
MAX_GRAD_NORM = 0.5


class RolloutBuffer:
    def __init__(self):
        self.grids = []
        self.scalars = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, grid, scalars, action, log_prob, value, reward, done):
        self.grids.append(grid)
        self.scalars.append(scalars)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.__init__()


def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Returns (advantages, returns), both the same length as rewards.

    `values` are the critic's estimate AT each step in the buffer.
    `last_value` is the critic's estimate for the state AFTER the last
    step in the buffer (bootstrapping the return for whatever wasn't
    observed yet, same idea as the bootstrapped TD target used
    elsewhere in this project -- just applied over a whole trajectory
    with an exponentially-weighted blend of different lookahead
    lengths, via `lam`, instead of a single one-step lookahead).
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(n)):
        next_non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def ppo_update(model, optimizer, buffer, last_value, device="cpu"):
    """
    Runs PPO_EPOCHS passes of minibatch updates over the collected
    rollout, then clears the buffer. Returns a dict of diagnostic stats
    (useful for confirming the update is behaving sensibly during
    training, e.g. entropy not collapsing to zero too fast).
    """
    n = len(buffer)
    advantages, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones, last_value)
    # Normalizing advantages is standard PPO practice: it keeps the scale
    # of the policy gradient consistent across batches with very
    # different raw reward magnitudes (sparse coin/kill rewards vs. dense
    # small penalties), instead of the update size being at the mercy of
    # whatever happened to be in this particular batch.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    grids = torch.tensor(np.array(buffer.grids), dtype=torch.float32, device=device)
    scalars = torch.tensor(np.array(buffer.scalars), dtype=torch.float32, device=device)
    actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    indices = np.arange(n)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}

    for _epoch in range(PPO_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, n, MINIBATCH_SIZE):
            mb_idx = indices[start:start + MINIBATCH_SIZE]
            if len(mb_idx) < 2:
                continue
            mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=device)

            logits, values_pred = model(grids[mb_idx_t], scalars[mb_idx_t])
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[mb_idx_t])
            entropy = dist.entropy().mean()

            # The ratio between the new and old policy's probability of
            # having taken the action that was actually taken. PPO's
            # defining trick: clip this ratio so a single update can
            # never push the policy arbitrarily far based on one batch,
            # regardless of how large the advantage estimate is.
            ratio = torch.exp(new_log_probs - old_log_probs[mb_idx_t])
            mb_adv = advantages_t[mb_idx_t]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_pred, returns_t[mb_idx_t])

            # Entropy bonus: rewards keeping the policy's action
            # distribution from collapsing into a single, rigid choice
            # too early, which would reduce exploration and risk the
            # kind of deterministic, exploitable behaviour that caused
            # real problems for other agents in this project.
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["updates"] += 1

    if stats["updates"] > 0:
        stats["policy_loss"] /= stats["updates"]
        stats["value_loss"] /= stats["updates"]
        stats["entropy"] /= stats["updates"]

    buffer.clear()
    return stats
