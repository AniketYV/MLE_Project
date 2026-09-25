"""
Rollout buffer for PPO.

Unlike a DQN replay buffer, this is NOT persistent storage you sample
from repeatedly. It holds exactly ONE batch of experience collected by
the CURRENT policy. After one PPO update pass, you call reset() and
start collecting the next batch fresh. Using stale data here would
defeat the point of PPO's on-policy clipping — this is the #1 mistake
that produces silently-wrong PPO agents, so don't reuse a buffer across
iterations.

Usage pattern (Part 4 will wire this into an actual training loop):
    buffer = RolloutBuffer()
    for step in range(ROLLOUT_STEPS):
        ... run policy, get action/log_prob/value ...
        buffer.add(state, action, log_prob, reward, done, value)
    buffer.compute_returns_and_advantages(last_value)
    batches = buffer.get_minibatches(MINIBATCH_SIZE)
    buffer.reset()
"""

import numpy as np
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import GAMMA, GAE_LAMBDA
except ImportError:
    from agent_code.ppo_agent.config import GAMMA, GAE_LAMBDA


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.action_masks = []
        # Filled in by compute_returns_and_advantages():
        self.advantages = None
        self.returns = None

    def add(self, state, action, log_prob, reward, done, value, action_mask=None):
        """
        Call once per environment step.
        state:        np.ndarray, shape [STATE_DIM]
        action:       int
        log_prob:     float — log_prob of the action under the policy that chose it
        reward:       float
        done:         bool — True if this step ended the episode
        value:        float — V(state) estimate from the critic at the time of the step
        action_mask:  optional list/array of length ACTION_DIM (1=allowed, 0=masked).
                      MUST be the exact mask used when the action was sampled —
                      it gets reused during the PPO update for evaluate_actions(),
                      since the importance-sampling ratio requires the same
                      masking at both sampling and update time.
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.action_masks.append(action_mask)

    def __len__(self):
        return len(self.states)

    def compute_returns_and_advantages(self, last_value: float):
        """
        Computes GAE advantages and returns for every step in the buffer.
        Must be called once, after collection is finished, before any
        minibatches are drawn.

        last_value: V(s) for the state AFTER the final stored step
                    (bootstrap value — 0.0 if the episode ended there,
                    otherwise the critic's estimate of the state it left off in).
        """
        n = len(self)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + GAMMA * next_value * next_non_terminal - self.values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)

        self.advantages = advantages
        self.returns = returns

    def get_minibatches(self, minibatch_size: int):
        """
        Returns a list of minibatch dicts, each with torch tensors ready
        for the PPO update step. Advantages are normalized (standard PPO
        trick — stabilizes training a lot).

        Must be called after compute_returns_and_advantages().
        """
        if self.advantages is None:
            raise RuntimeError(
                "Call compute_returns_and_advantages() before get_minibatches()."
            )

        n = len(self)
        indices = np.arange(n)
        np.random.shuffle(indices)

        states = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        returns = torch.tensor(self.returns, dtype=torch.float32)

        advantages = self.advantages.copy()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # action_masks may be all-None (masking not used) — only build the
        # tensor if at least one real mask was ever stored, otherwise pass
        # None through so evaluate_actions() skips masking entirely.
        has_masks = any(m is not None for m in self.action_masks)
        if has_masks:
            action_masks = torch.tensor(np.array(self.action_masks), dtype=torch.float32)
        else:
            action_masks = None

        batches = []
        for start in range(0, n, minibatch_size):
            batch_idx = indices[start:start + minibatch_size]
            batch = {
                "states": states[batch_idx],
                "actions": actions[batch_idx],
                "old_log_probs": old_log_probs[batch_idx],
                "returns": returns[batch_idx],
                "advantages": advantages[batch_idx],
            }
            if action_masks is not None:
                batch["action_masks"] = action_masks[batch_idx]
            batches.append(batch)
        return batches


if __name__ == "__main__":
    # Sanity check with fake data — 10 fake steps, STATE_DIM doesn't
    # matter here since we're just testing the buffer's own logic.
    buf = RolloutBuffer()
    for i in range(10):
        fake_state = np.zeros(28, dtype=np.float32)
        fake_mask = [1, 1, 1, 1, 1, 0] if i % 3 == 0 else [1, 1, 1, 1, 1, 1]
        buf.add(
            state=fake_state,
            action=i % 6,
            log_prob=-1.2,
            reward=1.0 if i == 9 else 0.0,
            done=(i == 9),
            value=0.5,
            action_mask=fake_mask,
        )

    buf.compute_returns_and_advantages(last_value=0.0)
    print(f"Buffer length: {len(buf)}")
    print(f"Advantages: {buf.advantages}")
    print(f"Returns: {buf.returns}")

    batches = buf.get_minibatches(minibatch_size=4)
    print(f"Number of minibatches: {len(batches)}")
    print(f"First minibatch states shape: {batches[0]['states'].shape}")
    print(f"First minibatch has action_masks: {'action_masks' in batches[0]}")
    if "action_masks" in batches[0]:
        print(f"First minibatch action_masks shape: {batches[0]['action_masks'].shape}")
