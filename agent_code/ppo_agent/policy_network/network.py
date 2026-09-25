"""
Policy network (the "actor").

Input:  a state feature vector (shape: [batch, STATE_DIM])
Output: logits over ACTION_DIM discrete actions.

Kept deliberately separate from the value network so you can test,
save, load, and debug it in complete isolation.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import STATE_DIM, ACTION_DIM, HIDDEN_DIM, NUM_HIDDEN_LAYERS
except ImportError:
    from agent_code.ppo_agent.config import STATE_DIM, ACTION_DIM, HIDDEN_DIM, NUM_HIDDEN_LAYERS


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden_dim: int = HIDDEN_DIM, num_hidden_layers: int = NUM_HIDDEN_LAYERS):
        super().__init__()

        layers = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape [batch, action_dim]."""
        return self.net(state)

    def get_distribution(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> Categorical:
        """
        Returns a Categorical distribution over actions for the given state(s).
        action_mask: optional, shape [batch, action_dim], 1 = allowed, 0 = masked
        out. Masked actions get -inf logits BEFORE softmax, so they get exactly
        zero probability — a hard constraint, not just a learned discouragement.
        """
        logits = self.forward(state)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float("-inf"))
        return Categorical(logits=logits)

    def act(self, state: torch.Tensor, action_mask: torch.Tensor = None):
        """
        Sample an action from the policy.
        Returns: (action_index, log_prob_of_action)
        Used during rollout collection (Part 4).
        """
        dist = self.get_distribution(state, action_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor, action_masks: torch.Tensor = None):
        """
        Given a batch of states and the actions that were actually taken,
        return (log_probs, entropy) under the CURRENT policy.
        Used during the PPO update step (Part 6), where the policy has
        changed since the actions were originally sampled. action_masks
        MUST be the same masks used when the actions were originally
        sampled (same state -> same mask, since masking is deterministic
        from game_state) — otherwise the PPO importance-sampling ratio
        becomes invalid.
        """
        dist = self.get_distribution(states, action_masks)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


if __name__ == "__main__":
    # Quick sanity check — run this file directly to verify shapes work:
    #   python policy_network/network.py
    net = PolicyNetwork()
    dummy_state = torch.randn(1, STATE_DIM)
    action, log_prob = net.act(dummy_state)
    print(f"Sampled action index: {action.item()}, log_prob: {log_prob.item():.4f}")

    batch_states = torch.randn(5, STATE_DIM)
    batch_actions = torch.randint(0, ACTION_DIM, (5,))
    log_probs, entropy = net.evaluate_actions(batch_states, batch_actions)
    print(f"Batch log_probs shape: {log_probs.shape}, entropy shape: {entropy.shape}")

    # Masking sanity check: mask out the last action (BOMB, index 5) and
    # confirm it's NEVER sampled across many draws.
    mask = torch.ones(1, ACTION_DIM)
    mask[0, 5] = 0
    sampled = [net.act(dummy_state, action_mask=mask)[0].item() for _ in range(200)]
    assert 5 not in sampled, "Masked action was sampled — masking is broken!"
    print("Masking check passed: BOMB action never sampled when masked out.")
