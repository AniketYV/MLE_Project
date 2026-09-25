"""
Value network (the "critic").

Input:  a state feature vector (shape: [batch, STATE_DIM])
Output: a single scalar V(s) — the estimated expected return from that state.

Kept deliberately separate from the policy network. If training instability
shows up, you can freeze/inspect/replace this network without touching the
policy at all, and vice versa.
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import STATE_DIM, HIDDEN_DIM, NUM_HIDDEN_LAYERS
except ImportError:
    from agent_code.ppo_agent.config import STATE_DIM, HIDDEN_DIM, NUM_HIDDEN_LAYERS


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM,
                 hidden_dim: int = HIDDEN_DIM, num_hidden_layers: int = NUM_HIDDEN_LAYERS):
        super().__init__()

        layers = [nn.Linear(state_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns V(s), shape [batch, 1]. Squeeze the last dim when consuming it."""
        return self.net(state)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning V(s) as shape [batch] (squeezed)."""
        return self.forward(state).squeeze(-1)


if __name__ == "__main__":
    # Quick sanity check — run this file directly to verify shapes work:
    #   python value_network/network.py
    net = ValueNetwork()
    dummy_state = torch.randn(1, STATE_DIM)
    value = net.get_value(dummy_state)
    print(f"Single state value shape: {value.shape}, value: {value.item():.4f}")

    batch_states = torch.randn(5, STATE_DIM)
    batch_values = net.get_value(batch_states)
    print(f"Batch value shape: {batch_values.shape}")
