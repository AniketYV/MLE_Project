"""
Actor-critic network for the PPO agent.

A small CNN reads the spatial board encoding (see observation.py) plus a
few scalar extras, and produces both:
  - a POLICY: a probability distribution over the 6 possible actions
  - a VALUE estimate: how good the current state is expected to be

This is a genuinely different kind of function approximator from the
hand-engineered linear model used elsewhere in this project -- the
network learns its own internal representations from the raw spatial
layout, rather than being handed pre-computed directional features.

Two stride-2 convolutions downsample the 17x17 board (17 -> 9 -> 5)
before flattening, keeping the first fully-connected layer's parameter
count reasonable and inference fast enough to comfortably clear the
tournament's per-step time limit on CPU.
"""
import torch
import torch.nn as nn

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


class ActorCritic(nn.Module):
    def __init__(self, n_channels=8, board_size=17, n_scalars=8, n_actions=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 17 -> 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 9 -> 5
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, board_size, board_size)
            conv_out_size = self.conv(dummy).flatten(start_dim=1).shape[1]

        self.trunk = nn.Sequential(
            nn.Linear(conv_out_size + n_scalars, 256),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(256, n_actions)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, grid, scalars):
        """grid: (B, C, H, W) float tensor. scalars: (B, S) float tensor.
        Returns (logits, value): logits (B, n_actions), value (B,)."""
        x = self.conv(grid)
        x = x.flatten(start_dim=1)
        x = torch.cat([x, scalars], dim=1)
        x = self.trunk(x)
        logits = self.actor_head(x)
        value = self.critic_head(x).squeeze(-1)
        return logits, value
