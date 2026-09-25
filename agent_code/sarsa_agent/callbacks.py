"""
sarsa_agent: same feature representation and network architecture as
dqn_agent, but trained with on-policy SARSA instead of off-policy Q-learning
(Double DQN). Built as a genuinely separate model for comparison, not a
copy-paste duplicate to inflate the "two models" requirement -- the
learning algorithm itself is different (see train.py).

Kept self-contained (not importing from agent_code/dqn_agent/) since each
agent_code/<name>/ folder must stand alone for the tournament.

Pure RL: no hard-coded action-overriding rules anywhere in this agent.
"""
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

FEATURE_DIM = 19
MODEL_FILE = "sarsa-model.pt"

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT


class QNet(nn.Module):
    """Same small 2-hidden-layer MLP as dqn_agent, for a fair comparison
    that isolates the learning algorithm as the actual variable."""

    def __init__(self, in_dim=FEATURE_DIM, out_dim=len(ACTIONS), hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def setup(self):
    self.device = torch.device("cpu")
    self.model = QNet().to(self.device)

    if os.path.isfile(MODEL_FILE):
        self.logger.info("Loading model from saved state.")
        self.model.load_state_dict(torch.load(MODEL_FILE, map_location=self.device))
    else:
        self.logger.info("No saved model found, initializing new model from scratch.")
    self.model.eval()

    if not hasattr(self, "epsilon"):
        self.epsilon = 0.05 if not self.train else 1.0


def act(self, game_state: dict) -> str:
    if game_state is None:
        return np.random.choice(ACTIONS)

    features = state_to_features(game_state)
    with torch.no_grad():
        state_t = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.model(state_t).squeeze(0).numpy()

    if self.train and np.random.rand() < self.epsilon:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    action_idx = int(np.argmax(q_values))
    return ACTIONS[action_idx]


def _bfs_nearest(start, targets, free_mask):
    if not targets:
        return None, None, None
    target_set = set(targets)
    w, h = free_mask.shape
    visited = np.zeros_like(free_mask, dtype=bool)
    visited[start] = True
    queue = deque([(start, 0)])
    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) in target_set:
            return x - start[0], y - start[1], dist
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nx < w and 0 <= ny < h) or visited[nx, ny]:
                continue
            if free_mask[nx, ny] or (nx, ny) in target_set:
                visited[nx, ny] = True
                queue.append(((nx, ny), dist + 1))
    return None, None, None


def state_to_features(game_state: dict) -> np.array:
    """Identical feature layout to dqn_agent -- see that agent's docstring
    for the full field-by-field description. Kept identical so any
    performance difference reflects the learning algorithm, not the inputs.
    """
    if game_state is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    field = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    others = game_state['others']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']

    w, h = field.shape

    danger = np.array(explosion_map > 0, dtype=np.float32)
    for (bx, by), timer in bombs:
        urgency = float(np.clip(1.0 - (timer / s.BOMB_TIMER), 0.0, 1.0))
        danger[bx, by] = max(danger[bx, by], urgency)
        for dx_dir, dy_dir in DIRECTIONS:
            for i in range(1, s.BOMB_POWER + 1):
                sx, sy = bx + dx_dir * i, by + dy_dir * i
                if not (0 <= sx < w and 0 <= sy < h) or field[sx, sy] == -1:
                    break
                danger[sx, sy] = max(danger[sx, sy], urgency)

    occupied = {pos for _, _, _, pos in others} | {pos for pos, _ in bombs}

    def is_free(px, py):
        if not (0 <= px < w and 0 <= py < h):
            return False
        return field[px, py] == 0 and (px, py) not in occupied

    move_free, move_danger = [], []
    for dx_dir, dy_dir in DIRECTIONS:
        nx, ny = x + dx_dir, y + dy_dir
        move_free.append(1.0 if is_free(nx, ny) else 0.0)
        in_bounds = 0 <= nx < w and 0 <= ny < h
        move_danger.append(float(danger[nx, ny]) if in_bounds else 1.0)

    here_danger = float(danger[x, y])
    bomb_available = 1.0 if bombs_left > 0 else 0.0

    free_mask = (field == 0)
    norm = float(max(w, h))

    def relative(targets):
        dx, dy, _dist = _bfs_nearest((x, y), targets, free_mask)
        reachable = 1.0 if dx is not None else 0.0
        return (dx or 0) / norm, (dy or 0) / norm, reachable

    coin_dx, coin_dy, has_coin = relative(coins)
    crate_positions = list(zip(*np.where(field == 1)))
    crate_dx, crate_dy, has_crate = relative(crate_positions)
    opp_positions = [pos for _, _, _, pos in others]
    opp_dx, opp_dy, has_opp = relative(opp_positions)

    features = np.array([
        *move_free,
        *move_danger,
        here_danger,
        bomb_available,
        coin_dx, coin_dy, has_coin,
        crate_dx, crate_dy, has_crate,
        opp_dx, opp_dy, has_opp,
    ], dtype=np.float32)

    assert features.shape[0] == FEATURE_DIM
    return features
