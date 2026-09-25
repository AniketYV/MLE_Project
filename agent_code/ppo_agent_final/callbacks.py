"""
ppo_agent / callbacks.py

PPO actor-critic for bomberman_rl with a hard SAFETY SHIELD.

The shield is what makes this agent hard to beat: the policy can only choose
actions that (a) are legal, (b) do not step into a blast, and (c) leave a
guaranteed escape route in the (x, y, time) space. Bombs are only allowed if a
crate or an opponent is inside the blast AND the agent can escape its own bomb.
=> ~zero suicides, PPO only has to learn WHERE to go and WHEN to bomb.
"""
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT (same order as ACTIONS[:4])

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, 'my-saved-model.pt')

# a bomb with observed timer t is lethal on step-indices t .. t+EXPLOSION_TIMER-1
HORIZON = s.BOMB_TIMER + s.EXPLOSION_TIMER + 1
N_CHANNELS = 11


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        side = (s.COLS + 1) // 2
        self.body = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * side * side, 256), nn.ReLU(),
        )
        self.pi = nn.Linear(256, len(ACTIONS))
        self.v = nn.Linear(256, 1)
        for m in self.body:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, 2 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, x, mask=None):
        h = self.body(x)
        logits = self.pi(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# Danger / safety computation
# --------------------------------------------------------------------------- #
def blast_tiles(field, x, y):
    tiles = [(x, y)]
    for dx, dy in MOVES:
        for i in range(1, s.BOMB_POWER + 1):
            nx, ny = x + dx * i, y + dy * i
            if field[nx, ny] == -1:  # only stone walls stop a blast, crates do not
                break
            tiles.append((nx, ny))
    return tiles


def lethal_map(state, extra_bomb=None):
    """L[k, x, y] == True  <=>  standing on (x, y) after the k-th upcoming step is fatal."""
    field = state['field']
    L = np.zeros((HORIZON,) + field.shape, dtype=bool)
    L[0] |= state['explosion_map'] > 0
    bombs = [((bx, by), t) for (bx, by), t in state['bombs']]
    if extra_bomb is not None:
        bombs.append(extra_bomb)
    for (bx, by), t in bombs:
        tiles = blast_tiles(field, bx, by)
        for k in range(t, t + s.EXPLOSION_TIMER):
            if 0 <= k < HORIZON:
                for (x, y) in tiles:
                    L[k, x, y] = True
    return L


def opponent_distance(field, free, others):
    """Steps an opponent needs to reach each tile (multi-source BFS). Used to not plan escapes through them."""
    dist = np.full(field.shape, 99, dtype=np.int32)
    q = deque()
    for (ox, oy) in others:
        dist[ox, oy] = 0
        q.append((ox, oy))
    while q:
        cx, cy = q.popleft()
        for dx, dy in MOVES:
            nx, ny = cx + dx, cy + dy
            if free[nx, ny] and dist[nx, ny] == 99:
                dist[nx, ny] = dist[cx, cy] + 1
                q.append((nx, ny))
    return dist


def can_survive(start, k_start, L, free, odist):
    """Is there any move sequence from `start` (positions from step k_start on) that avoids all blasts?
    Tiles an opponent can reach no later than we can (ties are decided randomly by the game) count as blocked.
    odist=None disables that (used when no bomb threatens us anyway)."""
    frontier = {start}
    for k in range(k_start, HORIZON):
        nxt = set()
        for (x, y) in frontier:
            for dx, dy in ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = x + dx, y + dy
                if dx or dy:
                    if not free[nx, ny] or (odist is not None and odist[nx, ny] <= k + 1):
                        continue
                if not L[k, nx, ny]:
                    nxt.add((nx, ny))
        if not nxt:
            return False
        frontier = nxt
    return True


def action_mask(state, L):
    field = state['field']
    x, y = state['self'][3]
    bombs_left = state['self'][2]
    others = {tuple(o[3]) for o in state['others']}

    free = field == 0
    for (bx, by), _ in state['bombs']:
        free[bx, by] = False

    odist = opponent_distance(field, free, others)
    odist_move = odist if L.any() else None  # only plan around opponents when a blast is looming
    valid = np.zeros(len(ACTIONS), dtype=bool)
    safe = np.zeros(len(ACTIONS), dtype=bool)

    for i, (dx, dy) in enumerate(MOVES):
        nx, ny = x + dx, y + dy
        if free[nx, ny] and (nx, ny) not in others:
            valid[i] = True
            contested = odist_move is not None and odist[nx, ny] <= 1
            safe[i] = (not L[0, nx, ny]) and not contested and can_survive((nx, ny), 1, L, free, odist_move)

    valid[4] = True
    safe[4] = (not L[0, x, y]) and can_survive((x, y), 1, L, free, odist_move)

    if bombs_left:
        blast = blast_tiles(field, x, y)
        useful = any(field[tx, ty] == 1 or (tx, ty) in others for tx, ty in blast)
        if useful:
            valid[5] = True
            L2 = lethal_map(state, extra_bomb=((x, y), s.BOMB_TIMER))
            free2 = free.copy()
            free2[x, y] = False
            safe[5] = (not L2[0, x, y]) and can_survive((x, y), 1, L2, free2, odist)

    mask = valid & safe
    if not mask.any():  # no perfectly safe action: at least avoid immediate death
        mask = valid.copy()
        mask[5] = False
        for i, (dx, dy) in enumerate(MOVES):
            if mask[i] and L[0, x + dx, y + dy]:
                mask[i] = False
        if L[0, x, y]:
            mask[4] = False
        if not mask.any():
            mask[:] = True
    return mask


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def encode(state):
    """-> (features float32 [C, W, H], action mask bool [6])"""
    field = state['field']
    x, y = state['self'][3]
    L = lethal_map(state)

    f = np.zeros((N_CHANNELS,) + field.shape, dtype=np.float32)
    f[0] = field == -1
    f[1] = field == 1
    for cx, cy in state['coins']:
        f[2, cx, cy] = 1.0
    for (bx, by), t in state['bombs']:
        f[3, bx, by] = (s.BOMB_TIMER - t) / s.BOMB_TIMER
    f[4:8] = L[:4]
    f[8, x, y] = 1.0
    for o in state['others']:
        f[9, o[3][0], o[3][1]] = 1.0
    f[10] = float(state['self'][2])
    return f, action_mask(state, L)


def target_distance(state):
    """BFS distance to the most relevant goal: nearest coin > nearest crate > nearest opponent (used for reward shaping)."""
    field = state['field']
    start = tuple(state['self'][3])
    coins = {tuple(c) for c in state['coins']}
    opps = {tuple(o[3]) for o in state['others']}
    dist = {start: 0}
    q = deque([start])
    best_coin = best_crate = best_opp = None
    while q:
        cur = q.popleft()
        d = dist[cur]
        for dx, dy in MOVES:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in dist or field[nxt] == -1:
                continue
            dist[nxt] = d + 1
            if field[nxt] == 1:
                if best_crate is None:
                    best_crate = d + 1
                continue
            if best_coin is None and nxt in coins:
                best_coin = d + 1
            if best_opp is None and nxt in opps:
                best_opp = d + 1
            q.append(nxt)
    for v in (best_coin, best_crate, best_opp):
        if v is not None:
            return v
    return 0


# --------------------------------------------------------------------------- #
# Framework callbacks
# --------------------------------------------------------------------------- #
def setup(self):
    torch.set_num_threads(1)
    self.net = ActorCritic()
    self._roll = None
    if os.path.isfile(MODEL_PATH):
        try:
            self.net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            self.logger.info('Loaded PPO weights.')
        except Exception as err:  # stale checkpoint (architecture / feature change)
            print(f'[ppo_agent] could not load {MODEL_PATH}: {err}\n[ppo_agent] -> starting from scratch')
    else:
        self.logger.info('No checkpoint found, starting from scratch.')
    self.net.eval()


def act(self, game_state: dict) -> str:
    feats, mask = encode(game_state)
    x = torch.from_numpy(feats).unsqueeze(0)
    m = torch.from_numpy(mask).unsqueeze(0)
    with torch.no_grad():
        logits, value = self.net(x, m)
    if self.train:
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        self._roll = (feats, mask, int(a), float(dist.log_prob(a)), float(value))
        return ACTIONS[int(a)]
    return ACTIONS[int(logits.argmax(dim=1))]  # tournament: greedy
