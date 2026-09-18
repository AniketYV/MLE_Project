import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

FEATURE_DIM = 19
MODEL_FILE = "dqn-model.pt"

# --- Compliance switch -----------------------------------------------------
# The course spec says the submitted agent's decisions must come from a
# learned model, not a hard-coded rule ("a feature that deterministically
# returns the action which results in the best move ... would disallow").
# The safety shield and pre-bomb escape check below ARE hard-coded rules --
# they override the network's chosen action in specific situations. This is
# closer to published "safe RL via action shielding" (Alshiekh et al. 2018)
# than to a disallowed hand-coded-answer feature, since the network still
# has to learn everything else and the override only fires in a narrow,
# safety-critical slice of states -- but it's a genuine judgment call, not
# an obviously-safe one. ASK ON DISCORD (#final-project-questions) whether
# this is acceptable before submitting. If told no, or if in doubt, set this
# to False for your tournament submission -- training can still use it
# (it only affects what data training sees, not what's disallowed at
# evaluation time) by leaving TRAIN_USES_SAFETY_PATCHES on its own switch.
USE_SAFETY_PATCHES_AT_INFERENCE = True
USE_SAFETY_PATCHES_DURING_TRAINING = True

# Directions in the same order used for the 4 "move" features below.
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT


class QNet(nn.Module):
    """A small 2-hidden-layer MLP. Cheap to train, plenty for a hand-crafted
    feature vector this size."""

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
    """Called once when the agent is loaded (both in play and train mode)."""
    self.device = torch.device("cpu")
    self.model = QNet().to(self.device)

    if os.path.isfile(MODEL_FILE):
        self.logger.info("Loading model from saved state.")
        self.model.load_state_dict(torch.load(MODEL_FILE, map_location=self.device))
    else:
        self.logger.info("No saved model found, initializing new model from scratch.")
    self.model.eval()

    # Epsilon is only used while self.train is True. train.py's
    # setup_training() will reset/override this with a proper schedule.
    if not hasattr(self, "epsilon"):
        self.epsilon = 0.05 if not self.train else 1.0

    # Tracked purely for reporting: how often do the safety patches actually
    # change the network's chosen action? Useful evidence for the "is this
    # still fundamentally a learned agent" discussion in your report.
    if not hasattr(self, "shield_interventions"):
        self.shield_interventions = 0
        self.total_decisions = 0


# Danger urgency (see state_to_features docstring) above which the safety
# shield below considers the agent to be standing somewhere lethal. Keep
# in sync with train.py's DANGER_THRESHOLD (same concept, used for reward
# shaping there vs a hard behavioral override here).
DANGER_THRESHOLD = 0.2


def act(self, game_state: dict) -> str:
    if game_state is None:
        return np.random.choice(ACTIONS)

    features = state_to_features(game_state)
    with torch.no_grad():
        state_t = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.model(state_t).squeeze(0).numpy()

    if self.train and np.random.rand() < self.epsilon:
        self.logger.debug("Choosing action purely at random (epsilon-greedy).")
        # Bias slightly away from bombing/waiting while exploring.
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        action_idx = int(np.argmax(q_values))
        action = ACTIONS[action_idx]
        self.logger.debug(f"Q-values: {q_values.round(2)} -> {action}")

    use_patches = (USE_SAFETY_PATCHES_DURING_TRAINING if self.train
                   else USE_SAFETY_PATCHES_AT_INFERENCE)

    self.total_decisions += 1
    if use_patches:
        shielded = _apply_safety_shield(action, features, q_values, game_state)
        shielded = _apply_bomb_safety_check(shielded, features, q_values, game_state)
        if shielded != action:
            self.shield_interventions += 1
        action = shielded

    if self.total_decisions % 1000 == 0:
        rate = self.shield_interventions / self.total_decisions
        self.logger.info(f"Safety-patch intervention rate so far: "
                          f"{self.shield_interventions}/{self.total_decisions} ({rate:.1%})")

    return action


def _can_reach_safety(start, field, blocked, danger_tiles, max_steps):
    """Generalized version of the BFS used by _can_escape_bomb: from
    `start`, is a tile outside `danger_tiles` reachable within `max_steps`,
    walking only through currently-free, unblocked tiles? Used both to
    proactively check a hypothetical new bomb (via _can_escape_bomb) and,
    now, to make the reactive shield look more than one step ahead."""
    if start not in danger_tiles:
        return True
    w, h = field.shape
    visited = {start}
    frontier = [start]
    for _ in range(max_steps):
        next_frontier = []
        for (x, y) in frontier:
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in visited or not (0 <= nx < w and 0 <= ny < h):
                    continue
                if field[nx, ny] != 0 or (nx, ny) in blocked:
                    continue
                visited.add((nx, ny))
                if (nx, ny) not in danger_tiles:
                    return True
                next_frontier.append((nx, ny))
        frontier = next_frontier
        if not frontier:
            break
    return False


def _apply_safety_shield(action, features, q_values, game_state):
    """A hard, non-learned override for standing in danger and not moving
    toward safety. Log analysis found the dominant real failure mode isn't
    freezing (WAIT/BOMB) -- it's OSCILLATING: e.g. two steps fleeing a
    bomb, then one step back toward it, netting too little distance to
    clear the blast in time.

    This has two layers, checked in order of how much they can verify:
    1. Prefer a candidate direction that's PROVABLY escapable -- BFS
       forward from that tile to confirm a real escape route exists, not
       just that it's momentarily less dangerous than its neighbors. A
       real game.log trace showed the original 1-step-lookahead version
       of this shield pick a locally-safer-looking tile that turned out
       to be a dead end on the real (crate-filled) board, still dying.
    2. If no candidate is provably escapable (e.g. genuinely cornered),
       fall back to the locally-least-dangerous tile as before -- better
       than nothing even if it can't be verified to fully work out.

    Q-values still break ties among equally-good directions in both
    layers, so the learned policy still matters.
    """
    here_danger = features[8]
    if here_danger <= DANGER_THRESHOLD:
        return action  # not in immediate danger, trust the learned policy

    move_free = features[0:4]
    move_danger = features[4:8]
    candidates = [i for i in range(4) if move_free[i] > 0.5]
    if not candidates:
        return action  # truly no legal escape tile; nothing we can do about it

    field = game_state['field']
    x, y = game_state['self'][3]
    others = {p for _, _, _, p in game_state['others']}
    bombs = game_state['bombs']
    blocked = others | {p for p, _ in bombs}

    danger_tiles = set()
    min_timer = s.BOMB_TIMER
    for (bx, by), timer in bombs:
        danger_tiles |= _blast_tiles((bx, by), field)
        min_timer = min(min_timer, timer)
    max_steps = max(min_timer, 1)

    neighbor_of = {0: (x, y - 1), 1: (x + 1, y), 2: (x, y + 1), 3: (x - 1, y)}
    escapable = [i for i in candidates
                 if _can_reach_safety(neighbor_of[i], field, blocked, danger_tiles, max_steps)]

    if escapable:
        if action in ACTION_TO_IDX and ACTION_TO_IDX[action] in escapable:
            return action  # already heading toward a provably-escapable tile
        best_i = max(escapable, key=lambda i: q_values[i])
        return ACTIONS[best_i]

    # Nothing provably escapes -- fall back to locally-least-dangerous.
    min_danger = min(move_danger[i] for i in candidates)
    safest = [i for i in candidates if move_danger[i] <= min_danger + 1e-6]
    if action in ACTION_TO_IDX and ACTION_TO_IDX[action] in safest:
        return action
    best_i = max(safest, key=lambda i: q_values[i]) if len(safest) > 1 else safest[0]
    return ACTIONS[best_i]


def _blast_tiles(pos, field):
    """Tiles that would be hit by a bomb dropped at pos, matching the real
    game rule: stops at walls, passes straight through crates (see
    items.py's get_blast_coords)."""
    x, y = pos
    w, h = field.shape
    tiles = {(x, y)}
    for dx_dir, dy_dir in DIRECTIONS:
        for i in range(1, s.BOMB_POWER + 1):
            sx, sy = x + dx_dir * i, y + dy_dir * i
            if not (0 <= sx < w and 0 <= sy < h) or field[sx, sy] == -1:
                break
            tiles.add((sx, sy))
    return tiles


def _can_escape_bomb(pos, field, blocked, extra_danger_tiles=frozenset()):
    """Can the agent reach a tile that is safe from EVERY currently
    relevant blast (the hypothetical new bomb at pos, plus any bombs
    already ticking on the board) within BOMB_TIMER steps, walking only
    through tiles that are currently free? `extra_danger_tiles` should be
    the union of blast patterns for any bombs already on the field --
    otherwise a route that dodges the new bomb but walks straight into an
    existing one would incorrectly be called safe."""
    blast = _blast_tiles(pos, field) | extra_danger_tiles
    w, h = field.shape
    visited = {pos}
    frontier = [pos]
    for _ in range(s.BOMB_TIMER):
        next_frontier = []
        for (x, y) in frontier:
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in visited or not (0 <= nx < w and 0 <= ny < h):
                    continue
                if field[nx, ny] != 0 or (nx, ny) in blocked:
                    continue
                visited.add((nx, ny))
                if (nx, ny) not in blast:
                    return True
                next_frontier.append((nx, ny))
        frontier = next_frontier
        if not frontier:
            break
    return False


def _apply_bomb_safety_check(action, features, q_values, game_state):
    """A second, proactive safety rule: don't drop a bomb unless there's an
    actual escape route. The reactive shield above only kicks in once the
    agent is already standing in danger -- but if dropping a bomb here
    leaves no tile reachable outside the blast within BOMB_TIMER steps
    (e.g. a dead end, or a spot too close to a wall), reacting after the
    fact is already too late. This checks before the bomb is placed.
    Crucially, it also accounts for any bombs already ticking on the board
    (own or opponents'), not just the new one -- an escape route that
    dodges the new bomb but walks into an existing blast is not safe."""
    if action != 'BOMB' or features[9] <= 0.5:  # not bombing, or can't anyway
        return action

    field = game_state['field']
    pos = game_state['self'][3]
    blocked = {p for _, _, _, p in game_state['others']} | {p for p, _ in game_state['bombs']}

    existing_danger = set()
    for (bx, by), _timer in game_state['bombs']:
        existing_danger |= _blast_tiles((bx, by), field)

    if _can_escape_bomb(pos, field, blocked, existing_danger):
        return action  # safe to bomb, keep the network's choice

    # No safe escape: fall back to the network's best action that is
    # actually legal (WAIT, or a move into a currently-free tile). Blindly
    # taking argmax over all non-BOMB actions risked repeatedly picking a
    # move into a wall/crate/other agent -- still blocked by the real game
    # engine as INVALID_ACTION, just via a different action. Restricting
    # the candidate set to what's genuinely walkable fixes that.
    move_free = features[0:4]  # UP, RIGHT, DOWN, LEFT
    legal_indices = [i for i in range(4) if move_free[i] > 0.5] + [4]  # + WAIT
    fallback_idx = max(legal_indices, key=lambda i: q_values[i])
    return ACTIONS[fallback_idx]


def _bfs_nearest(start, targets, free_mask):
    """Shortest-path (dx, dy, dist) from start to the closest target,
    walking through tiles where free_mask is True. Targets themselves
    (e.g. crates, which are NOT free) may always be entered as the final
    step, so this correctly finds the nearest crate/coin/opponent even
    though the target tile itself isn't walkable. Returns
    (None, None, None) if no target is reachable or there are none."""
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
    """Turns the game_state dict into a fixed-size feature vector.

    Layout (19 floats):
      [0:4]   can I step {UP, RIGHT, DOWN, LEFT}? (1 = free, 0 = blocked)
      [4:8]   danger URGENCY of that tile: 0 = safe, ramps up toward 1 as
              a bomb affecting it gets closer to exploding, 1 = live explosion
      [8]     danger urgency of my current tile right now
      [9]     can I drop a bomb?
      [10:13] (dx, dy, reachable?) to the nearest coin, dx/dy normalized
      [13:16] (dx, dy, reachable?) to the nearest crate
      [16:19] (dx, dy, reachable?) to the nearest opponent
    """
    if game_state is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    field = game_state['field']  # -1 wall, 0 free, 1 crate
    _, _, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']  # [((bx, by), timer), ...]
    others = game_state['others']  # [(name, score, bombs_left, (ox, oy)), ...]
    coins = game_state['coins']  # [(cx, cy), ...]
    explosion_map = game_state['explosion_map']

    w, h = field.shape

    # --- Danger urgency map: 0 (safe) to 1 (about to / already exploding).
    # A live explosion is maximum urgency. A ticking bomb's urgency rises
    # as its timer counts down, so the network can learn "I have N steps
    # left to get clear" rather than just a flat yes/no danger flag.
    danger = np.array(explosion_map > 0, dtype=np.float32)  # already 0/1
    for (bx, by), timer in bombs:
        urgency = 1.0 - (timer / s.BOMB_TIMER)
        urgency = float(np.clip(urgency, 0.0, 1.0))
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
