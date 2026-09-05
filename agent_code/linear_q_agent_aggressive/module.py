"""
Shared game-model helpers and the linear Q-learning model for linear_q_agent.

This file has four jobs, in order:
  1. World-model helpers  -- reproduce the real engine's rules (blast radius,
     what counts as dangerous, what counts as walkable) exactly.
  2. Pathfinding (BFS)     -- "which direction is nearest coin/crate/safety".
  3. Feature extraction    -- turn a raw game_state into a fixed vector.
  4. The potential function + QModel + symmetry augmentation.

Kept separate from callbacks.py / train.py so both import the exact same
logic instead of risking two slightly-different copies drifting apart.
"""
from collections import deque

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Cardinal directions, in the SAME order as every one-hot direction slot
# used throughout this file: UP, RIGHT, DOWN, LEFT. This order also matches
# the first four entries of ACTIONS -- that alignment is what makes the
# symmetry-augmentation code near the bottom of this file simple.
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# Hardcoded to match settings.py (BOMB_POWER = 3). Not imported directly so
# this file has no dependency on the game's working-directory setup.
BOMB_POWER = 3


# ===========================================================================
# PART 1: World-model helpers
# ===========================================================================
# These functions answer "what does the real engine consider true" -- blast
# shape, danger, walkability. Every one of these was the site of a real bug
# at some point in this project (blast stopping at crates when it shouldn't;
# danger tiles treated as impassable walls when they're not). Treat this
# section as the most safety-critical code in the file: if it's wrong, the
# agent's whole sense of danger is built on a false premise, and training
# more just reinforces the wrong lesson faster.

def get_blast_coords(pos, field, power=BOMB_POWER):
    """
    Reproduces Bomb.get_blast_coords() from items.py EXACTLY: a blast is
    only stopped by a stone wall (-1). It passes straight through crates (1),
    destroying them, and continues to the full power range beyond them.
    Do not "fix" this to stop at crates -- that would no longer match the
    real engine and would make every danger calculation downstream wrong.
    """
    x, y = pos
    width, height = field.shape
    coords = [(x, y)]
    for dx, dy in DIRS:
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                break
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def compute_danger_tiles(game_state):
    """
    Tiles that are lethal right now (active explosion) or will become
    lethal once a currently-ticking bomb goes off. Deliberately
    conservative: marks a pending bomb's FULL eventual blast area as
    dangerous immediately, rather than reasoning precisely about "will I
    actually be here when it detonates". That precision could be added
    later, but the conservative version is the safe default -- better to
    occasionally treat a tile as more dangerous than it turns out to be
    than to walk confidently into a real blast.
    """
    field = game_state['field']
    danger = set()

    explosion_map = game_state['explosion_map']
    xs, ys = np.nonzero(explosion_map > 0)
    danger.update(zip(xs.tolist(), ys.tolist()))

    for bomb_pos, _timer in game_state['bombs']:
        danger.update(get_blast_coords(bomb_pos, field))

    return danger


def compute_opponent_threat_tiles(game_state):
    """
    Tiles that would become dangerous if an opponent who is currently ABLE
    to drop a bomb (not on cooldown) did so right now, from their current
    position. This is deliberately a softer signal than compute_danger_tiles
    -- it's a real bomb yet, so it gets its own, smaller potential penalty
    rather than being merged into actual danger. The point is purely to
    stop the agent from being caught flat-footed the instant an opponent
    who was already in bombing range actually pulls the trigger, which
    real danger_tiles has no way to anticipate since it only sees bombs
    that already exist.
    """
    field = game_state['field']
    threat = set()
    for (_name, _score, bomb_available, pos) in game_state['others']:
        if bomb_available:
            threat.update(get_blast_coords(pos, field))
    return threat


def walkable(pos, field, occupied):
    """A tile you could physically step onto right now: on the board, not a
    wall or crate, and not currently occupied by a bomb or another agent.
    Note this does NOT check danger -- danger tiles are still walkable,
    they're just risky. Conflating "blocked" with "dangerous" was exactly
    the escape-search bug from earlier in this project."""
    x, y = pos
    width, height = field.shape
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    if field[x, y] != 0:
        return False
    if pos in occupied:
        return False
    return True


# ===========================================================================
# PART 2: Pathfinding (breadth-first search)
# ===========================================================================
# BFS explores the board outward in rings of increasing distance, so it's
# the right tool whenever every step "costs" the same (true here -- moving
# one tile always costs exactly one step). The first matching tile BFS
# finds is guaranteed to be the closest one, no need to check further.
#
# Two variants, because we need two different answers to the same search:
#   - bfs_first_step: "which of my 4 immediate directions leads toward it"
#     (used to build one-hot direction features for the Q-function)
#   - bfs_distance:    "how many steps away is it"
#     (used by the potential function, which needs an actual number to
#     measure whether a move made things better or worse)

def bfs_first_step(start, is_target, field, occupied, extra_blocked=frozenset()):
    """
    Returns the index (0..3, matching DIRS = UP,RIGHT,DOWN,LEFT) of the
    first step of the shortest path from `start` to the nearest tile
    satisfying is_target(pos), or None if unreachable.

    If `start` itself already satisfies is_target, returns None -- there's
    nothing further to walk toward. (Omitting this check was a real bug:
    without it, an agent already standing next to a crate could "walk
    toward" a different equally-qualifying neighbouring tile forever,
    farming approach-reward without ever committing to bomb anything.)
    """
    if is_target(start):
        return None

    width, height = field.shape
    visited = {start}
    queue = deque()

    for dir_idx, (dx, dy) in enumerate(DIRS):
        nxt = (start[0] + dx, start[1] + dy)
        if nxt in extra_blocked or nxt in visited:
            continue
        if walkable(nxt, field, occupied):
            visited.add(nxt)
            queue.append((nxt, dir_idx))

    while queue:
        pos, first_dir = queue.popleft()
        if is_target(pos):
            return first_dir
        for dx, dy in DIRS:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in extra_blocked or nxt in visited:
                continue
            if 0 <= nxt[0] < width and 0 <= nxt[1] < height and walkable(nxt, field, occupied):
                visited.add(nxt)
                queue.append((nxt, first_dir))

    return None


def bfs_distance(start, is_target, field, occupied, extra_blocked=frozenset(), max_dist=40):
    """
    Same search as bfs_first_step, but returns the actual shortest-path
    DISTANCE instead of the first direction. Capped at max_dist so that
    "genuinely far away" and "unreachable" resolve to the same finite
    value -- a potential function built on this never sees an
    infinite/undefined jump, which would otherwise destabilize learning.
    """
    if is_target(start):
        return 0
    width, height = field.shape
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        pos, dist = queue.popleft()
        if dist >= max_dist:
            continue
        for dx, dy in DIRS:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in extra_blocked or nxt in visited:
                continue
            if 0 <= nxt[0] < width and 0 <= nxt[1] < height and walkable(nxt, field, occupied):
                if is_target(nxt):
                    return dist + 1
                visited.add(nxt)
                queue.append((nxt, dist + 1))
    return max_dist


def one_hot_direction(direction_idx):
    vec = [0.0, 0.0, 0.0, 0.0]
    if direction_idx is not None:
        vec[direction_idx] = 1.0
    return vec


# ===========================================================================
# PART 3: Feature extraction
# ===========================================================================
# Feature layout (29 total):
#  0      bias (always 1.0) -- lets the linear model learn a baseline
#         value for each action, not just a value relative to features
#  1      bomb action currently available
#  2      current tile is in danger right now
#  3-6    blocked[UP,RIGHT,DOWN,LEFT]      -- can't walk there at all
#  7-10   risky[UP,RIGHT,DOWN,LEFT]        -- walking there enters danger
#  11-14  one-hot: first step toward nearest coin
#  15-18  one-hot: first step toward a tile adjacent to nearest crate
#  19-22  one-hot: first step toward nearest safe (non-danger) tile
#  23-26  one-hot: first step toward a tile adjacent to nearest opponent
#  27     dropping a bomb here would hit a crate/opponent AND leaves an
#         escape route afterward
#  28     mobility: fraction (0-1) of the 4 adjacent tiles that are
#         currently safe to step onto -- a preemptive "how boxed in am I"
#         signal, checked before any bomb makes things overtly dangerous
#  29     currently standing where an ARMED opponent (bomb available) could
#         hit me if they dropped a bomb THIS instant -- anticipates a bomb
#         that doesn't exist yet, instead of only reacting once it does
N_FEATURES = 30


def _count_safe_neighbors(pos, field, occupied, danger_tiles):
    count = 0
    for dx, dy in DIRS:
        nxt = (pos[0] + dx, pos[1] + dy)
        if walkable(nxt, field, occupied) and nxt not in danger_tiles:
            count += 1
    return count


def state_to_features(game_state):
    """Converts the raw game_state dict into the fixed-length feature
    vector described in the layout comment above."""
    if game_state is None:
        return np.zeros(N_FEATURES, dtype=np.float32)

    field = game_state['field']
    width, height = field.shape
    _name, _score, bomb_available, (ax, ay) = game_state['self']
    pos = (ax, ay)

    coins = set(game_state['coins'])
    bombs = game_state['bombs']
    others = [p for (_n, _s, _b, p) in game_state['others']]
    others_set = set(others)

    occupied = set(others)
    for (bx, by), _t in bombs:
        occupied.add((bx, by))

    danger_tiles = compute_danger_tiles(game_state)
    in_danger_now = pos in danger_tiles

    features = np.zeros(N_FEATURES, dtype=np.float32)
    features[0] = 1.0  # bias
    features[1] = 1.0 if bomb_available else 0.0
    features[2] = 1.0 if in_danger_now else 0.0

    # 3-6 blocked, 7-10 risky-to-move-there
    for i, (dx, dy) in enumerate(DIRS):
        nxt = (ax + dx, ay + dy)
        is_walkable = walkable(nxt, field, occupied)
        features[3 + i] = 0.0 if is_walkable else 1.0
        features[7 + i] = 1.0 if (not is_walkable or nxt in danger_tiles) else 0.0

    # 11-14 direction to nearest coin
    dir_coin = bfs_first_step(pos, lambda p: p in coins, field, occupied) if coins else None
    features[11:15] = one_hot_direction(dir_coin)

    # 15-18 direction to a tile adjacent to the nearest crate
    def near_crate(p):
        x, y = p
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and field[nx, ny] == 1:
                return True
        return False
    dir_crate = bfs_first_step(pos, near_crate, field, occupied)
    features[15:19] = one_hot_direction(dir_crate)

    # 19-22 escape direction: nearest tile NOT in danger. Deliberately does
    # NOT block transit through danger_tiles (see walkable() note above) --
    # you often have to step through the edge of a blast zone to actually
    # get clear of it before detonation.
    dir_escape = bfs_first_step(pos, lambda p: p not in danger_tiles, field, occupied) if in_danger_now else None
    features[19:23] = one_hot_direction(dir_escape)

    # 23-26 direction to a tile adjacent to the nearest opponent
    def near_opponent(p):
        x, y = p
        for dx, dy in DIRS:
            if (x + dx, y + dy) in others_set:
                return True
        return False
    dir_opp = bfs_first_step(pos, near_opponent, field, occupied) if others_set else None
    features[23:27] = one_hot_direction(dir_opp)

    # 27 good bomb spot: hits a crate or opponent AND an escape route
    # remains afterward
    if bomb_available:
        blast = set(get_blast_coords(pos, field))
        hits_crate = any(
            0 <= x < width and 0 <= y < height and field[x, y] == 1
            for (x, y) in blast
        )
        hits_opponent = any(o in blast for o in others_set)
        if hits_crate or hits_opponent:
            hypothetical_danger = danger_tiles | blast
            escape_after = bfs_first_step(pos, lambda p: p not in hypothetical_danger, field, occupied)
            features[27] = 1.0 if escape_after is not None else 0.0

    # 28 mobility: how many of my 4 neighbours are safe right now
    features[28] = _count_safe_neighbors(pos, field, occupied, danger_tiles) / 4.0

    threat_tiles = compute_opponent_threat_tiles(game_state)
    features[29] = 1.0 if pos in threat_tiles else 0.0

    return features


# ===========================================================================
# PART 4: Potential function for potential-based reward shaping
# ===========================================================================
# F(s,a,s') = gamma*Phi(s') - Phi(s), for ANY potential function Phi,
# provably does not change the optimal policy (Ng, Harada & Russell 1999)
# -- it only changes how fast the agent finds it. This replaces earlier
# ad-hoc "reward moving toward X" shaping, which was exploitable: a fixed
# per-step bonus doesn't care whether you're making real progress or just
# oscillating, so the agent learned to farm it by wandering near crates
# instead of bombing them. A round trip nets zero potential-based shaping
# by construction, so that exploit is structurally impossible here.

POTENTIAL_MAX_DIST = 40
POTENTIAL_WEIGHTS = {"coin": 1.0, "opponent": 0.9, "crate": 0.3}  # AGGRESSIVE variant: opponent weight raised well above the original 0.6
POTENTIAL_DANGER_PENALTY = 15.0  # AGGRESSIVE variant: lowered from 25 -- still dominates single-step approach shaping, but tolerates more risk to actually engage
POTENTIAL_OPPONENT_THREAT_PENALTY = 8.0  # softer than real danger -- it's
                                          # a risk, not a certainty
POTENTIAL_MOBILITY_WEIGHT = 1.0  # softer, preemptive nudge away from
                                  # self-cornering, well below the danger
                                  # penalty since it's a precaution, not
                                  # an active emergency


def compute_potential(game_state):
    """Higher is better. Combines (negative) distance to the nearest coin,
    opponent, and crate -- each only contributing when that target type
    still exists -- plus a danger penalty and a mobility term."""
    if game_state is None:
        return 0.0

    field = game_state['field']
    width, height = field.shape
    _name, _score, _bomb_avail, (ax, ay) = game_state['self']
    pos = (ax, ay)

    coins = set(game_state['coins'])
    others = [p for (_n, _s, _b, p) in game_state['others']]
    others_set = set(others)

    occupied = set(others)
    for (bx, by), _t in game_state['bombs']:
        occupied.add((bx, by))

    phi = 0.0

    if coins:
        d = bfs_distance(pos, lambda p: p in coins, field, occupied, max_dist=POTENTIAL_MAX_DIST)
        phi -= POTENTIAL_WEIGHTS["coin"] * d

    if others_set:
        def near_opponent(p):
            x, y = p
            return any((x + dx, y + dy) in others_set for dx, dy in DIRS)
        d = bfs_distance(pos, near_opponent, field, occupied, max_dist=POTENTIAL_MAX_DIST)
        phi -= POTENTIAL_WEIGHTS["opponent"] * d

    if (field == 1).any():
        def near_crate(p):
            x, y = p
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and field[nx, ny] == 1:
                    return True
            return False
        d = bfs_distance(pos, near_crate, field, occupied, max_dist=POTENTIAL_MAX_DIST)
        phi -= POTENTIAL_WEIGHTS["crate"] * d

    danger_tiles = compute_danger_tiles(game_state)
    if pos in danger_tiles:
        phi -= POTENTIAL_DANGER_PENALTY

    threat_tiles = compute_opponent_threat_tiles(game_state)
    if pos in threat_tiles and pos not in danger_tiles:
        phi -= POTENTIAL_OPPONENT_THREAT_PENALTY

    safe_neighbors = _count_safe_neighbors(pos, field, occupied, danger_tiles)
    phi -= POTENTIAL_MOBILITY_WEIGHT * (4 - safe_neighbors)

    return phi


# ===========================================================================
# PART 5: Linear Q-learning model
# ===========================================================================
# One linear weight vector per action: Q(s, a) = w_a . phi(s).
#
# Why linear, not a neural network: with only 29 hand-designed features,
# a linear model is expressive enough (each feature is already a
# meaningful, pre-digested signal, not a raw pixel), trains in a fraction
# of a second per update, and is fast enough at inference to comfortably
# clear the tournament's 0.5s-per-step limit on CPU. The tradeoff is that
# it can only learn a WEIGHTED SUM of the features -- it can't discover
# interactions between them on its own the way a network could. That's
# exactly why the feature engineering in Part 3 has to do the heavy
# lifting: e.g. feature 27 (good_bomb_spot) hand-combines "hits a crate"
# AND "has an escape route" into one signal, because a linear model
# can't learn that AND relationship from the two facts separately.
#
# Trained with TD(0) semi-gradient Q-learning:
#     target = r + gamma * max_a' Q(s', a')      (bootstrapped estimate of
#                                                   total future reward)
#     w_a <- w_a + lr * (target - Q(s,a)) * phi(s)
#
# "Bootstrapped" is the key word: the target uses the model's OWN current
# estimate of the next state's value, not just the immediate reward. This
# is what lets credit propagate backward through time -- e.g. the reward
# for a coin revealed by a bomb 4 steps ago can still influence the value
# of the action that dropped that bomb, because the value estimate at each
# state already incorporates what tends to follow it.

class QModel:
    """
    One linear weight vector per action: Q(s, a) = w_a . phi(s).

    Two stability mechanisms on top of plain TD(0), both standard practice
    for Q-learning with function approximation (same idea DQN uses, just
    applied to a linear model instead of a network):

    - Target network: `target_weights` is a slow-moving copy of `weights`,
      used ONLY to compute the bootstrap target (max_a' Q(s',a')), synced
      to the live weights periodically via sync_target() rather than every
      step. Without this, the target shifts under every single update,
      which is a big part of why training against a real adversary kept
      destabilizing -- the thing being chased kept moving.
    - Experience replay buffer (see ReplayBuffer below): updates happen on
      randomly sampled PAST transitions, not just whatever just happened,
      breaking the strong correlation between consecutive updates.
    """

    def __init__(self, n_features=N_FEATURES, n_actions=len(ACTIONS)):
        self.weights = np.zeros((n_actions, n_features), dtype=np.float32)
        self.target_weights = self.weights.copy()
        self.epsilon = 1.0  # persisted here so it survives checkpoint reloads

    def predict(self, features):
        """Live weights -- used for actually choosing actions."""
        return self.weights @ features

    def predict_target(self, features):
        """Slow-moving weights -- used only inside the TD target."""
        return self.target_weights @ features

    def update(self, features, action_idx, target, lr):
        prediction = float(self.weights[action_idx] @ features)
        td_error = target - prediction
        self.weights[action_idx] += lr * td_error * features
        return td_error

    def sync_target(self):
        self.target_weights = self.weights.copy()


class ReplayBuffer:
    """
    Fixed-size buffer of past transitions. push() stores one, sample()
    returns a random batch. Using random past transitions instead of only
    the newest one is what breaks the correlation between consecutive
    updates that was contributing to training instability.

    Stores raw (features, action, reward, next_features, done) tuples --
    NOT pre-expanded via symmetry -- so memory use stays proportional to
    real gameplay experience. Symmetry expansion happens at update time in
    train.py instead (apply all 8 symmetries to whatever gets sampled).
    """

    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, features, action_idx, reward, next_features, done):
        self.buffer.append((features, action_idx, reward, next_features, done))

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


# ===========================================================================
# PART 6: Symmetry augmentation
# ===========================================================================
# The 8 symmetries of a square (identity, rotations by 90/180/270 degrees,
# and each of those composed with a mirror flip) form the dihedral group
# D4. Each is represented below as a permutation of the 4 direction
# indices: SYMMETRIES[name][i] is where index i's content moves to under
# that transform.

SYMMETRIES = {
    "identity":  (0, 1, 2, 3),
    "rot90":     (1, 2, 3, 0),
    "rot180":    (2, 3, 0, 1),
    "rot270":    (3, 0, 1, 2),
    "flip_h":    (0, 3, 2, 1),   # mirror left-right
    "flip_v":    (2, 1, 0, 3),   # mirror up-down
    "diag_main": (1, 0, 3, 2),
    "diag_anti": (3, 2, 1, 0),
}

# Every one-hot direction BLOCK in the feature vector, as (start, length).
_DIRECTIONAL_BLOCKS = [(3, 4), (7, 4), (11, 4), (15, 4), (19, 4), (23, 4)]


def apply_symmetry_to_features(features, perm):
    """Returns a NEW feature vector as if the whole board had been
    transformed by `perm`. Scalar features (bias, danger flag,
    good_bomb_spot, mobility) are unchanged -- rotating or mirroring the
    world doesn't change whether you're in danger, only WHICH direction
    things are in."""
    out = features.copy()
    for start, length in _DIRECTIONAL_BLOCKS:
        block = features[start:start + length]
        new_block = np.zeros(length, dtype=features.dtype)
        for i in range(length):
            new_block[perm[i]] = block[i]
        out[start:start + length] = new_block
    return out


def apply_symmetry_to_action(action_idx, perm):
    """UP/RIGHT/DOWN/LEFT (indices 0-3, matching DIRS) get relabeled by the
    same permutation used on the features. WAIT (4) and BOMB (5) are
    direction-independent and pass through unchanged."""
    if action_idx < 4:
        return perm[action_idx]
    return action_idx
