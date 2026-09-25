"""
Observation encoding for the PPO agent.

Deliberately independent from the feature engineering used elsewhere in
this project. Instead of a hand-designed summary vector (BFS-computed
directions, danger flags, and so on), the board is encoded as a stack of
2D grids ("channels") -- the same idea used for feeding images to a CNN --
so the network can learn its own useful spatial patterns rather than
relying on features we chose by hand. A small vector of scalar extras
(things with no natural location on the board) is fed in alongside it.
"""
import numpy as np

BOARD_SIZE = 17     # matches settings.py: COLS = ROWS = 17
N_CHANNELS = 8
N_SCALARS = 8
BOMB_TIMER = 4       # matches settings.py
BOMB_POWER = 3       # matches settings.py

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# Channel layout:
#   0  stone walls
#   1  crates
#   2  coins
#   3  own position
#   4  opponent positions
#   5  bomb danger, weighted by urgency (closer to exploding = higher value)
#   6  active explosion right now
#   7  recommended escape tile (1.0 at the single tile the escape search
#      says to move to next, when in danger; all zero otherwise). Diagnosed
#      via dedicated logging: the equivalent information as a small scalar
#      one-hot was essentially ignored by the policy (followed only 12% of
#      the time despite being correct), plausibly because 4 numbers out of
#      1608 total inputs were too easy for training to not bother with.
#      Painting it spatially puts it inside the same rich representation
#      the CNN already prioritizes, instead of a small, easily-ignored side
#      channel.


def _distance_to_safety(start, field, occupied, danger_tiles, max_dist=15):
    """
    Same search as _has_escape_route, but returns the actual shortest-path
    DISTANCE to the nearest safe tile instead of a yes/no answer. Capped
    at max_dist -- "genuinely far" and "unreachable" both resolve to the
    same finite value. Used for potential-based escape-progress shaping:
    rewarding the CHANGE in this distance from one step to the next,
    rather than a flat bonus for merely surviving another step. This
    matters because a flat "still alive" bonus pays out identically
    whether the agent is standing still or actually escaping, for as long
    as the bomb hasn't detonated yet -- which turned out to give the
    policy a real incentive to stall instead of escape (confirmed
    directly in training: the agent would WAIT for several consecutive
    steps while a known escape direction was available, then die the
    instant the bomb went off). A potential-based reward can't be
    exploited this way: standing still doesn't change the distance, so it
    nets exactly zero, not a reward.
    """
    from collections import deque

    if start not in danger_tiles:
        return 0

    width, height = field.shape
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        cur, dist = queue.popleft()
        if cur not in danger_tiles:
            return dist
        if dist >= max_dist:
            continue
        for dx, dy in DIRS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if (0 <= nxt[0] < width and 0 <= nxt[1] < height
                    and _walkable(nxt, field) and nxt not in occupied and nxt not in visited):
                visited.add(nxt)
                queue.append((nxt, dist + 1))

    return max_dist  # no route found within range -- treat as "very far"


def _blast_coords(pos, field, power=BOMB_POWER):
    """Mirrors Bomb.get_blast_coords() in items.py exactly: a blast is
    stopped only by a stone wall, and passes straight through crates."""
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


def _walkable(pos, field):
    x, y = pos
    width, height = field.shape
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    return field[x, y] == 0


def _count_safe_neighbors(pos, field, danger_tiles):
    """How many of the 4 adjacent tiles are currently safe to step onto:
    on the board, not a wall/crate, and not in the current blast-danger
    set. A dense, explicit "how many ways out do I have right now" signal
    -- the network could in principle infer this from the raw danger
    channel alone, but giving it directly removes that burden and lets
    training focus on using the signal rather than first discovering it."""
    count = 0
    for dx, dy in DIRS:
        nxt = (pos[0] + dx, pos[1] + dy)
        if _walkable(nxt, field) and nxt not in danger_tiles:
            count += 1
    return count


def _has_escape_route(start, field, occupied, danger_tiles, max_dist=15):
    """
    Same search as _bfs_escape_direction, but just answers yes/no: is
    there a reachable tile not in danger_tiles? Shares the same
    "can walk through current danger, only the destination must be safe"
    logic. Used both for the escape-direction signal below, and (from
    train.py) to check a HYPOTHETICAL danger set -- e.g. "if I dropped a
    bomb right now, would a real escape route still exist afterward" --
    which is what the previous bombing-decision reward was missing: it
    only checked whether a bomb would hit a crate, never whether the spot
    was actually survivable.
    """
    from collections import deque

    if start not in danger_tiles:
        return True

    width, height = field.shape
    visited = {start}
    queue = deque()
    for dx, dy in DIRS:
        nxt = (start[0] + dx, start[1] + dy)
        if _walkable(nxt, field) and nxt not in occupied and nxt not in visited:
            visited.add(nxt)
            queue.append((nxt, 1))

    while queue:
        cur, dist = queue.popleft()
        if cur not in danger_tiles:
            return True
        if dist >= max_dist:
            continue
        for dx, dy in DIRS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if (0 <= nxt[0] < width and 0 <= nxt[1] < height
                    and _walkable(nxt, field) and nxt not in occupied and nxt not in visited):
                visited.add(nxt)
                queue.append((nxt, dist + 1))

    return False


def _bfs_escape_direction(start, field, occupied, danger_tiles, max_dist=15):
    """
    Returns a one-hot 4-vector: which of UP/RIGHT/DOWN/LEFT is the first
    step of the shortest path from `start` to the nearest tile NOT
    currently in danger. All zeros if already safe, or if no route is
    found within max_dist steps.

    Critically, this search is allowed to walk THROUGH currently-dangerous
    tiles as intermediate steps -- only the final destination needs to be
    safe. A bomb takes several turns to explode, so a tile that's inside
    a blast radius right now is still perfectly walkable at this instant;
    treating it as an impassable wall would make many real escape routes
    invisible to the search (this exact mistake was found and fixed in a
    different agent earlier in this project, and is deliberately avoided
    here from the start).
    """
    from collections import deque

    if start not in danger_tiles:
        return [0.0, 0.0, 0.0, 0.0]

    width, height = field.shape
    visited = {start}
    queue = deque()
    for dir_idx, (dx, dy) in enumerate(DIRS):
        nxt = (start[0] + dx, start[1] + dy)
        if _walkable(nxt, field) and nxt not in occupied and nxt not in visited:
            visited.add(nxt)
            queue.append((nxt, dir_idx, 1))

    while queue:
        cur, first_dir, dist = queue.popleft()
        if cur not in danger_tiles:
            result = [0.0, 0.0, 0.0, 0.0]
            result[first_dir] = 1.0
            return result
        if dist >= max_dist:
            continue
        for dx, dy in DIRS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if (0 <= nxt[0] < width and 0 <= nxt[1] < height
                    and _walkable(nxt, field) and nxt not in occupied and nxt not in visited):
                visited.add(nxt)
                queue.append((nxt, first_dir, dist + 1))

    return [0.0, 0.0, 0.0, 0.0]  # no route found -- genuinely trapped


def compute_danger_tiles(game_state):
    """Standalone version of the danger-tile computation used inside
    encode() -- exposed separately so train.py can check "am I in danger
    right now" without needing the full grid/scalar encoding."""
    field = game_state['field']
    danger_tiles = set()
    for (bpos, _timer) in game_state['bombs']:
        danger_tiles.update(_blast_coords(bpos, field))
    explosion_map = game_state['explosion_map']
    width, height = field.shape
    for x in range(width):
        for y in range(height):
            if explosion_map[x, y] > 0:
                danger_tiles.add((x, y))
    return danger_tiles


def encode(game_state):
    """Returns (grid, scalars).
    grid:    float32 array, shape (N_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    scalars: float32 array, shape (N_SCALARS,)
    """
    if game_state is None:
        return (np.zeros((N_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
                np.zeros(N_SCALARS, dtype=np.float32))

    field = game_state['field']
    _name, _score, bomb_available, (sx, sy) = game_state['self']
    others = game_state['others']
    bombs = game_state['bombs']
    coins = game_state['coins']

    grid = np.zeros((N_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    grid[0] = (field == -1).astype(np.float32)
    grid[1] = (field == 1).astype(np.float32)
    for (cx, cy) in coins:
        grid[2, cx, cy] = 1.0
    grid[3, sx, sy] = 1.0
    for (_n, _s, _b, (ox, oy)) in others:
        grid[4, ox, oy] = 1.0
    danger_tiles = compute_danger_tiles(game_state)
    for (bpos, timer) in bombs:
        urgency = 1.0 - (timer / BOMB_TIMER)
        for (bx, by) in _blast_coords(bpos, field):
            grid[5, bx, by] = max(grid[5, bx, by], urgency)
    grid[6] = (game_state['explosion_map'] > 0).astype(np.float32)

    occupied = set((ox, oy) for (_n, _s, _b, (ox, oy)) in others)
    occupied.update(bpos for (bpos, _t) in bombs)

    escape_direction = _bfs_escape_direction((sx, sy), field, occupied, danger_tiles)
    if sum(escape_direction) > 0:
        dir_idx = escape_direction.index(1.0)
        dx, dy = DIRS[dir_idx]
        ex, ey = sx + dx, sy + dy
        if 0 <= ex < BOARD_SIZE and 0 <= ey < BOARD_SIZE:
            grid[7, ex, ey] = 1.0

    scalars = np.array([
        1.0 if bomb_available else 0.0,
        len(others) / 3.0,
        game_state['step'] / 400.0,
        _count_safe_neighbors((sx, sy), field, danger_tiles) / 4.0,
        *escape_direction,
    ], dtype=np.float32)

    return grid, scalars
