"""
Converts the framework's raw `game_state` dict into a fixed-length
numpy feature vector consumed by both the policy and value networks.

IMPORTANT: this assumes the standard ukoethe/bomberman_rl game_state shape:
    game_state['field']          -> 2D array, field[x, y]: -1 wall, 1 crate, 0 free
    game_state['self']           -> (name, score, bombs_left, (x, y))
    game_state['others']         -> list of (name, score, bombs_left, (x, y))
    game_state['bombs']          -> list of ((x, y), countdown)
    game_state['coins']          -> list of (x, y)
    game_state['explosion_map']  -> 2D array, >0 where an explosion is active

If your actual game_state differs (key names, coordinate order), tell me
the mismatch and I'll adjust — easier to fix one file than debug later.
"""

from collections import deque
import numpy as np

BOMB_POWER = 3  # blast radius in bomberman_rl; adjust if your rules differ
BOMB_TIMER = 4  # steps until a placed bomb explodes; check settings.py if unsure

DIRECTIONS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
MOVE_ORDER = ["UP", "DOWN", "LEFT", "RIGHT"]

MAX_BFS_DISTANCE = 30  # for normalizing distances; board is smaller than this


def _get_danger_tiles(field, bombs, explosion_map):
    """
    Returns a set of (x, y) coordinates that are currently unsafe:
    active explosions, plus every tile inside a ticking bomb's blast
    footprint (blocked by walls, like real bomb blasts).
    """
    danger = set()

    # Active explosions right now.
    xs, ys = np.where(explosion_map > 0)
    danger.update(zip(xs.tolist(), ys.tolist()))

    # Footprint of each placed bomb, regardless of countdown (conservative:
    # treat the whole eventual blast area as dangerous, not just last-second).
    for (bx, by), _countdown in bombs:
        danger.add((bx, by))
        for dx, dy in DIRECTIONS.values():
            for step in range(1, BOMB_POWER + 1):
                x, y = bx + dx * step, by + dy * step
                if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                    break
                if field[x, y] == -1:  # wall blocks the blast
                    break
                danger.add((x, y))
                if field[x, y] == 1:  # crate stops the blast after this tile
                    break

    return danger


def _bfs_direction_and_distance(start, targets, field):
    """
    BFS from `start` over free tiles (field == 0) to the nearest tile in
    `targets`. Returns:
        direction_one_hot: length-5 list [UP, DOWN, LEFT, RIGHT, NONE]
            — NONE=1 means no target reachable (or no targets at all)
        normalized_distance: float in [0, 1], 1.0 if unreachable
    """
    one_hot = [0, 0, 0, 0, 0]
    targets = set(targets)

    if not targets:
        one_hot[4] = 1
        return one_hot, 1.0

    if start in targets:
        one_hot[4] = 1  # already there, no direction needed
        return one_hot, 0.0

    visited = {start}
    # queue holds (position, first_move_direction_name_or_None)
    queue = deque([(start, None)])
    dist_from_start = {start: 0}

    while queue:
        (cx, cy), first_dir = queue.popleft()

        if (cx, cy) in targets:
            idx = MOVE_ORDER.index(first_dir)
            one_hot[idx] = 1
            norm_dist = min(dist_from_start[(cx, cy)] / MAX_BFS_DISTANCE, 1.0)
            return one_hot, norm_dist

        for dir_name in MOVE_ORDER:
            dx, dy = DIRECTIONS[dir_name]
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nx, ny] != 0 and (nx, ny) not in targets:
                continue  # blocked by wall/crate, unless it IS the target tile
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            dist_from_start[(nx, ny)] = dist_from_start[(cx, cy)] + 1
            queue.append(((nx, ny), first_dir if first_dir is not None else dir_name))

    # No path found.
    one_hot[4] = 1
    return one_hot, 1.0


def state_to_features(game_state: dict) -> np.ndarray:
    """
    Main entry point. Returns a 1D float32 numpy array — this length
    MUST match STATE_DIM in config.py (currently 28).
    """
    field = game_state["field"]
    _, _, bombs_left, (ax, ay) = game_state["self"]
    others = [pos for (_, _, _, pos) in game_state["others"]]
    bombs = game_state["bombs"]
    coins = game_state["coins"]
    explosion_map = game_state["explosion_map"]

    danger_tiles = _get_danger_tiles(field, bombs, explosion_map)

    # --- Directional features (5 each: UP, DOWN, LEFT, RIGHT, NONE) ---
    coin_dir, coin_dist = _bfs_direction_and_distance((ax, ay), coins, field)

    crate_coords = list(zip(*np.where(field == 1)))
    crate_dir, crate_dist = _bfs_direction_and_distance((ax, ay), crate_coords, field)

    enemy_dir, enemy_dist = _bfs_direction_and_distance((ax, ay), others, field)

    # --- Danger flags for staying put + each of the 4 move directions (5) ---
    danger_flags = [1 if (ax, ay) in danger_tiles else 0]  # index 0 = staying
    for dir_name in MOVE_ORDER:
        dx, dy = DIRECTIONS[dir_name]
        nx, ny = ax + dx, ay + dy
        in_bounds = 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
        danger_flags.append(1 if in_bounds and (nx, ny) in danger_tiles else 0)

    # --- Valid move mask (4): is the tile free to walk into right now? ---
    valid_move = []
    occupied = set(others)
    for dir_name in MOVE_ORDER:
        dx, dy = DIRECTIONS[dir_name]
        nx, ny = ax + dx, ay + dy
        in_bounds = 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
        free = in_bounds and field[nx, ny] == 0 and (nx, ny) not in occupied
        valid_move.append(1 if free else 0)

    can_bomb = 1 if bombs_left > 0 else 0
    in_explosion_now = 1 if (ax, ay) in danger_tiles else 0

    features = (
        coin_dir + [coin_dist]
        + crate_dir + [crate_dist]
        + enemy_dir + [enemy_dist]
        + danger_flags
        + valid_move
        + [can_bomb, in_explosion_now]
    )

    return np.array(features, dtype=np.float32)


def _has_escape_route(self_pos, field, blast_tiles, occupied, max_steps=BOMB_TIMER) -> bool:
    """
    BFS up to max_steps moves from self_pos over free tiles. Returns True
    if a tile OUTSIDE blast_tiles is reachable within that many moves —
    i.e., there's actually time to get clear before the bomb goes off.
    occupied: positions currently held by other agents — treated as
    blocked, same as a wall, since you can't walk through another player.
    Ignoring this was a real bug: it let the mask say "safe" for an
    escape tile an opponent was actually standing on.
    """
    visited = {self_pos}
    queue = deque([(self_pos, 0)])
    while queue:
        (x, y), steps = queue.popleft()
        if steps >= max_steps:
            continue
        for dx, dy in DIRECTIONS.values():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nx, ny] != 0:
                continue  # wall or crate blocks movement
            if (nx, ny) in occupied:
                continue  # another agent is standing here right now
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            if (nx, ny) not in blast_tiles:
                return True
            queue.append(((nx, ny), steps + 1))
    return False


def has_escape_route_if_bomb_now(game_state: dict) -> bool:
    """
    If the agent placed a bomb at its CURRENT position right now, would it
    have a reachable safe tile to retreat to before the bomb explodes?
    Used by train.py/callbacks.py to MASK the BOMB action when this is
    False — a hard constraint, not a learned preference, so the policy
    cannot select a bomb placement with no escape route at all.

    NOTE: this treats opponents' CURRENT positions as fixed obstacles for
    the whole BFS. It's a snapshot, not a prediction of where they'll be
    several steps from now — opponents move too, so this is a reasonable
    approximation, not a perfect guarantee. Good enough to catch "the only
    exit is currently blocked," not perfect foresight of future blocking.
    """
    field = game_state["field"]
    _, _, _, self_pos = game_state["self"]
    blast_tiles = get_blast_tiles(self_pos, field)
    occupied = set(pos for (_, _, _, pos) in game_state["others"])
    return _has_escape_route(self_pos, field, blast_tiles, occupied)


def is_position_in_danger(game_state: dict) -> bool:
    """
    Public helper (used by train.py's shaping): is the agent's CURRENT
    position inside an active explosion or a ticking bomb's blast
    footprint right now? Used to stop rewarding "closer to objective"
    shaping when that movement actually walks into danger — otherwise
    the dense approach-bonus can pull the agent back into a blast zone
    chasing a coin/crate, overriding whatever self-preservation it's
    otherwise learned.
    """
    field = game_state["field"]
    _, _, _, self_pos = game_state["self"]
    danger_tiles = _get_danger_tiles(field, game_state["bombs"], game_state["explosion_map"])
    return self_pos in danger_tiles


def get_blast_tiles(position, field) -> set:
    """
    Returns the set of tiles a bomb placed at `position` would hit —
    same blast-footprint logic as _get_danger_tiles, but for a
    hypothetical bomb rather than ones already on the board. Used by
    train.py to check whether a just-placed bomb threatens an opponent.
    """
    x0, y0 = position
    tiles = {(x0, y0)}
    for dx, dy in DIRECTIONS.values():
        for step in range(1, BOMB_POWER + 1):
            x, y = x0 + dx * step, y0 + dy * step
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                break
            if field[x, y] == -1:
                break
            tiles.add((x, y))
            if field[x, y] == 1:
                break
    return tiles


def get_nearest_enemy_distance(game_state: dict) -> float:
    """
    Standalone helper (reused by train.py for reward shaping) — normalized
    BFS distance in [0, 1] to the nearest opponent. 1.0 means unreachable
    or no opponents remain.
    """
    field = game_state["field"]
    _, _, _, (ax, ay) = game_state["self"]
    others = [pos for (_, _, _, pos) in game_state["others"]]
    _, dist = _bfs_direction_and_distance((ax, ay), others, field)
    return dist


def get_nearest_coin_distance(game_state: dict) -> float:
    """Normalized BFS distance in [0, 1] to the nearest coin. 1.0 = none/unreachable."""
    field = game_state["field"]
    _, _, _, (ax, ay) = game_state["self"]
    _, dist = _bfs_direction_and_distance((ax, ay), game_state["coins"], field)
    return dist


def get_nearest_crate_distance(game_state: dict) -> float:
    """Normalized BFS distance in [0, 1] to the nearest crate. 1.0 = none/unreachable."""
    field = game_state["field"]
    _, _, _, (ax, ay) = game_state["self"]
    crate_coords = list(zip(*np.where(field == 1)))
    _, dist = _bfs_direction_and_distance((ax, ay), crate_coords, field)
    return dist


def build_bomb_safety_mask(game_state: dict) -> list:
    """
    Returns a length-6 mask [UP, DOWN, LEFT, RIGHT, WAIT, BOMB], each 1 if
    the action is allowed, 0 if masked out. Only BOMB can ever be masked
    here (no bombs left, or no escape route) — movement/wait stay always
    allowed; illegal moves are still handled by the existing INVALID_ACTION
    reward penalty, not masking, to keep this scoped to bomb-safety only.
    """
    _, _, bombs_left, _ = game_state["self"]
    can_bomb_safely = bombs_left > 0 and has_escape_route_if_bomb_now(game_state)
    return [1, 1, 1, 1, 1, 1 if can_bomb_safely else 0]


if __name__ == "__main__":
    # Minimal synthetic game_state to sanity-check shape and that it runs.
    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[:, 0] = -1
    field[3, 3] = 1  # a crate

    fake_state = {
        "field": field,
        "self": ("me", 0, 1, (1, 1)),
        "others": [],
        "bombs": [],
        "coins": [(5, 5)],
        "explosion_map": np.zeros((7, 7), dtype=int),
    }

    feats = state_to_features(fake_state)
    print(f"Feature vector length: {len(feats)}")
    print(feats)
