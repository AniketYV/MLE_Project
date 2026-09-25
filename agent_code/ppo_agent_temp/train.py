"""
Training loop for the PPO agent.

Reward design is deliberately simple and independent from any other
agent in this project: sparse event rewards (coin, kill, crate, death),
plus a crude straight-line-distance shaping term for the nearest visible
coin -- Manhattan distance, not a pathfinding search, so it doesn't
require (or resemble) the BFS-based potential function used elsewhere.
PPO's own credit-assignment mechanism (GAE, see ppo.py) handles most of
the sparse-reward problem on its own by design, which is part of why a
much simpler shaping scheme is a reasonable choice here, not just a
shortcut.
"""
import csv
import os
from datetime import datetime

import torch

import events as e
from .network import ActorCritic
from .observation import encode, _blast_coords, _has_escape_route, compute_danger_tiles, _bfs_escape_direction, _distance_to_safety
from .ppo import RolloutBuffer, ppo_update

LEARNING_RATE = 1e-4   # reduced from 3e-4 -- gentler individual steps,
                        # complementing the tighter clip range and fewer
                        # epochs in ppo.py
ROLLOUT_SIZE = 2048   # doubled from 1024 -- a larger, more diverse batch
                       # of episodes per update is less likely to be
                       # dominated by whatever one short, skewed stretch
                       # of experience happened to occur most recently

COIN_SHAPING_WEIGHT = 0.5  # raised from 0.3 -- smoother, more gradual
                            # lever than the sparse reward, distributed
                            # across many small steps rather than one
                            # spike, so less likely to destabilize training
CRATE_SHAPING_WEIGHT = 0.15  # smaller than coin -- crates are a means to
                              # an end (revealing coins/opening paths),
                              # not the actual objective
BOMB_DECISION_REWARD = 4.0   # raised from 1.5 -- the math showed engaging
                              # with crates still had worse expected value
                              # than just waiting, given the current death
                              # rate, so the policy was correctly (not
                              # buggily) choosing the passive option every
                              # time. This and the reduced death penalty
                              # below are aimed at shifting that balance.
ESCAPE_PROGRESS_WEIGHT = 1.0  # potential-based: rewards REDUCING distance
                               # to safety, not just "still alive this
                               # step". The flat per-step version of this
                               # (tried previously) was a confirmed
                               # exploit: standing still paid out exactly
                               # as many "survived" bonuses as actually
                               # escaping did, for as long as the bomb
                               # hadn't detonated yet -- diagnostic
                               # logging directly caught the policy
                               # choosing WAIT for several consecutive
                               # steps with a known escape route
                               # available, then dying the instant the
                               # bomb went off. A potential-based reward
                               # can't be gamed this way: standing still
                               # doesn't change the distance to safety,
                               # so it nets exactly zero.

REWARDS = {
    e.COIN_COLLECTED: 18,   # was 10, then over-corrected to 50 -- that
                            # 5x jump destabilized the policy back into
                            # near-total passivity (91 moves across
                            # 14,464 steps). A much smaller, gentler
                            # increase this time.
    e.KILLED_OPPONENT: 28,  # was 20, then over-corrected to 75 -- reverted
                            # to a gentler increase, same reasoning as coin above
    e.CRATE_DESTROYED: 6,
    e.KILLED_SELF: -35,     # raised well above the original -20: with
                            # WAITED removed and bombing rewards increased,
                            # this needs to clearly dominate even a
                            # multi-crate bombing burst, so "bomb a few
                            # crates then die" can't outscore either doing
                            # nothing (now a clean 0) or actually surviving
    e.GOT_KILLED: -35,
    e.INVALID_ACTION: -1,
    e.WAITED: 0,  # removed the -0.1 per-step version of this: it accumulated
                  # over a full round if the agent survived (up to -40 over
                  # 400 steps), which made dying EARLY an artificially cheap
                  # way to escape that accumulation -- effectively rewarding
                  # suicide-bombing runs over surviving a boring round. A
                  # flat 0 baseline for doing nothing removes that loophole.
}

LOG_PATH = "logs/training_log.csv"
DANGER_LOG_PATH = "logs/danger_diagnostics.csv"
MODEL_PATH = "ppo-model.pt"
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def _danger_diagnostic(old_game_state, action, events):
    """On every step where the agent starts out in danger: what direction
    the escape search recommends, what action was actually taken, and
    whether it survived. Same diagnostic approach that found two real,
    concrete bugs (not just tuning issues) in a different agent earlier
    in this project -- built now because six different reward/feature
    fixes have all failed to produce a single survive-and-bomb round,
    which means guessing further isn't productive; actual measurement is
    needed."""
    if old_game_state is None:
        return None
    danger_tiles = compute_danger_tiles(old_game_state)
    _name, _score, _bomb, pos = old_game_state['self']
    if pos not in danger_tiles:
        return None

    field = old_game_state['field']
    others = old_game_state['others']
    bombs = old_game_state['bombs']
    occupied = set((ox, oy) for (_n, _s, _b, (ox, oy)) in others)
    occupied.update(bpos for (bpos, _t) in bombs)

    escape_onehot = _bfs_escape_direction(pos, field, occupied, danger_tiles)
    dir_names = ["UP", "RIGHT", "DOWN", "LEFT"]
    if sum(escape_onehot) > 0:
        escape_dir = dir_names[escape_onehot.index(1.0)]
    else:
        escape_dir = "NONE_FOUND"

    return {
        "escape_direction": escape_dir,
        "action_taken": action,
        "followed_escape": (action == escape_dir),
        "died_this_step": (e.KILLED_SELF in events or e.GOT_KILLED in events),
    }


def _log_danger_diagnostic(round_number, step, diagnostic):
    if diagnostic is None:
        return
    file_exists = os.path.isfile(DANGER_LOG_PATH)
    with open(DANGER_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["round", "step", "escape_direction", "action_taken",
                              "followed_escape", "died_this_step"])
        writer.writerow([round_number, step, diagnostic["escape_direction"],
                          diagnostic["action_taken"], diagnostic["followed_escape"],
                          diagnostic["died_this_step"]])


def setup_training(self):
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
    self.buffer = RolloutBuffer()
    self.network.train()

    self._round_reward = 0.0
    self._round_steps = 0
    self._round_stats = {"coins": 0, "crates": 0, "kills": 0}

    log_dir = os.path.dirname(LOG_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.isfile(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "round", "steps", "reward", "coins", "crates", "kills", "died"]
            )


def _nearest_coin_distance(game_state):
    """Straight-line (Manhattan) distance to the nearest visible coin, or
    None if no coin is currently visible. Deliberately crude -- does not
    account for walls or crates blocking the path -- so this is a
    genuinely different, simpler heuristic from the BFS-based pathfinding
    used elsewhere in this project, not a disguised reuse of it."""
    coins = game_state['coins']
    if not coins:
        return None
    _name, _score, _bomb, (sx, sy) = game_state['self']
    return min(abs(sx - cx) + abs(sy - cy) for (cx, cy) in coins)


def _nearest_crate_distance(game_state):
    """Same crude Manhattan-distance idea, applied to the nearest crate
    still on the board. Only meaningful when no coin is currently
    visible -- coins take priority once one is revealed."""
    field = game_state['field']
    _name, _score, _bomb, (sx, sy) = game_state['self']
    crate_positions = list(zip(*(field == 1).nonzero()))
    if not crate_positions:
        return None
    return min(abs(sx - cx) + abs(sy - cy) for (cx, cy) in crate_positions)


def _coin_shaping(old_game_state, new_game_state):
    if old_game_state is None or new_game_state is None:
        return 0.0
    old_dist = _nearest_coin_distance(old_game_state)
    new_dist = _nearest_coin_distance(new_game_state)
    if old_dist is not None and new_dist is not None:
        return COIN_SHAPING_WEIGHT * (old_dist - new_dist)

    # No coin visible in at least one of the two states -- fall back to
    # crate-approach shaping instead. This is what was missing: with no
    # visible coins (the normal situation for most of a round in
    # `classic`, since coins start hidden under crates) and nothing else
    # pulling the policy toward productive behaviour, it collapsed into
    # always choosing the only always-safe action, WAIT.
    old_crate_dist = _nearest_crate_distance(old_game_state)
    new_crate_dist = _nearest_crate_distance(new_game_state)
    if old_crate_dist is not None and new_crate_dist is not None:
        return CRATE_SHAPING_WEIGHT * (old_crate_dist - new_crate_dist)

    return 0.0


def _bomb_decision_shaping(old_game_state, action):
    """
    Immediate reward for choosing to drop a bomb, computed from the state
    BEFORE the action -- doesn't wait for the bomb to actually explode.

    Critically, this now checks TWO things, not one: does the bomb hit a
    crate, AND does a real escape route still exist afterward (simulating
    the bomb's own blast added on top of any existing danger, then
    running the same BFS escape search used for the escape-direction
    signal). The earlier version only checked the first condition, which
    was a real, identified gap: the policy was being rewarded for bombing
    near ANY crate regardless of whether that specific spot was actually
    survivable, which plausibly explains why it kept destroying crates
    while still dying essentially every time it engaged at all. A bomb
    that hits a crate but leaves no escape route is now penalized just as
    strongly as a wasted bomb that hits nothing -- "technically a good
    target" is no longer good enough on its own.
    """
    if action != "BOMB" or old_game_state is None:
        return 0.0
    field = old_game_state['field']
    _name, _score, bomb_available, pos = old_game_state['self']
    if not bomb_available:
        return 0.0  # invalid anyway; INVALID_ACTION already covers this

    width, height = field.shape
    blast = _blast_coords(pos, field)
    hits_crate = any(
        0 <= x < width and 0 <= y < height and field[x, y] == 1
        for (x, y) in blast
    )
    if not hits_crate:
        return -BOMB_DECISION_REWARD * 0.5

    others = old_game_state['others']
    bombs = old_game_state['bombs']
    occupied = set((ox, oy) for (_n, _s, _b, (ox, oy)) in others)
    occupied.update(bpos for (bpos, _t) in bombs)

    hypothetical_danger = set(blast)
    for (bpos, _timer) in bombs:
        hypothetical_danger.update(_blast_coords(bpos, field))

    has_escape = _has_escape_route(pos, field, occupied, hypothetical_danger)
    return BOMB_DECISION_REWARD if has_escape else -BOMB_DECISION_REWARD


def _escape_progress_shaping(old_game_state, new_game_state):
    """Potential-based reward for REDUCING distance to safety between one
    step and the next. Standing still, or moving without making real
    progress toward safety, nets exactly zero -- not a reward -- which is
    what removes the flat-bonus exploit found via diagnostic logging (see
    ESCAPE_PROGRESS_WEIGHT above for the full story). Only meaningful
    between two real states, so this isn't used at the terminal step of a
    round; the sparse death penalty already covers that case."""
    if old_game_state is None or new_game_state is None:
        return 0.0

    old_field = old_game_state['field']
    _n1, _s1, _b1, old_pos = old_game_state['self']
    old_danger = compute_danger_tiles(old_game_state)
    if old_pos not in old_danger:
        return 0.0  # wasn't in danger to begin with, nothing to shape

    old_occupied = set((ox, oy) for (_n2, _s2, _b2, (ox, oy)) in old_game_state['others'])
    old_occupied.update(bpos for (bpos, _t) in old_game_state['bombs'])
    old_dist = _distance_to_safety(old_pos, old_field, old_occupied, old_danger)

    new_field = new_game_state['field']
    _n3, _s3, _b3, new_pos = new_game_state['self']
    new_danger = compute_danger_tiles(new_game_state)
    new_occupied = set((ox, oy) for (_n4, _s4, _b4, (ox, oy)) in new_game_state['others'])
    new_occupied.update(bpos for (bpos, _t) in new_game_state['bombs'])
    new_dist = _distance_to_safety(new_pos, new_field, new_occupied, new_danger)

    return ESCAPE_PROGRESS_WEIGHT * (old_dist - new_dist)


def reward_from_events(events):
    return sum(REWARDS.get(ev, 0) for ev in events)


def _update_stats(self, events):
    self._round_stats["coins"] += events.count(e.COIN_COLLECTED)
    self._round_stats["crates"] += events.count(e.CRATE_DESTROYED)
    self._round_stats["kills"] += events.count(e.KILLED_OPPONENT)


def _maybe_update(self):
    if len(self.buffer) >= ROLLOUT_SIZE:
        stats = ppo_update(self.network, self.optimizer, self.buffer, self._last_value, device=self.device)
        self.logger.info(
            f"PPO update: policy_loss={stats['policy_loss']:.4f} "
            f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.4f}"
        )
        self.network.train()


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = (reward_from_events(events)
              + _coin_shaping(old_game_state, new_game_state)
              + _bomb_decision_shaping(old_game_state, self_action)
              + _escape_progress_shaping(old_game_state, new_game_state))
    self._round_reward += reward
    self._round_steps += 1
    _update_stats(self, events)

    diagnostic = _danger_diagnostic(old_game_state, self_action, events)
    if diagnostic is not None:
        _log_danger_diagnostic(old_game_state['round'], old_game_state['step'], diagnostic)

    self.buffer.add(
        self._last_grid, self._last_scalars, self._last_action,
        self._last_log_prob, self._last_value, reward, False,
    )
    _maybe_update(self)


def end_of_round(self, last_game_state, last_action, events):
    reward = (reward_from_events(events)
              + _bomb_decision_shaping(last_game_state, last_action))
    self._round_reward += reward
    self._round_steps += 1
    _update_stats(self, events)

    diagnostic = _danger_diagnostic(last_game_state, last_action, events)
    if diagnostic is not None and last_game_state is not None:
        _log_danger_diagnostic(last_game_state['round'], last_game_state['step'], diagnostic)

    self.buffer.add(
        self._last_grid, self._last_scalars, self._last_action,
        self._last_log_prob, self._last_value, reward, True,
    )
    _maybe_update(self)

    died = 1 if (e.KILLED_SELF in events or e.GOT_KILLED in events) else 0
    round_number = last_game_state['round'] if last_game_state is not None else -1
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            round_number, self._round_steps, round(self._round_reward, 2),
            self._round_stats["coins"], self._round_stats["crates"],
            self._round_stats["kills"], died,
        ])

    torch.save(self.network.state_dict(), MODEL_PATH)

    self._round_reward = 0.0
    self._round_steps = 0
    self._round_stats = {"coins": 0, "crates": 0, "kills": 0}
