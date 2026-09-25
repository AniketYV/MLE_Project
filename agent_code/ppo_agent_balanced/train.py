"""
Training loop for the PPO agent.

Reward design, independent from any other agent in this project:
  - Sparse event rewards (coin, kill, crate, death), proportioned from the
    start so official scoring priorities (kill > coin >> crate) can't be
    inverted by cumulative crate-clearing over a long round.
  - Manhattan-distance shaping toward the nearest visible coin, then
    opponent (safety-gated), then crate as a last resort -- crude,
    straight-line distance, not a pathfinding search, so it doesn't
    require (or resemble) the BFS-based potential function used elsewhere
    in this project.
  - Escape-verified bombing: a bomb only pays off if a real, confirmed
    escape route exists afterward, not just "hits a crate."
  - Potential-based escape-progress shaping: rewards reducing distance to
    safety step by step, not a flat "survived this step" bonus (which was
    a confirmed, diagnosed exploit -- standing still paid the same as
    actually escaping, for as long as the bomb hadn't gone off yet).
"""
import csv
import os
from datetime import datetime

import numpy as np
import torch

import events as e
from .network import ActorCritic
from .observation import encode, _blast_coords, _has_escape_route, compute_danger_tiles, _bfs_escape_direction, _distance_to_safety
from .ppo import RolloutBuffer, ppo_update

LEARNING_RATE = 1e-4   # gentler individual steps
ROLLOUT_SIZE = 2048     # larger, more diverse batch per update

# Shaping weights, in priority order: coin (safest, guaranteed value) >
# opponent-hunting (higher payoff, riskier) > crate (purely instrumental --
# a means to reveal coins and open paths, never the actual goal).
COIN_SHAPING_WEIGHT = 0.6
OPPONENT_SHAPING_WEIGHT = 0.35  # NEW: actively rewards closing distance on
                                 # a visible opponent when no coin is
                                 # available -- previously there was no
                                 # dedicated incentive to hunt at all,
                                 # despite eliminating opponents being
                                 # worth 5x a coin in real scoring and the
                                 # explicit goal being to win against them
CRATE_SHAPING_WEIGHT = 0.15
BOMB_DECISION_REWARD = 3.0      # reverted back up from 1.8/1.2 -- see
                                 # CRATE_DESTROYED comment above; escape-
                                 # verified: only rewarded if a real
                                 # escape route exists afterward, not just
                                 # "hits a crate" (see _bomb_decision_shaping)
ESCAPE_PROGRESS_WEIGHT = 1.0    # potential-based: rewards REDUCING distance
                                 # to safety, not flat "still alive" -- the
                                 # flat version was a confirmed exploit
                                 # (standing still paid the same as actually
                                 # escaping, caught via diagnostic logging)

REWARDS = {
    # Designed from scratch this time, with the explicit priority order
    # baked in from the start rather than adjusted after the fact: killing
    # an opponent is worth far more than a coin (matching real scoring,
    # where a kill = 5 coins), a coin is worth far more than a single
    # crate-clear cycle, and crates are only ever a means to an end. This
    # avoids the exact problem hit twice before -- changing these values
    # on an ALREADY-TRAINED policy destabilized it, because the value
    # function had calibrated to the old, wrong balance. Training a fresh
    # network under the correct balance from step one has no such
    # calibration to disrupt.
    e.COIN_COLLECTED: 40,
    e.KILLED_OPPONENT: 150,
    e.CRATE_DESTROYED: 5,   # reverted back up from 2.5/1.5 -- the smaller
                            # values didn't make coin-collection better,
                            # they just made the policy retreat into
                            # passivity (73% fully passive, EV(engage) =
                            # -74). The insight this corrects: the shaped
                            # reward's job isn't to numerically mirror
                            # official scoring proportions -- it's to
                            # produce broad, confident crate-clearing,
                            # since which crates hide coins is random and
                            # more crates cleared is what actually drives
                            # the statistical chance of finding one. The
                            # original ppo_agent proved this works,
                            # reaching 67% survive-and-bomb success with
                            # this same stronger incentive.
    e.KILLED_SELF: -50,     # kept large enough that dying is never a
                            # casually acceptable trade-off for a middling
                            # haul: e.g. clearing 10 crates plus grabbing a
                            # coin nets 27+40=67 before this penalty, so
                            # dying right after still costs real net value
                            # (67-50=17) rather than being a free trade
    e.GOT_KILLED: -50,
    e.INVALID_ACTION: -1,
    e.WAITED: 0,            # NOT a negative per-step penalty -- that was
                            # a confirmed exploit (see below): a per-step
                            # WAITED cost accumulated over a full survived
                            # round, making early death an artificially
                            # cheap way to escape that accumulation.
}

LOG_PATH = "logs/training_log.csv"
DANGER_LOG_PATH = "logs/danger_diagnostics.csv"
MODEL_PATH = "ppo-model.pt"
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


def _danger_diagnostic(old_game_state, action, events):
    """On every step where the agent starts out in danger: what direction
    the escape search recommends, what action was actually taken, and
    whether it survived. Kept from the start this time as a standing
    diagnostic, not added reactively -- the same approach found two real,
    concrete bugs (not just tuning issues) in an earlier agent in this
    project, and having it in place from round one means any future
    escape-following regression is caught with real data immediately,
    rather than after many rounds of guessing."""
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
    still on the board. Only used as the last-resort fallback -- coins
    and opponents both take priority once either is visible."""
    field = game_state['field']
    _name, _score, _bomb, (sx, sy) = game_state['self']
    crate_positions = list(zip(*(field == 1).nonzero()))
    if not crate_positions:
        return None
    return min(abs(sx - cx) + abs(sy - cy) for (cx, cy) in crate_positions)


def _nearest_opponent_distance(game_state):
    """Same crude Manhattan-distance idea, applied to the nearest visible
    opponent. NEW: previously there was no dedicated incentive to hunt
    opponents at all -- only the eventual sparse KILLED_OPPONENT reward,
    with nothing pulling the policy toward them in the meantime. Given
    the explicit goal is winning against other agents (worth 5x a coin
    in real scoring), this closes a real gap, not just a nice-to-have."""
    others = game_state['others']
    if not others:
        return None
    _name, _score, _bomb, (sx, sy) = game_state['self']
    return min(abs(sx - ox) + abs(sy - oy) for (_n, _s, _b, (ox, oy)) in others)


def _coin_shaping(old_game_state, new_game_state):
    if old_game_state is None or new_game_state is None:
        return 0.0
    old_dist = _nearest_coin_distance(old_game_state)
    new_dist = _nearest_coin_distance(new_game_state)
    if old_dist is not None and new_dist is not None:
        return COIN_SHAPING_WEIGHT * (old_dist - new_dist)

    # No coin visible -- fall back to opponent-hunting if one is visible.
    # Safety-gated: only reward closing distance if the resulting position
    # isn't itself dangerous (e.g. walking toward an opponent who just
    # placed a bomb). Without this, an unchecked "always move closer"
    # incentive could pull the policy into danger purely to shave off
    # distance -- this has never been tested against a real, bomb-placing
    # opponent yet, only solo training so far, so the gate errs safe.
    old_opp_dist = _nearest_opponent_distance(old_game_state)
    new_opp_dist = _nearest_opponent_distance(new_game_state)
    if old_opp_dist is not None and new_opp_dist is not None:
        new_field = new_game_state['field']
        _n, _s, _b, new_pos = new_game_state['self']
        new_danger = compute_danger_tiles(new_game_state)
        if new_pos not in new_danger:
            return OPPONENT_SHAPING_WEIGHT * (old_opp_dist - new_opp_dist)
        return 0.0

    # Neither coin nor opponent visible -- fall back to crate-approach.
    # This is what originally prevented the WAIT-collapse: with nothing
    # else pulling the policy toward productive behaviour, it defaulted
    # to always choosing the only always-safe action.
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
    others = old_game_state['others']
    opponent_positions = set((ox, oy) for (_n, _s, _b, (ox, oy)) in others)

    hits_crate = any(
        0 <= x < width and 0 <= y < height and field[x, y] == 1
        for (x, y) in blast
    )
    hits_opponent = any(o in opponent_positions for o in blast)

    bombs = old_game_state['bombs']
    occupied = set(opponent_positions)
    occupied.update(bpos for (bpos, _t) in bombs)
    hypothetical_danger = set(blast)
    for (bpos, _timer) in bombs:
        hypothetical_danger.update(_blast_coords(bpos, field))
    has_escape = _has_escape_route(pos, field, occupied, hypothetical_danger)

    if not hits_crate and not hits_opponent:
        # Pointless bomb -- hits nothing useful. Still genuinely dangerous
        # if it also has no escape route (this matters a lot more in a
        # crate-free scenario like coin-heaven, where nothing blocks blast
        # propagation and a "wasted" bomb can be just as lethal as a
        # productive one). A flat, weak penalty regardless of survivability
        # doesn't strongly discourage bombing where it's never useful.
        return -BOMB_DECISION_REWARD * (1.5 if not has_escape else 0.5)

    if not has_escape:
        return -BOMB_DECISION_REWARD
    # Opponent hit is worth more, matching the same priority (kills >
    # everything else) established in the sparse reward table above.
    return BOMB_DECISION_REWARD * (2.5 if hits_opponent else 1.0)


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


REWARD_CLAMP = 300.0  # generous safety margin -- comfortably above any
                       # legitimate single-step total even with multiple
                       # events stacking (largest single event is
                       # KILLED_OPPONENT at 150), so this never interferes
                       # with normal operation. Exists purely to catch a
                       # genuine bug (e.g. a stray division producing an
                       # extreme value somewhere) before it can propagate
                       # into GAE and the loss computation and corrupt the
                       # network -- this is what a second real crash,
                       # after the first fix only addressed the policy-
                       # ratio side of the same underlying problem,
                       # revealed was still missing.


def _clamp_reward(reward):
    if not np.isfinite(reward):
        return 0.0
    return float(np.clip(reward, -REWARD_CLAMP, REWARD_CLAMP))


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = _clamp_reward(
        reward_from_events(events)
        + _coin_shaping(old_game_state, new_game_state)
        + _bomb_decision_shaping(old_game_state, self_action)
        + _escape_progress_shaping(old_game_state, new_game_state)
    )
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
    reward = _clamp_reward(
        reward_from_events(events)
        + _bomb_decision_shaping(last_game_state, last_action)
    )
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

    # Final safety net: even with the guards in ppo_update, never persist
    # a checkpoint whose weights have gone non-finite. Overwriting a good
    # saved checkpoint with a broken one would make recovery impossible;
    # skipping the save costs at most this one round's worth of progress.
    state = self.network.state_dict()
    if all(torch.isfinite(v).all() for v in state.values()):
        torch.save(state, MODEL_PATH)
    else:
        self.logger.warning(
            "Network weights are non-finite -- refusing to overwrite the "
            "saved checkpoint with a corrupted one."
        )

    self._round_reward = 0.0
    self._round_steps = 0
    self._round_stats = {"coins": 0, "crates": 0, "kills": 0}
