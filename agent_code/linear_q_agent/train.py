import csv
import os
import pickle
from datetime import datetime

import numpy as np

import events as e
from .module import (
    ACTIONS, N_FEATURES, ReplayBuffer, SYMMETRIES, apply_symmetry_to_action,
    apply_symmetry_to_features, compute_potential, state_to_features,
)

# --- hyperparameters --------------------------------------------------
# Learning rate reduced from 0.05 -> 0.02: with replay, each environment
# step now triggers many more gradient updates than before (one immediate
# update plus a replayed batch, each x8 for symmetry), so each individual
# update is made gentler to compensate and avoid overshooting.
LEARNING_RATE = 0.008  # was 0.02 -- gentler updates so adapting to a
                        # harder opponent doesn't overwrite an already-good
                        # policy in large destructive steps
DISCOUNT = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99  # reverted -- slow decay (0.995) was tested and made
                       # things WORSE: prolonged exploration means more
                       # random BOMB placements near a genuinely dangerous,
                       # mobile opponent, which is far costlier than the
                       # same randomness in solo play

REPLAY_BATCH_SIZE = 8       # extra past transitions replayed per step
REPLAY_CAPACITY = 150000    # was 20000 -- confirmed too small: a single
                            # 400-round burst against a real opponent
                            # generates ~68k transitions, which fully
                            # evicted the buffer within one burst. At
                            # 150k, a burst that size only replaces
                            # ~45% of it, so stable prior experience
                            # (solo classic, coin-heaven) survives
                            # alongside the new, harder experience.
TARGET_SYNC_EVERY_N_ROUNDS = 5   # how often target_weights catches up to weights
BUFFER_SAVE_EVERY_N_ROUNDS = 20  # saving 150k transitions every round would
                                  # be a real I/O cost for no benefit;
                                  # periodic saving is enough since it only
                                  # needs to survive across separate runs

GOOD_BOMB_PLACED = "GOOD_BOMB_PLACED"
BAD_BOMB_PLACED = "BAD_BOMB_PLACED"

REWARDS = {
    e.COIN_COLLECTED: 10,
    e.KILLED_OPPONENT: 20,
    e.CRATE_DESTROYED: 2,
    e.KILLED_SELF: -20,
    e.GOT_KILLED: -20,
    e.INVALID_ACTION: -1,
    e.WAITED: -0.1,
    GOOD_BOMB_PLACED: 1.0,
    BAD_BOMB_PLACED: -1.0,
}

LOG_PATH = "logs/training_log.csv"
DANGER_LOG_PATH = "logs/danger_diagnostics.csv"
MODEL_PATH = "my-saved-model.pt"
REPLAY_PATH = "replay-buffer.pt"
CORE_REPLAY_PATH = "replay-buffer-core.pt"
DIR_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


def setup_training(self):
    self._round_reward = 0.0
    self._round_steps = 0
    self._round_stats = {"coins": 0, "crates": 0, "kills": 0}
    self._rounds_since_sync = 0

    # Persisted to disk (unlike before) so bursts of training in SEPARATE
    # `python main.py` invocations still share accumulated experience,
    # instead of every burst starting the decorrelation benefit from zero.
    if not hasattr(self, "replay_buffer"):
        if os.path.isfile(REPLAY_PATH):
            with open(REPLAY_PATH, "rb") as f:
                self.replay_buffer = pickle.load(f)
            # Safety check: if the feature vector size ever changes (e.g. a
            # new feature gets added), old buffer entries become the wrong
            # shape and would crash a replayed update. Detect that and
            # start a fresh buffer instead of failing mid-training.
            if len(self.replay_buffer) > 0:
                sample_features = self.replay_buffer.buffer[0][0]
                if len(sample_features) != N_FEATURES:
                    self.logger.warning(
                        f"Replay buffer has stale {len(sample_features)}-dim "
                        f"transitions but current features are {N_FEATURES}-dim "
                        f"-- discarding stale buffer and starting fresh."
                    )
                    self.replay_buffer = ReplayBuffer(capacity=REPLAY_CAPACITY)
        else:
            self.replay_buffer = ReplayBuffer(capacity=REPLAY_CAPACITY)

    if not hasattr(self, "core_buffer"):
        if os.path.isfile(CORE_REPLAY_PATH):
            with open(CORE_REPLAY_PATH, "rb") as f:
                self.core_buffer = pickle.load(f)
        else:
            self.core_buffer = None

    log_dir = os.path.dirname(LOG_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.isfile(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "round", "steps", "reward",
                 "coins", "crates", "kills", "died", "epsilon"]
            )


def _danger_diagnostic(old_game_state, action, events):
    if old_game_state is None:
        return None
    features = state_to_features(old_game_state)
    if features[2] != 1.0:
        return None
    escape_onehot = features[19:23]
    if escape_onehot.sum() > 0:
        escape_dir = DIR_NAMES[int(np.argmax(escape_onehot))]
    else:
        escape_dir = "NONE_FOUND"
    return {
        "escape_direction": escape_dir,
        "action_taken": action,
        "followed_escape": (action == escape_dir),
        "action_invalid": (e.INVALID_ACTION in events),
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
                              "followed_escape", "action_invalid", "died_this_step"])
        writer.writerow([round_number, step, diagnostic["escape_direction"],
                          diagnostic["action_taken"], diagnostic["followed_escape"],
                          diagnostic["action_invalid"], diagnostic["died_this_step"]])


def _bomb_decision_event(old_game_state, action):
    if old_game_state is None or action != "BOMB":
        return []
    features = state_to_features(old_game_state)
    return [GOOD_BOMB_PLACED if features[27] == 1.0 else BAD_BOMB_PLACED]


def reward_from_events(events):
    return sum(REWARDS.get(ev, 0) for ev in events)


def _update_stats(self, events):
    self._round_stats["coins"] += events.count(e.COIN_COLLECTED)
    self._round_stats["crates"] += events.count(e.CRATE_DESTROYED)
    self._round_stats["kills"] += events.count(e.KILLED_OPPONENT)


def _symmetric_update(self, old_features, action_idx, reward, new_features, done):
    """
    TD update, once per board symmetry (8x). Uses the TARGET network
    (model.predict_target) for the bootstrap term, not the live weights --
    that's what stops the thing we're chasing from shifting under every
    single update.
    """
    for perm in SYMMETRIES.values():
        t_features = apply_symmetry_to_features(old_features, perm)
        t_action = apply_symmetry_to_action(action_idx, perm)
        if not done:
            t_new_features = apply_symmetry_to_features(new_features, perm)
            target = reward + DISCOUNT * float(np.max(self.model.predict_target(t_new_features)))
        else:
            target = reward
        self.model.update(t_features, t_action, target, LEARNING_RATE)


def _learn_from_transition(self, old_features, action_idx, reward, new_features, done):
    """
    Push the transition to the replay buffer, then learn ONLY from sampled
    batches (recent + protected core) -- no immediate update on the live
    transition itself. This was a real bug: an unconditional immediate
    update on every single fresh step is, by definition, unprotected by
    the core buffer (which only covers the *replayed* portion), so it was
    quietly reintroducing the exact online/correlated-update instability
    the whole replay mechanism exists to prevent. Standard DQN-style
    methods don't do an immediate update either, for this same reason.
    """
    self.replay_buffer.push(old_features, action_idx, reward, new_features, done)

    batch = self.replay_buffer.sample(REPLAY_BATCH_SIZE)
    if self.core_buffer is not None and len(self.core_buffer) > 0:
        # Half the replayed batch always comes from the frozen core --
        # this is what makes it immune to eviction, unlike the recent
        # buffer which does still turn over.
        batch = batch[:REPLAY_BATCH_SIZE // 2] + self.core_buffer.sample(REPLAY_BATCH_SIZE // 2)
    for (f, a, r, nf, d) in batch:
        _symmetric_update(self, f, a, r, nf, d)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    custom = _bomb_decision_event(old_game_state, self_action)
    all_events = events + custom
    sparse_reward = reward_from_events(all_events)

    shaping = DISCOUNT * compute_potential(new_game_state) - compute_potential(old_game_state)
    reward = sparse_reward + shaping

    self._round_reward += reward
    self._round_steps += 1
    _update_stats(self, all_events)

    diagnostic = _danger_diagnostic(old_game_state, self_action, events)
    if diagnostic is not None:
        _log_danger_diagnostic(old_game_state['round'], old_game_state['step'], diagnostic)

    if old_game_state is not None and self_action in ACTIONS:
        old_features = state_to_features(old_game_state)
        new_features = state_to_features(new_game_state)
        action_idx = ACTIONS.index(self_action)
        _learn_from_transition(self, old_features, action_idx, reward, new_features, done=False)


def end_of_round(self, last_game_state, last_action, events):
    custom = _bomb_decision_event(last_game_state, last_action)
    all_events = events + custom
    sparse_reward = reward_from_events(all_events)

    shaping = -compute_potential(last_game_state)
    reward = sparse_reward + shaping

    self._round_reward += reward
    self._round_steps += 1
    _update_stats(self, all_events)

    diagnostic = _danger_diagnostic(last_game_state, last_action, events)
    if diagnostic is not None and last_game_state is not None:
        _log_danger_diagnostic(last_game_state['round'], last_game_state['step'], diagnostic)

    if last_game_state is not None and last_action in ACTIONS:
        old_features = state_to_features(last_game_state)
        action_idx = ACTIONS.index(last_action)
        _learn_from_transition(self, old_features, action_idx, reward, None, done=True)

    self._rounds_since_sync += 1
    if self._rounds_since_sync >= TARGET_SYNC_EVERY_N_ROUNDS:
        self.model.sync_target()
        self._rounds_since_sync = 0

    self.model.epsilon = max(EPSILON_END, self.model.epsilon * EPSILON_DECAY)

    died = 1 if (e.KILLED_SELF in events or e.GOT_KILLED in events) else 0
    round_number = last_game_state['round'] if last_game_state is not None else -1
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            round_number,
            self._round_steps,
            round(self._round_reward, 2),
            self._round_stats["coins"],
            self._round_stats["crates"],
            self._round_stats["kills"],
            died,
            round(self.model.epsilon, 3),
        ])

    tmp_model = MODEL_PATH + ".tmp"
    with open(tmp_model, "wb") as file:
        pickle.dump(self.model, file)
    os.replace(tmp_model, MODEL_PATH)

    tmp_replay = REPLAY_PATH + ".tmp"
    if round_number % BUFFER_SAVE_EVERY_N_ROUNDS == 0:
        with open(tmp_replay, "wb") as file:
            pickle.dump(self.replay_buffer, file)
        os.replace(tmp_replay, REPLAY_PATH)

    self._round_reward = 0.0
    self._round_steps = 0
    self._round_stats = {"coins": 0, "crates": 0, "kills": 0}
