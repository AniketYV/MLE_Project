import json
import os
import random
import time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import events as e
from .callbacks import ACTION_TO_IDX, MODEL_FILE, QNet, state_to_features

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# --- Hyperparameters (tune these) ---
BUFFER_SIZE = 20_000
BATCH_SIZE = 64
GAMMA = 0.95
LR = 1e-3
TARGET_UPDATE_EVERY = 500   # gradient steps between target-network syncs
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_ROUNDS = 2_000  # rounds over which epsilon decays START -> END
EPSILON_STEP = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_ROUNDS

# Danger urgency (see callbacks.state_to_features) above which we consider
# the agent "in danger" for shaping purposes.
DANGER_THRESHOLD = 0.2

# --- Files used to persist training progress across separate process runs.
# Each `python main.py play --train 1 ...` invocation is a fresh process, so
# without this, epsilon and the round counter would reset to their initial
# values every single time you (re)start training -- which silently breaks
# any multi-burst / curriculum training setup (see dqn_train_pipeline.py).
STATE_FILE = "dqn-train-state.json"
# A training pipeline can drop a number in this file to force a specific
# epsilon for the upcoming burst (e.g. re-raising exploration when moving
# to a harder scenario). Consumed once, then deleted.
EPSILON_OVERRIDE_FILE = "dqn-epsilon-override.txt"

# --- Custom shaping events (on top of the built-in ones in events.py) ---
MOVED_CLOSER_TO_COIN = "MOVED_CLOSER_TO_COIN"
MOVED_FURTHER_FROM_COIN = "MOVED_FURTHER_FROM_COIN"
STAYED_PUT_IN_DANGER = "STAYED_PUT_IN_DANGER"
FLED_DANGER = "FLED_DANGER"
USELESS_BOMB_SPAM = "USELESS_BOMB_SPAM"


def setup_training(self):
    """Called once after callbacks.setup(), only when self.train is True."""
    self.buffer = deque(maxlen=BUFFER_SIZE)
    self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    self.target_model = QNet().to(self.device)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()

    self.gradient_steps = 0

    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        self.epsilon = state.get("epsilon", EPSILON_START)
        self.rounds_played = state.get("rounds_played", 0)
        self.logger.info(f"Resumed training state: epsilon={self.epsilon:.3f}, "
                          f"rounds_played={self.rounds_played}")
    else:
        self.epsilon = EPSILON_START
        self.rounds_played = 0

    if os.path.isfile(EPSILON_OVERRIDE_FILE):
        with open(EPSILON_OVERRIDE_FILE) as f:
            self.epsilon = float(f.read().strip())
        os.remove(EPSILON_OVERRIDE_FILE)
        self.logger.info(f"Epsilon overridden for this burst: {self.epsilon:.3f}")


def _nearest_coin_manhattan(game_state):
    if game_state is None or not game_state['coins']:
        return None
    x, y = game_state['self'][3]
    return min(abs(cx - x) + abs(cy - y) for cx, cy in game_state['coins'])


def _danger_here(game_state):
    """Danger urgency (0..1) of the agent's own tile, recomputed directly
    rather than re-deriving it from the feature vector by index, so this
    stays correct even if the feature layout changes later."""
    if game_state is None:
        return 0.0
    feat = state_to_features(game_state)
    return float(feat[8])  # index 8 = here_danger, see callbacks.py docstring


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    # --- Shaping 1: encourage closing the distance to the nearest coin ---
    old_dist = _nearest_coin_manhattan(old_game_state)
    new_dist = _nearest_coin_manhattan(new_game_state)
    if old_dist is not None and new_dist is not None:
        if new_dist < old_dist:
            events.append(MOVED_CLOSER_TO_COIN)
        elif new_dist > old_dist:
            events.append(MOVED_FURTHER_FROM_COIN)

    # --- Shaping 2: punish standing still on a dangerous tile, reward
    # actually leaving one. This directly targets the failure mode where
    # the agent dropped a bomb and then just stood there repeating BOMB
    # (an invalid no-op) until its own bomb killed it. ---
    was_in_danger = _danger_here(old_game_state) > DANGER_THRESHOLD
    if was_in_danger:
        if self_action in ('WAIT', 'BOMB'):
            events.append(STAYED_PUT_IN_DANGER)
        else:
            events.append(FLED_DANGER)

    # --- Shaping 3: extra penalty for repeatedly trying to drop a bomb
    # that can't be dropped (already have one active). A single invalid
    # action is a minor mistake; spamming BOMB while unable to place one
    # is the specific pattern that got the agent killed. ---
    if self_action == 'BOMB' and e.INVALID_ACTION in events:
        events.append(USELESS_BOMB_SPAM)

    reward = reward_from_events(self, events)
    old_feat = state_to_features(old_game_state)
    new_feat = state_to_features(new_game_state)
    action_idx = ACTION_TO_IDX[self_action]

    self.buffer.append(Transition(old_feat, action_idx, new_feat, reward, False))
    _optimize(self)


def _save_with_retry(write_fn, path, max_retries=6, initial_delay=0.15):
    """Writes to a .tmp file and atomically renames it into place, retrying
    on transient file-lock errors (e.g. OneDrive/Dropbox/antivirus briefly
    locking the file mid-sync -- Windows error 32, "sharing violation").
    Logs a warning and gives up (without crashing the whole training run)
    if it still fails after all retries."""
    tmp_path = f"{path}.tmp"
    delay = initial_delay
    last_exc = None
    for attempt in range(max_retries):
        try:
            write_fn(tmp_path)
            os.replace(tmp_path, path)  # atomic on the same filesystem
            return
        except (OSError, RuntimeError) as exc:
            last_exc = exc
            time.sleep(delay)
            delay *= 1.7
    self_logger = getattr(_save_with_retry, "_logger", None)
    msg = (f"Failed to save {path} after {max_retries} retries "
           f"(likely a cloud-sync file lock, e.g. OneDrive): {last_exc}")
    if self_logger is not None:
        self_logger.warning(msg)
    else:
        print(f"[dqn_agent] WARNING: {msg}")


def _write_state_file(path, epsilon, rounds_played):
    with open(path, "w") as f:
        json.dump({"epsilon": epsilon, "rounds_played": rounds_played}, f)


def _save_checkpoint(self):
    _save_with_retry._logger = self.logger
    _save_with_retry(lambda p: torch.save(self.model.state_dict(), p), MODEL_FILE)
    _save_with_retry(
        lambda p: _write_state_file(p, self.epsilon, self.rounds_played),
        STATE_FILE,
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    was_in_danger = _danger_here(last_game_state) > DANGER_THRESHOLD
    if was_in_danger and last_action in ('WAIT', 'BOMB'):
        events.append(STAYED_PUT_IN_DANGER)
    if last_action == 'BOMB' and e.INVALID_ACTION in events:
        events.append(USELESS_BOMB_SPAM)

    reward = reward_from_events(self, events)
    last_feat = state_to_features(last_game_state)
    action_idx = ACTION_TO_IDX[last_action]
    zero_next = np.zeros_like(last_feat)

    self.buffer.append(Transition(last_feat, action_idx, zero_next, reward, True))
    _optimize(self)

    # Linear epsilon decay, once per finished round (stabler than per-step).
    self.rounds_played += 1
    self.epsilon = max(EPSILON_END, self.epsilon - EPSILON_STEP)
    self.logger.info(f"Round {self.rounds_played} done. epsilon={self.epsilon:.3f}, "
                      f"buffer={len(self.buffer)}")

    _save_checkpoint(self)


def _optimize(self):
    if len(self.buffer) < BATCH_SIZE:
        return

    batch = random.sample(self.buffer, BATCH_SIZE)
    states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=self.device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
    next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
    dones = torch.tensor([float(t.done) for t in batch], dtype=torch.float32, device=self.device)

    q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Double DQN: pick the best next action using the ONLINE network,
        # but evaluate its value using the TARGET network. Plain DQN uses
        # the target network for both, which systematically overestimates
        # Q-values (the max operator tends to pick actions whose value
        # estimate is high due to noise, then that same noisy estimate is
        # used as the value -- errors reinforce themselves). Decoupling
        # selection from evaluation removes that bias and gives more
        # stable training.
        next_actions = self.model(next_states).argmax(1)
        next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + GAMMA * next_q * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.gradient_steps += 1
    if self.gradient_steps % TARGET_UPDATE_EVERY == 0:
        self.target_model.load_state_dict(self.model.state_dict())


def reward_from_events(self, events: list) -> float:
    game_rewards = {
        e.COIN_COLLECTED: 1.0,
        e.KILLED_OPPONENT: 5.0,
        e.KILLED_SELF: -5.0,
        e.GOT_KILLED: -5.0,
        e.CRATE_DESTROYED: 0.3,
        e.INVALID_ACTION: -0.5,
        e.WAITED: -0.05,
        e.SURVIVED_ROUND: 0.5,
        MOVED_CLOSER_TO_COIN: 0.05,
        MOVED_FURTHER_FROM_COIN: -0.05,
        STAYED_PUT_IN_DANGER: -1.0,
        FLED_DANGER: 0.3,
        USELESS_BOMB_SPAM: -0.3,
    }
    reward_sum = sum(game_rewards.get(event, 0.0) for event in events)
    self.logger.debug(f"Awarded {reward_sum:.2f} for events {', '.join(events)}")
    return reward_sum
