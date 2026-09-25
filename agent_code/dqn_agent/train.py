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


Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward", "done"),
)

BUFFER_SIZE = 30_000
BATCH_SIZE = 64
GAMMA = 0.97
LR = 5e-4

TARGET_UPDATE_EVERY = 500

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_ROUNDS = 10_000
EPSILON_STEP = (
    (EPSILON_START - EPSILON_END) / EPSILON_DECAY_ROUNDS
)

DANGER_THRESHOLD = 0.2

STATE_FILE = "dqn-train-state.json"
EPSILON_OVERRIDE_FILE = "dqn-epsilon-override.txt"

MOVED_CLOSER_TO_COIN = "MOVED_CLOSER_TO_COIN"
MOVED_FURTHER_FROM_COIN = "MOVED_FURTHER_FROM_COIN"
STAYED_PUT_IN_DANGER = "STAYED_PUT_IN_DANGER"
FLED_DANGER = "FLED_DANGER"
USELESS_BOMB_SPAM = "USELESS_BOMB_SPAM"


def setup_training(self):
    self.buffer = deque(maxlen=BUFFER_SIZE)
    self.optimizer = optim.Adam(
        self.model.parameters(),
        lr=LR,
        weight_decay=1e-5,
    )

    self.target_model = QNet().to(self.device)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()

    self.gradient_steps = 0

    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)

        self.epsilon = state.get("epsilon", EPSILON_START)
        self.rounds_played = state.get("rounds_played", 0)
    else:
        self.epsilon = EPSILON_START
        self.rounds_played = 0

    if os.path.isfile(EPSILON_OVERRIDE_FILE):
        with open(EPSILON_OVERRIDE_FILE) as f:
            self.epsilon = float(f.read().strip())
        os.remove(EPSILON_OVERRIDE_FILE)


def _nearest_coin_manhattan(game_state):
    if game_state is None or not game_state["coins"]:
        return None

    x, y = game_state["self"][3]

    return min(
        abs(cx - x) + abs(cy - y)
        for cx, cy in game_state["coins"]
    )


def _danger_here(game_state):
    if game_state is None:
        return 0.0

    feat = state_to_features(game_state)
    return float(feat[33])


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: list,
):
    old_dist = _nearest_coin_manhattan(old_game_state)
    new_dist = _nearest_coin_manhattan(new_game_state)

    # Light shaping only; actual game objectives dominate.
    if old_dist is not None and new_dist is not None:
        if new_dist < old_dist:
            events.append(MOVED_CLOSER_TO_COIN)
        elif new_dist > old_dist:
            events.append(MOVED_FURTHER_FROM_COIN)

    was_in_danger = _danger_here(old_game_state) > DANGER_THRESHOLD

    if was_in_danger:
        if self_action in ("WAIT", "BOMB"):
            events.append(STAYED_PUT_IN_DANGER)
        else:
            events.append(FLED_DANGER)

    if self_action == "BOMB" and e.INVALID_ACTION in events:
        events.append(USELESS_BOMB_SPAM)

    reward = reward_from_events(self, events)

    old_feat = state_to_features(old_game_state)
    new_feat = state_to_features(new_game_state)
    action_idx = ACTION_TO_IDX[self_action]

    self.buffer.append(
        Transition(
            old_feat,
            action_idx,
            new_feat,
            reward,
            False,
        )
    )

    _optimize(self)


def _save_with_retry(write_fn, path, max_retries=6, initial_delay=0.15):
    tmp_path = f"{path}.tmp"
    delay = initial_delay

    for _ in range(max_retries):
        try:
            write_fn(tmp_path)
            os.replace(tmp_path, path)
            return
        except (OSError, RuntimeError):
            time.sleep(delay)
            delay *= 1.7


def _write_state_file(path, epsilon, rounds_played):
    with open(path, "w") as f:
        json.dump(
            {
                "epsilon": epsilon,
                "rounds_played": rounds_played,
            },
            f,
        )


def _save_checkpoint(self):
    _save_with_retry(
        lambda p: torch.save(self.model.state_dict(), p),
        MODEL_FILE,
    )

    _save_with_retry(
        lambda p: _write_state_file(
            p,
            self.epsilon,
            self.rounds_played,
        ),
        STATE_FILE,
    )


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: list,
):
    was_in_danger = _danger_here(last_game_state) > DANGER_THRESHOLD

    if was_in_danger and last_action in ("WAIT", "BOMB"):
        events.append(STAYED_PUT_IN_DANGER)

    if last_action == "BOMB" and e.INVALID_ACTION in events:
        events.append(USELESS_BOMB_SPAM)

    reward = reward_from_events(self, events)

    last_feat = state_to_features(last_game_state)
    action_idx = ACTION_TO_IDX[last_action]
    zero_next = np.zeros_like(last_feat)

    self.buffer.append(
        Transition(
            last_feat,
            action_idx,
            zero_next,
            reward,
            True,
        )
    )

    _optimize(self)

    self.rounds_played += 1
    self.epsilon = max(
        EPSILON_END,
        self.epsilon - EPSILON_STEP,
    )

    _save_checkpoint(self)


def _optimize(self):
    if len(self.buffer) < BATCH_SIZE:
        return

    batch = random.sample(self.buffer, BATCH_SIZE)

    states = torch.tensor(
        np.array([t.state for t in batch]),
        dtype=torch.float32,
        device=self.device,
    )

    actions = torch.tensor(
        [t.action for t in batch],
        dtype=torch.long,
        device=self.device,
    )

    next_states = torch.tensor(
        np.array([t.next_state for t in batch]),
        dtype=torch.float32,
        device=self.device,
    )

    rewards = torch.tensor(
        [t.reward for t in batch],
        dtype=torch.float32,
        device=self.device,
    )

    dones = torch.tensor(
        [t.done for t in batch],
        dtype=torch.bool,
        device=self.device,
    )

    current_q_values = (
        self.model(states)
        .gather(1, actions.unsqueeze(1))
        .squeeze(1)
    )

    with torch.no_grad():
        # Double DQN:
        # online network selects the action
        next_online_q_values = self.model(next_states)
        best_next_actions = torch.argmax(
            next_online_q_values,
            dim=1,
            keepdim=True,
        )

        # target network evaluates that action
        next_target_q_values = self.target_model(next_states)
        next_max_q = (
            next_target_q_values
            .gather(1, best_next_actions)
            .squeeze(1)
        )

        next_max_q[dones] = 0.0

        expected_q_values = rewards + GAMMA * next_max_q

    loss = nn.functional.smooth_l1_loss(
        current_q_values,
        expected_q_values,
    )

    self.optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        self.model.parameters(),
        max_norm=1.0,
    )

    self.optimizer.step()

    self.gradient_steps += 1

    if self.gradient_steps % TARGET_UPDATE_EVERY == 0:
        self.target_model.load_state_dict(
            self.model.state_dict()
        )


def reward_from_events(self, events: list) -> float:
    game_rewards = {
        # Main objectives
        e.COIN_COLLECTED: 2.0,
        e.KILLED_OPPONENT: 5.0,
        e.CRATE_DESTROYED: 0.4,

        # Survival
        e.KILLED_SELF: -6.0,
        e.GOT_KILLED: -5.0,

        # Bad actions
        e.INVALID_ACTION: -0.75,
        e.WAITED: -0.05,
        USELESS_BOMB_SPAM: -0.8,

        # Small survival shaping
        STAYED_PUT_IN_DANGER: -1.5,
        FLED_DANGER: 0.5,

        # Very light navigation shaping
        MOVED_CLOSER_TO_COIN: 0.03,
        MOVED_FURTHER_FROM_COIN: -0.03,

        # End-of-round survival
        e.SURVIVED_ROUND: 0.5,
    }

    total_reward = 0.0

    for event in events:
        if event in game_rewards:
            total_reward += game_rewards[event]

    return total_reward
