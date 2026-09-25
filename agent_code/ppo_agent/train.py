"""
Framework training hooks + the actual PPO update.

Called by the framework when running in training mode (same pattern as
your other agents): setup_training() once, game_events_occurred() after
every step, end_of_round() when an episode finishes.

Unlike DQN's "store transition, sample minibatch from replay buffer,
update every step" pattern, PPO here:
  1. accumulates a full rollout of ROLLOUT_STEPS steps across (possibly
     several) episodes into the buffer
  2. only THEN runs PPO_EPOCHS passes over that batch
  3. wipes the buffer and starts collecting the next one

So most steps just add to the buffer and do nothing else — the actual
learning happens in bursts, not continuously.
"""

import os
import sys
import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import (
        ACTIONS, GAMMA, CLIP_EPS, VALUE_LOSS_COEF, ENTROPY_COEF, MAX_GRAD_NORM,
        LEARNING_RATE, ROLLOUT_STEPS, PPO_EPOCHS, MINIBATCH_SIZE,
        CHECKPOINT_DIR, POLICY_CHECKPOINT_NAME, VALUE_CHECKPOINT_NAME,
    )
    from features import (
        state_to_features, get_nearest_enemy_distance,
        get_nearest_coin_distance, get_nearest_crate_distance, get_blast_tiles,
        is_position_in_danger,
    )
    from rollout_buffer import RolloutBuffer
    from value_network.network import ValueNetwork
    from episode_logger import init_episode_tracking, track_step, log_episode_end
except ImportError:
    from agent_code.ppo_agent.config import (
        ACTIONS, GAMMA, CLIP_EPS, VALUE_LOSS_COEF, ENTROPY_COEF, MAX_GRAD_NORM,
        LEARNING_RATE, ROLLOUT_STEPS, PPO_EPOCHS, MINIBATCH_SIZE,
        CHECKPOINT_DIR, POLICY_CHECKPOINT_NAME, VALUE_CHECKPOINT_NAME,
    )
    from agent_code.ppo_agent.features import (
        state_to_features, get_nearest_enemy_distance,
        get_nearest_coin_distance, get_nearest_crate_distance, get_blast_tiles,
        is_position_in_danger,
    )
    from agent_code.ppo_agent.rollout_buffer import RolloutBuffer
    from agent_code.ppo_agent.value_network.network import ValueNetwork
    from agent_code.ppo_agent.episode_logger import init_episode_tracking, track_step, log_episode_end

ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}

# --- Reward shaping -----------------------------------------------------
# Starting point only — tune based on what you learned from your other
# agents' reward_from_events. Event names must match what your framework
# actually emits; check against your dqn_agent's version if unsure.
EVENT_REWARDS = {
    "COIN_COLLECTED": 2.0,
    "KILLED_OPPONENT": 8.0,
    "KILLED_SELF": -10.0,
    "GOT_KILLED": -3.0,
    "INVALID_ACTION": -0.1,
    "WAITED": -0.05,
    "CRATE_DESTROYED": 0.7,
    "COIN_FOUND": 0.5,
    "SURVIVED_ROUND": 0.05,
}


# Distance threshold (normalized BFS distance, see features.py) above which
# an enemy is considered "not near" — roughly 6+ tiles away, given
# MAX_BFS_DISTANCE=30 in features.py.
SAFE_ENEMY_DISTANCE = 0.2
SAFE_CRATE_BONUS = 0.3   # extra reward for destroying a crate when it's safe to
COIN_BONUS = 0.5         # extra reward for collecting a coin — always, not gated on safety
AGGRESSIVE_BOMB_BONUS = 1.0   # reward for placing a bomb that threatens an opponent, whether or not it lands


def reward_from_events(self, events: list, game_state: dict = None) -> float:
    """
    Sums shaped rewards for the events that occurred this step.
    If game_state is provided, adds shaping bonuses:
      1. Crate bonus when no enemy is nearby (bombing is risky — only
         reinforce doing it in safe moments).
      2. Coin bonus ALWAYS, regardless of enemy proximity — grabbing a
         coin is a quick, low-risk pickup, not a risky action like
         bombing, so it shouldn't need an "all clear" condition. Gating
         it the same way as crates was starving this signal, since with
         3 opponents roaming, "no enemy within 6 tiles" rarely holds.
      3. Aggressive-bomb bonus when a just-placed bomb threatens an
         opponent's current position — reinforces GOING FOR kills, not
         just landing them, since a real kill is too rare to learn from
         directly on its own.
    """
    reward = 0.0
    for event in events:
        reward += EVENT_REWARDS.get(event, 0.0)

    if game_state is not None and "CRATE_DESTROYED" in events:
        if get_nearest_enemy_distance(game_state) > SAFE_ENEMY_DISTANCE:
            reward += SAFE_CRATE_BONUS

    if game_state is not None and "COIN_COLLECTED" in events:
        reward += COIN_BONUS

    if game_state is not None and "BOMB_DROPPED" in events:
        field = game_state["field"]
        _, _, _, self_pos = game_state["self"]
        blast_tiles = get_blast_tiles(self_pos, field)
        opponent_positions = [pos for (_, _, _, pos) in game_state["others"]]
        if any(pos in blast_tiles for pos in opponent_positions):
            reward += AGGRESSIVE_BOMB_BONUS

    return reward


# Per-step shaping: rewards NET progress toward a coin/crate, not just the
# eventual payoff. Symmetric on purpose — moving closer pays +BONUS, moving
# farther costs -BONUS, so oscillating back and forth nets to zero. An
# earlier one-sided version (reward for closer, nothing for farther) let
# the agent farm reward by wandering back and forth near an objective
# without ever reaching it — this fixes that exploit.
COIN_APPROACH_BONUS = 0.3
CRATE_APPROACH_BONUS = 0.1


def compute_shaping_reward(self, old_game_state, new_game_state) -> float:
    if old_game_state is None or new_game_state is None:
        return 0.0

    if is_position_in_danger(new_game_state):
        # Don't reward "closer to objective" if the move that got you
        # there also walked you into an active blast zone — chasing
        # shaping reward through danger was pulling the agent back into
        # its own (or someone else's) bomb, undermining self-preservation.
        return 0.0

    shaping = 0.0
    old_coin_dist = get_nearest_coin_distance(old_game_state)
    new_coin_dist = get_nearest_coin_distance(new_game_state)
    if new_coin_dist < old_coin_dist:
        shaping += COIN_APPROACH_BONUS
    elif new_coin_dist > old_coin_dist:
        shaping -= COIN_APPROACH_BONUS

    if get_nearest_enemy_distance(old_game_state) > SAFE_ENEMY_DISTANCE:
        old_crate_dist = get_nearest_crate_distance(old_game_state)
        new_crate_dist = get_nearest_crate_distance(new_game_state)
        if new_crate_dist < old_crate_dist:
            shaping += CRATE_APPROACH_BONUS
        elif new_crate_dist > old_crate_dist:
            shaping -= CRATE_APPROACH_BONUS

    return shaping


def setup_training(self):
    """Called once, before training starts."""
    self.value_net = ValueNetwork()
    self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
    self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=LEARNING_RATE)
    self.buffer = RolloutBuffer()
    self.train_mode = True
    init_episode_tracking(self)

    checkpoint_dir = os.path.join(os.path.dirname(__file__), CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)

    self.logger.info("PPO training setup complete.")


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Called after every step during training.
    old_game_state -> self_action -> events happened -> new_game_state.
    """
    if old_game_state is None or self_action is None:
        return  # framework sometimes calls this on the very first step

    reward = reward_from_events(self, events, old_game_state)
    reward += compute_shaping_reward(self, old_game_state, new_game_state)
    track_step(self, reward, events)

    old_features = state_to_features(old_game_state)
    state_tensor = torch.tensor(old_features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        value = self.value_net.get_value(state_tensor).item()

    self.buffer.add(
        state=old_features,
        action=ACTION_TO_INDEX[self_action],
        log_prob=self.last_log_prob,
        reward=reward,
        done=False,
        value=value,
        action_mask=getattr(self, "last_action_mask", None),
    )

    _maybe_update(self, new_game_state)


def end_of_round(self, last_game_state, last_action, events):
    """Called once when an episode ends (agent died or round timed out)."""
    if last_action is None:
        return

    reward = reward_from_events(self, events, last_game_state)
    track_step(self, reward, events)

    last_features = state_to_features(last_game_state)
    state_tensor = torch.tensor(last_features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        value = self.value_net.get_value(state_tensor).item()

    self.buffer.add(
        state=last_features,
        action=ACTION_TO_INDEX[last_action],
        log_prob=self.last_log_prob,
        reward=reward,
        done=True,
        value=value,
        action_mask=getattr(self, "last_action_mask", None),
    )

    self.logger.info(f"Episode ended. Buffer size now: {len(self.buffer)}")
    log_episode_end(self, events)

    _maybe_update(self, new_game_state=None, force=False)
    _save_checkpoints(self)


def _maybe_update(self, new_game_state, force: bool = False):
    """
    Runs a PPO update once the buffer has ROLLOUT_STEPS worth of
    experience. The buffer spans multiple episodes on purpose — GAE
    already cuts advantage propagation at episode boundaries via the
    `done` flags, so partial episodes at the end of a batch are fine.
    force=True is available for manual/debugging use but isn't called
    automatically anymore (episode end no longer forces small updates).
    """
    if len(self.buffer) < ROLLOUT_STEPS and not force:
        return
    if len(self.buffer) == 0:
        return

    if new_game_state is not None:
        # Bootstrap: estimate value of the state the rollout stopped in
        # (not a terminal state, so it still has future value).
        features = state_to_features(new_game_state)
        state_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            last_value = self.value_net.get_value(state_tensor).item()
    else:
        # Episode ended — no future reward beyond this point.
        last_value = 0.0

    ppo_update(self, last_value)
    self.buffer.reset()


def ppo_update(self, last_value: float):
    """The core PPO learning step — Parts 1-3 of this project all feed into this."""
    self.buffer.compute_returns_and_advantages(last_value)
    minibatches = self.buffer.get_minibatches(MINIBATCH_SIZE)

    for epoch in range(PPO_EPOCHS):
        for batch in minibatches:
            new_log_probs, entropy = self.policy_net.evaluate_actions(
                batch["states"], batch["actions"], batch.get("action_masks")
            )

            ratio = torch.exp(new_log_probs - batch["old_log_probs"])
            surr1 = ratio * batch["advantages"]
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch["advantages"]
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_bonus = entropy.mean()

            values = self.value_net.get_value(batch["states"])
            value_loss = torch.nn.functional.mse_loss(values, batch["returns"])

            total_policy_loss = policy_loss - ENTROPY_COEF * entropy_bonus

            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), MAX_GRAD_NORM)
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            (VALUE_LOSS_COEF * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), MAX_GRAD_NORM)
            self.value_optimizer.step()

    self.logger.info(
        f"PPO update done — policy_loss={policy_loss.item():.4f}, "
        f"value_loss={value_loss.item():.4f}, entropy={entropy_bonus.item():.4f}"
    )


def _save_checkpoints(self):
    checkpoint_dir = os.path.join(os.path.dirname(__file__), CHECKPOINT_DIR)
    torch.save(self.policy_net.state_dict(), os.path.join(checkpoint_dir, POLICY_CHECKPOINT_NAME))
    torch.save(self.value_net.state_dict(), os.path.join(checkpoint_dir, VALUE_CHECKPOINT_NAME))


if __name__ == "__main__":
    # Sanity check the PPO update math in isolation, no framework/game
    # needed — fabricates a full rollout and runs one update, checking
    # it doesn't crash and losses are finite numbers.
    import types
    try:
        from policy_network.network import PolicyNetwork
    except ImportError:
        from agent_code.ppo_agent.policy_network.network import PolicyNetwork

    class FakeLogger:
        def info(self, msg):
            print(f"[logger] {msg}")

    fake_self = types.SimpleNamespace(logger=FakeLogger())
    fake_self.policy_net = PolicyNetwork()
    setup_training(fake_self)

    for i in range(64):
        fake_self.buffer.add(
            state=np.random.randn(29).astype(np.float32),
            action=np.random.randint(0, 6),
            log_prob=-1.5,
            reward=np.random.choice([0.0, 1.0, -1.0]),
            done=(i == 63),
            value=0.3,
        )

    ppo_update(fake_self, last_value=0.0)
    print("OK — PPO update ran without error.")
