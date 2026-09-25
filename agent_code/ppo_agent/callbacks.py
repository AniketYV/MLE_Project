"""
The framework calls setup() once at agent creation, then act() once per
game step. This file's only job is wiring: game_state -> features ->
policy network -> action string. No training logic lives here — that's
train.py's job (Part 5/6).
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import ACTIONS, CHECKPOINT_DIR, POLICY_CHECKPOINT_NAME
    from features import state_to_features, build_bomb_safety_mask
    from policy_network.network import PolicyNetwork
except ImportError:
    from agent_code.ppo_agent.config import ACTIONS, CHECKPOINT_DIR, POLICY_CHECKPOINT_NAME
    from agent_code.ppo_agent.features import state_to_features, build_bomb_safety_mask
    from agent_code.ppo_agent.policy_network.network import PolicyNetwork


def setup(self):
    """
    Called once when the agent is created (by the framework).
    self is the agent object the framework gives you — same pattern as
    your other agents (dqn_agent, linear_q_agent, etc).
    """
    self.policy_net = PolicyNetwork()

    checkpoint_path = os.path.join(
        os.path.dirname(__file__), CHECKPOINT_DIR, POLICY_CHECKPOINT_NAME
    )
    if os.path.exists(checkpoint_path):
        self.policy_net.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        self.policy_net.eval()
        self.logger.info(f"Loaded policy checkpoint from {checkpoint_path}")
    else:
        self.logger.info("No policy checkpoint found — using randomly initialized network.")

    # Used by train.py to know whether it should explore/collect log_probs
    # or just play greedily. Defaults to eval-style play.
    self.train_mode = getattr(self, "train", False)


def act(self, game_state: dict) -> str:
    """
    Called once per game step. Must return one of the strings in
    config.ACTIONS.
    """
    features = state_to_features(game_state)
    state_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, STATE_DIM]

    mask = build_bomb_safety_mask(game_state)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, ACTION_DIM]

    with torch.no_grad():
        if self.train_mode:
            # Sample from the distribution — exploration during training.
            # Mask applied here means BOMB literally cannot be sampled
            # when unsafe, not just discouraged via reward.
            action_idx, log_prob = self.policy_net.act(state_tensor, action_mask=mask_tensor)
            # Stash for train.py to pick up and add to the rollout buffer.
            self.last_log_prob = log_prob.item()
            self.last_value = None  # filled in by train.py using the value network
            self.last_action_mask = mask  # train.py needs the SAME mask for the PPO update
            action_idx = action_idx.item()
        else:
            # Greedy play — take the most likely ALLOWED action, no exploration.
            logits = self.policy_net(state_tensor)
            logits = logits.masked_fill(mask_tensor == 0, float("-inf"))
            action_idx = torch.argmax(logits, dim=-1).item()

    return ACTIONS[action_idx]


if __name__ == "__main__":
    # Standalone sanity check without the full framework — fakes just
    # enough of `self` and a game_state to confirm act() returns a
    # valid action string without crashing.
    import types

    class FakeLogger:
        def info(self, msg):
            print(f"[logger] {msg}")

    fake_self = types.SimpleNamespace(logger=FakeLogger(), train=False)
    setup(fake_self)

    field = np.zeros((7, 7), dtype=int)
    field[0, :] = -1
    field[:, 0] = -1
    fake_state = {
        "field": field,
        "self": ("me", 0, 1, (1, 1)),
        "others": [],
        "bombs": [],
        "coins": [(5, 5)],
        "explosion_map": np.zeros((7, 7), dtype=int),
    }

    chosen_action = act(fake_self, fake_state)
    print(f"Chosen action: {chosen_action}")
    assert chosen_action in ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]
    print("OK — valid action returned.")

    # Masking check: agent with 0 bombs left must never choose BOMB.
    fake_state_no_bombs = dict(fake_state)
    fake_state_no_bombs["self"] = ("me", 0, 0, (1, 1))
    fake_self.train_mode = True  # exercise the sampling path, not just greedy
    for _ in range(50):
        action = act(fake_self, fake_state_no_bombs)
        assert action != "BOMB", "Masked BOMB action was still chosen!"
    print("OK — BOMB never chosen when masked out (0 bombs left), across 50 samples.")
