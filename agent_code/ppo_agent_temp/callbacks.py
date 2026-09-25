"""
setup() and act() for the PPO agent -- the interface the game framework
actually calls. All the real logic lives in network.py (the model),
observation.py (state encoding), and ppo.py (the learning algorithm).
"""
import os

import torch

from .network import ActorCritic, ACTIONS
from .observation import encode, compute_danger_tiles, _bfs_escape_direction

MODEL_PATH = "ppo-model.pt"
DIR_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


def setup(self):
    """
    Called once before the first round. Always loads an existing
    checkpoint if one is present, for the same reason as every other
    agent in this project: training happens across many separate
    `python main.py` invocations, and discarding progress every time
    would throw away everything except the current session.
    """
    self.device = "cpu"  # tournament runs on CPU; keep inference on CPU too
    self.network = ActorCritic()
    if os.path.isfile(MODEL_PATH):
        self.logger.info("Loading PPO model from saved state.")
        self.network.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
    else:
        self.logger.info("No saved PPO model found, starting from scratch.")
    self.network.to(self.device)
    self.network.eval()


def act(self, game_state: dict) -> str:
    grid, scalars = encode(game_state)
    grid_t = torch.tensor(grid, dtype=torch.float32, device=self.device).unsqueeze(0)
    scalars_t = torch.tensor(scalars, dtype=torch.float32, device=self.device).unsqueeze(0)

    with torch.no_grad():
        logits, value = self.network(grid_t, scalars_t)
        dist = torch.distributions.Categorical(logits=logits)
        # PPO's policy is a probability distribution by construction, and
        # sampled from even during real (non-training) play, not just
        # during training. Unlike a purely deterministic argmax policy,
        # this can never get stuck in the kind of infinite tie-breaking
        # loop found and fixed in an earlier agent in this project --
        # there's no such thing as an exact tie forcing identical
        # repeated behaviour, since the action is always drawn from a
        # genuine distribution.
        action_idx = dist.sample()

        # Hard safety override: extensive diagnosis (danger_diagnostics.csv)
        # showed the learned policy was following a genuinely-existing
        # escape route in roughly 0-12% of opportunities across many
        # training attempts, tried across six distinct reward/architecture
        # fixes -- the signal was correct and available, the policy just
        # wasn't reliably using it. Rather than keep hoping further reward
        # engineering eventually teaches this specific behaviour, a
        # non-learned rule directly enforces it: if a real escape route
        # exists right now, take it, full stop. This is the same principle
        # recommended earlier in this project for what makes a robust
        # tournament agent in general -- a hard safety layer on top of a
        # learned policy, not a replacement for one.
        field = game_state['field']
        _name, _score, _bomb, pos = game_state['self']
        danger_tiles = compute_danger_tiles(game_state)
        if pos in danger_tiles:
            others = game_state['others']
            bombs = game_state['bombs']
            occupied = set((ox, oy) for (_n, _s, _b, (ox, oy)) in others)
            occupied.update(bpos for (bpos, _t) in bombs)
            escape_onehot = _bfs_escape_direction(pos, field, occupied, danger_tiles)
            if sum(escape_onehot) > 0:
                escape_dir_idx = escape_onehot.index(1.0)
                action_idx = torch.tensor(escape_dir_idx)

        # Recompute log_prob for whichever action is actually being taken
        # (whether the network's own sample or the safety override) so
        # PPO's training math stays internally consistent -- the stored
        # log_prob must always reflect the probability the CURRENT policy
        # assigns to the action actually executed, not the action it
        # originally happened to sample.
        log_prob = dist.log_prob(action_idx)

    # Stashed here so train.py can record this transition once the game
    # reports what actually happened as a result of this action.
    self._last_grid = grid
    self._last_scalars = scalars
    self._last_action = int(action_idx.item())
    self._last_log_prob = float(log_prob.item())
    self._last_value = float(value.item())

    return ACTIONS[self._last_action]
