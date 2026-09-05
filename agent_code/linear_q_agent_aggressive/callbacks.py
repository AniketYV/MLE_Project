import os
import pickle
import random

import numpy as np

from .module import ACTIONS, QModel, state_to_features

# Kept nonzero even outside training. A purely deterministic greedy policy
# (epsilon=0) can get stuck in an infinite loop: if two neighbouring tiles
# happen to have symmetric Q-values pulling toward each other, the agent
# bounces between them forever with nothing to break the tie. Confirmed
# happening in real evaluation -- the agent oscillated UP/DOWN between the
# same two tiles for an entire 400-step round, never reaching a crate. This
# is live during actual tournament play too, since self.train is False
# there -- so this can't be training-only exploration, it has to always
# apply.
INFERENCE_EPSILON = 0.08  # AGGRESSIVE variant: raised from 0.03


def setup(self):
    """
    Called once before the first round. Always loads an existing checkpoint
    if one is present, regardless of self.train -- training happens across
    many separate `python main.py` invocations over time, and discarding
    the saved model every time self.train is True would throw away all
    progress from every earlier session except the current one.
    """
    model_path = "my-saved-model.pt"
    if os.path.isfile(model_path):
        self.logger.info("Loading model from saved state.")
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)
    else:
        self.logger.info("No saved model found, starting from scratch.")
        self.model = QModel()


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)

    effective_epsilon = self.model.epsilon if self.train else INFERENCE_EPSILON
    if random.random() < effective_epsilon:
        self.logger.debug("Exploring: random action.")
        # Slightly favour movement over WAIT/BOMB during exploration so the
        # agent actually covers the board instead of dithering.
        return np.random.choice(ACTIONS, p=[.19, .19, .19, .19, .14, .10])

    q_values = self.model.predict(features)
    action = ACTIONS[int(np.argmax(q_values))]
    self.logger.debug(f"Exploiting: chose {action} (Q={q_values}).")
    return action
