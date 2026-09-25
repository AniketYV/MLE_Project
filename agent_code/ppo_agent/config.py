"""
All PPO hyperparameters and constants live here.
Keeping them in one file means if training goes wrong, you have exactly
one place to check/adjust — not scattered magic numbers across files.
"""

# --- Action space -----------------------------------------------------
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]
ACTION_DIM = len(ACTIONS)

# --- State representation ---------------------------------------------
# Matches the length of the vector returned by features.state_to_features().
# If you ever change what features.py returns, update this number to match
# (run features.py directly — it prints the length).
STATE_DIM = 29

# --- Network architecture ----------------------------------------------
HIDDEN_DIM = 128          # width of hidden layers, both networks
NUM_HIDDEN_LAYERS = 2     # depth, both networks

# --- PPO core hyperparameters -------------------------------------------
GAMMA = 0.99               # discount factor
GAE_LAMBDA = 0.95          # GAE smoothing parameter
CLIP_EPS = 0.2             # PPO clipping range
VALUE_LOSS_COEF = 0.5      # weight of value loss in total loss
ENTROPY_COEF = 0.01        # weight of entropy bonus (exploration)
MAX_GRAD_NORM = 0.5        # gradient clipping, helps stability

# --- Training loop -------------------------------------------------------
LEARNING_RATE = 3e-4
ROLLOUT_STEPS = 2048       # steps collected before each update
PPO_EPOCHS = 4             # passes over each rollout batch
MINIBATCH_SIZE = 64
NUM_ITERATIONS = 1000      # total training iterations (tune later)

# --- Checkpointing ---------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
POLICY_CHECKPOINT_NAME = "policy-network.pt"
VALUE_CHECKPOINT_NAME = "value-network.pt"
