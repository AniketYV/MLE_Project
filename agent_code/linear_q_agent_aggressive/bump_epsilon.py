"""
Run this ONCE before continuing training on a scenario the model hasn't
seen much of yet (e.g. switching from coin-heaven to classic). It raises
epsilon back up so the agent actually explores the new dynamics instead of
just exploiting a policy tuned for the old scenario -- without discarding
any learned weights. Works from any working directory.
"""
import os
import sys
import pickle

NEW_EPSILON = 0.3

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "my-saved-model.pt")
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(model_path, "rb") as f:
    model = pickle.load(f)

print(f"Current epsilon: {model.epsilon:.3f}")
model.epsilon = NEW_EPSILON
print(f"New epsilon:     {model.epsilon:.3f}")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("Saved. Weights unchanged, only epsilon was bumped.")
