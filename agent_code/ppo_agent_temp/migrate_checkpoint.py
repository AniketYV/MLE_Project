"""
Migrates an existing PPO checkpoint into the CURRENT architecture defined
in network.py, preserving as much learned progress as possible instead of
starting over. Generically handles ANY parameter whose shape grew along
exactly one dimension (e.g. conv.0.weight when a channel is added,
trunk.0.weight when a scalar is added) -- works regardless of which part
of the architecture changed, or how many times it's changed before.

For each such parameter, old values are copied into the first N entries
along whichever dimension grew; the newly added entries keep their random
initialization, since there's no prior experience to transfer for
something that didn't exist before. Parameters with unchanged shape
(all three conv layers' later weights, both actor/critic heads, biases
generally) transfer over completely as-is.

Run this ONCE, from inside agent_code/ppo_agent/, before continuing
training, any time observation.py's N_CHANNELS or N_SCALARS changes:
    python migrate_checkpoint.py
"""
import os

import torch

from network import ActorCritic

MODEL_PATH = "ppo-model.pt"


def _grow_dim(old_tensor, new_shape):
    """Returns a new tensor of new_shape, with old_tensor's values copied
    into the first old_tensor.shape[i] entries along whichever dimension
    differs. The newly added entries are zero-initialized (not random) --
    this makes the migrated network's output IDENTICAL to the
    pre-migration network at the moment of migration, since a zero-weight
    contributes nothing regardless of its input. Training will gradually
    learn to use the new channel from there, without any migration-
    induced randomness destabilizing behaviour right after the switch."""
    diffs = [i for i in range(len(new_shape)) if new_shape[i] != old_tensor.shape[i]]
    if len(diffs) == 0:
        return old_tensor
    if len(diffs) > 1:
        raise RuntimeError(f"More than one dimension changed: old={old_tensor.shape} new={new_shape} -- can't auto-migrate, handle manually.")
    dim = diffs[0]
    if new_shape[dim] < old_tensor.shape[dim]:
        raise RuntimeError(f"Dimension {dim} SHRANK ({old_tensor.shape[dim]} -> {new_shape[dim]}) -- this script only handles growing. Migrate manually.")
    new_tensor = torch.zeros(new_shape, dtype=old_tensor.dtype)
    slicer = [slice(None)] * len(new_shape)
    slicer[dim] = slice(0, old_tensor.shape[dim])
    new_tensor[tuple(slicer)] = old_tensor
    return new_tensor


def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"No checkpoint found at {MODEL_PATH} -- nothing to migrate.")
        return

    old_state = torch.load(MODEL_PATH, map_location="cpu")
    new_net = ActorCritic()  # uses whatever N_CHANNELS/N_SCALARS are currently in the code
    new_state = new_net.state_dict()

    if all(old_state[k].shape == new_state[k].shape for k in new_state if k in old_state):
        print("Checkpoint already matches the current architecture -- nothing to do.")
        return

    changed_keys = [k for k in new_state if k in old_state and old_state[k].shape != new_state[k].shape]
    backup_suffix = "_".join(changed_keys[:1]) if changed_keys else "unknown"
    backup_path = f"ppo-model.before_migration_{backup_suffix.replace('.', '_')}.pt"
    os.rename(MODEL_PATH, backup_path)
    print(f"Backed up old checkpoint to {backup_path}")

    for key in new_state:
        if key not in old_state:
            print(f"  {key}: new parameter, keeping random initialization")
            continue
        if new_state[key].shape == old_state[key].shape:
            new_state[key] = old_state[key]
        else:
            print(f"  {key}: shape {tuple(old_state[key].shape)} -> {tuple(new_state[key].shape)}, transplanting old values")
            new_state[key] = _grow_dim(old_state[key], new_state[key].shape)

    new_net.load_state_dict(new_state)
    torch.save(new_net.state_dict(), MODEL_PATH)
    print(f"Migrated checkpoint saved to {MODEL_PATH}. "
          f"All unchanged-shape layers transferred exactly; changed layers kept their old values where possible.")


if __name__ == "__main__":
    main()
