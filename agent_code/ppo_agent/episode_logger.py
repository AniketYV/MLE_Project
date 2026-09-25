"""
Logs one row per episode to a CSV file: reward, coins collected,
opponents killed, whether the agent survived, and episode length.

This is what actually tells you whether the agent is improving — loss
values alone can look "fine" while the agent isn't getting any better
at the game. Plot episode_reward or coins_collected over time once you
have a few hundred/thousand rows.
"""

import os
import csv

LOG_FILENAME = "episode_log.csv"
CSV_HEADERS = [
    "episode", "steps", "reward", "coins_collected",
    "opponents_killed", "survived", "self_destructed", "stage",
]


def _current_stage() -> str:
    """Reads the stage name from the PPO_STAGE_NAME env var, set by
    train_pipeline.py before each stage's subprocess. Empty string for
    ad-hoc training runs outside the curriculum pipeline."""
    return os.environ.get("PPO_STAGE_NAME", "")


def _log_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILENAME)


def init_episode_tracking(self):
    """Call once in setup_training(). Resets per-episode counters and
    creates the CSV with headers if it doesn't already exist."""
    self.episode_reward = 0.0
    self.episode_steps = 0
    self.episode_coins = 0
    self.episode_kills = 0
    self.episode_count = getattr(self, "episode_count", 0)

    path = _log_path()
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(CSV_HEADERS)


def track_step(self, reward: float, events: list):
    """Call once per step (from game_events_occurred) to accumulate
    this episode's running totals."""
    self.episode_reward += reward
    self.episode_steps += 1
    if "COIN_COLLECTED" in events:
        self.episode_coins += 1
    if "KILLED_OPPONENT" in events:
        self.episode_kills += 1


def log_episode_end(self, events: list):
    """Call once per episode (from end_of_round, AFTER track_step has
    been called for the final step) to write the row and reset counters."""
    self.episode_count += 1
    survived = 1 if ("GOT_KILLED" not in events and "KILLED_SELF" not in events) else 0
    self_destructed = 1 if "KILLED_SELF" in events else 0

    with open(_log_path(), "a", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow([
            self.episode_count, self.episode_steps, round(self.episode_reward, 3),
            self.episode_coins, self.episode_kills, survived, self_destructed,
            _current_stage(),
        ])

    self.logger.info(
        f"Episode {self.episode_count} — reward={self.episode_reward:.2f}, "
        f"steps={self.episode_steps}, coins={self.episode_coins}, "
        f"kills={self.episode_kills}, survived={survived}"
    )

    # Reset for the next episode.
    self.episode_reward = 0.0
    self.episode_steps = 0
    self.episode_coins = 0
    self.episode_kills = 0


if __name__ == "__main__":
    import types

    class FakeLogger:
        def info(self, msg):
            print(f"[logger] {msg}")

    fake_self = types.SimpleNamespace(logger=FakeLogger())
    init_episode_tracking(fake_self)

    # Simulate a 5-step episode: coin at step 2, dies at the end.
    track_step(fake_self, reward=0.0, events=[])
    track_step(fake_self, reward=1.0, events=["COIN_COLLECTED"])
    track_step(fake_self, reward=0.0, events=[])
    track_step(fake_self, reward=0.0, events=[])
    track_step(fake_self, reward=-5.0, events=["GOT_KILLED"])
    log_episode_end(fake_self, events=["GOT_KILLED"])

    print(f"CSV written to: {_log_path()}")
    with open(_log_path()) as f:
        print(f.read())
