"""
ppo_agent / train.py  --  PPO (clip) + GAE, updates every ROLLOUT steps.

Resumes automatically from my-saved-model.pt (delete it to start from scratch).
"""
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import events as e
from .callbacks import MODEL_PATH, target_distance

# ---- hyper-parameters ------------------------------------------------------
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
LR = 3e-4
EPOCHS = 4
BATCH = 256
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD = 0.5
ROLLOUT = int(os.environ.get('PPO_ROLLOUT', 2048))  # env steps collected between updates
SHAPING = 0.03                                      # potential-based "walk towards goal" reward

BEST_PATH = MODEL_PATH.replace('.pt', '.best.pt')

REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.OPPONENT_ELIMINATED: 0.3,
    e.CRATE_DESTROYED: 0.15,
    e.COIN_FOUND: 0.3,
    e.GOT_KILLED: -5.0,
    e.KILLED_SELF: -3.0,   # (self-kill also triggers GOT_KILLED -> total -8)
    e.SURVIVED_ROUND: 2.0,
    e.INVALID_ACTION: -1.0,
    e.WAITED: -0.05,
}
STEP_PENALTY = -0.01


def reward_from_events(events: List[str]) -> float:
    return STEP_PENALTY + sum(REWARDS.get(ev, 0.0) for ev in events)


def env_score(events: List[str]) -> int:
    return events.count(e.COIN_COLLECTED) + 5 * events.count(e.KILLED_OPPONENT)


# ---- callbacks -------------------------------------------------------------
def setup_training(self):
    self.opt = torch.optim.Adam(self.net.parameters(), lr=LR, eps=1e-5)
    self.buf = {k: [] for k in ('obs', 'mask', 'act', 'logp', 'val', 'rew', 'done')}
    self._roll = None
    self.stats = dict(score=[], ret=[], kills=0, deaths=0, suicides=0, rounds=0)
    self.ep_score = 0
    self.ep_ret = 0.0
    self.n_updates = 0
    self.best = -1e9


def _push(self, reward, done):
    feats, mask, a, logp, val = self._roll
    b = self.buf
    b['obs'].append(feats)
    b['mask'].append(mask)
    b['act'].append(a)
    b['logp'].append(logp)
    b['val'].append(val)
    b['rew'].append(reward)
    b['done'].append(float(done))
    self._roll = None
    self.ep_ret += reward


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if self._roll is None or old_game_state is None or new_game_state is None:
        return
    r = reward_from_events(events)
    r += SHAPING * (target_distance(old_game_state) - GAMMA * target_distance(new_game_state))
    self.ep_score += env_score(events)
    self.stats['kills'] += events.count(e.KILLED_OPPONENT)
    _push(self, r, False)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    b = self.buf
    if self._roll is not None:
        # agent died: its last step never went through game_events_occurred
        _push(self, reward_from_events(events), True)
        self.ep_score += env_score(events)
        self.stats['kills'] += events.count(e.KILLED_OPPONENT)
        self.stats['deaths'] += e.GOT_KILLED in events
        self.stats['suicides'] += e.KILLED_SELF in events
    elif b['done'] and not b['done'][-1]:
        # survived: last step was already recorded, close the episode
        if e.SURVIVED_ROUND in events:
            b['rew'][-1] += REWARDS[e.SURVIVED_ROUND]
            self.ep_ret += REWARDS[e.SURVIVED_ROUND]
        b['done'][-1] = 1.0

    self.stats['score'].append(self.ep_score)
    self.stats['ret'].append(self.ep_ret)
    self.stats['rounds'] += 1
    self.ep_score, self.ep_ret = 0, 0.0

    if len(b['act']) >= ROLLOUT:
        _update(self)


# ---- PPO -------------------------------------------------------------------
def _update(self):
    b = self.buf
    obs = torch.from_numpy(np.stack(b['obs']))
    mask = torch.from_numpy(np.stack(b['mask']))
    act = torch.tensor(b['act'], dtype=torch.long)
    old_logp = torch.tensor(b['logp'], dtype=torch.float32)
    val = torch.tensor(b['val'], dtype=torch.float32)
    rew = torch.tensor(b['rew'], dtype=torch.float32)
    done = torch.tensor(b['done'], dtype=torch.float32)
    n = len(act)

    adv = torch.zeros(n)
    last = 0.0
    for t in reversed(range(n)):
        nonterm = 1.0 - done[t]
        next_val = val[t + 1] if t + 1 < n else 0.0
        delta = rew[t] + GAMMA * next_val * nonterm - val[t]
        last = delta + GAMMA * LAMBDA * nonterm * last
        adv[t] = last
    ret = adv + val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            logits, v = self.net(obs[idx], mask[idx])
            dist = Categorical(logits=logits)
            ratio = (dist.log_prob(act[idx]) - old_logp[idx]).exp()
            pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - CLIP, 1 + CLIP) * adv[idx]).mean()
            vf = F.mse_loss(v, ret[idx])
            ent = dist.entropy().mean()
            loss = pg + VF_COEF * vf - ENT_COEF * ent
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD)
            self.opt.step()
    torch.set_num_threads(1)

    self.n_updates += 1
    st = self.stats
    mean_score = float(np.mean(st['score']))
    print(f"[ppo] update {self.n_updates:4d} | steps {n} | rounds {st['rounds']:3d} | "
          f"score {mean_score:5.2f} | return {np.mean(st['ret']):6.2f} | "
          f"kills {st['kills']} | killed {st['deaths']} | suicides {st['suicides']} | "
          f"entropy {float(ent.detach()):.2f}")

    torch.save(self.net.state_dict(), MODEL_PATH)
    if mean_score > self.best and st['rounds'] >= 5:
        self.best = mean_score
        torch.save(self.net.state_dict(), BEST_PATH)

    for k in b:
        b[k].clear()
    self.stats = dict(score=[], ret=[], kills=0, deaths=0, suicides=0, rounds=0)
