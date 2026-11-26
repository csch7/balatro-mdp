"""Ablation of UCT components on the full game (Big Blind).

One component is changed at a time; all variants see the same starting hands as the main run.
"""
from __future__ import annotations

import math
import random
import time

from game.config import Config
from game.mdp import BalatroMDP, State
from game.simulator import Sim, run_episode
from planners.uct import UCT
from policies.baselines import greedy_policy, heuristic_policy


def plain_key(s: State):
    """No suit-symmetry reduction."""
    return (s.hand, s.deck, s.hands_left, s.discards_left, s.score)


def random_rollout(sim: Sim, s: State, rng) -> float:
    total = 0.0
    while not sim.is_terminal(s):
        s, r, _ = sim.step(s, rng.choice(sim.legal_actions(s)), rng)
        total += r
    return total


def zero_rollout(sim: Sim, s: State, rng) -> float:
    return 0.0


def shuffled_order(mdp, s):
    acts = mdp.legal_actions(s)
    random.shuffle(acts)
    return acts


ABLATIONS = {   # name -> UCT keyword overrides (everything else = the reported configuration)
    "full (reported)": {},
    "plain key (no suit merging)": {"key": plain_key},
    "no progressive widening": {"widen": False},
    "random action order": {"action_order": shuffled_order},
    "zero leaf value (no rollout)": {"rollout": zero_rollout},
    "random rollout": {"rollout": random_rollout},
    "c = 0.25": {"c": 0.25},
    "c = 2.0": {"c": 2.0},
    "100 iterations": {"n_iter": 100},
}


def experiment_ablation(episodes=100, iters=300, seed=0, target=450, variants=None):
    mdp = BalatroMDP(Config(reward_mode="target", target=target))
    w = sum(run_episode(mdp, greedy_policy, random.Random(seed * 100_000 + ep)) for ep in range(episodes))
    p = w / episodes
    print(f"{'greedy play':32s} win={p:.3f} +/- {math.sqrt(p * (1 - p) / episodes):.3f}", flush=True)
    w = sum(run_episode(mdp, heuristic_policy(), random.Random(seed * 100_000 + ep)) for ep in range(episodes))
    p = w / episodes
    print(f"{'discard heuristic (no search)':32s} win={p:.3f} +/- {math.sqrt(p * (1 - p) / episodes):.3f}", flush=True)
    for name in (variants or list(ABLATIONS)):
        sim = Sim(mdp)
        uct = UCT(sim, **{"n_iter": iters, **ABLATIONS[name]})
        t0, wins = time.time(), 0
        for ep in range(episodes):
            random.seed(ep)                      # for shuffled_order / random_rollout
            wins += run_episode(mdp, lambda m, s, r: uct.plan(s, r), random.Random(seed * 100_000 + ep))
        p = wins / episodes
        reuse = uct.hits / max(uct.hits + uct.expansions, 1)
        print(f"{name:32s} win={p:.3f} +/- {math.sqrt(p * (1 - p) / episodes):.3f}  "
              f"{(time.time() - t0) / episodes:5.1f}s/game  {sim.calls / episodes:7.0f} calls/game  "
              f"node reuse {100 * reuse:4.1f}%", flush=True)
