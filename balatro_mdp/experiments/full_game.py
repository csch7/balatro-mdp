"""Experiments on the real 52-card game: greedy vs UCT, no-search baselines, search vs heuristic."""
from __future__ import annotations

import math
import random
import time

from ..mdp import BalatroMDP, Config, Sim, run_episode
from ..policies import (greedy_policy, greedy_rollout, greedy_smallest_policy, heuristic_policy,
                        heuristic_rollout)
from ..uct import UCT, FlatMC


def experiment_full(episodes=100, iters=300, seed=0, target=450):
    """P(reach target) on the real 52-card game: greedy-play vs UCT, same initial hands."""
    mdp = BalatroMDP(Config(reward_mode="target", target=target))
    sim = Sim(mdp); uct = UCT(sim, n_iter=iters)
    wins, t0 = {"greedy-play": 0, "UCT": 0}, time.time()
    for ep in range(episodes):
        for name, pol in (("greedy-play", greedy_policy), ("UCT", lambda m, s, r: uct.plan(s, r))):
            wins[name] += run_episode(mdp, pol, random.Random(seed * 100_000 + ep))
        if (ep + 1) % 10 == 0:
            print(f"  {ep + 1}/{episodes} episodes, {time.time() - t0:.0f}s", flush=True)
    for name, w in wins.items():
        p = w / episodes
        print(f"{name:12s} P(reach {target}) = {p:.3f} +/- {math.sqrt(p * (1 - p) / episodes):.3f}")


def experiment_baselines(games=1500, targets=(300, 450, 600), seed=11):
    """No-search policies on the full game at each Ante-1 blind (fresh starting hands)."""
    pols = {
        "greedy, smallest best subset (bad tie-break)": greedy_smallest_policy,
        "greedy, no discards (5-card plays)": greedy_policy,
        "discard junk if High Card": heuristic_policy(("High Card",)),
        "discard junk if High Card/Pair": heuristic_policy(("High Card", "Pair")),
        "discard junk if <= Two Pair": heuristic_policy(("High Card", "Pair", "Two Pair")),
    }
    for tgt in targets:
        mdp = BalatroMDP(Config(reward_mode="target", target=tgt))
        print(f"\nTarget {tgt}  ({games} games each)")
        for name, pol in pols.items():
            w = sum(run_episode(mdp, pol, random.Random(seed * 1_000_000 + ep)) for ep in range(games))
            p = w / games
            print(f"  {name:46s} {p:.3f} +/- {math.sqrt(p * (1 - p) / games):.3f}", flush=True)


def experiment_search(episodes=100, iters=300, target=450, seed=0, methods=None):
    """Search vs the discard heuristic on the same starting hands; paired difference vs heuristic."""
    mdp = BalatroMDP(Config(reward_mode="target", target=target))
    makers = {
        "greedy, no discards": lambda: (None, None),
        "discard heuristic (no search)": lambda: (None, None),
        "UCT, greedy rollouts": lambda: _planner(mdp, UCT, iters, greedy_rollout),
        "UCT, heuristic rollouts": lambda: _planner(mdp, UCT, iters, heuristic_rollout),
        "flat MC, heuristic rollouts": lambda: _planner(mdp, FlatMC, iters, heuristic_rollout),
    }
    names = methods or list(makers)
    outcomes = {}
    for name in names:
        plan, sim = makers[name]()
        t0, res = time.time(), []
        for ep in range(episodes):
            rng = random.Random(seed * 100_000 + ep)
            if name == "greedy, no discards":
                res.append(run_episode(mdp, greedy_policy, rng))
            elif name == "discard heuristic (no search)":
                res.append(run_episode(mdp, heuristic_policy(), rng))
            else:
                res.append(run_episode(mdp, lambda m, s, r: plan.plan(s, r), rng))
        outcomes[name] = res
        p = sum(res) / episodes
        extra = f"{(time.time() - t0) / episodes:5.1f}s/game  {sim.calls / episodes:7.0f} calls/game" if sim else ""
        line = f"{name:32s} win={p:.3f} +/- {math.sqrt(p * (1 - p) / episodes):.3f}  {extra}"
        ref = outcomes.get("discard heuristic (no search)")
        if ref and name != "discard heuristic (no search)":
            d = [a - b for a, b in zip(res, ref)]
            md = sum(d) / episodes
            sd = math.sqrt(sum((x - md) ** 2 for x in d) / (episodes - 1) / episodes)
            line += f"  vs heuristic: {md:+.3f} +/- {sd:.3f}"
        print(line, flush=True)
    return outcomes


def _planner(mdp, cls, iters, rollout):
    sim = Sim(mdp)
    if cls is UCT:
        return UCT(sim, n_iter=iters, rollout=rollout), sim
    return FlatMC(sim, m=20, n=max(iters // 20, 1), rollout=rollout), sim     # same iteration budget
