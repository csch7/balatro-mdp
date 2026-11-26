"""Flat Monte Carlo policy improvement over the top-ranked actions (no tree)."""
from __future__ import annotations

from ..game.rules import Action
from ..game.simulator import Sim
from ..policies.baselines import heuristic_rollout
from ..policies.ordering import ordered_actions


class FlatMC:
    """Flat Monte Carlo policy improvement (no tree): evaluate the top-m ranked actions with n
    rollouts each (first step sampled, then `rollout`) and play the best mean."""
    def __init__(self, sim: Sim, m=20, n=10, rollout=heuristic_rollout):
        self.sim, self.m, self.n, self.rollout = sim, m, n, rollout

    def plan(self, s, rng) -> Action:
        best, best_val = None, -1.0
        for a in ordered_actions(self.sim.mdp, s)[:self.m]:
            tot = 0.0
            for _ in range(self.n):
                ns, r, _ = self.sim.step(s, a, rng)
                tot += r + self.rollout(self.sim, ns, rng)
            if tot / self.n > best_val:
                best, best_val = a, tot / self.n
        return best
