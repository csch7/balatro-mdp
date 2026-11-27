"""Flat Monte Carlo policy improvement over the top-ranked actions (no tree)."""
from __future__ import annotations

from game.rules import Action
from game.simulator import Sim, rollout
from policies.base import Policy
from policies.ordering import ordered_actions


class FlatMC(Policy):
    """Evaluate the top-m ranked actions with n rollouts each (first step sampled, then
    `rollout_policy`) and play the best mean."""
    def __init__(self, sim: Sim, rollout_policy: Policy | None, m=20, n=10):
        self.sim, self.rollout_policy, self.m, self.n = sim, rollout_policy, m, n

    def act(self, s, rng) -> Action:
        best, best_val = None, -1.0
        for a in ordered_actions(self.sim.mdp, s)[:self.m]:
            tot = 0.0
            for _ in range(self.n):
                ns, r, _ = self.sim.step(s, a, rng)
                tot += r + rollout(self.sim, ns, self.rollout_policy, rng)
            if tot / self.n > best_val:
                best, best_val = a, tot / self.n
        return best
