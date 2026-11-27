"""Sparse sampling planner (Kearns, Mansour & Ng, 2002)."""
from __future__ import annotations

from game.rules import Action
from game.simulator import Sim, rollout
from policies.base import Policy


class SparseSampling(Policy):
    """Depth-H lookahead tree: every action at every node, C sampled next states each.
    Cost O((k*C)^H) simulator calls per decision, independent of |S|. Finite-horizon,
    undiscounted. Leaves at depth H are valued by a rollout of `leaf_policy` (None = 0);
    H=None looks ahead to the end of the round."""
    def __init__(self, sim: Sim, leaf_policy: Policy | None = None, C: int = 1, H: int | None = None):
        self.sim, self.leaf_policy, self.C, self.H = sim, leaf_policy, C, H

    def _v(self, s, h, rng):
        if self.sim.is_terminal(s):
            return 0.0
        if h == 0:
            return rollout(self.sim, s, self.leaf_policy, rng)
        return max(q for _, q in self.q_values(s, rng, h))

    def q_values(self, s, rng, h=None):
        h = h if h is not None else (self.H or s.hands_left + s.discards_left)
        out = []
        for a in self.sim.legal_actions(s):
            tot = 0.0
            for _ in range(self.C):
                ns, r, _ = self.sim.step(s, a, rng)
                tot += r + self._v(ns, h - 1, rng)
            out.append((a, tot / self.C))
        return out

    def act(self, s, rng) -> Action:
        return max(self.q_values(s, rng), key=lambda t: t[1])[0]
