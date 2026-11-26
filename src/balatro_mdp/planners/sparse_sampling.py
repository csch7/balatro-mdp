"""Sparse sampling planner (Kearns, Mansour & Ng, 2002)."""
from __future__ import annotations

from ..game.rules import Action
from ..game.simulator import Sim


class SparseSampling:
    """Depth-H lookahead tree: every action at every node, C sampled next states each.
    Cost O((k*C)^H) simulator calls per decision, independent of |S|.
    Finite-horizon, undiscounted. `leaf(sim, s, rng)` replaces the zero leaf value."""
    def __init__(self, sim: Sim, C: int = 1, H: int | None = None, leaf=None):
        self.sim, self.C, self.H, self.leaf = sim, C, H, leaf

    def _v(self, s, h, rng):
        if self.sim.is_terminal(s):
            return 0.0
        if h == 0:
            return self.leaf(self.sim, s, rng) if self.leaf else 0.0
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

    def plan(self, s, rng) -> Action:
        return max(self.q_values(s, rng), key=lambda t: t[1])[0]
