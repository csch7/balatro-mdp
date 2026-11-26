"""Deliberately simple second exact solver, used only to cross-check ExactSolver."""
from __future__ import annotations


class ReferenceSolver:
    """Deliberately simple second implementation (no shortcuts) used only to cross-check ExactSolver."""
    def __init__(self, mdp):
        self.mdp, self.memo = mdp, {}

    def q(self, s, a):
        return sum(p * (r + (0.0 if self.mdp.is_terminal(ns) else self.v(ns)))
                   for p, ns, r in self.mdp.transitions(s, a))

    def v(self, s):
        key = self.mdp.canonical(s)
        if key not in self.memo:
            self.memo[key] = max(self.q(s, a) for a in self.mdp.legal_actions(s))
        return self.memo[key]
