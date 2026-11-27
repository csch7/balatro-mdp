"""Deliberately simple second exact solver, used only to cross-check ExactSolver."""
from __future__ import annotations


class ReferenceSolver:
    """Plain recursion over mdp.transitions (no shortcuts), memoized on the canonical state."""
    def __init__(self, mdp, max_outcomes: int):
        self.mdp, self.max_outcomes, self.memo = mdp, max_outcomes, {}

    def q(self, s, a):
        return sum(p * (r + (0.0 if self.mdp.is_terminal(ns) else self.v(ns)))
                   for p, ns, r in self.mdp.transitions(s, a, self.max_outcomes))

    def v(self, s):
        key = self.mdp.canonical(s)
        if key not in self.memo:
            self.memo[key] = max(self.q(s, a) for a in self.mdp.legal_actions(s))
        return self.memo[key]
