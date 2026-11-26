"""Black-box simulator wrapper that counts generative-model calls, and an episode runner."""
from __future__ import annotations

from .mdp import BalatroMDP


class Sim:
    """Black-box generative interface that counts simulator calls."""
    def __init__(self, mdp: BalatroMDP):
        self.mdp, self.cfg, self.calls = mdp, mdp.cfg, 0

    def step(self, s, a, rng):
        self.calls += 1
        return self.mdp.step(s, a, rng)

    def legal_actions(self, s): return self.mdp.legal_actions(s)
    def is_terminal(self, s): return self.mdp.is_terminal(s)
    def canonical(self, s): return self.mdp.canonical(s)


def run_episode(mdp: BalatroMDP, policy, rng) -> float:
    s, total = mdp.initial_state(rng), 0.0
    while not mdp.is_terminal(s):
        s, r, _ = mdp.step(s, policy(mdp, s, rng), rng)
        total += r
    return total
