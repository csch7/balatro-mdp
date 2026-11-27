"""Black-box simulator wrapper that counts generative-model calls, and rollouts on it."""
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


def rollout(sim: Sim, s, policy, rng) -> float:
    """Return of following `policy` from s in the simulator (sampled draws). policy=None gives 0."""
    if policy is None:
        return 0.0
    total = 0.0
    while not sim.is_terminal(s):
        s, r, _ = sim.step(s, policy.act(s, rng), rng)
        total += r
    return total
