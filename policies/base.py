"""The common policy interface: everything that picks actions (baselines and planners) has
`act(state, rng) -> action`. Planners also expose `sim` (simulator-call counter) and `stats()`."""
from __future__ import annotations

from game.rules import Action


class Policy:
    label: str = ""
    sim = None                     # set by planners: the Sim whose calls they make

    def act(self, s, rng) -> Action:
        raise NotImplementedError

    def stats(self) -> dict:
        """Extra numbers to report for this policy (e.g. UCT node reuse)."""
        return {}
