"""The fast exact solver must agree with the plain reference implementation."""
import random

from game import BalatroMDP, deck_config
from solvers import ExactSolver, ReferenceSolver, TooBig


def test_exact_matches_reference_on_tiny_deck():
    mdp = BalatroMDP(deck_config("tiny", reward_mode="target"))
    s = mdp.initial_state(random.Random(3))
    fast, ref = ExactSolver(mdp).solve(s), ReferenceSolver(mdp)
    assert max(abs(fast[a] - ref.q(s, a)) for a in fast) < 1e-9


def test_full_game_is_refused():
    mdp = BalatroMDP(deck_config("standard", reward_mode="target"))
    try:
        ExactSolver(mdp).solve(mdp.initial_state(random.Random(0)))
    except TooBig:
        return
    raise AssertionError("full game should exceed the exact-solve limit")
