"""Rule spot checks carried over from the original notebook, plus a small solver cross-check."""
import random

from balatro_mdp import BalatroMDP, Config, ExactSolver, ReferenceSolver, score_hand


def test_royal_flush_score():
    assert score_hand([(14, 0), (13, 0), (12, 0), (11, 0), (10, 0)]) == (100 + 51) * 8


def test_pair_score():
    assert score_hand([(5, 0), (5, 1)]) == (10 + 10) * 2


def test_wheel_straight_score():
    assert score_hand([(14, 0), (2, 1), (3, 2), (4, 3), (5, 0)]) == (30 + 11 + 2 + 3 + 4 + 5) * 4


def test_opening_action_count():
    mdp = BalatroMDP()
    assert len(mdp.legal_actions(mdp.initial_state(random.Random(0)))) == 436


def test_exact_matches_reference_on_tiny_deck():
    mdp = BalatroMDP(Config.tiny())
    s = mdp.initial_state(random.Random(3))
    fast, ref = ExactSolver(mdp).solve(s), ReferenceSolver(mdp)
    assert max(abs(fast[a] - ref.q(s, a)) for a in fast) < 1e-9
