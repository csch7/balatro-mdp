"""Scoring spot checks carried over from the original notebook."""
from balatro_mdp.game import score_hand


def test_royal_flush_score():
    assert score_hand([(14, 0), (13, 0), (12, 0), (11, 0), (10, 0)]) == (100 + 51) * 8


def test_pair_score():
    assert score_hand([(5, 0), (5, 1)]) == (10 + 10) * 2


def test_wheel_straight_score():
    assert score_hand([(14, 0), (2, 1), (3, 2), (4, 3), (5, 0)]) == (30 + 11 + 2 + 3 + 4 + 5) * 4
