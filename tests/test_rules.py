"""Scoring spot checks carried over from the original notebook."""
from game import deck_config, score_hand

TABLE = deck_config().hand_table


def test_royal_flush_score():
    assert score_hand([(14, 0), (13, 0), (12, 0), (11, 0), (10, 0)], TABLE) == (100 + 51) * 8


def test_pair_score():
    assert score_hand([(5, 0), (5, 1)], TABLE) == (10 + 10) * 2


def test_wheel_straight_score():
    assert score_hand([(14, 0), (2, 1), (3, 2), (4, 3), (5, 0)], TABLE) == (30 + 11 + 2 + 3 + 4 + 5) * 4
