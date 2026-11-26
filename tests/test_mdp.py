"""Generative model and suit-symmetry checks."""
import random

from game import BalatroMDP, Config


def test_opening_action_count():
    mdp = BalatroMDP()
    assert len(mdp.legal_actions(mdp.initial_state(random.Random(0)))) == 436


def test_transitions_sum_to_one():
    mdp = BalatroMDP(Config.tiny())
    s = mdp.initial_state(random.Random(1))
    a = ("discard", (0, 1))
    assert abs(sum(p for p, _, _ in mdp.transitions(s, a)) - 1.0) < 1e-12


def test_canonical_ignores_suit_relabeling():
    mdp = BalatroMDP(Config.tiny())
    s = mdp.initial_state(random.Random(2))
    swapped = type(s)(tuple(sorted((r, 1 - su) for r, su in s.hand)),
                      tuple(sorted((r, 1 - su) for r, su in s.deck)),
                      s.hands_left, s.discards_left, s.score)
    assert mdp.canonical(s) == mdp.canonical(swapped)
