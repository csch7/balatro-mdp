"""Smoke tests: each method built from a config spec returns a legal action."""
import random

import pytest

from game import BalatroMDP, deck_config
from planners import build_policy

SPECS = ["greedy", "heuristic", "random", {"type": "greedy", "tie_break": "smallest"},
         {"type": "uct", "iters": 50}, {"type": "uct", "iters": 50, "rollout": None, "key": "plain"},
         {"type": "uct", "iters": 50, "action_order": "shuffled", "rollout": "random"},
         {"type": "sparse", "H": 1, "C": 1, "leaf": "greedy"}, {"type": "flat_mc", "m": 5, "n": 2}]


@pytest.mark.parametrize("spec", SPECS)
def test_method_returns_legal_action(spec):
    mdp = BalatroMDP(deck_config("tiny", reward_mode="target"))
    s = mdp.initial_state(random.Random(4))
    assert build_policy(spec, mdp).act(s, random.Random(0)) in mdp.legal_actions(s)


def test_bad_specs_are_rejected():
    mdp = BalatroMDP(deck_config("tiny"))
    with pytest.raises(ValueError):
        build_policy("mcts", mdp)
    with pytest.raises(ValueError):
        build_policy({"type": "uct", "rollout": "uct"}, mdp)


def test_planner_counts_simulator_calls():
    mdp = BalatroMDP(deck_config("tiny", reward_mode="target"))
    uct = build_policy({"type": "uct", "iters": 30}, mdp)
    uct.act(mdp.initial_state(random.Random(5)), random.Random(0))
    assert uct.sim.calls > 0 and 0.0 <= uct.stats()["node_reuse"] <= 1.0
