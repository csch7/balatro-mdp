"""Smoke tests: each planner returns a legal action quickly."""
import random

from game import BalatroMDP, Config, Sim
from planners import UCT, FlatMC, SparseSampling
from policies import greedy_rollout


def _setup():
    mdp = BalatroMDP(Config.tiny())
    return mdp, Sim(mdp), mdp.initial_state(random.Random(4))


def test_uct_returns_legal_action():
    mdp, sim, s = _setup()
    assert UCT(sim, n_iter=50).plan(s, random.Random(0)) in mdp.legal_actions(s)


def test_sparse_sampling_returns_legal_action():
    mdp, sim, s = _setup()
    a = SparseSampling(sim, C=1, H=1, leaf=greedy_rollout).plan(s, random.Random(0))
    assert a in mdp.legal_actions(s)


def test_flat_mc_returns_legal_action():
    mdp, sim, s = _setup()
    assert FlatMC(sim, m=5, n=2).plan(s, random.Random(0)) in mdp.legal_actions(s)
