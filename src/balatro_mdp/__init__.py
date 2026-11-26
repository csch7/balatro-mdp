"""Balatro Ante-1 as a finite-horizon MDP: simulator, exact solver and sampling-based planners.

Subpackages:
    game         rules, Config, BalatroMDP, Sim
    policies     greedy / discard-heuristic baselines and action ordering
    solvers      exact backward induction and a reference solver
    planners     sparse sampling, UCT, flat Monte Carlo
    experiments  the experiments behind the paper (run via the CLI)
"""
from .game import (HAND_TABLE, Action, BalatroMDP, Card, Config, Sim, State, card_chips, evaluate,
                   run_episode, score_hand)
from .planners import UCT, FlatMC, SparseSampling
from .policies import (best_play, greedy_policy, greedy_rollout, greedy_smallest_policy,
                       heuristic_action, heuristic_policy, heuristic_rollout, ordered_actions)
from .solvers import ExactSolver, ReferenceSolver, TooBig, simulate_optimal, tree_size

__all__ = [
    "Action", "BalatroMDP", "Card", "Config", "ExactSolver", "FlatMC", "HAND_TABLE",
    "ReferenceSolver", "Sim", "SparseSampling", "State", "TooBig", "UCT", "best_play",
    "card_chips", "evaluate", "greedy_policy", "greedy_rollout", "greedy_smallest_policy",
    "heuristic_action", "heuristic_policy", "heuristic_rollout", "ordered_actions",
    "run_episode", "score_hand", "simulate_optimal", "tree_size",
]
