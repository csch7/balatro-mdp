"""No-search policies: greedy play, the discard heuristic, and their rollouts."""
from __future__ import annotations

import itertools

from game.config import Config
from game.mdp import State
from game.rules import Action, evaluate, score_hand
from game.simulator import Sim


def best_play(cfg: Config, hand):
    """Highest-scoring play. Extra cards never lower a score, so a best play exists among
    full-size (max_cards) subsets. Always playing 5 cards also refills the hand, which is why
    this tie-break beats 'play the smallest best subset' (win rate ~0.89 vs ~0.67 at target 300)."""
    m = min(cfg.max_cards, len(hand))
    best, best_score = None, -1
    for idx in itertools.combinations(range(len(hand)), m):
        sc = score_hand([hand[i] for i in idx])
        if sc > best_score:
            best, best_score = idx, sc
    return ("play", best), best_score


def greedy_policy(mdp, s, rng) -> Action:
    return best_play(mdp.cfg, s.hand)[0]


def greedy_rollout(sim: Sim, s: State, rng) -> float:
    """Return of 'always play the best hand now' from s (counts simulator calls)."""
    total = 0.0
    while not sim.is_terminal(s):
        s, r, _ = sim.step(s, best_play(sim.cfg, s.hand)[0], rng)
        total += r
    return total


def greedy_smallest_policy(mdp, s, rng) -> Action:
    """Greedy with the WRONG tie-break (smallest subset reaching the best score); used only to
    show that the tie-break matters (e.g. only the 2 paired cards are played)."""
    best, best_score = None, -1
    for size in range(1, min(mdp.cfg.max_cards, len(s.hand)) + 1):
        for idx in itertools.combinations(range(len(s.hand)), size):
            sc = score_hand([s.hand[i] for i in idx])
            if sc > best_score:
                best, best_score = idx, sc
    return ("play", best)


DISCARD_TYPES = ("High Card", "Pair", "Two Pair")


def heuristic_action(cfg: Config, s: State, types=DISCARD_TYPES) -> Action:
    """Greedy play, except: while discards remain and the best playable hand is one of `types`,
    discard the (up to 5 lowest) cards that do not score in it. No search."""
    play, _ = best_play(cfg, s.hand)
    name, scoring = evaluate([s.hand[i] for i in play[1]])
    if s.discards_left > 0 and s.deck and name in types:
        keep = set(scoring)
        junk = [i for i, c in enumerate(s.hand) if c not in keep]
        if junk:
            return ("discard", tuple(junk[:cfg.max_cards]))
    return play


def heuristic_policy(types=DISCARD_TYPES):
    return lambda mdp, s, rng: heuristic_action(mdp.cfg, s, types)


def heuristic_rollout(sim: Sim, s: State, rng) -> float:
    """Return of the discard heuristic from s (counts simulator calls)."""
    total = 0.0
    while not sim.is_terminal(s):
        s, r, _ = sim.step(s, heuristic_action(sim.cfg, s), rng)
        total += r
    return total
