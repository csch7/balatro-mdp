"""No-search policies: greedy play, the discard heuristic and uniform random play."""
from __future__ import annotations

import itertools

from game.config import Config
from game.mdp import BalatroMDP, State
from game.rules import Action, evaluate, score_hand

from .base import Policy


def best_play(cfg: Config, hand, tie_break: str = "largest"):
    """Highest-scoring play and its score.

    tie_break="largest": search only full-size (max_cards) subsets. Extra cards never lower a
      score, so a best play exists among them, and playing 5 cards also refills the hand, which is
      why this beats the alternative (win rate ~0.89 vs ~0.67 at target 300).
    tie_break="smallest": the smallest subset reaching the best score (e.g. only the 2 paired cards).
    """
    m = min(cfg.max_cards, len(hand))
    sizes = [m] if tie_break == "largest" else range(1, m + 1)
    best, best_score = None, -1
    for size in sizes:
        for idx in itertools.combinations(range(len(hand)), size):
            sc = score_hand([hand[i] for i in idx], cfg.hand_table)
            if sc > best_score:
                best, best_score = idx, sc
    return ("play", best), best_score


class GreedyPolicy(Policy):
    """Always play the best hand now; never discard."""
    def __init__(self, mdp: BalatroMDP, tie_break: str = "largest"):
        if tie_break not in ("largest", "smallest"):
            raise ValueError(f"tie_break must be 'largest' or 'smallest', not {tie_break!r}")
        self.cfg, self.tie_break = mdp.cfg, tie_break

    def act(self, s: State, rng) -> Action:
        return best_play(self.cfg, s.hand, self.tie_break)[0]


class DiscardHeuristic(Policy):
    """Greedy play, except: while discards remain and the best playable hand is one of
    `discard_on`, discard the (up to max_cards lowest) cards that do not score in it. No search."""
    def __init__(self, mdp: BalatroMDP, discard_on=("High Card", "Pair", "Two Pair")):
        self.cfg, self.discard_on = mdp.cfg, tuple(discard_on)

    def act(self, s: State, rng) -> Action:
        play, _ = best_play(self.cfg, s.hand)
        name, scoring = evaluate([s.hand[i] for i in play[1]])
        if s.discards_left > 0 and s.deck and name in self.discard_on:
            keep = set(scoring)
            junk = [i for i, c in enumerate(s.hand) if c not in keep]
            if junk:
                return ("discard", tuple(junk[:self.cfg.max_cards]))
        return play


class RandomPolicy(Policy):
    """A uniformly random legal action."""
    def __init__(self, mdp: BalatroMDP):
        self.mdp = mdp

    def act(self, s: State, rng) -> Action:
        return rng.choice(self.mdp.legal_actions(s))
