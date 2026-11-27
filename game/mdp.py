"""The Balatro Ante-1 MDP: state, generative model, exact transitions and suit-symmetry key."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from math import comb

from .config import Config
from .rules import Action, Card, score_hand


@dataclass(frozen=True)
class State:
    hand: tuple[Card, ...]      # sorted
    deck: tuple[Card, ...]      # undrawn cards, stored as a sorted set
    hands_left: int
    discards_left: int
    score: int                  # tracked (capped at target) only in "target" mode, else 0


class BalatroMDP:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def initial_state(self, rng: random.Random) -> State:
        deck = list(self.cfg.full_deck())
        hand = rng.sample(deck, self.cfg.hand_size)
        rest = [c for c in deck if c not in hand]
        return State(tuple(sorted(hand)), tuple(rest), self.cfg.hands, self.cfg.discards, 0)

    def is_terminal(self, s: State) -> bool:
        if s.hands_left == 0 or not s.hand:
            return True
        return self.cfg.reward_mode == "target" and s.score >= self.cfg.target

    def legal_actions(self, s: State):
        if self.is_terminal(s):
            return []
        kinds = []
        if s.hands_left > 0: kinds.append("play")
        if s.discards_left > 0 and s.deck: kinds.append("discard")   # empty deck: pointless
        top = min(self.cfg.max_cards, len(s.hand))
        return [(k, idx) for k in kinds for n in range(1, top + 1)
                for idx in itertools.combinations(range(len(s.hand)), n)]

    def _apply(self, s: State, a: Action):
        """Deterministic part of a transition."""
        kind, idx = a
        chosen = [s.hand[i] for i in idx]
        kept = [c for i, c in enumerate(s.hand) if i not in idx]
        pts = score_hand(chosen) if kind == "play" else 0
        hl = s.hands_left - 1 if kind == "play" else s.hands_left
        dl = s.discards_left if kind == "play" else s.discards_left - 1
        if self.cfg.reward_mode == "target":
            new_score = min(s.score + pts, self.cfg.target)
            reward = 1.0 if (s.score < self.cfg.target <= s.score + pts) else 0.0
        else:
            new_score, reward = 0, float(pts)
        k = min(len(chosen), len(s.deck))       # refill the hand; it shrinks if the deck runs out
        return kept, hl, dl, new_score, reward, k

    def step(self, s: State, a: Action, rng: random.Random):
        """Generative model G(s, a) -> (s', r, done)."""
        kept, hl, dl, sc, r, k = self._apply(s, a)
        drawn = rng.sample(s.deck, k)
        deck = [c for c in s.deck if c not in drawn]
        ns = State(tuple(sorted(kept + drawn)), tuple(deck), hl, dl, sc)
        return ns, r, self.is_terminal(ns)

    def transitions(self, s: State, a: Action, max_outcomes: int = 200_000):
        """Exact next-state distribution (small decks only)."""
        kept, hl, dl, sc, r, k = self._apply(s, a)
        n_out = comb(len(s.deck), k)
        if n_out > max_outcomes:
            raise ValueError(f"{n_out} outcomes: use a smaller deck or sample with step()")
        out = []
        for drawn in itertools.combinations(s.deck, k):
            deck = tuple(c for c in s.deck if c not in drawn)
            out.append((1.0 / n_out, State(tuple(sorted(kept + list(drawn))), deck, hl, dl, sc), r))
        return out

    def canonical(self, s: State):
        """Suit-symmetry reduction: smallest key over all joint relabelings of the suits."""
        best = None
        for perm in itertools.permutations(range(self.cfg.n_suits)):
            h = tuple(sorted((r, perm[su]) for r, su in s.hand))
            d = tuple(sorted((r, perm[su]) for r, su in s.deck))
            key = (h, d, s.hands_left, s.discards_left, s.score)
            if best is None or key < best:
                best = key
        return best
