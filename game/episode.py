"""Real games. The deck is shuffled once and cards come off the top, as in Balatro.

Two policies run with the same seed therefore face the same card order (common random numbers),
whatever they do. The policy gets its own random generator, so search never changes the cards.
"""
from __future__ import annotations

import random

from .mdp import BalatroMDP, State


def episode_rngs(seed) -> tuple[random.Random, random.Random]:
    """Independent (deck, policy) generators for one episode; `seed` may be an int or a str."""
    return random.Random(f"deck:{seed}"), random.Random(f"policy:{seed}")


class Episode:
    def __init__(self, mdp: BalatroMDP, deck_rng: random.Random):
        cfg = mdp.cfg
        order = list(cfg.full_deck())
        deck_rng.shuffle(order)
        h = cfg.hand_size
        self.mdp, self._order, self._pos = mdp, order, h
        self.state = State(tuple(sorted(order[:h])), tuple(sorted(order[h:])), cfg.hands, cfg.discards, 0)
        self.total = 0.0

    @property
    def done(self) -> bool:
        return self.mdp.is_terminal(self.state)

    def step(self, a):
        k = self.mdp.draw_count(self.state, a)
        drawn = self._order[self._pos:self._pos + k]
        self._pos += k
        self.state, r, done = self.mdp.apply_draw(self.state, a, drawn)
        self.total += r
        return self.state, r, done


def run_episode(mdp: BalatroMDP, policy, seed) -> float:
    """Total reward of `policy` in one game; in target mode, 1 for a win and 0 for a loss."""
    deck_rng, policy_rng = episode_rngs(seed)
    ep = Episode(mdp, deck_rng)
    while not ep.done:
        ep.step(policy.act(ep.state, policy_rng))
    return ep.total
