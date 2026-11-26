"""Game configuration: the real Ante-1 rules and the reduced toy decks used for exact solving."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    ranks: tuple[int, ...] = tuple(range(2, 15))
    n_suits: int = 4
    hand_size: int = 8
    hands: int = 4                 # Balatro defaults: 8-card hand, 4 hands, 3 discards
    discards: int = 3
    max_cards: int = 5
    target: int = 300              # Ante 1 blinds: 300 small, 450 big, 600 boss
    reward_mode: str = "points"    # "points": reward = score of each play
    #                                "target": reward = 1 on first reaching target (returns in [0,1])

    def full_deck(self):
        return tuple(sorted((r, s) for r in self.ranks for s in range(self.n_suits)))

    @staticmethod
    def small() -> "Config":
        """14 cards (ranks 8-A, 2 suits), 5-card hand, 2 hands + 1 discard, win = 300.
        Largest config solved exactly in ~10-20 s per root."""
        return Config(ranks=tuple(range(8, 15)), n_suits=2, hand_size=5, hands=2,
                      discards=1, target=300, reward_mode="target")

    @staticmethod
    def tiny() -> "Config":
        """12 cards (ranks 9-A, 2 suits), 5-card hand, 2 hands + 1 discard, win = 300."""
        return Config(ranks=tuple(range(9, 15)), n_suits=2, hand_size=5, hands=2,
                      discards=1, target=300, reward_mode="target")

    @staticmethod
    def medium() -> "Config":
        """21 cards, 3 hands + 2 discards: NOT exactly solvable (~3.6e16 evaluations)."""
        return Config(ranks=tuple(range(8, 15)), n_suits=3, hand_size=5, hands=3,
                      discards=2, target=150)
