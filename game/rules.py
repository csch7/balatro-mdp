"""Balatro scoring rules: cards, poker-hand evaluation and chip x mult scoring.

The chip/mult values per hand live in configs/scoring/*.yaml and reach here as `table`
(a Config's `hand_table`).
"""
from __future__ import annotations

from collections import Counter

Card = tuple[int, int]                # (rank 2..14 with 14 = Ace, suit index)
Action = tuple[str, tuple[int, ...]]  # ("play" | "discard", indices into state.hand)

HAND_NAMES = ("High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush",
              "Full House", "Four of a Kind", "Straight Flush")


def card_chips(rank: int) -> int:
    return 11 if rank == 14 else min(rank, 10)


def evaluate(cards):
    """Return (hand name, scoring cards). Only scoring cards add their chips."""
    ranks = [r for r, _ in cards]
    counts = Counter(ranks)
    shape = sorted(counts.values(), reverse=True)
    five = len(cards) == 5
    flush = five and len({s for _, s in cards}) == 1
    uniq = sorted(set(ranks))
    straight = five and len(uniq) == 5 and (uniq[-1] - uniq[0] == 4 or uniq == [2, 3, 4, 5, 14])

    def of_rank_count(n):
        return [c for c in cards if counts[c[0]] >= n]

    if straight and flush: return "Straight Flush", list(cards)
    if shape[0] == 4: return "Four of a Kind", of_rank_count(4)
    if shape[:2] == [3, 2]: return "Full House", list(cards)
    if flush: return "Flush", list(cards)
    if straight: return "Straight", list(cards)
    if shape[0] == 3: return "Three of a Kind", of_rank_count(3)
    if shape[:2] == [2, 2]: return "Two Pair", of_rank_count(2)
    if shape[0] == 2: return "Pair", of_rank_count(2)
    return "High Card", [max(cards)]


def score_hand(cards, table: dict[str, tuple[int, int]]) -> int:
    """(base chips + chips of the scoring cards) x mult, with (chips, mult) from `table`."""
    name, scoring = evaluate(cards)
    chips, mult = table[name]
    return (chips + sum(card_chips(r) for r, _ in scoring)) * mult
