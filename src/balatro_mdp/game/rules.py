"""Balatro scoring rules: cards, poker-hand evaluation and level-1 chip x mult values."""
from __future__ import annotations

from collections import Counter

Card = tuple[int, int]                # (rank 2..14 with 14 = Ace, suit index)


Action = tuple[str, tuple[int, ...]]  # ("play" | "discard", indices into state.hand)


HAND_TABLE = {  # level-1 Balatro hands: (base chips, mult)
    "High Card": (5, 1), "Pair": (10, 2), "Two Pair": (20, 2), "Three of a Kind": (30, 3),
    "Straight": (30, 4), "Flush": (35, 4), "Full House": (40, 4),
    "Four of a Kind": (60, 7), "Straight Flush": (100, 8),
}


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


def score_hand(cards) -> int:
    name, scoring = evaluate(cards)
    chips, mult = HAND_TABLE[name]
    return (chips + sum(card_chips(r) for r, _ in scoring)) * mult
