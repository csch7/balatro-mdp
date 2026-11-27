"""Orderings of legal actions for progressive widening: heuristic best-first, or random."""
from __future__ import annotations

import itertools

from game.mdp import BalatroMDP, State
from game.rules import score_hand


def ordered_actions(mdp: BalatroMDP, s: State, rng=None):
    """Legal actions ranked best-first for progressive widening.
      play D    : score of the hand D
      discard D : best play still available among the kept cards
    Plays and discards are ranked separately, then interleaved; ties prefer more cards."""
    n, m, table = len(s.hand), mdp.cfg.max_cards, mdp.cfg.hand_table
    sub = {}
    for size in range(1, min(m, n) + 1):
        for idx in itertools.combinations(range(n), size):
            sub[sum(1 << i for i in idx)] = score_hand([s.hand[i] for i in idx], table)
    best = [0] * (1 << n)                       # best[mask] = best play inside mask
    for mask in range(1, 1 << n):
        b = sub.get(mask, 0)
        for i in range(n):
            if mask >> i & 1:
                b = max(b, best[mask & ~(1 << i)])
        best[mask] = b
    full = (1 << n) - 1
    acts = mdp.legal_actions(s)
    mask_of = lambda a: sum(1 << i for i in a[1])
    plays = sorted((a for a in acts if a[0] == "play"), key=lambda a: (-sub[mask_of(a)], -len(a[1])))
    discs = sorted((a for a in acts if a[0] == "discard"),
                   key=lambda a: (-best[full & ~mask_of(a)], -len(a[1])))
    out = []
    for p, d in itertools.zip_longest(plays, discs):
        out += [x for x in (p, d) if x is not None]
    return out


def shuffled_actions(mdp: BalatroMDP, s: State, rng):
    """Legal actions in random order (ablation of ordered_actions)."""
    acts = mdp.legal_actions(s)
    rng.shuffle(acts)
    return acts


ACTION_ORDERS = {"ordered": ordered_actions, "shuffled": shuffled_actions}
