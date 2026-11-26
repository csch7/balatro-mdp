"""Exact backward induction (ground truth) and its feasibility estimator."""
from __future__ import annotations

import itertools
from math import comb

from ..game.config import Config
from ..game.mdp import BalatroMDP, State
from ..game.rules import Action, score_hand


def tree_size(cfg: Config) -> float:
    """Upper bound on (action, draw-outcome) evaluations from ONE root, ignoring memoization
    and early termination; exact count of the decision tree by DP over
    (plays used, discards used, deck size)."""
    n_deck = len(cfg.full_deck()) - cfg.hand_size
    h, top = cfg.hand_size, min(cfg.max_cards, cfg.hand_size)
    layer, work = {(0, 0, n_deck): 1}, 0
    while layer:
        nxt = {}
        for (p, d, m), w in layer.items():
            for j in range(1, top + 1):
                n_act = comb(h, j)
                for kind in ("play", "discard"):
                    if kind == "play":
                        if p >= cfg.hands: continue
                        np_, nd = p + 1, d
                    else:
                        if d >= cfg.discards or m == 0: continue
                        np_, nd = p, d + 1
                    if np_ == cfg.hands:          # last play: reward only
                        work += w * n_act
                        continue
                    k = min(j, m)
                    outcomes = n_act * comb(m, k)
                    work += w * outcomes
                    nxt[(np_, nd, m - k)] = nxt.get((np_, nd, m - k), 0) + w * outcomes
        layer = nxt
    return float(work)


class TooBig(Exception):
    pass


class ExactSolver:
    """V_t(s) = max_a [ r(s,a) + E_draw V_{t+1}(s') ], memoized over reachable states,
    with suit-symmetry merging, no draw enumeration for episode-ending plays, and a closed
    form for (hands_left=1, discards_left=0)."""
    def __init__(self, mdp: BalatroMDP, max_tree: float = 5e8):
        self.mdp, self.cfg, self.max_tree, self.memo = mdp, mdp.cfg, max_tree, {}

    def _best_play_reward(self, s: State) -> float:
        m = min(self.cfg.max_cards, len(s.hand))
        best = max(score_hand([s.hand[i] for i in idx])
                   for idx in itertools.combinations(range(len(s.hand)), m))
        if self.cfg.reward_mode == "target":
            return 1.0 if s.score + best >= self.cfg.target else 0.0
        return float(best)

    def q(self, s: State, a: Action) -> float:
        kept, hl, dl, sc, r, k = self.mdp._apply(s, a)
        if hl == 0 or (self.cfg.reward_mode == "target" and sc >= self.cfg.target):
            return r
        n_out, total = comb(len(s.deck), k), 0.0
        for drawn in itertools.combinations(s.deck, k):
            deck = tuple(c for c in s.deck if c not in drawn)
            total += self.v(State(tuple(sorted(kept + list(drawn))), deck, hl, dl, sc))
        return r + total / n_out

    def v(self, s: State) -> float:
        if self.mdp.is_terminal(s):
            return 0.0
        if s.hands_left == 1 and s.discards_left == 0:
            return self._best_play_reward(s)
        key = self.mdp.canonical(s)
        if key not in self.memo:
            self.memo[key] = max(self.q(s, a) for a in self.mdp.legal_actions(s))
        return self.memo[key]

    def solve(self, s: State):
        """Exact Q*(s, a) for every legal action."""
        size = tree_size(self.cfg)
        if size > self.max_tree:
            raise TooBig(f"decision tree ~{size:.1e} evaluations > limit {self.max_tree:.0e}")
        return {a: self.q(s, a) for a in self.mdp.legal_actions(s)}


def simulate_optimal(solver: ExactSolver, s0: State, episodes: int, rng):
    """Mean +/- s.e. of the exact-optimal policy run in the simulator; should equal V*(s0)."""
    mdp, cache, tot, tot2 = solver.mdp, {}, 0.0, 0.0
    for _ in range(episodes):
        s, g = s0, 0.0
        while not mdp.is_terminal(s):
            if s not in cache:
                q = {a: solver.q(s, a) for a in mdp.legal_actions(s)}
                cache[s] = max(q, key=q.get)
            s, r, _ = mdp.step(s, cache[s], rng)
            g += r
        tot += g; tot2 += g * g
    mean = tot / episodes
    return mean, (max(tot2 / episodes - mean * mean, 0.0) / episodes) ** 0.5
