"""UCT (Kocsis & Szepesvari, 2006) with a suit-symmetric transposition table and progressive widening."""
from __future__ import annotations

import math

from ..game.rules import Action
from ..game.simulator import Sim
from ..policies.baselines import greedy_rollout
from ..policies.ordering import ordered_actions


class _Node:
    __slots__ = ("rep", "actions", "N", "Na", "Wa")

    def __init__(self, rep, actions):
        self.rep, self.actions, self.N = rep, actions, 0
        self.Na, self.Wa = [0] * len(actions), [0.0] * len(actions)


class UCT:
    """UCB1 tree policy + greedy rollout evaluation of new leaves.
      * transposition table keyed by sim.canonical(state) (suit symmetry); the walk continues
        from the node's stored representative state, valid because values are suit-symmetric
      * progressive widening: a node with N visits may use the first 1 + floor(widen_c*N^alpha)
        actions of ordered_actions (an addition beyond the UCT paper)
      * returns should lie in [0,1] (reward_mode='target'); otherwise pass scale ~ max return."""
    def __init__(self, sim: Sim, n_iter=500, c=1.0, rollout=greedy_rollout, widen=True,
                 widen_c=2.0, widen_alpha=0.5, scale=1.0, key=None, action_order=None):
        self.sim, self.n_iter, self.c, self.rollout = sim, n_iter, c, rollout
        self.widen, self.widen_c, self.widen_alpha, self.scale = widen, widen_c, widen_alpha, scale
        self.key = key or sim.canonical                  # state key for the transposition table
        self.action_order = action_order or ordered_actions
        self.hits = self.expansions = 0   # node-reuse diagnostics (tree steps landing on a known node)

    def _select(self, node: _Node) -> int:
        n_act = len(node.actions)
        if self.widen:
            n_act = min(n_act, 1 + int(self.widen_c * node.N ** self.widen_alpha))
        for i in range(n_act):
            if node.Na[i] == 0:
                return i
        log_n = math.log(node.N)
        return max(range(n_act), key=lambda i: node.Wa[i] / node.Na[i]
                   + self.c * math.sqrt(log_n / node.Na[i]))

    def _iterate(self, root, table, rng):
        path, s = [], root
        while True:
            if self.sim.is_terminal(s):
                G = 0.0
                break
            k = self.key(s)
            node = table.get(k)
            if node is None:                                   # expand, evaluate by rollout
                table[k] = _Node(s, self.action_order(self.sim.mdp, s))
                self.expansions += 1
                G = self.rollout(self.sim, s, rng) / self.scale
                break
            if path:
                self.hits += 1
            i = self._select(node)
            s, r, _ = self.sim.step(node.rep, node.actions[i], rng)
            path.append((node, i, r / self.scale))
        for node, i, r in reversed(path):
            G = r + G
            node.N += 1; node.Na[i] += 1; node.Wa[i] += G

    def search(self, root, rng):
        table = {}
        for _ in range(self.n_iter):
            self._iterate(root, table, rng)
        return table[self.key(root)]

    def plan(self, root, rng) -> Action:
        node = self.search(root, rng)
        return node.actions[max(range(len(node.actions)), key=lambda i: node.Na[i])]
