"""UCT (Kocsis & Szepesvari, 2006) with a suit-symmetric transposition table and progressive widening."""
from __future__ import annotations

import math

from game.rules import Action
from game.simulator import Sim, rollout
from policies.base import Policy
from policies.ordering import ACTION_ORDERS


class _Node:
    __slots__ = ("rep", "actions", "N", "Na", "Wa")

    def __init__(self, rep, actions):
        self.rep, self.actions, self.N = rep, actions, 0
        self.Na, self.Wa = [0] * len(actions), [0.0] * len(actions)


class UCT(Policy):
    """UCB1 tree policy + rollout evaluation of new leaves.
      * transposition table keyed by the canonical state (suit symmetry; key="plain" turns it off);
        the walk continues from the node's stored representative state, valid because values are
        suit-symmetric
      * progressive widening: a node with N visits may use the first 1 + floor(widen_c*N^alpha)
        actions of `action_order` (an addition beyond the UCT paper)
      * rollout_policy=None values new leaves at 0
      * returns should lie in [0,1] (reward_mode='target'); otherwise pass scale ~ max return."""
    def __init__(self, sim: Sim, rollout_policy: Policy | None, iters=300, c=1.0, widen=True,
                 widen_c=2.0, widen_alpha=0.5, scale=1.0, key="canonical", action_order="ordered"):
        self.sim, self.rollout_policy, self.iters, self.c = sim, rollout_policy, iters, c
        self.widen, self.widen_c, self.widen_alpha, self.scale = widen, widen_c, widen_alpha, scale
        self.key = {"canonical": sim.canonical, "plain": sim.mdp.plain_key}[key]
        self.action_order = ACTION_ORDERS[action_order]
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
                table[k] = _Node(s, self.action_order(self.sim.mdp, s, rng))
                self.expansions += 1
                G = rollout(self.sim, s, self.rollout_policy, rng) / self.scale
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
        for _ in range(self.iters):
            self._iterate(root, table, rng)
        return table[self.key(root)]

    def act(self, root, rng) -> Action:
        node = self.search(root, rng)
        return node.actions[max(range(len(node.actions)), key=lambda i: node.Na[i])]

    def stats(self) -> dict:
        return {"node_reuse": self.hits / max(self.hits + self.expansions, 1)}
