"""Regret against exact values on toy decks, and the feasibility table."""
from __future__ import annotations

import random
import time

from ..game.config import Config
from ..game.mdp import BalatroMDP
from ..game.simulator import Sim
from ..planners.sparse_sampling import SparseSampling
from ..planners.uct import UCT
from ..policies.baselines import greedy_policy, greedy_rollout, heuristic_action
from ..solvers.exact import ExactSolver, tree_size


def experiment_exact(cfg: Config, n_states=10, reps=3, seed=0):
    """Regret V*(s) - Q*(s, chosen action) against exact values (win probability)."""
    mdp = BalatroMDP(cfg); solver = ExactSolver(mdp); rng = random.Random(seed)
    states = [mdp.initial_state(rng) for _ in range(n_states)]
    t0 = time.time()
    truth = [solver.solve(s) for s in states]
    print(f"exact solve: {len(solver.memo)} memoized states, {time.time() - t0:.1f}s")
    sim = Sim(mdp)
    methods = {"random": None, "greedy-play": None, "discard heuristic": None}
    for H, C, leaf in [(3, 1, None), (2, 1, greedy_rollout), (2, 2, greedy_rollout), (2, 4, greedy_rollout)]:
        methods[f"sparse H={H} C={C}" + (" +greedy leaf" if leaf else "")] = \
            lambda H=H, C=C, leaf=leaf: SparseSampling(sim, C=C, H=H, leaf=leaf)
    for n in (25, 100, 400, 1600, 6400):
        methods[f"UCT {n} iters"] = lambda n=n: UCT(sim, n_iter=n)
    print(f"\n{'method':26s} {'regret':>8s} {'optimal%':>9s} {'sim calls/decision':>19s}")
    rows = []
    for name, make in methods.items():
        regrets, hits, calls = [], 0, 0
        for s, q in zip(states, truth):
            v = max(q.values())
            for _ in range(reps):
                if name == "random":
                    a = rng.choice(list(q))
                elif name == "greedy-play":
                    a = greedy_policy(mdp, s, rng)
                elif name == "discard heuristic":
                    a = heuristic_action(cfg, s)
                else:
                    sim.calls = 0
                    a = make().plan(s, rng)
                    calls += sim.calls
                regrets.append(v - q[a]); hits += q[a] >= v - 1e-9
        n_dec = len(regrets)
        rows.append((name, calls / n_dec, sum(regrets) / n_dec, hits / n_dec))
        print(f"{name:26s} {sum(regrets) / n_dec:8.4f} {100 * hits / n_dec:8.1f}% {calls / n_dec:19.0f}")
    return rows


def feasibility_table():
    rows = [("21 cards, 3 hands, 2 discards", Config.medium()),
            ("21 cards, 2 hands, 1 discard", Config(ranks=tuple(range(8, 15)), n_suits=3, hand_size=5, hands=2, discards=1)),
            ("small(): 14 cards, 2 hands, 1 discard", Config.small()),
            ("tiny(): 12 cards, 2 hands, 1 discard", Config.tiny())]
    for name, c in rows:
        print(f"  {name:42s} {tree_size(c):.2e}")
