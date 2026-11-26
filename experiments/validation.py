"""Validation of the exact solver, and diagnostics on the 12-card variant."""
from __future__ import annotations

import random
import time
from dataclasses import replace

from game.config import Config
from game.mdp import BalatroMDP
from game.simulator import Sim
from planners.sparse_sampling import SparseSampling
from planners.uct import UCT
from policies.baselines import greedy_rollout
from solvers.exact import ExactSolver, simulate_optimal
from solvers.reference import ReferenceSolver


def experiment_validation(tiny_cfg: Config, small_cfg: Config, n_tiny=8, n_small=6, n_ref_small=2,
                          episodes=8000, seed=7):
    """(i) ExactSolver vs ReferenceSolver; (ii) executing the optimal policy matches V*."""
    rng = random.Random(seed)
    mdp = BalatroMDP(tiny_cfg); fast, ref = ExactSolver(mdp), ReferenceSolver(mdp)
    worst = 0.0
    for _ in range(n_tiny):
        s = mdp.initial_state(rng)
        a = fast.solve(s)
        worst = max(worst, max(abs(a[k] - ref.q(s, k)) for k in a))
    print(f"independent solver, {n_tiny} tiny roots: max |dQ| = {worst:.1e}", flush=True)
    mdp = BalatroMDP(small_cfg); solver = ExactSolver(mdp); ref = ReferenceSolver(mdp)
    zs, t0 = [], time.time()
    for i in range(n_small):
        s0 = mdp.initial_state(rng)
        q = solver.solve(s0); v = max(q.values())
        m, se = simulate_optimal(solver, s0, episodes, random.Random(100 + i))
        zs.append((m - v) / se)
        print(f"  small root {i}: V*={v:.4f} simulated={m:.4f}+/-{se:.4f} z={zs[-1]:+.2f}  [{time.time() - t0:.0f}s]", flush=True)
        if i < n_ref_small:
            worst = max(abs(q[k] - ref.q(s0, k)) for k in q)
            print(f"    independent solver on this root: max |dQ| = {worst:.1e}", flush=True)
    print(f"mean z = {sum(zs) / len(zs):+.2f} (should be ~0 with sd ~1); max |z| = {max(abs(z) for z in zs):.2f}")


def experiment_tiny_diagnostics(base: Config, seed=0, n_states=10,
                                sparse=((3, 1, False), (2, 1, True), (2, 4, True)),
                                variants=((1, 1, 200), (2, 0, 300), (2, 1, 300)),
                                uct_iters=(1000, 8000), c=0.2, n_roots=6):
    """Small-deck diagnostics: (i) sparse sampling ties at C=1; (ii) UCT convergence on shallower
    variants of `base`, each given as (hands, discards, target)."""
    mdp = BalatroMDP(base); sim = Sim(mdp); rng = random.Random(seed)
    states = [mdp.initial_state(rng) for _ in range(n_states)]
    print("Sparse sampling: number of root actions tied for the maximum estimate")
    for H, C, use_leaf in sparse:
        leaf = greedy_rollout if use_leaf else None
        ties = []
        for s in states:
            qs = SparseSampling(sim, C=C, H=H, leaf=leaf).q_values(s, rng)
            top = max(q for _, q in qs)
            ties.append(sum(q == top for _, q in qs))
        print(f"  H={H} C={C}: mean {sum(ties) / len(ties):.1f} of {len(qs)} actions tied")
    print(f"UCT (c={c}): regret and bias of the chosen action's estimate, {n_roots} roots")
    for hands, disc, target in variants:
        cfg = replace(base, hands=hands, discards=disc, target=target)
        m2 = BalatroMDP(cfg); sol = ExactSolver(m2); s2 = Sim(m2); r2 = random.Random(2)
        roots = [m2.initial_state(r2) for _ in range(n_roots)]
        for n in uct_iters:
            reg, bias = [], []
            for s in roots:
                q = sol.solve(s); v = max(q.values())
                node = UCT(s2, n_iter=n, c=c).search(s, r2)
                mv = max(range(len(node.actions)), key=lambda i: node.Na[i]); a = node.actions[mv]
                reg.append(v - q[a]); bias.append(node.Wa[mv] / node.Na[mv] - q[a])
            print(f"  hands={hands} discards={disc} n={n}: regret={sum(reg) / n_roots:.3f}  bias={sum(bias) / n_roots:+.3f}", flush=True)
