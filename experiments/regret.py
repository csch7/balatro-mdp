"""Regret against exact values on a deck small enough to solve (configs/regret_table.yaml),
and the decision-tree size table."""
from __future__ import annotations

import random
import time
from pathlib import Path

from game.config import Config, config_from_dict
from game.mdp import BalatroMDP
from planners.registry import build_policy
from solvers.exact import ExactSolver, tree_size


def regret_table(cfg: Config, methods: list, n_states: int, reps: int, seed, max_tree: float) -> list[dict]:
    """Regret V*(s) - Q*(s, chosen action) (lost win probability) for each method spec, averaged
    over `n_states` random starting hands x `reps` decisions."""
    mdp = BalatroMDP(cfg); solver = ExactSolver(mdp, max_tree); rng = random.Random(seed)
    states = [mdp.initial_state(rng) for _ in range(n_states)]
    t0 = time.time()
    truth = [solver.solve(s) for s in states]
    print(f"exact solve: {len(solver.memo)} memoized states, {time.time() - t0:.1f}s")
    print(f"\n{'method':26s} {'regret':>8s} {'optimal%':>9s} {'sim calls/decision':>19s}")
    rows = []
    for spec in methods:
        policy = build_policy(spec, mdp)
        regrets, hits, calls = [], 0, 0
        for s, q in zip(states, truth):
            v = max(q.values())
            for _ in range(reps):
                if policy.sim is not None:
                    policy.sim.calls = 0
                a = policy.act(s, rng)
                calls += policy.sim.calls if policy.sim is not None else 0
                regrets.append(v - q[a]); hits += q[a] >= v - 1e-9
        n_dec = len(regrets)
        row = {"method": policy.label, "type": spec["type"] if isinstance(spec, dict) else spec,
               "calls_per_decision": calls / n_dec, "regret": sum(regrets) / n_dec, "optimal_frac": hits / n_dec}
        rows.append(row)
        print(f"{row['method']:26s} {row['regret']:8.4f} {100 * row['optimal_frac']:8.1f}% {row['calls_per_decision']:19.0f}")
    return rows


def feasibility_table(rows) -> list[dict]:
    """Print and return tree_size for (name, Config) rows."""
    out = []
    for name, c in rows:
        out.append({"deck": name, "tree_size": tree_size(c)})
        print(f"  {name:42s} {out[-1]['tree_size']:.2e}")
    return out


def run_regret_table(conf, out_dir):
    cfg = config_from_dict(conf["game"])
    print(f"Regret table on the {conf['game'].get('deck', 'standard')} deck:")
    rows = regret_table(cfg, conf["methods"], **conf["params"])
    results = {"rows": rows}
    if conf.get("plot"):
        from .plotting import plot_regret   # optional matplotlib dependency
        results["figure"] = plot_regret(rows, Path(out_dir) / "regret_vs_calls.png")
        print(f"saved {results['figure']}")
    return results
