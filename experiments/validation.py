"""Validation of the exact solver (configs/validation.yaml), and diagnostics on a small deck
(configs/tiny_diagnostics.yaml)."""
from __future__ import annotations

import random
import time
from dataclasses import replace
from pathlib import Path

import yaml

from game.config import config_from_dict
from game.mdp import BalatroMDP
from planners.registry import build_policy
from solvers.exact import ExactSolver, simulate_optimal
from solvers.reference import ReferenceSolver

from .regret import regret_table


def run_validation(conf, out_dir):
    """(i) ExactSolver vs ReferenceSolver on tiny_game roots; (ii) executing the optimal policy
    on small_game roots matches V*."""
    p, lim = conf["params"], conf["limits"]
    rng = random.Random(p["seed"])
    mdp = BalatroMDP(config_from_dict(conf["tiny_game"]))
    fast, ref = ExactSolver(mdp, lim["max_tree"]), ReferenceSolver(mdp, lim["max_outcomes"])
    worst = 0.0
    for _ in range(p["n_tiny"]):
        s = mdp.initial_state(rng)
        a = fast.solve(s)
        worst = max(worst, max(abs(a[k] - ref.q(s, k)) for k in a))
    print(f"independent solver, {p['n_tiny']} tiny roots: max |dQ| = {worst:.1e}", flush=True)
    results = {"tiny_max_abs_dq": worst, "small_roots": []}
    mdp = BalatroMDP(config_from_dict(conf["small_game"]))
    solver, ref = ExactSolver(mdp, lim["max_tree"]), ReferenceSolver(mdp, lim["max_outcomes"])
    zs, t0 = [], time.time()
    for i in range(p["n_small"]):
        s0 = mdp.initial_state(rng)
        q = solver.solve(s0); v = max(q.values())
        m, se = simulate_optimal(solver, s0, p["episodes"], random.Random(p["sim_seed"] + i))
        zs.append((m - v) / se)
        row = {"v_star": v, "simulated": m, "se": se, "z": zs[-1]}
        print(f"  small root {i}: V*={v:.4f} simulated={m:.4f}+/-{se:.4f} z={zs[-1]:+.2f}  [{time.time() - t0:.0f}s]", flush=True)
        if i < p["n_ref_small"]:
            row["ref_max_abs_dq"] = max(abs(q[k] - ref.q(s0, k)) for k in q)
            print(f"    independent solver on this root: max |dQ| = {row['ref_max_abs_dq']:.1e}", flush=True)
        results["small_roots"].append(row)
    print(f"mean z = {sum(zs) / len(zs):+.2f} (should be ~0 with sd ~1); max |z| = {max(abs(z) for z in zs):.2f}")
    return results


def run_tiny_diagnostics(conf, out_dir):
    """(i) how many root actions sparse sampling leaves tied; (ii) UCT regret and bias on shallower
    variants of the deck, each [hands, discards, target]; (iii) the regret table on the deck."""
    base = config_from_dict(conf["game"])
    p = conf["params"]
    mdp = BalatroMDP(base); rng = random.Random(p["seed"])
    states = [mdp.initial_state(rng) for _ in range(p["n_states"])]
    results = {"sparse_ties": [], "uct": []}
    print("Sparse sampling: number of root actions tied for the maximum estimate")
    for spec in conf["sparse"]:
        planner = build_policy(spec, mdp)
        ties = []
        for s in states:
            qs = planner.q_values(s, rng)
            top = max(q for _, q in qs)
            ties.append(sum(q == top for _, q in qs))
        results["sparse_ties"].append({"method": planner.label, "mean_ties": sum(ties) / len(ties), "actions": len(qs)})
        print(f"  {planner.label}: mean {sum(ties) / len(ties):.1f} of {len(qs)} actions tied")
    n_roots, uct_spec = p["n_roots"], conf["uct"]
    print(f"UCT (c={uct_spec.get('c', 1.0)}): regret and bias of the chosen action's estimate, {n_roots} roots")
    for hands, disc, target in conf["variants"]:
        cfg = replace(base, hands=hands, discards=disc, target=target)
        m2 = BalatroMDP(cfg); sol = ExactSolver(m2, conf["limits"]["max_tree"]); r2 = random.Random(p["root_seed"])
        roots = [m2.initial_state(r2) for _ in range(n_roots)]
        for n in conf["uct_iters"]:
            planner = build_policy({**uct_spec, "iters": n}, m2)
            reg, bias = [], []
            for s in roots:
                q = sol.solve(s); v = max(q.values())
                node = planner.search(s, r2)
                mv = max(range(len(node.actions)), key=lambda i: node.Na[i]); a = node.actions[mv]
                reg.append(v - q[a]); bias.append(node.Wa[mv] / node.Na[mv] - q[a])
            results["uct"].append({"hands": hands, "discards": disc, "target": target, "iters": n,
                                   "regret": sum(reg) / n_roots, "bias": sum(bias) / n_roots})
            print(f"  hands={hands} discards={disc} n={n}: regret={sum(reg) / n_roots:.3f}  bias={sum(bias) / n_roots:+.3f}", flush=True)
    regret = dict(conf["regret"])
    if "methods" not in regret:      # default: the same methods as the main regret table
        with open(Path(__file__).resolve().parent.parent / "configs" / "regret_table.yaml") as f:
            regret["methods"] = yaml.safe_load(f)["methods"]
    print(f"\nRegret table on the {conf['game'].get('deck', 'standard')} deck:")
    results["regret_rows"] = regret_table(base, regret.pop("methods"), max_tree=conf["limits"]["max_tree"], **regret)
    return results
