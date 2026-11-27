"""Quick demo (configs/demo.yaml, ~15 s): decision-tree sizes, an exact solve of one starting hand,
a simulation check of the optimal policy, and a few methods' regret on that hand."""
from __future__ import annotations

import random
import time

from game.config import config_from_dict
from game.mdp import BalatroMDP
from planners.registry import build_policy
from solvers.exact import ExactSolver, simulate_optimal

from .regret import feasibility_table


def run_demo(conf, out_dir):
    print("Decision-tree size from one root (evaluations):")
    results = {"feasibility": feasibility_table([(row["name"], config_from_dict(row)) for row in conf["feasibility"]])}

    deck = conf["game"].get("deck", "standard")
    rng = random.Random(conf["seed"])
    mdp = BalatroMDP(config_from_dict(conf["game"]))
    s0 = mdp.initial_state(rng)
    print(f"\nExact solve of the {deck} deck ({len(mdp.cfg.full_deck())} cards), seed-{conf['seed']} root...")
    t0 = time.time()
    solver = ExactSolver(mdp, conf["limits"]["max_tree"])
    q = solver.solve(s0)
    v_star = max(q.values())
    print(f"  V*(s0) = {v_star:.4f}  ({len(solver.memo)} states, {time.time() - t0:.1f}s)")
    mean, se = simulate_optimal(solver, s0, conf["optimal_episodes"], random.Random(conf["optimal_seed"]))
    print(f"  optimal policy simulated: {mean:.4f} +/- {se:.4f}  (should match V*)")
    results.update({"v_star": v_star, "optimal_simulated": mean, "optimal_se": se, "methods": []})

    print("\nRegret on this root (V* - Q* of chosen action; one root is noisy, so planners are")
    print("repeated; the 10-root table in the regret-table experiment is the real evidence):")
    for entry in conf["methods"]:
        entry = dict(entry)
        reps = entry.pop("reps", 1)
        policy = build_policy(entry, mdp)
        regrets, calls = [], 0
        for _ in range(reps):
            if policy.sim is not None:
                policy.sim.calls = 0
            regrets.append(v_star - q[policy.act(s0, rng)])
            calls += policy.sim.calls if policy.sim is not None else 0
        results["methods"].append({"method": policy.label, "reps": reps, "regret": sum(regrets) / reps,
                                   "calls_per_decision": calls // reps})
        print(f"  {policy.label:28s} mean regret = {sum(regrets) / reps:.4f} over {reps} run(s)"
              f"   ({calls // reps} simulator calls/decision)")
    return results
