"""Command-line entry point; replaces the RUN_* flags of the original notebook.

    python -m balatro_mdp                     # quick demo (~15 s): feasibility table + one exact root
    python -m balatro_mdp regret-table        # pick any of the experiments below, in any order
    python -m balatro_mdp all                 # every experiment (2-3 hours on one core)

Approximate runtimes on one core: regret-table ~5 min (Table 2), full-game ~6 min,
baselines ~1-10 min, search ~40-50 min, validation ~5 min, tiny-diagnostics ~5-12 min,
ablation ~45 min.
"""
from __future__ import annotations

import argparse
import random
import time

from .exact import ExactSolver, simulate_optimal
from .experiments.ablation import experiment_ablation
from .experiments.full_game import experiment_baselines, experiment_full, experiment_search
from .experiments.regret import experiment_exact, feasibility_table, plot_regret
from .experiments.validation import experiment_tiny_diagnostics, experiment_validation
from .mdp import BalatroMDP, Config, Sim
from .policies import greedy_policy, greedy_rollout
from .sparse import SparseSampling
from .uct import UCT


def demo():
    """Feasibility table, exact solve of one Config.small() root, and regret on that root."""
    print("Decision-tree size from one root (evaluations):")
    feasibility_table()

    rng = random.Random(42)
    mdp = BalatroMDP(Config.small())
    s0 = mdp.initial_state(rng)
    print("\nExact solve of Config.small(), seed-42 root...")
    t0 = time.time()
    solver = ExactSolver(mdp)
    q = solver.solve(s0)
    v_star = max(q.values())
    print(f"  V*(s0) = {v_star:.4f}  ({len(solver.memo)} states, {time.time() - t0:.1f}s)")
    mean, se = simulate_optimal(solver, s0, 20000, random.Random(1))
    print(f"  optimal policy simulated: {mean:.4f} +/- {se:.4f}  (should match V*)")

    sim = Sim(mdp)
    print("\nRegret on this root (V* - Q* of chosen action; one root is noisy, so planners are")
    print("repeated; the 10-root table in experiment_exact is the real evidence):")
    cands = {   # name: (planner call, repetitions)
        "greedy-play": (lambda: greedy_policy(mdp, s0, rng), 1),
        "UCT 1600 iters": (lambda: UCT(sim, n_iter=1600).plan(s0, rng), 6),
        "sparse H=2 C=4 +greedy leaf": (lambda: SparseSampling(sim, C=4, H=2, leaf=greedy_rollout).plan(s0, rng), 3),
    }
    for name, (f, reps) in cands.items():
        regrets, calls = [], 0
        for _ in range(reps):
            sim.calls = 0
            regrets.append(v_star - q[f()])
            calls += sim.calls
        print(f"  {name:28s} mean regret = {sum(regrets) / reps:.4f} over {reps} run(s)"
              f"   ({calls // reps} simulator calls/decision)")


def regret_table(plot=False):
    print("\nRegret table on Config.small():")
    rows = experiment_exact(Config.small())
    if plot:
        print(f"saved {plot_regret(rows)}")


def full_game():
    print("\nFull 52-card game, Big Blind (target 450):")
    experiment_full(episodes=100, iters=300, target=450)


def baselines():
    print("\nNo-search baselines on the full game:")
    experiment_baselines()


def search():
    for tgt, n_games in ((450, 300), (600, 150)):
        print(f"\nSearch vs discard heuristic, target {tgt} ({n_games} games):")
        experiment_search(episodes=n_games, iters=300, target=tgt)


def validation():
    print("\nExact-solver validation:")
    experiment_validation()


def tiny_diagnostics():
    print("\n12-card diagnostics:")
    experiment_tiny_diagnostics()
    print("\nRegret table on the 12-card Config.tiny():")
    experiment_exact(Config.tiny())


def ablation():
    print("\nUCT ablations, Big Blind (target 450):")
    experiment_ablation(episodes=100, iters=300, target=450)


EXPERIMENTS = {   # same order as the RUN_* flags in the original notebook
    "regret-table": regret_table,
    "full-game": full_game,
    "baselines": baselines,
    "search": search,
    "validation": validation,
    "tiny-diagnostics": tiny_diagnostics,
    "ablation": ablation,
}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m balatro_mdp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="*", choices=["demo", "all", *EXPERIMENTS],
                   help="what to run (default: demo)")
    p.add_argument("--plot", action="store_true",
                   help="with regret-table: save regret_vs_calls.png (needs matplotlib)")
    args = p.parse_args(argv)
    names = args.experiments or ["demo"]
    if "all" in names:
        names = ["demo", *EXPERIMENTS]
    for name in names:
        if name == "demo":
            demo()
        elif name == "regret-table":
            regret_table(plot=args.plot)
        else:
            EXPERIMENTS[name]()


if __name__ == "__main__":
    main()
