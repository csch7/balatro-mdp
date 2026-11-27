"""Command-line entry point. Each experiment reads its settings from configs/<name>.yaml.

    python main.py                                   # quick demo (~15 s)
    python main.py regret-table --plot               # one or more experiments, in order
    python main.py full-game --set params.episodes=20 --set game.target=600
    python main.py full-game --config my_run.yaml    # use another config file
    python main.py all                               # every experiment (2-3 hours on one core)

Approximate runtimes on one core with the default configs: regret-table ~5 min (Table 2),
full-game ~6 min, baselines ~1-10 min, search ~40-50 min, validation ~5 min,
tiny-diagnostics ~5-12 min, ablation ~45 min.
"""
from __future__ import annotations

import argparse
import copy
import random
import time
from dataclasses import replace
from pathlib import Path

import yaml

from experiments.ablation import experiment_ablation
from experiments.full_game import experiment_baselines, experiment_full, experiment_search
from experiments.regret import experiment_exact, feasibility_table
from experiments.validation import experiment_tiny_diagnostics, experiment_validation
from game import BalatroMDP, Sim, config_from_dict
from planners import UCT, SparseSampling
from policies import greedy_policy, greedy_rollout
from solvers import ExactSolver, simulate_optimal

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def demo(conf):
    """Feasibility table, exact solve of one small-deck root, and regret on that root."""
    print("Decision-tree size from one root (evaluations):")
    feasibility_table([(row["name"], config_from_dict(row)) for row in conf["feasibility"]])

    rng = random.Random(conf["seed"])
    mdp = BalatroMDP(config_from_dict(conf["game"]))
    s0 = mdp.initial_state(rng)
    print(f"\nExact solve of Config.{conf['game'].get('preset', 'full')}(), seed-{conf['seed']} root...")
    t0 = time.time()
    solver = ExactSolver(mdp)
    q = solver.solve(s0)
    v_star = max(q.values())
    print(f"  V*(s0) = {v_star:.4f}  ({len(solver.memo)} states, {time.time() - t0:.1f}s)")
    mean, se = simulate_optimal(solver, s0, conf["optimal_episodes"], random.Random(1))
    print(f"  optimal policy simulated: {mean:.4f} +/- {se:.4f}  (should match V*)")

    sim = Sim(mdp)
    print("\nRegret on this root (V* - Q* of chosen action; one root is noisy, so planners are")
    print("repeated; the 10-root table in experiment_exact is the real evidence):")
    u, sp = conf["uct"], conf["sparse"]
    leaf = greedy_rollout if sp["greedy_leaf"] else None
    cands = {   # name: (planner call, repetitions)
        "greedy-play": (lambda: greedy_policy(mdp, s0, rng), 1),
        f"UCT {u['iters']} iters": (lambda: UCT(sim, n_iter=u["iters"]).plan(s0, rng), u["reps"]),
        f"sparse H={sp['H']} C={sp['C']}" + (" +greedy leaf" if leaf else ""):
            (lambda: SparseSampling(sim, C=sp["C"], H=sp["H"], leaf=leaf).plan(s0, rng), sp["reps"]),
    }
    for name, (f, reps) in cands.items():
        regrets, calls = [], 0
        for _ in range(reps):
            sim.calls = 0
            regrets.append(v_star - q[f()])
            calls += sim.calls
        print(f"  {name:28s} mean regret = {sum(regrets) / reps:.4f} over {reps} run(s)"
              f"   ({calls // reps} simulator calls/decision)")


def regret_table(conf, out_dir="results"):
    cfg = config_from_dict(conf["game"])
    print(f"\nRegret table on {_describe(conf['game'])}:")
    rows = experiment_exact(cfg, **conf.get("params", {}))
    if conf.get("plot"):
        from experiments.plotting import plot_regret   # optional matplotlib dependency
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print(f"saved {plot_regret(rows, path=str(Path(out_dir) / 'regret_vs_calls.png'))}")


def full_game(conf):
    cfg = config_from_dict(conf["game"])
    print(f"\nGreedy play vs UCT on {_describe(conf['game'])}, target {cfg.target}:")
    experiment_full(cfg, **conf.get("params", {}))


def baselines(conf):
    print(f"\nNo-search baselines on {_describe(conf['game'])}:")
    experiment_baselines(config_from_dict(conf["game"]), **conf.get("params", {}))


def search(conf):
    cfg = config_from_dict(conf["game"])
    for run in conf["runs"]:
        print(f"\nSearch vs discard heuristic, target {run['target']} ({run['episodes']} games):")
        experiment_search(replace(cfg, target=run["target"]), episodes=run["episodes"], **conf.get("params", {}))


def validation(conf):
    print("\nExact-solver validation:")
    experiment_validation(config_from_dict(conf["tiny_game"]), config_from_dict(conf["small_game"]),
                          **conf.get("params", {}))


def tiny_diagnostics(conf):
    cfg = config_from_dict(conf["game"])
    print(f"\nDiagnostics on {_describe(conf['game'])}:")
    experiment_tiny_diagnostics(cfg, **conf.get("params", {}))
    print(f"\nRegret table on {_describe(conf['game'])}:")
    experiment_exact(cfg, **conf.get("regret_params", {}))


def ablation(conf):
    cfg = config_from_dict(conf["game"])
    print(f"\nUCT ablations, {_describe(conf['game'])}, target {cfg.target}:")
    experiment_ablation(cfg, **conf.get("params", {}))


EXPERIMENTS = {   # name -> runner; each reads configs/<name with _ for ->.yaml
    "demo": demo,
    "regret-table": regret_table,
    "full-game": full_game,
    "baselines": baselines,
    "search": search,
    "validation": validation,
    "tiny-diagnostics": tiny_diagnostics,
    "ablation": ablation,
}


def _describe(game: dict) -> str:
    preset = game.get("preset", "full")
    return "the full 52-card game" if preset == "full" and "ranks" not in game else f"Config.{preset}()"


def load_config(path: Path, overrides=()) -> dict:
    """Read a YAML config and apply `--set dotted.key=value` overrides (values parsed as YAML)."""
    with open(path) as f:
        conf = yaml.safe_load(f) or {}
    conf = copy.deepcopy(conf)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        parts = key.split(".")
        node = conf
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return conf


def main(argv=None):
    p = argparse.ArgumentParser(prog="python main.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="*", choices=["all", *EXPERIMENTS],
                   help="what to run (default: demo)")
    p.add_argument("--config", type=Path,
                   help="config file to use instead of configs/<experiment>.yaml (one experiment only)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set params.episodes=20 (repeatable)")
    p.add_argument("--plot", action="store_true", help="with regret-table: save regret_vs_calls.png")
    p.add_argument("--out-dir", default="results", help="where figures are written (default: results/)")
    args = p.parse_args(argv)
    names = args.experiments or ["demo"]
    if "all" in names:
        names = list(EXPERIMENTS)
    if args.config and len(names) > 1:
        p.error("--config works with a single experiment")
    for name in names:
        path = args.config or CONFIG_DIR / f"{name.replace('-', '_')}.yaml"
        conf = load_config(path, args.overrides)
        if name == "regret-table":
            conf["plot"] = conf.get("plot") or args.plot
            regret_table(conf, out_dir=args.out_dir)
        else:
            EXPERIMENTS[name](conf)


if __name__ == "__main__":
    main()
