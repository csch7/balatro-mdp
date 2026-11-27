"""Command line: runs experiments with the settings in configs/<experiment>.yaml and saves each
run (config, log, results, figures) to results/<experiment>/<timestamp>/.

    python main.py                                   # quick demo (~15 s)
    python main.py regret-table --plot               # one or more experiments, in order
    python main.py full-game --deck red --set params.episodes=20
    python main.py full-game --config my_run.yaml    # use another config file
    python main.py all                               # every experiment (2-3 hours on one core)
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from experiments import EXPERIMENTS
from experiments.output import RunRecorder
from game import available_decks

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(path: Path, overrides=()) -> dict:
    """Read a YAML config and apply `dotted.key=value` overrides (values parsed as YAML)."""
    with open(path) as f:
        conf = copy.deepcopy(yaml.safe_load(f) or {})
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        parts, node = key.split("."), conf
        for i, part in enumerate(parts):           # list items are addressed by index: methods.1.iters
            k = int(part) if isinstance(node, list) else part
            if i == len(parts) - 1:
                node[k] = yaml.safe_load(raw)
            elif isinstance(node, list):
                node = node[k]
            else:
                node = node.setdefault(k, {})
    return conf


def main(argv=None):
    p = argparse.ArgumentParser(prog="python main.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="*", choices=["all", *EXPERIMENTS], help="what to run (default: demo)")
    p.add_argument("--config", type=Path,
                   help="config file to use instead of configs/<experiment>.yaml (one experiment only)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set params.episodes=20 (repeatable)")
    p.add_argument("--deck", choices=available_decks(), help="same as --set game.deck=NAME")
    p.add_argument("--plot", action="store_true", help="same as --set plot=true (regret-table)")
    p.add_argument("--out-dir", default="results", help="where runs are saved (default: results/)")
    p.add_argument("--no-save", action="store_true", help="print only; write nothing to --out-dir")
    args = p.parse_args(argv)
    names = args.experiments or ["demo"]
    if "all" in names:
        names = list(EXPERIMENTS)
    if args.config and len(names) > 1:
        p.error("--config works with a single experiment")
    overrides = args.overrides + ([f"game.deck={args.deck}"] if args.deck else []) + (["plot=true"] if args.plot else [])
    for name in names:
        conf = load_config(args.config or CONFIG_DIR / f"{name.replace('-', '_')}.yaml", overrides)
        print(f"\n=== {name} ===")
        with RunRecorder(args.out_dir, name, conf, enabled=not args.no_save) as run:
            run.save(EXPERIMENTS[name](conf, run.dir))


if __name__ == "__main__":
    main()
