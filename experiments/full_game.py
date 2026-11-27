"""Win-rate experiments on a full deck: greedy vs UCT, no-search baselines, search vs heuristic."""
from __future__ import annotations

from dataclasses import replace

from game.config import config_from_dict

from .compare import compare_policies


def run_full_game(conf, out_dir):
    """configs/full_game.yaml: every method on the same games at one target."""
    cfg = config_from_dict(conf["game"])
    print(f"Win rate on the {conf['game'].get('deck', 'standard')} deck, target {cfg.target}:")
    rows = compare_policies(cfg, conf["methods"], **conf["params"])
    return {"rows": rows}


def run_baselines(conf, out_dir):
    """configs/baselines.yaml: no-search methods at each target."""
    cfg = config_from_dict(conf["game"])
    params = dict(conf["params"])
    targets = params.pop("targets")
    rows = []
    for tgt in targets:
        print(f"\nTarget {tgt} ({params['episodes']} games each)")
        rows += compare_policies(replace(cfg, target=tgt), conf["methods"], **params)
    return {"rows": rows}


def run_search(conf, out_dir):
    """configs/search.yaml: search methods vs a reference method, one block per entry in `runs`."""
    cfg = config_from_dict(conf["game"])
    rows = []
    for run in conf["runs"]:
        print(f"\nTarget {run['target']} ({run['episodes']} games), paired against {conf['reference']!r}:")
        rows += compare_policies(replace(cfg, target=run["target"]), conf["methods"], run["episodes"],
                                 conf["params"]["seed"], reference=conf["reference"])
    return {"rows": rows}
