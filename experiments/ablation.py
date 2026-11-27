"""Ablation of UCT components (configs/ablation.yaml): the reported UCT configuration with one
setting changed at a time, plus no-search baselines, all on the same games."""
from __future__ import annotations

from game.config import config_from_dict

from .compare import compare_policies


def run_ablation(conf, out_dir):
    cfg = config_from_dict(conf["game"])
    base = conf["uct"]
    methods = list(conf.get("baselines", []))
    for name, overrides in conf["variants"].items():
        methods.append({**base, **(overrides or {}), "label": name})
    print(f"UCT ablations on the {conf['game'].get('deck', 'standard')} deck, target {cfg.target}:")
    return {"rows": compare_policies(cfg, methods, **conf["params"])}
