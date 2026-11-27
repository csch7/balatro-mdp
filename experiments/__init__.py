"""Experiments. Each takes its config (configs/<name>.yaml) and an output folder, prints a
report, and returns its results as a dict (saved to results.json by main.py)."""
from .ablation import run_ablation
from .demo import run_demo
from .full_game import run_baselines, run_full_game, run_search
from .regret import run_regret_table
from .validation import run_tiny_diagnostics, run_validation

EXPERIMENTS = {   # CLI name -> runner; the config is configs/<name with - replaced by _>.yaml
    "demo": run_demo,
    "regret-table": run_regret_table,
    "full-game": run_full_game,
    "baselines": run_baselines,
    "search": run_search,
    "validation": run_validation,
    "tiny-diagnostics": run_tiny_diagnostics,
    "ablation": run_ablation,
}

__all__ = ["EXPERIMENTS"]
