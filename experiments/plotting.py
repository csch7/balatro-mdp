"""Figures for the paper (needs matplotlib: pip install matplotlib)."""
from __future__ import annotations


def plot_regret(rows, path) -> str:
    """Regret vs simulator calls (log x) for UCT, sparse sampling and the zero-call baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    uct = [(r["calls_per_decision"], r["regret"]) for r in rows if r["type"] == "uct"]
    sp = [(r["calls_per_decision"], r["regret"]) for r in rows if r["type"] == "sparse"]
    if uct:
        ax.plot(*zip(*uct), "o-", label="UCT")
    if sp:
        ax.plot(*zip(*sp), "s--", label="Sparse sampling (various H, C)")
    styles = iter([":", "-.", "--", (0, (1, 3))])
    for r in rows:
        if r["type"] not in ("uct", "sparse", "flat_mc"):
            ax.axhline(r["regret"], color="gray", ls=next(styles, "--"), lw=1)
            ax.text(ax.get_xlim()[0] * 1.1 if uct or sp else 1, r["regret"] + 0.005, r["method"],
                    fontsize=8, color="gray", ha="left")
    ax.set_xscale("log"); ax.set_xlabel("simulator calls per decision"); ax.set_ylabel("regret vs exact V*")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)
