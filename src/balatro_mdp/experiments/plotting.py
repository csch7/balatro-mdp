"""Figures for the paper (needs matplotlib: pip install -e '.[plot]')."""
from __future__ import annotations


def plot_regret(rows, path="regret_vs_calls.png"):
    """Regret vs simulator calls (log x) for UCT, sparse sampling and the zero-call baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    uct = [(c, r) for n, c, r, _ in rows if n.startswith("UCT")]
    sp = [(c, r) for n, c, r, _ in rows if n.startswith("sparse")]
    ax.plot(*zip(*uct), "o-", label="UCT (25 to 6,400 iterations)")
    ax.plot(*zip(*sp), "s--", label="Sparse sampling (various H, C)")
    for n, c, r, _ in rows:
        if n in ("random", "greedy-play", "discard heuristic"):
            ax.axhline(r, color="gray", ls={"random": ":", "greedy-play": "-.", "discard heuristic": "--"}[n], lw=1)
            x, dy, ha = {"random": (3e4, 0.007, "right"), "discard heuristic": (3e4, -0.017, "right"),
                         "greedy-play": (55, 0.007, "left")}[n]
            ax.text(x, r + dy, n, fontsize=8, color="gray", ha=ha)
    ax.set_xscale("log"); ax.set_xlabel("simulator calls per decision"); ax.set_ylabel("regret vs exact V*")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=200)
    return path
