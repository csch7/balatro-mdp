"""Win rates of several methods on the same games. Every method plays the same `episodes` games:
game i uses seed "<seed>:<i>", so all methods see the same shuffled decks (paired comparison)."""
from __future__ import annotations

import math
import time

from game.config import Config
from game.episode import run_episode
from game.mdp import BalatroMDP
from planners.registry import build_policy


def compare_policies(cfg: Config, methods: list, episodes: int, seed, reference: str | None = None,
                     progress_every: int = 25) -> list[dict]:
    """Run each method spec for `episodes` games; if `reference` names a method listed earlier, also
    report the paired difference against it. Returns one row per method."""
    mdp = BalatroMDP(cfg)
    rows, outcomes = [], {}
    for spec in methods:
        policy = build_policy(spec, mdp)
        t0, res = time.time(), []
        for ep in range(episodes):
            res.append(run_episode(mdp, policy, f"{seed}:{ep}"))
            if policy.sim is not None and progress_every and (ep + 1) % progress_every == 0 and ep + 1 < episodes:
                print(f"    {policy.label}: {ep + 1}/{episodes} games, {time.time() - t0:.0f}s", flush=True)
        p = sum(res) / episodes
        row = {"method": policy.label, "target": cfg.target, "episodes": episodes, "wins": sum(res),
               "win_rate": p, "se": math.sqrt(p * (1 - p) / episodes)}
        line = f"  {policy.label:46s} win={p:.3f} +/- {row['se']:.3f}"
        if policy.sim is not None:
            row["sec_per_game"] = (time.time() - t0) / episodes
            row["calls_per_game"] = policy.sim.calls / episodes
            line += f"  {row['sec_per_game']:5.1f}s/game  {row['calls_per_game']:7.0f} calls/game"
        for k, v in policy.stats().items():
            row[k] = v
            line += f"  {k.replace('_', ' ')} {v:.3f}"
        if reference in outcomes and policy.label != reference:
            d = [a - b for a, b in zip(res, outcomes[reference])]
            md = sum(d) / episodes
            sd = math.sqrt(sum((x - md) ** 2 for x in d) / max(episodes - 1, 1) / episodes)
            row["vs_reference"], row["vs_reference_se"] = md, sd
            line += f"  vs {reference}: {md:+.3f} +/- {sd:.3f}"
        row["outcomes"] = res
        print(line, flush=True)
        rows.append(row)
        outcomes[policy.label] = res
    return rows
