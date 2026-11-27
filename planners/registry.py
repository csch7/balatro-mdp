"""Build any policy or planner from a config-file spec, so experiment configs can list the
methods to compare. A spec is a type name or a dict:

    {type: greedy, tie_break: largest}                      no search
    {type: heuristic, discard_on: [High Card, Pair]}        no search
    {type: random}
    {type: uct, iters: 300, c: 1.0, rollout: greedy, widen: true, widen_c: 2.0,
     widen_alpha: 0.5, key: canonical, action_order: ordered}
    {type: flat_mc, m: 20, n: 15, rollout: heuristic}
    {type: sparse, H: 2, C: 4, leaf: greedy}

`rollout` / `leaf` take a no-search spec, or null for a zero leaf value. `label` names the method
in printed tables and results (default: built from the spec).
"""
from __future__ import annotations

from game.mdp import BalatroMDP
from game.simulator import Sim
from policies.base import Policy
from policies.baselines import DiscardHeuristic, GreedyPolicy, RandomPolicy

from .flat_mc import FlatMC
from .sparse_sampling import SparseSampling
from .uct import UCT

NO_SEARCH = {"greedy": GreedyPolicy, "heuristic": DiscardHeuristic, "random": RandomPolicy}
PLANNERS = {"uct": (UCT, "rollout", "greedy"), "flat_mc": (FlatMC, "rollout", "heuristic"),
            "sparse": (SparseSampling, "leaf", None)}


def _label(spec: dict) -> str:
    return spec.get("label") or " ".join([spec["type"]] + [f"{k}={v}" for k, v in spec.items() if k != "type"])


def build_policy(spec, mdp: BalatroMDP) -> Policy:
    spec = {"type": spec} if isinstance(spec, str) else dict(spec)
    label = _label(spec)
    spec.pop("label", None)
    kind = spec.pop("type")
    if kind in NO_SEARCH:
        policy = NO_SEARCH[kind](mdp, **spec)
    elif kind in PLANNERS:
        cls, inner_key, default_inner = PLANNERS[kind]
        inner = spec.pop(inner_key, default_inner)
        inner_policy = None if inner is None else build_policy(inner, mdp)
        if inner_policy is not None and inner_policy.sim is not None:
            raise ValueError(f"{inner_key} must be a no-search policy, got {inner!r}")
        policy = cls(Sim(mdp), inner_policy, **spec)
    else:
        raise ValueError(f"unknown method type {kind!r}; choose from {sorted(NO_SEARCH) + sorted(PLANNERS)}")
    policy.label = label
    return policy
