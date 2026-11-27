"""Game configuration. Decks live in configs/decks/*.yaml; this module loads them into a Config."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DECK_DIR = Path(__file__).resolve().parent.parent / "configs" / "decks"


@dataclass(frozen=True)
class Config:
    ranks: tuple[int, ...]         # 2..10, 11 = J, 12 = Q, 13 = K, 14 = A
    n_suits: int
    hand_size: int                 # cards held
    hands: int                     # plays per round
    discards: int                  # discards per round
    max_cards: int = 5             # cards per play or discard
    target: int = 300              # Ante 1 blinds: 300 small, 450 big, 600 boss
    reward_mode: str = "points"    # "points": reward = score of each play
    #                                "target": reward = 1 on first reaching target (returns in [0,1])

    def full_deck(self):
        return tuple(sorted((r, s) for r in self.ranks for s in range(self.n_suits)))


_FIELDS = {f.name for f in fields(Config)}
_DECK_META = {"name", "description", "base"}


def available_decks() -> list[str]:
    return sorted(p.stem for p in DECK_DIR.glob("*.yaml"))


def load_deck(name: str) -> dict:
    """Settings from configs/decks/<name>.yaml, following `base:` to the deck it extends."""
    path = DECK_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown deck {name!r}; available: {available_decks()}")
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    settings = load_deck(d["base"]) if "base" in d else {}
    settings.update({k: v for k, v in d.items() if k not in _DECK_META})
    unknown = set(settings) - _FIELDS
    if unknown:
        raise ValueError(f"{path}: unknown setting(s) {sorted(unknown)}; valid: {sorted(_FIELDS)}")
    return settings


def config_from_dict(d: dict | None) -> Config:
    """Build a Config from a config-file `game` section: `deck` (a file in configs/decks, default
    standard) plus any Config fields to set or override, e.g. {"deck": "red", "target": 450}."""
    d = dict(d or {})
    d.pop("name", None)                       # label used by some config files
    settings = load_deck(d.pop("deck", "standard"))
    unknown = set(d) - _FIELDS
    if unknown:
        raise ValueError(f"unknown game setting(s) {sorted(unknown)}; valid: {sorted(_FIELDS)}")
    settings.update(d)
    settings["ranks"] = tuple(settings["ranks"])
    return Config(**settings)


def deck_config(name: str = "standard", **overrides) -> Config:
    """Shorthand for code and tests: deck_config("tiny", reward_mode="target")."""
    return config_from_dict({"deck": name, **overrides})
