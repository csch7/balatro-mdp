"""Game configuration. Decks live in configs/decks/*.yaml and hand values in configs/scoring/*.yaml;
this module loads them into a Config."""
from __future__ import annotations

from dataclasses import dataclass, fields
from functools import cached_property
from pathlib import Path

import yaml

from .rules import HAND_NAMES

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DECK_DIR = CONFIG_DIR / "decks"
SCORING_DIR = CONFIG_DIR / "scoring"


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
    scoring: str = "level1"        # a file in configs/scoring/

    def full_deck(self):
        return tuple(sorted((r, s) for r in self.ranks for s in range(self.n_suits)))

    @cached_property
    def hand_table(self) -> dict[str, tuple[int, int]]:
        """{hand name: (base chips, mult)} from configs/scoring/<scoring>.yaml."""
        return load_scoring(self.scoring)


_FIELDS = {f.name for f in fields(Config)}
_DECK_META = {"name", "description", "base"}


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def available_decks() -> list[str]:
    return sorted(p.stem for p in DECK_DIR.glob("*.yaml"))


def load_deck(name: str) -> dict:
    """Settings from configs/decks/<name>.yaml, following `base:` to the deck it extends."""
    path = DECK_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown deck {name!r}; available: {available_decks()}")
    d = _read_yaml(path)
    settings = load_deck(d["base"]) if "base" in d else {}
    settings.update({k: v for k, v in d.items() if k not in _DECK_META})
    unknown = set(settings) - _FIELDS
    if unknown:
        raise ValueError(f"{path}: unknown setting(s) {sorted(unknown)}; valid: {sorted(_FIELDS)}")
    return settings


def load_scoring(name: str) -> dict[str, tuple[int, int]]:
    """{hand name: (base chips, mult)} from configs/scoring/<name>.yaml."""
    path = SCORING_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"unknown scoring {name!r}; available: {sorted(p.stem for p in SCORING_DIR.glob('*.yaml'))}")
    hands = _read_yaml(path).get("hands", {})
    missing, extra = set(HAND_NAMES) - set(hands), set(hands) - set(HAND_NAMES)
    if missing or extra:
        raise ValueError(f"{path}: missing hands {sorted(missing)}, unknown hands {sorted(extra)}")
    return {name: (int(chips), int(mult)) for name, (chips, mult) in hands.items()}


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
    cfg = Config(**settings)
    cfg.hand_table                            # fail early on a bad scoring file
    return cfg


def deck_config(name: str = "standard", **overrides) -> Config:
    """Shorthand for code and tests: deck_config("tiny", reward_mode="target")."""
    return config_from_dict({"deck": name, **overrides})
