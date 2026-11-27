"""Config files and the game-section parser."""
from pathlib import Path

import pytest

from game import Config, config_from_dict
from main import EXPERIMENTS, load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_presets_match_code():
    assert config_from_dict({"preset": "small"}) == Config.small()
    assert config_from_dict(None) == Config()


def test_overrides_and_abandoned_deck():
    cfg = config_from_dict({"preset": "full", "ranks": [2, 3, 4, 5, 6, 7, 8, 9, 10, 14], "target": 450})
    assert len(cfg.full_deck()) == 40 and cfg.target == 450 and cfg.hands == 4


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError):
        config_from_dict({"preset": "full", "hand_sise": 8})


@pytest.mark.parametrize("name", list(EXPERIMENTS))
def test_every_experiment_has_a_valid_config(name):
    conf = load_config(CONFIGS / f"{name.replace('-', '_')}.yaml")
    for key, section in conf.items():
        if key == "game" or key.endswith("_game"):
            config_from_dict(section)


def test_set_override():
    conf = load_config(CONFIGS / "full_game.yaml", ["params.episodes=5", "game.target=600"])
    assert conf["params"]["episodes"] == 5 and conf["game"]["target"] == 600
