"""Config files, deck files and the game-section parser."""
from pathlib import Path

import pytest

from game import available_decks, config_from_dict, deck_config, load_deck
from experiments import EXPERIMENTS
from game import load_scoring
from main import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_standard_deck_is_the_real_game():
    cfg = deck_config("standard")
    assert len(cfg.full_deck()) == 52 and (cfg.hand_size, cfg.hands, cfg.discards) == (8, 4, 3)


def test_toy_decks_match_the_paper():
    small, tiny = deck_config("small"), deck_config("tiny")
    assert len(small.full_deck()) == 14 and len(tiny.full_deck()) == 12
    assert (small.hand_size, small.hands, small.discards) == (5, 2, 1)


def test_base_inheritance():
    assert deck_config("red").discards == 4 and deck_config("red").hands == 4
    assert len(deck_config("abandoned").full_deck()) == 40


@pytest.mark.parametrize("name", available_decks())
def test_every_deck_loads(name):
    assert load_deck(name)["n_suits"] >= 1


def test_overrides():
    cfg = config_from_dict({"deck": "blue", "target": 450, "reward_mode": "target"})
    assert cfg.hands == 5 and cfg.target == 450


def test_unknown_key_and_deck_are_rejected():
    with pytest.raises(ValueError):
        config_from_dict({"deck": "standard", "hand_sise": 8})
    with pytest.raises(ValueError):
        config_from_dict({"deck": "purple"})


@pytest.mark.parametrize("name", list(EXPERIMENTS))
def test_every_experiment_has_a_valid_config(name):
    conf = load_config(CONFIGS / f"{name.replace('-', '_')}.yaml")
    for key, section in conf.items():
        if key == "game" or key.endswith("_game"):
            config_from_dict(section)


def test_set_override():
    conf = load_config(CONFIGS / "full_game.yaml", ["params.episodes=5", "game.deck=red"])
    assert conf["params"]["episodes"] == 5 and config_from_dict(conf["game"]).discards == 4


def test_scoring_file():
    table = load_scoring("level1")
    assert table["Pair"] == (10, 2) and len(table) == 9
    assert deck_config("standard").hand_table == table


def test_set_override_into_a_list():
    conf = load_config(CONFIGS / "full_game.yaml", ["methods.1.iters=20"])
    assert conf["methods"][1]["iters"] == 20
