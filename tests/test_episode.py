"""Real episodes: a fixed shuffled deck per seed, independent of the policy's randomness."""
import random

from game import BalatroMDP, Episode, deck_config, episode_rngs, run_episode
from planners import build_policy


def _cards_seen(mdp, seed, policy):
    deck_rng, policy_rng = episode_rngs(seed)
    ep, seen = Episode(mdp, deck_rng), []
    seen.append(ep.state.hand)
    while not ep.done:
        ep.step(policy.act(ep.state, policy_rng))
        seen.append(ep.state.hand)
    return seen


def test_same_seed_same_cards_whatever_the_planner_draws():
    mdp = BalatroMDP(deck_config("standard", reward_mode="target", target=450))
    greedy = build_policy("greedy", mdp)

    class NoisyGreedy:            # same actions as greedy, but burns random numbers like a planner
        def act(self, s, rng):
            for _ in range(1000):
                rng.random()
            return greedy.act(s, rng)

    assert _cards_seen(mdp, "7:3", greedy) == _cards_seen(mdp, "7:3", NoisyGreedy())


def test_draws_come_off_the_top_of_one_shuffle():
    mdp = BalatroMDP(deck_config("tiny", reward_mode="target"))
    deck_rng, _ = episode_rngs(0)
    order = list(mdp.cfg.full_deck()); random.Random("deck:0").shuffle(order)
    ep = Episode(mdp, deck_rng)
    assert set(ep.state.hand) == set(order[:5])
    ep.step(("discard", (0, 1)))
    assert set(order[5:7]) <= set(ep.state.hand)


def test_run_episode_is_reproducible():
    mdp = BalatroMDP(deck_config("standard", reward_mode="target", target=450))
    uct = build_policy({"type": "uct", "iters": 20}, mdp)
    assert run_episode(mdp, uct, "1:2") == run_episode(mdp, uct, "1:2")
