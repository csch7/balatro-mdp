# balatro-mdp

Ante 1 of Balatro modeled as a finite-horizon Markov decision process (STOR 543, Dynamic Decision Analysis).
The code includes a generative simulator, an exact backward-induction solver for small decks, sparse sampling,
UCT with progressive widening, flat Monte Carlo, no-search baselines, and the experiments from the paper.
See [docs/model.md](docs/model.md) for the model and methods.

## Project layout

```
balatro-mdp/
├── main.py                  command line: runs experiments from their config files and saves the results
├── configs/
│   ├── <experiment>.yaml    one per experiment: game, methods to compare, run parameters
│   ├── decks/               one per deck: standard, red, blue, black, painted, abandoned, small, tiny, medium
│   └── scoring/             chip/mult values per poker hand (level1.yaml)
├── game/
│   ├── rules.py             cards, hand evaluation, chips x mult scoring
│   ├── config.py            Config; loads decks and scoring files
│   ├── mdp.py               State, BalatroMDP (step, apply_draw, transitions, canonical)
│   ├── simulator.py         Sim (counts simulator calls), rollout
│   └── episode.py           real games: one shuffled deck per seed, separate policy randomness
├── policies/
│   ├── base.py              the Policy interface: act(state, rng)
│   ├── baselines.py         GreedyPolicy, DiscardHeuristic, RandomPolicy
│   └── ordering.py          action orderings for progressive widening
├── planners/
│   ├── uct.py, flat_mc.py, sparse_sampling.py
│   └── registry.py          build_policy(spec): any method from a config entry
├── solvers/
│   ├── exact.py             tree_size, ExactSolver, simulate_optimal
│   └── reference.py         ReferenceSolver (cross-check)
├── experiments/
│   ├── demo.py, regret.py, full_game.py, ablation.py, validation.py
│   ├── compare.py           win rates of several methods on the same games
│   ├── output.py            saves each run to results/
│   └── plotting.py          regret_vs_calls figure
├── tests/
├── docs/model.md            the MDP, decks, methods and experiments
├── results/                 saved runs (see results/README.md)
└── requirements.txt         pyyaml, matplotlib (plots), pytest (tests)
```

## Running

Python 3.10+, run from the repository root after `pip install -r requirements.txt`.

```
python main.py                            # quick demo (~15 s)
python main.py regret-table --plot        # Table 2 + regret figure (~5 min)
python main.py full-game baselines        # several experiments, in order
python main.py all                        # everything (2-3 hours on one core)
pytest
```

Experiments: `demo`, `regret-table`, `full-game`, `baselines`, `search`, `validation`, `tiny-diagnostics`, `ablation`.

Every run is saved to `results/<experiment>/<timestamp>/`: `config.yaml` (the exact settings, including
command-line overrides), `log.txt` (everything printed), `results.json` (all numbers, including per-game
outcomes) and any figures. Use `--no-save` to only print.

## Configs

Each experiment reads `configs/<experiment>.yaml` (dashes become underscores):

```yaml
# configs/full_game.yaml
game:
  deck: standard        # a file in configs/decks/
  reward_mode: target   # any other key sets or overrides that Config field
  target: 450
params:
  episodes: 100
  seed: 0               # game i uses seed "<seed>:<i>"; every method plays the same games
methods:
  - {label: greedy-play, type: greedy}
  - {label: UCT, type: uct, iters: 300, c: 1.0, rollout: greedy}
```

Methods are specs for `planners/registry.py`: `greedy`, `heuristic`, `random`, `uct`, `flat_mc` and `sparse`,
with their settings (see the docstring there). To change a run, edit the file, use your own, or override values:

```
python main.py full-game --config my_run.yaml
python main.py full-game --set params.episodes=20 --set methods.1.iters=1000
python main.py full-game --deck abandoned
```

Decks (`configs/decks/*.yaml`) hold the cards and round rules; `base: standard` lets a deck change only what
differs. The blind (`target`, `reward_mode`) belongs to the experiment config. See
[configs/decks/README.md](configs/decks/README.md) for which Balatro decks are modeled.

## Randomness

A real game shuffles the deck once (from the episode seed) and deals from the top, as Balatro does. Planners get a
separate generator for their own simulations, so searching never changes the cards that come next, and every
method in an experiment faces exactly the same card order (paired comparison).

Library use:

```python
from game import BalatroMDP, deck_config, run_episode
from planners import build_policy

mdp = BalatroMDP(deck_config("standard", reward_mode="target", target=450))
uct = build_policy({"type": "uct", "iters": 300}, mdp)
win = run_episode(mdp, uct, seed="0:0")
```
