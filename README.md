# balatro-mdp

Ante 1 of Balatro modeled as a finite-horizon Markov decision process (STOR 543, Dynamic Decision Analysis).
The code includes a generative simulator, an exact backward-induction solver for small decks, sparse sampling,
UCT with progressive widening, flat Monte Carlo, no-search baselines, and the experiments from the paper.
See [docs/model.md](docs/model.md) for the model and methods.

## Project layout

```
balatro-mdp/
├── main.py                  command line for the demo and experiments (replaces the notebook's RUN_* flags)
├── configs/                 one YAML file per experiment: game settings + run parameters
│   └── decks/               one YAML file per deck: standard, red, blue, black, painted, abandoned, small, tiny, medium
├── game/
│   ├── rules.py             cards, hand evaluation, chips x mult scoring
│   ├── config.py            Config, deck loading (load_deck, config_from_dict, deck_config)
│   ├── mdp.py               State, BalatroMDP (step, transitions, canonical)
│   └── simulator.py         Sim (counts simulator calls), run_episode
├── policies/
│   ├── baselines.py         greedy play, discard heuristic, their rollouts
│   └── ordering.py          ordered_actions for progressive widening
├── solvers/
│   ├── exact.py             tree_size, ExactSolver, simulate_optimal
│   └── reference.py         ReferenceSolver (cross-check)
├── planners/
│   ├── sparse_sampling.py   SparseSampling
│   ├── uct.py               UCT
│   └── flat_mc.py           FlatMC
├── experiments/
│   ├── regret.py            regret vs exact V*, feasibility table
│   ├── full_game.py         52-card game: baselines, UCT, search vs heuristic
│   ├── ablation.py          UCT ablations
│   ├── validation.py        solver validation, 12-card diagnostics
│   └── plotting.py          regret_vs_calls figure
├── tests/                   rules, MDP, solvers, planners
├── docs/model.md            the MDP, configurations, methods and experiments
├── results/                 experiment outputs (figures, logs)
└── requirements.txt         pyyaml (configs), matplotlib (plots), pytest (tests)
```

## Running

Python 3.10+, run from the repository root after `pip install -r requirements.txt`.

```
python main.py                            # quick demo (~15 s)
python main.py regret-table --plot        # Table 2 + results/regret_vs_calls.png (~5 min)
python main.py full-game baselines        # several experiments, in order
python main.py all                        # everything (2-3 hours on one core)
pytest
```

Experiments: `demo`, `regret-table`, `full-game`, `baselines`, `search`, `validation`, `tiny-diagnostics`, `ablation`.

## Configs

Each experiment reads `configs/<experiment>.yaml` (dashes become underscores), and the defaults reproduce the paper's numbers.
A file has a `game` section and a `params` section:

```yaml
# configs/full_game.yaml
game:
  deck: standard        # a file in configs/decks/
  reward_mode: target   # any other key sets or overrides that Config field
  target: 450
params:
  episodes: 100
  iters: 300
  seed: 0
```

To change a run, edit the file, point at your own file, or override single values:

```
python main.py full-game --config my_run.yaml
python main.py full-game --set params.episodes=20 --set game.target=600
python main.py full-game --deck abandoned                # any deck in configs/decks/
```

Decks (`configs/decks/*.yaml`) hold the cards and round rules (`ranks`, `n_suits`, `hand_size`, `hands`, `discards`);
`base: standard` lets a deck change only what differs. The blind (`target`, `reward_mode`) belongs to the experiment config.
See [configs/decks/README.md](configs/decks/README.md) for which Balatro decks are modeled.

Library use:

```python
import random
from game import BalatroMDP, Sim, deck_config, run_episode
from planners import UCT

mdp = BalatroMDP(deck_config("standard", reward_mode="target", target=450))
uct = UCT(Sim(mdp), n_iter=300)
win = run_episode(mdp, lambda m, s, r: uct.plan(s, r), random.Random(0))
```
