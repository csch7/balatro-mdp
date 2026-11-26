# balatro-mdp

Ante 1 of Balatro modeled as a finite-horizon Markov decision process (STOR 543, Dynamic Decision Analysis).
The code includes a generative simulator, an exact backward-induction solver for small decks, sparse sampling,
UCT with progressive widening, flat Monte Carlo, no-search baselines, and the experiments from the paper.
See [docs/model.md](docs/model.md) for the model and methods.

## Project layout

```
balatro-mdp/
├── pyproject.toml
├── README.md
├── docs/
│   └── model.md                 the MDP, configurations, methods and experiments
├── results/                     experiment outputs (figures, logs)
├── src/balatro_mdp/
│   ├── cli.py                   `balatro-mdp` command (replaces the notebook's RUN_* flags)
│   ├── game/
│   │   ├── rules.py             cards, hand evaluation, chips x mult scoring
│   │   ├── config.py            Config: full game and small/tiny/medium toy decks
│   │   ├── mdp.py               State, BalatroMDP (step, transitions, canonical)
│   │   └── simulator.py         Sim (counts simulator calls), run_episode
│   ├── policies/
│   │   ├── baselines.py         greedy play, discard heuristic, their rollouts
│   │   └── ordering.py          ordered_actions for progressive widening
│   ├── solvers/
│   │   ├── exact.py             tree_size, ExactSolver, simulate_optimal
│   │   └── reference.py         ReferenceSolver (cross-check)
│   ├── planners/
│   │   ├── sparse_sampling.py   SparseSampling
│   │   ├── uct.py               UCT
│   │   └── flat_mc.py           FlatMC
│   └── experiments/
│       ├── regret.py            regret vs exact V*, feasibility table
│       ├── full_game.py         52-card game: baselines, UCT, search vs heuristic
│       ├── ablation.py          UCT ablations
│       ├── validation.py        solver validation, 12-card diagnostics
│       └── plotting.py          regret_vs_calls figure
└── tests/                       rules, MDP, solvers, planners
```

## Setup

Python 3.10+. The core package uses only the standard library.

```
pip install -e ".[dev,plot]"     # dev = pytest, plot = matplotlib
```

## Running

```
balatro-mdp                            # quick demo (~15 s)
balatro-mdp regret-table --plot        # Table 2 + results/regret_vs_calls.png (~5 min)
balatro-mdp full-game baselines        # several experiments, in order
balatro-mdp all                        # everything (2-3 hours on one core)
pytest
```

Experiments: `regret-table`, `full-game`, `baselines`, `search`, `validation`, `tiny-diagnostics`, `ablation`.
Without installing, use `PYTHONPATH=src python -m balatro_mdp ...`.

Library use:

```python
import random
from balatro_mdp.game import BalatroMDP, Config, Sim, run_episode
from balatro_mdp.planners import UCT

mdp = BalatroMDP(Config(reward_mode="target", target=450))
uct = UCT(Sim(mdp), n_iter=300)
win = run_episode(mdp, lambda m, s, r: uct.plan(s, r), random.Random(0))
```
