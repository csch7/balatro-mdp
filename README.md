# balatro-mdp

Ante 1 of Balatro modeled as a finite-horizon Markov decision process (STOR 543, Dynamic Decision Analysis).
The code includes a generative simulator, an exact backward-induction solver for small decks, sparse sampling,
UCT with progressive widening, flat Monte Carlo, no-search baselines, and the experiments from the paper.
See [docs/model.md](docs/model.md) for the model and methods.

## Project layout

```
balatro-mdp/
├── main.py                  command line for the demo and experiments (replaces the notebook's RUN_* flags)
├── game/
│   ├── rules.py             cards, hand evaluation, chips x mult scoring
│   ├── config.py            Config: full game and small/tiny/medium toy decks
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
└── requirements.txt         matplotlib (plots) and pytest (tests); the code itself needs only the standard library
```

## Running

Python 3.10+, run from the repository root.

```
python main.py                            # quick demo (~15 s)
python main.py regret-table --plot        # Table 2 + results/regret_vs_calls.png (~5 min)
python main.py full-game baselines        # several experiments, in order
python main.py all                        # everything (2-3 hours on one core)
pytest                                    # tests (pip install -r requirements.txt)
```

Experiments: `regret-table`, `full-game`, `baselines`, `search`, `validation`, `tiny-diagnostics`, `ablation`.

Library use:

```python
import random
from game import BalatroMDP, Config, Sim, run_episode
from planners import UCT

mdp = BalatroMDP(Config(reward_mode="target", target=450))
uct = UCT(Sim(mdp), n_iter=300)
win = run_episode(mdp, lambda m, s, r: uct.plan(s, r), random.Random(0))
```
