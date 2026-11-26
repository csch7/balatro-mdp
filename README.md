# balatro-mdp

Ante 1 of Balatro modeled as a finite-horizon Markov decision process (STOR 543, Dynamic Decision Analysis).
The code includes a generative simulator, an exact backward-induction solver for small decks, sparse sampling,
UCT with progressive widening, flat Monte Carlo, simple no-search baselines, and the experiments from the paper.

## Layout

```
balatro_mdp/
  rules.py        cards, poker-hand evaluation, level-1 chips x mult scoring
  mdp.py          Config (full game + small/tiny/medium toy decks), State, BalatroMDP, Sim, run_episode
  policies.py     greedy play, discard heuristic, their rollouts, ordered_actions (for widening)
  exact.py        tree_size feasibility bound, ExactSolver, simulate_optimal, ReferenceSolver
  sparse.py       SparseSampling (Kearns, Mansour & Ng, 2002)
  uct.py          UCT (Kocsis & Szepesvari, 2006) and FlatMC
  experiments/
    regret.py       regret vs exact V* on toy decks, feasibility table, regret plot
    full_game.py    52-card game: greedy vs UCT, no-search baselines, search vs heuristic
    ablation.py     UCT ablations (suit merging, widening, action order, rollouts, c, iterations)
    validation.py   solver cross-check and 12-card diagnostics
  __main__.py     command line, replaces the notebook's RUN_* flags
tests/            rule spot checks and a solver cross-check
```

## Running

Python 3.10+ with the standard library only. matplotlib is needed only for `--plot`.

```
python -m balatro_mdp                          # quick demo (~15 s)
python -m balatro_mdp regret-table --plot      # Table 2 + regret_vs_calls.png (~5 min)
python -m balatro_mdp full-game baselines      # run several experiments in order
python -m balatro_mdp all                      # everything (2-3 hours on one core)
python -m pytest tests
```

Experiments: `regret-table`, `full-game`, `baselines`, `search`, `validation`, `tiny-diagnostics`, `ablation`.

Library use:

```python
import random
from balatro_mdp import BalatroMDP, Config, Sim, UCT, run_episode

mdp = BalatroMDP(Config(reward_mode="target", target=450))
uct = UCT(Sim(mdp), n_iter=300)
win = run_episode(mdp, lambda m, s, r: uct.plan(s, r), random.Random(0))
```
