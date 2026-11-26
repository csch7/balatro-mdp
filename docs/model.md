# Model and methods

## The MDP

- **State** `(hand, deck, hands_left, discards_left, score)`. The deck is the unordered set of undrawn cards, so the state is Markov: draws are uniform from that set.
- **Actions** choose 1 to 5 cards from the hand to play or to discard. An 8-card hand gives 218 subsets x 2 = 436 actions.
- **Transitions** remove the chosen cards and refill the hand with a uniform draw. Each k-card draw from an m-card deck has probability 1 / C(m, k). `BalatroMDP.step` samples a draw (the generative model) and `BalatroMDP.transitions` enumerates all of them (small decks only).
- **Rewards.** In `reward_mode="points"` each play earns its score, (base chips + chips of scoring cards) x mult with level-1 values. In `reward_mode="target"` the reward is 1 when the score first reaches the blind, so V(s) is the probability of beating the blind.
- **Suit symmetry.** `BalatroMDP.canonical` maps a state to the smallest key over all suit relabelings, so states that differ only by suit share a value and a UCT node.

Simplifications compared with real Balatro: one blind of Ante 1, no jokers, tarot/planet cards, money, shop or hand levels.

## Configurations

| Config | Deck | Hand | Hands / discards | Exact solve |
|---|---|---|---|---|
| `Config()` | 52 cards | 8 | 4 / 3 | no (about 10^46 evaluations) |
| `Config.medium()` | 21 cards | 5 | 3 / 2 | no (about 3.6 x 10^16) |
| `Config.small()` | 14 cards (8 to A, 2 suits) | 5 | 2 / 1 | yes, about 10 to 20 s per root |
| `Config.tiny()` | 12 cards (9 to A, 2 suits) | 5 | 2 / 1 | yes, a few seconds |

`solvers.exact.tree_size` gives these bounds, and `ExactSolver.solve` raises `TooBig` above 5 x 10^8.

## Methods

- **Exact backward induction** (`solvers/exact.py`): V(s) = max_a [ r(s,a) + E V(s') ], memoized on the canonical key. Plays that end the episode skip draw enumeration, and the hands_left = 1, discards_left = 0 case has a closed form. `ReferenceSolver` is a shortcut-free second implementation used for cross-checking.
- **Sparse sampling** (`planners/sparse_sampling.py`, Kearns, Mansour & Ng 2002): every action at every node, C sampled next states, depth H, optional leaf evaluator. Cost is about (|A| C)^H simulator calls, independent of |S|.
- **UCT** (`planners/uct.py`, Kocsis & Szepesvari 2006): UCB1 selection W_a/N_a + c sqrt(ln N / N_a), progressive widening over the first 1 + floor(2 sqrt N) actions of `policies.ordering.ordered_actions`, greedy-rollout leaf evaluation, and a transposition table keyed by the canonical state. The move played is the most visited root action.
- **Flat Monte Carlo** (`planners/flat_mc.py`): the top 20 ranked actions, n rollouts each, best mean wins.
- **Baselines** (`policies/baselines.py`): greedy play (best 5-card hand, never discard) and the discard heuristic (discard non-scoring cards while the best hand is at most Two Pair).

## Experiments

| CLI name | Module | What it measures |
|---|---|---|
| `regret-table` | `experiments/regret.py` | regret V*(s) - Q*(s, a) on `Config.small()` vs simulator calls (Table 2, regret plot) |
| `full-game` | `experiments/full_game.py` | greedy vs UCT win rate on the 52-card Big Blind |
| `baselines` | `experiments/full_game.py` | no-search policies at targets 300, 450, 600 |
| `search` | `experiments/full_game.py` | UCT and flat MC vs the discard heuristic, paired on identical seeds |
| `validation` | `experiments/validation.py` | ExactSolver vs ReferenceSolver; simulated optimal play vs V* |
| `tiny-diagnostics` | `experiments/validation.py` | sparse-sampling ties at C = 1, UCT bias on shallow variants |
| `ablation` | `experiments/ablation.py` | UCT with one component changed at a time |
