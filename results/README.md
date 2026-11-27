# results

`python main.py <experiment>` saves each run to `results/<experiment>/<timestamp>/`:

- `config.yaml`: the exact settings used, including command-line overrides
- `log.txt`: everything the run printed
- `results.json`: every number in the report, plus per-game outcomes for win-rate experiments
- figures such as `regret_vs_calls.png`

Commit the runs that back numbers in the paper.
