# dev_visualization: experiment data and visualization

`experiments/` is where the exploration stack and the baselines write all run output. It is git-ignored.

```
experiments/
├── node_logs/                    per-node stdout from scripts/start_*.sh
├── online_train/                 AdExPro training logs (online_train_*.csv, online_samples_*.csv)
├── runs/<world>/<method>/        per-run CSVs from the experiment runner
├── logs/                         runner pane logs
└── summary.csv                   one row per runner run
```

## Scripts

- `plot_experiments.py <experiments_dir>` plots the mean ± 1σ exploration curves for each world, metric and method from the runner output.
