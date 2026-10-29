# dev_visualization: ROS visualization packages

Built in the same colcon workspace (the repo root) as `dev_explore/`, with the system Python
(step 1 of the build in the top-level README). Run output goes to `../dev_results/experiments/`.

```
dev_visualization/
├── prediction_eval/      scores every /predicted_map against the ground-truth /gt_map and records
│                         per-cell accuracy (headless)
└── prediction_heatmap/   live matplotlib window drawing prediction_eval's results
```

## prediction_eval

Only cells that were unknown in the visible map the predictor saw, and that it filled in, are
compared with GT. Per-cell outcome counts accumulate on the GT grid. Output goes to
`$EXPRO_EXPERIMENTS_DIR/prediction_eval/<method>_<world>_<stamp>/` (`accuracy.npz`, rewritten every
5 s, and `cycles.csv`). Every 2 s it publishes the per-cell results on the GT grid as
`prob_msgs/ProbGrid` (transient-local):

| Topic | Per cell |
|---|---|
| `/prediction_eval/accuracy` | share of predictions that matched GT (NaN: never scored) |
| `/prediction_eval/error_balance` | (phantom − missed) / predictions (NaN: never wrong) |
| `/prediction_eval/n_eval` | times the cell was scored |

A phantom obstacle is a cell predicted occupied that GT has free; a missed obstacle is predicted
free but occupied in GT.

`prediction_eval.launch world:= method:= [include_gt:=true]` starts the GT octomap
(`gt_mapping_<world>.launch`) and the node. The experiment runner uses it for `expro`, `adexpro`
and `oracle` runs.

## prediction_heatmap

Subscribes to those topics plus `/gt_map` and `/state_estimation`, and redraws a window every 2 s:

- **accuracy**: per cell, the share of its predictions that matched GT; the robot's path, its
  start (black) and current position (orange)
- **error type**: per cell, phantom obstacles (red) vs missed obstacles (blue)

Cells scored fewer than `min_count` (default 3) times are hidden. Closing the window does not stop
prediction_eval.

`prediction_heatmap.launch world:= method:=` = `prediction_eval.launch` + the window. This is what
`scripts/start_{expro,adexpro,oracle}.sh` run in pane 4 (`EVAL=0` turns both off). By hand:

```bash
ros2 launch prediction_heatmap prediction_heatmap.launch world:=indoor method:=adexpro
```
