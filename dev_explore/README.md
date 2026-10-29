# dev_explore: the exploration system

ROS 2 packages for ExPro / AdExPro, the simulator stack and the batch experiment runner. The
colcon workspace is the repo root; setup, Docker, build and run instructions are in the
[top-level README](../README.md). Baselines live in [../dev_baselines](../dev_baselines/README.md).
Run output goes to `../dev_visualization/experiments/`.

## Packages

```
dev_explore/
├── autonomy_stack/        [submodule] CMU simulator and low-level navigation (local planner,
│                          terrain analysis, velodyne sim, visualization_tools metrics)
├── planner/               the method
│   ├── map_prediction/    FPUnet map prediction (Python, torch)
│   │   ├── map_prediction_node.py          ExPro predictor (FPUnet, optional TRT)
│   │   ├── self_adapt.py                   AdExPro: trainer thread + adapt policies
│   │   ├── map_prediction_trainer.py       OnlineTrainer (masked CE, L2-SP, distill / entropy)
│   │   ├── oracle_map_prediction_node.py   GT-filled windows (upper bound)
│   │   ├── config/                         map_prediction / online_update / oracle params
│   │   └── models/                         weights (git-ignored, download: models/readme.md)
│   ├── global_planner/    C++: reachability → frontiers → A* distance matrix → RPV visit order
│   ├── waypoint_bridge/   global goal → waypoints for the local planner
│   ├── occupancy_mapping/ occupancy-grid mapping
│   ├── simulator/         launch + config per world (mapping / gt_mapping / planner),
│   │                      the three prediction launches, rviz/expro.rviz
│   ├── msgs/              prob_msgs (ProbGrid confidence), rpv_msgs (planning stats)
│   └── plugins/           RViz plugins
└── experiment_runner/     batch runs from YAML (config/); used by scripts/run_experiments.sh
```

## Data flow

```
simulator ──scan──▶ octomap ──/visible_map──▶ map_prediction (expro | adexpro | oracle)
                                               │ /predicted_map, /confidence_map, /all_frontier
                                               ▼
           local planner ◀──waypoints── waypoint_bridge ◀──path── global_planner
```
