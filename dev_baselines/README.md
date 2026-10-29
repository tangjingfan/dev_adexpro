# dev_baselines: baseline exploration planners

ROS 2 baselines, built in the same colcon workspace (the repo root) as `dev_explore/`. Each one
has a `scripts/start_<name>.sh` launcher.

```
dev_baselines/
├── tare_planner/   [submodule] TARE (C++)
├── gbplanner2/     GBPlanner2 + voxblox / minkindr / kdtree deps (C++)
├── nbvplanner/     receding-horizon NBV planner (C++)
├── greedy/         nearest-frontier greedy (C++)
├── ariadne/        ARiADNE (Python, torch; weights in model/, git-ignored)
├── cogniplan/      CogniPlan (Python, torch; weights in cogniplan/model/, git-ignored)
└── mapex/          MapEx ROS wrapper (Python, torch)
    └── MapEx/      [submodule] upstream MapEx (+ lama submodule); weights in
                    pretrained_models/weights/ (git-ignored)
```

The torch packages (`ariadne cogniplan mapex`) are built with the `adexpro` conda Python; see the
top-level README. MapEx weight paths in `mapex/config/` are relative to the repo root, which is
where the start scripts launch from.
