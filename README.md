# Adaptive Exploration via Probabilistic Planning and Online Map Prediction

Two methods:
- **ExPro**: probabilistic planning over an FPUnet map prediction.
- **AdExPro**: ExPro plus online self-supervised adaptation of the predictor.

## Layout

```
.
├── scripts/           entry points: docker_run, start_<method>, run_experiments
├── docker/            Dockerfile + conda env (environment.yml); build context = repo root
├── dev_explore/       the exploration system: autonomy stack, planner, map prediction,
│                      self-adapt, experiment runner
├── dev_baselines/     baseline planners (TARE, GBPlanner2, NBVP, ARiADNE, CogniPlan, MapEx, greedy)
├── dev_visualization/ experiment data and visualization
│   └── experiments/   run output (git-ignored)
└── build/ install/ log/   the repo root is the colcon workspace (git-ignored)
```

The table maps the names used in code (baselines, scripts, tmux sessions) to the names used in the paper:

| Paper          | Code      |
|----------------|-----------|
| ExPro          | `expro`   |
| AdExPro        | `adexpro` |
| ExPro (oracle) | `oracle`  |

## Set Up

```sh
git clone <this repo>
git submodule update --init --recursive --remote
```

Model weights are git-ignored. Download the FPUnet weights (`best_fpunet_256_8.pth`,
`best_model.pth`) from
[Google Drive](https://drive.google.com/drive/folders/1jXtpTeI6MsttXhTSDTYItPNMMTF7igY6?usp=drive_link)
into `dev_explore/planner/map_prediction/models/`, see its
[readme](dev_explore/planner/map_prediction/models/readme.md).

## Environment (Docker)

The stack targets ROS 2 Humble (Ubuntu 22.04, Python 3.10), and the torch packages need the
`adexpro` conda env on the same Python, so everything runs in one Docker image that bundles both.
Requires Docker and the NVIDIA Container Toolkit on the host. Run all commands from the repo root:

```bash
# build the image once
docker build -f docker/Dockerfile -t adexpro:humble .

# open a shell in the long-lived `adexpro` container. GPU and X11 are passed through, the repo
# is mounted at the same path, and the shell starts at the repo root. The prompt starts with [docker].
./scripts/docker_run.sh
```

After changing `docker/`, rebuild the image and recreate the container
with `docker rm -f adexpro` before the next `./scripts/docker_run.sh`.

## Build

Inside the container:

```bash
# from the repo root (the colcon workspace)
# 1. everything except the torch packages, without conda
conda deactivate
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-ignore map_prediction ariadne cogniplan mapex
# 2. the torch packages, with the adexpro Python
conda activate adexpro
python -m colcon build --symlink-install --packages-select map_prediction ariadne cogniplan mapex
```

## Run

From the repo root, inside the container:

```bash
# one method: ./scripts/start_<method>.sh [world] [conda_env]
#   world defaults to garage (also: indoor | forest | tunnel | matterport)
./scripts/start_adexpro.sh indoor   # AdExPro
./scripts/start_expro.sh   indoor   # ExPro
./scripts/start_oracle.sh  indoor   # ExPro with an oracle predictor (GT-filled frontier windows)
#   baselines: start_{tare,gbp2,nbvplanner,ariadne,cogniplan,mapex,greedy}.sh

# kill it (the tmux session is named after the method)
tmux kill-session -t adexpro

# batch experiments (config: a path, or a name under dev_explore/experiment_runner/config/)
./scripts/run_experiments.sh indoor_self_adapt
```

All output lands in `dev_visualization/experiments/`, as described in [dev_visualization/README.md](dev_visualization/README.md).
