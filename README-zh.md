# 基于概率规划与在线地图预测的自适应探索

[English](README.md) | [中文](README-zh.md)

两个方法：
- **ExPro**：在 FPUnet 地图预测之上做概率规划。
- **AdExPro**：在 ExPro 的基础上，对预测器做在线自监督自适应。

## 目录结构

```
.
├── scripts/           入口脚本：docker_run、start_<method>、run_experiments
├── docker/            Dockerfile 和 conda 环境（environment.yml）；构建上下文是 repo 根目录
├── dev_explore/       探索系统：autonomy stack、规划器、地图预测、
│                      self-adapt、实验 runner
├── dev_baselines/     baseline 规划器（TARE、GBPlanner2、NBVP、ARiADNE、CogniPlan、MapEx、greedy）
├── dev_visualization/ 实验数据与可视化
│   └── experiments/   运行输出（git 忽略）
└── build/ install/ log/   repo 根目录就是 colcon workspace（git 忽略）
```

代码里用的名字（baseline、脚本、tmux session）和论文里的名字对应关系如下：

| 论文           | 代码      |
|----------------|-----------|
| ExPro          | `expro`   |
| AdExPro        | `adexpro` |
| ExPro (oracle) | `oracle`  |

## 准备

```sh
git clone <this repo>
git submodule update --init --recursive --remote
```

模型权重不在 git 里。请从
[Google Drive](https://drive.google.com/drive/folders/1jXtpTeI6MsttXhTSDTYItPNMMTF7igY6?usp=drive_link)
下载 FPUnet 权重（`best_fpunet_256_8.pth`、`best_model.pth`），放到
`dev_explore/planner/map_prediction/models/`，详见该目录下的
[readme](dev_explore/planner/map_prediction/models/readme.md)。

## 环境（Docker）

整个系统基于 ROS 2 Humble（Ubuntu 22.04，Python 3.10），而 torch 相关的包需要同一个 Python
版本的 `adexpro` conda 环境，所以两者都打包在同一个 Docker 镜像里运行。主机需要安装 Docker 和
NVIDIA Container Toolkit。以下命令都在 repo 根目录下执行：

```bash
# 构建镜像（只需一次）
docker build -f docker/Dockerfile -t adexpro:humble .

# 进入长期运行的 `adexpro` 容器。GPU 和 X11 会传进容器，repo 挂载在和主机相同的路径，
# shell 从 repo 根目录开始，提示符以 [docker] 开头。
./scripts/docker_run.sh
```

修改 `docker/` 之后，需要重新构建镜像，并在下次运行 `./scripts/docker_run.sh` 之前用
`docker rm -f adexpro` 删除旧容器，让它用新镜像重新创建。

## 编译

在容器里执行：

```bash
# 在 repo 根目录（即 colcon workspace）下
# 1. 不用 conda，编译除 torch 包以外的所有包
conda deactivate
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-ignore map_prediction ariadne cogniplan mapex
# 2. 用 adexpro 环境的 Python 编译 torch 包
conda activate adexpro
python -m colcon build --symlink-install --packages-select map_prediction ariadne cogniplan mapex
```

## 运行

在容器里、repo 根目录下执行：

```bash
# 运行一个方法：./scripts/start_<method>.sh [world] [conda_env]
#   world 默认是 garage（可选：indoor | forest | tunnel | matterport）
./scripts/start_adexpro.sh indoor   # AdExPro
./scripts/start_expro.sh   indoor   # ExPro
./scripts/start_oracle.sh  indoor   # 使用 oracle 预测器的 ExPro（用真值填充 frontier 窗口）
#   baseline：start_{tare,gbp2,nbvplanner,ariadne,cogniplan,mapex,greedy}.sh

# 关闭（tmux session 以方法名命名）
tmux kill-session -t adexpro

# 批量实验（config 可以是文件路径，也可以是 dev_explore/experiment_runner/config/ 下的名字）
./scripts/run_experiments.sh indoor_self_adapt
```

所有输出都写到 `dev_visualization/experiments/`，说明见
[dev_visualization/README.md](dev_visualization/README.md)。
