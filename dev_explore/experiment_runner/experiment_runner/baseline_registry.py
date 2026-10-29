"""Baseline launch specifications.

Each BaselineSpec encodes what the corresponding start_<baseline>.sh script
does, parameterized by `world`. The experiment_runner's tmux_launcher
materializes these into actual tmux send-keys commands.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class LaunchPane:
    label: str                # short id used in log filename and pane title
    cmd_template: str         # uses {world}
    conda_env: Optional[str] = None
    delay_s: float = 0.0      # sleep before launching (waits for upstream)


@dataclass(frozen=True)
class BaselineSpec:
    name: str                                 # tmux session base name
    planner: LaunchPane                       # required, the main planner pane
    mapping: Optional[LaunchPane] = None      # None when baseline brings its own mapping (tare)
    extras: Tuple[LaunchPane, ...] = ()       # e.g. map_prediction for adexpro

    def format(self, template: str, world: str) -> str:
        return template.format(world=world)


def _mapping(world_override: Optional[str] = None) -> LaunchPane:
    world_token = world_override if world_override else "{world}"
    return LaunchPane(
        label="mapping",
        cmd_template=f"ros2 launch simulator mapping_{world_token}.launch",
        delay_s=8.0,
    )


BASELINES = {
    # expro (ExPro): learned predictor, no online update.
    "expro": BaselineSpec(
        name="expro",
        mapping=_mapping(),
        extras=(
            LaunchPane(
                label="map_prediction",
                cmd_template="ros2 launch simulator map_prediction.launch",
                conda_env="adexpro",
                delay_s=10.0,
            ),
        ),
        planner=LaunchPane(
            label="planner",
            cmd_template="ros2 launch simulator planner_{world}.launch",
            delay_s=10.0,
        ),
    ),
    # adexpro (AdExPro): expro plus online self-supervised fine-tuning of the predictor
    # (forces eager inference, TRT off). Use expro vs adexpro for ON-vs-OFF A/B runs.
    # Switches to map_prediction_online_update.launch, which loads
    # online_update.yaml. The map_prediction.launch does not honor :=overrides,
    # so a separate launch file is required to flip the config.
    "adexpro": BaselineSpec(
        name="adexpro",
        mapping=_mapping(),
        extras=(
            LaunchPane(
                label="map_prediction",
                cmd_template="ros2 launch simulator map_prediction_online_update.launch",
                conda_env="adexpro",
                delay_s=10.0,
            ),
        ),
        planner=LaunchPane(
            label="planner",
            cmd_template="ros2 launch simulator planner_{world}.launch",
            delay_s=10.0,
        ),
    ),
    # Like expro, but swaps the learned predictor for the oracle node that
    # fills frontier-local windows from a GT octomap. Mirrors
    # start_oracle.sh: adds an extra pane running gt_mapping_<world>
    # to publish /gt_map for the oracle node to read.
    "oracle": BaselineSpec(
        name="oracle",
        mapping=_mapping(),
        extras=(
            LaunchPane(
                label="gt_mapping",
                cmd_template="ros2 launch simulator gt_mapping_{world}.launch",
                delay_s=8.0,
            ),
            LaunchPane(
                label="oracle_map_prediction",
                cmd_template="ros2 launch simulator oracle_map_prediction.launch",
                conda_env="adexpro",
                delay_s=12.0,
            ),
        ),
        planner=LaunchPane(
            label="planner",
            cmd_template="ros2 launch simulator planner_{world}.launch",
            delay_s=10.0,
        ),
    ),
    "greedy": BaselineSpec(
        name="greedy",
        mapping=_mapping(),
        planner=LaunchPane(
            label="planner",
            cmd_template="ros2 launch greedy greedy_{world}.launch",
            delay_s=10.0,
        ),
    ),
    "nbvplanner": BaselineSpec(
        name="nbvplanner",
        mapping=_mapping(),
        planner=LaunchPane(
            label="planner",
            cmd_template="ros2 launch nbvplanner nbvplanner.launch",
            delay_s=10.0,
        ),
    ),
    "ariadne": BaselineSpec(
        name="ariadne",
        mapping=_mapping(),
        planner=LaunchPane(
            label="planner",
            cmd_template="ros2 launch ariadne ariadne.launch.py",
            conda_env="adexpro",
            delay_s=10.0,
        ),
    ),
    "cogniplan": BaselineSpec(
        name="cogniplan",
        mapping=_mapping(),
        planner=LaunchPane(
            label="planner",
            cmd_template="ros2 launch cogniplan cogniplan.launch.py",
            conda_env="adexpro",
            delay_s=10.0,
        ),
    ),
    "mapex": BaselineSpec(
        name="mapex",
        mapping=_mapping(),
        planner=LaunchPane(
            label="planner",
            cmd_template="ros2 launch mapex mapex.launch.py",
            delay_s=10.0,
        ),
    ),
    "tare": BaselineSpec(
        name="tare",
        mapping=None,
        planner=LaunchPane(
            label="planner",
            cmd_template="ros2 launch tare_planner explore_{world}.launch",
            delay_s=8.0,
        ),
    ),
    "gbp2": BaselineSpec(
        name="gbp2",
        mapping=_mapping(),
        planner=LaunchPane(
            label="planner",
            cmd_template="ros2 launch gbplanner gbplanner.launch.py",
            delay_s=10.0,
        ),
    ),
}
