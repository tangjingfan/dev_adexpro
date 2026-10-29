#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable


def generate_launch_description():
    mapex_params_file = os.path.join(get_package_share_directory("mapex"), "config", "mapex.matterport.yaml")
    rviz_config_file = os.path.join(get_package_share_directory("mapex"), "rviz", "mapex.rviz")
    waypoint_bridge_params_file = os.path.join(
        get_package_share_directory("waypoint_bridge"), "config", "waypoint_bridge.yaml"
    )

    # ── Launch arguments (override YAML values at runtime) ────────────────────
    big_lama_model_path_arg = DeclareLaunchArgument(
        "big_lama_model_path",
        default_value="dev_baselines/mapex/MapEx/pretrained_models/weights/big_lama",
        description="Path to the big LaMa model directory (relative to launch CWD / repo root)",
    )

    ensemble_model_dir_arg = DeclareLaunchArgument(
        "ensemble_model_dir",
        default_value="dev_baselines/mapex/MapEx/pretrained_models/weights/lama_ensemble",
        description="Path to the ensemble models directory (relative to launch CWD / repo root)",
    )

    lama_device_arg = DeclareLaunchArgument(
        "lama_device",
        default_value="cuda:0",
        description="Torch device for LaMa inference",
    )

    planning_cycle_arg = DeclareLaunchArgument(
        "planning_cycle",
        default_value="2.0",
        description="Planning timer period in seconds",
    )

    pixel_per_meter_arg = DeclareLaunchArgument(
        "pixel_per_meter",
        default_value="2.0",
        description="Map resolution in pixels per metre (must match project_map)",
    )

    unknown_as_occ_arg = DeclareLaunchArgument(
        "unknown_as_occ",
        default_value="true",
        description="Treat unknown cells as occupied for A* planning",
    )

    # ── Node ──────────────────────────────────────────────────────────────────
    mapex_node = Node(
        package="mapex",
        executable="mapex_node",
        name="mapex_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            mapex_params_file,
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="mapex_rviz",
        output="screen",
        arguments=["-d", rviz_config_file, "--ros-args", "--log-level", "warn"],
    )

    waypoint_bridge_node = Node(
        package="waypoint_bridge",
        executable="waypoint_bridge",
        name="waypoint_bridge_node",
        parameters=[
            waypoint_bridge_params_file,
        ],
        remappings=[
            ("/padded_visible_map", "/projected_map"),
        ],
        arguments=["--ros-args", "--log-level", "warn"],
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("PYTHONUNBUFFERED", "1"),
            big_lama_model_path_arg,
            ensemble_model_dir_arg,
            lama_device_arg,
            planning_cycle_arg,
            pixel_per_meter_arg,
            unknown_as_occ_arg,
            mapex_node,
            waypoint_bridge_node,
            rviz_node,
        ]
    )
