"""Launch the gbplanner2 node + waypoint_bridge alongside the autonomy_stack.

Pipeline:
    gbplanner  ──/solution_path──>  waypoint_bridge  ──/way_point──>  local_planner
                                    ↑
                          /padded_visible_map ← (remap) ← /projected_map

Replanning runs at ~2 s (gbplanner). waypoint_bridge re-samples /solution_path
on /padded_visible_map at ~5 Hz so the published waypoint advances smoothly
between plan ticks.

Usage (from a built dev_adexpro workspace):
    ros2 launch gbplanner gbplanner.launch.py
    ros2 launch gbplanner gbplanner.launch.py use_rviz:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('gbplanner')
    default_config = os.path.join(pkg_share, 'config', 'garage',
                                  'gbplanner_config.yaml')
    default_rviz = os.path.join(pkg_share, 'rviz', 'gbplanner.rviz')
    waypoint_bridge_params = os.path.join(
        get_package_share_directory('waypoint_bridge'),
        'config', 'waypoint_bridge.yaml')

    config_file = LaunchConfiguration('config_file')
    rviz_config = LaunchConfiguration('rviz_config')
    use_rviz = LaunchConfiguration('use_rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Path to the gbplanner YAML config'),
        DeclareLaunchArgument(
            'rviz_config', default_value=default_rviz,
            description='Path to the RViz config'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Launch RViz alongside the planner'),

        Node(
            package='gbplanner',
            executable='gbplanner_node',
            name='gbplanner',
            output='screen',
            emulate_tty=True,
            parameters=[config_file],
            remappings=[
                # voxblox map_manager subscribes to "pointcloud". Feed the
                # sensor-frame scan (/sensor_scan, frame sensor_at_scan), NOT
                # /registered_scan: the latter is already in the map frame, so
                # voxblox would resolve map->map = identity and take the ray-cast
                # origin as (0,0,0) instead of the robot. That makes every TSDF
                # carving ray fan out from the world origin and erase wall
                # surfaces between it and the points (missing voxels, robot
                # planning out through walls). /sensor_scan carries the
                # map->sensor_at_scan TF so voxblox gets the true sensor origin.
                ('pointcloud', '/sensor_scan'),
            ],
        ),

        Node(
            package='waypoint_bridge',
            executable='waypoint_bridge',
            name='waypoint_bridge',
            output='screen',
            parameters=[waypoint_bridge_params],
            remappings=[
                ('/padded_visible_map', '/projected_map'),
            ],
            arguments=['--ros-args', '--log-level', 'warn'],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='gbplanner_rviz',
            output='log',
            condition=IfCondition(use_rviz),
            arguments=[
                '-d', rviz_config,
                '--ros-args', '--log-level', 'warn',
            ],
            respawn=True,
        ),
    ])
