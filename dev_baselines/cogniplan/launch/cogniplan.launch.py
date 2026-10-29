from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_share_dir = get_package_share_directory('cogniplan')

    ld_preload = SetEnvironmentVariable(
        'LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
    )

    return LaunchDescription([
        ld_preload,

        DeclareLaunchArgument('use_gpu', default_value='true'),
        DeclareLaunchArgument('sensor_range', default_value='20.0'),
        DeclareLaunchArgument('node_resolution', default_value='2.0'),
        DeclareLaunchArgument('replanning_frequency', default_value='2.5'), # 2.5
        DeclareLaunchArgument('min_utility', default_value='1'), # 3
        DeclareLaunchArgument('num_gen_sample', default_value='4'),
        DeclareLaunchArgument('waypoint_threshold', default_value='1.5'),
        DeclareLaunchArgument('avoid_waypoint_oscillation', default_value='true'),
        DeclareLaunchArgument('enable_save_mode', default_value='false'),
        DeclareLaunchArgument('waypoint_timeout', default_value='7.5'), 

        Node(
            package='cogniplan',
            executable='cogniplan_planner_env',
            name='cogniplan_planner',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'publish_graph': True},
                {'use_gpu': LaunchConfiguration('use_gpu')},
                {'replanning_frequency': LaunchConfiguration('replanning_frequency')},
                {'map_free_value': 0},
                {'map_occupied_value': 100},
                {'map_unknown_value': -1},
                {'sensor_range': LaunchConfiguration('sensor_range')},
                {'node_resolution': LaunchConfiguration('node_resolution')},
                {'frontier_downsample_factor': 1},
                {'utility_range_factor': 0.8},
                {'min_utility': LaunchConfiguration('min_utility')},
                {'waypoint_threshold': LaunchConfiguration('waypoint_threshold')},
                {'next_waypoint_threshold': 5.0},
                {'num_gen_sample': LaunchConfiguration('num_gen_sample')},
                {'avoid_waypoint_oscillation': LaunchConfiguration('avoid_waypoint_oscillation')},
                {'enable_save_mode': LaunchConfiguration('enable_save_mode')},
                {'waypoint_timeout': LaunchConfiguration('waypoint_timeout')},
            ],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='cogniplan_rviz',
            output='log',
            arguments=[
                '-d',
                PathJoinSubstitution([package_share_dir, 'rviz', 'cogniplan.rviz']),
                '--ros-args', '--log-level', 'warn',
            ],
            respawn=True,
        ),
    ])
