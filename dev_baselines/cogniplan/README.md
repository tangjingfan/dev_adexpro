# CogniPlan ROS2 baseline

This package ports the CogniPlan planner into the workspace's ROS2 baseline layout.

The online node subscribes to `/projected_map` and `/state_estimation`, then publishes `/way_point`.
Download the upstream CogniPlan checkpoints and either place them under `checkpoints/` before building, or pass the paths at launch time:

```bash
ros2 launch cogniplan cogniplan.launch.py \
  policy_checkpoint:=/path/to/cogniplan_exp_pred7/checkpoint.pth \
  generator_dir:=/path/to/wgan_inpainting
```
