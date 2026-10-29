import os
import time
import traceback
import warnings

import numpy as np
import rclpy
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32, Header
from visualization_msgs.msg import Marker

from .agent import Agent
from .mapinpaint.evaluator import Evaluator
from .mapinpaint.networks import Generator
from .model import PolicyNet
from .node_manager import NodeManager
from .utils import MapInfo, check_collision, get_cell_position_from_coords, get_free_area_coords, is_free
from . import parameter
from . import agent as agent_module
from . import prediction_node_manager as prediction_node_manager_module
from . import node_manager as node_manager_module
from . import utils as utils_module

warnings.simplefilter("ignore", UserWarning)


class CogniPlanPlanner(Node):
    def __init__(self):
        super().__init__('cogniplan_planner')

        self.map_info = None
        self.robot_location = None
        self.start = None
        self.next_waypoint = None
        self.next_waypoint_list = []
        self.history_waypoint_list = []
        self.done = False
        self.step = 0
        self._last_map_stats_log_time = 0.0
        self.save_mode = False
        self._waypoint_publish_time = None  # when current waypoint was first published
        self._waypoint_stuck_dist = None    # distance to waypoint when it was published

        self.declare_parameter('publish_graph', True)
        self.declare_parameter('use_gpu', False)
        self.declare_parameter('replanning_frequency', 2.5)
        self.declare_parameter('map_resolution', float(parameter.CELL_SIZE))
        self.declare_parameter('map_free_value', 0)
        self.declare_parameter('map_occupied_value', 100)
        self.declare_parameter('map_unknown_value', -1)
        self.declare_parameter('sensor_range', float(parameter.SENSOR_RANGE))
        self.declare_parameter('node_resolution', float(parameter.NODE_RESOLUTION))
        self.declare_parameter('frontier_downsample_factor', 1)
        self.declare_parameter('utility_range_factor', 0.5)
        self.declare_parameter('min_utility', int(parameter.MIN_UTILITY))
        self.declare_parameter('waypoint_threshold', 1.0)
        self.declare_parameter('next_waypoint_threshold', 5.0)
        self.declare_parameter('num_gen_sample', int(parameter.N_GEN_SAMPLE))
        self.declare_parameter('avoid_waypoint_oscillation', True)
        self.declare_parameter('enable_save_mode', True)
        self.declare_parameter('waypoint_timeout', 15.0)

        package_share_dir = get_package_share_directory('cogniplan')
        default_model_dir = os.path.join(package_share_dir, 'model')
        default_policy = os.path.join(default_model_dir, 'checkpoint.pth')
        default_generator_config = os.path.join(default_model_dir, 'config.yaml')
        default_generator_checkpoint = os.path.join(default_model_dir, 'generator.pt')

        self.declare_parameter('policy_checkpoint', default_policy)
        self.declare_parameter('generator_config', default_generator_config)
        self.declare_parameter('generator_checkpoint', default_generator_checkpoint)

        self.publish_graph = self.get_parameter('publish_graph').value
        self.avoid_oscillation = self.get_parameter('avoid_waypoint_oscillation').value
        self.enable_save_mode = self.get_parameter('enable_save_mode').value
        self.waypoint_timeout = float(self.get_parameter('waypoint_timeout').value)
        self.frontier_downsample_factor = int(self.get_parameter('frontier_downsample_factor').value)

        parameter.CELL_SIZE = float(self.get_parameter('map_resolution').value)
        parameter.FREE = int(self.get_parameter('map_free_value').value)
        parameter.OCCUPIED = int(self.get_parameter('map_occupied_value').value)
        parameter.UNKNOWN = int(self.get_parameter('map_unknown_value').value)
        parameter.SENSOR_RANGE = float(self.get_parameter('sensor_range').value)
        parameter.NODE_RESOLUTION = float(self.get_parameter('node_resolution').value)
        parameter.FRONTIER_CELL_SIZE = self.frontier_downsample_factor * parameter.CELL_SIZE
        parameter.UTILITY_RANGE = float(self.get_parameter('utility_range_factor').value) * parameter.SENSOR_RANGE
        parameter.MIN_UTILITY = int(self.get_parameter('min_utility').value)
        parameter.N_GEN_SAMPLE = int(self.get_parameter('num_gen_sample').value)
        parameter.UPDATING_MAP_SIZE = 4 * parameter.SENSOR_RANGE + 4 * parameter.NODE_RESOLUTION
        parameter.THR_NEXT_WAYPOINT = float(self.get_parameter('next_waypoint_threshold').value)
        self.sync_parameter_modules()

        self.waypoint_threshold = float(self.get_parameter('waypoint_threshold').value)
        self.device = torch.device('cuda' if self.get_parameter('use_gpu').value and torch.cuda.is_available() else 'cpu')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/projected_map', self.get_map_callback, qos_profile=qos)
        self.loc_sub = self.create_subscription(
            Odometry, '/state_estimation', self.get_loc_callback, qos_profile=qos)

        self.waypoint_pub = self.create_publisher(PointStamped, '/way_point', 10)
        self.run_time_pub = self.create_publisher(Float32, '/runtime', 10)
        self.edge_pub = self.create_publisher(Marker, '/edge', 10)
        self.node_pub = self.create_publisher(PointCloud2, '/node', 10)
        self.frontier_pub = self.create_publisher(PointCloud2, '/frontier', 10)
        self.updatebox_pub = self.create_publisher(Marker, '/update_box', 10)
        self.pred_map_pub = self.create_publisher(OccupancyGrid, '/predicted_map', 10)

        self.robot = self.init_agent()
        frequency = float(self.get_parameter('replanning_frequency').value)
        self.timer = self.create_timer(1.0 / frequency, self.run)
        self.get_logger().info(f"CogniPlan planner ready on {self.device}; waiting for map and odometry")

    def sync_parameter_modules(self):
        names = [
            'CELL_SIZE', 'NODE_RESOLUTION', 'FRONTIER_CELL_SIZE', 'FREE', 'OCCUPIED', 'UNKNOWN',
            'SENSOR_RANGE', 'UTILITY_RANGE', 'MIN_UTILITY', 'UPDATING_MAP_SIZE',
            'K_SIZE', 'NODE_PADDING_SIZE', 'N_GEN_SAMPLE',
            'THR_GRAPH_HARD_UPDATE', 'CLUSTER_RANGE', 'THR_NEXT_WAYPOINT',
        ]
        for module in (utils_module, node_manager_module, agent_module, prediction_node_manager_module):
            for name in names:
                if hasattr(parameter, name):
                    setattr(module, name, getattr(parameter, name))

    def run(self):
        t1 = time.time()
        if self.done:
            return
        if self.map_info is None or self.robot_location is None:
            return

        # save mode: follow path to nearest frontier
        if self.save_mode:
            if self.next_waypoint is not None and \
                    np.linalg.norm(self.next_waypoint - self.robot_location) > self.waypoint_threshold:
                return
            else:
                if len(self.next_waypoint_list) > 0:
                    next_waypoint = np.array(self.next_waypoint_list.pop(0))
                    while (len(self.next_waypoint_list) > 0 and
                           not check_collision(self.robot_location, next_waypoint, self.map_info) and
                           np.linalg.norm(self.robot_location - next_waypoint) <
                           (parameter.THR_NEXT_WAYPOINT + parameter.NODE_RESOLUTION)):
                        next_waypoint = np.array(self.next_waypoint_list.pop(0))
                    self.next_waypoint = next_waypoint
                    self.history_waypoint_list.append((self.next_waypoint[0], self.next_waypoint[1]))
                    self.publish_waypoint(self.next_waypoint, time.time() - t1)
                    return
                else:
                    self.save_mode = False
                    self.get_logger().warning("Switch back to RL")

        # waypoint timeout: if stuck on same unreachable waypoint, fall back to nearest frontier
        if (self.enable_save_mode and self.next_waypoint is not None and
                self._waypoint_publish_time is not None and not self.save_mode):
            dist = np.linalg.norm(self.next_waypoint - self.robot_location)
            elapsed = time.time() - self._waypoint_publish_time
            if elapsed > self.waypoint_timeout and dist > self.waypoint_threshold:
                self._waypoint_publish_time = None
                self.get_logger().warning(
                    f"Waypoint unreachable for {elapsed:.1f}s (dist={dist:.2f}m): switching to save mode")
                if self._activate_save_mode(t1):
                    return

        # check and solve oscillation between two waypoints
        if self.avoid_oscillation and len(self.history_waypoint_list) > 4:
            if (self.history_waypoint_list[-1] == self.history_waypoint_list[-3] and
                    self.history_waypoint_list[-2] == self.history_waypoint_list[-4]):
                self.next_waypoint_list = []
                if self.next_waypoint is not None and \
                        np.linalg.norm(self.next_waypoint - self.robot_location) > self.waypoint_threshold:
                    return

        # if planned one more step ahead, use it
        if len(self.next_waypoint_list) > 0:
            if self.next_waypoint is not None and \
                    np.linalg.norm(self.next_waypoint - self.robot_location) > self.waypoint_threshold:
                pass
            else:
                if self.next_waypoint is not None:
                    self.robot_location = self.next_waypoint
                self.next_waypoint = np.array(self.next_waypoint_list.pop(0))
                self.publish_waypoint(self.next_waypoint, time.time() - t1)
        self.next_waypoint_list = []

        # remove nodes on obstacles if any (must run before get_planning_location)
        self.robot.node_manager.check_valid_node(self.robot_location, self.map_info)

        # get planning location (nearest graph node to robot)
        planning_location = self.get_planning_location()
        if planning_location is None:
            return

        try:
            # update planning graph
            self.robot.update_planning_state(self.map_info, planning_location)

            # get RL observation
            observation = self.robot.get_observation()

            # check termination
            if self.robot.utility is None or not (self.robot.utility > 0).any():
                self.get_logger().info("\033[92mExploration Completed\033[0m")
                self.done = True
                self.publish_runtime(0.0)
                return

            self.get_logger().info(
                f"Planning: frontiers={len(self.robot.frontier)}, "
                f"max_utility={int(np.max(self.robot.utility))}, "
                f"nodes={len(self.robot.node_coords)}")

            # network inference to get next waypoint
            next_location, next_node_index = self.robot.select_next_waypoint(observation)

        except Exception as exc:
            self.get_logger().error(f"CogniPlan planning step failed: {exc}")
            self.get_logger().error(traceback.format_exc())
            return

        self.next_waypoint_list.append(next_location)
        waypoint_key = (float(next_location[0]), float(next_location[1]))
        if not self.history_waypoint_list or waypoint_key != self.history_waypoint_list[-1]:
            self.history_waypoint_list.append(waypoint_key)

        # planning one more step if next node's utility is zero
        next_entry = self.robot.node_manager.nodes_dict.find(next_location.tolist())
        if next_entry is not None:
            if next_entry.data.utility == 0:
                try:
                    next_observation = self.robot.get_next_observation(next_node_index, observation)
                    next_next_location, _ = self.robot.select_next_waypoint(next_observation)
                    if np.linalg.norm(next_location - self.robot_location) < parameter.NODE_RESOLUTION:
                        self.next_waypoint_list = []
                    self.next_waypoint_list.append(next_next_location)
                except Exception:
                    pass
        else:
            self.get_logger().warning("next location not in node dict")

        t4 = time.time()

        # if RL gets stuck, go to nearest frontier
        if self.enable_save_mode and self.detect_waypoint_loop():
            self.get_logger().warning("Switch to save mode")
            if self._activate_save_mode(t1):
                return

        # get and publish waypoint; reset timeout only when waypoint changes
        new_waypoint = np.array(self.next_waypoint_list.pop(0))
        if (self.next_waypoint is None or
                not np.allclose(new_waypoint, self.next_waypoint, atol=1e-3)):
            self._waypoint_publish_time = time.time()
        self.next_waypoint = new_waypoint
        self.publish_waypoint(self.next_waypoint, t4 - t1)

        # publish predicted map
        if self.robot.pred_mean_map_info is not None:
            self.pred_map_pub.publish(self.pub_predicted_map())

        self.step += 1
        if self.publish_graph:
            self.visualize_graph()

    def _activate_save_mode(self, t1):
        """Switch to save mode using belief-graph path. Returns True if activated and first waypoint published."""
        path = self.robot.node_manager.path_to_nearest_frontier
        if not path:
            return False
        self.next_waypoint_list = [np.array(p) for p in path]
        self.save_mode = True
        # skip waypoints already close and collision-free (same logic as save_mode block)
        next_wp = np.array(self.next_waypoint_list.pop(0))
        while (len(self.next_waypoint_list) > 0 and
               not check_collision(self.robot_location, next_wp, self.map_info) and
               np.linalg.norm(self.robot_location - next_wp) <
               (parameter.THR_NEXT_WAYPOINT + parameter.NODE_RESOLUTION)):
            next_wp = np.array(self.next_waypoint_list.pop(0))
        self.next_waypoint = next_wp
        self.history_waypoint_list.append((float(next_wp[0]), float(next_wp[1])))
        self._waypoint_publish_time = time.time()
        self.publish_waypoint(next_wp, time.time() - t1)
        return True

    def detect_waypoint_loop(self, max_length=6):
        if len(self.history_waypoint_list) < max_length:
            return False

        waypoint_list_to_check = self.history_waypoint_list[-max_length:]
        loop = []
        for i, waypoint in enumerate(waypoint_list_to_check[:-1]):
            if waypoint == waypoint_list_to_check[-1]:
                loop = waypoint_list_to_check[i:]

        if not loop:
            return False

        loop_length = len(loop)
        if len(self.history_waypoint_list) < 2 * loop_length + 1:
            return False
        waypoint_list_to_check2 = self.history_waypoint_list[-max_length - loop_length + 1:-loop_length + 1]
        for i, waypoint in enumerate(waypoint_list_to_check2[:-1]):
            if waypoint == waypoint_list_to_check2[-1]:
                return True
        return False

    def pub_predicted_map(self):
        pred = self.robot.pred_mean_map_info.map
        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = 'map'
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.info.width = pred.shape[1]
        grid_msg.info.height = pred.shape[0]
        grid_msg.info.resolution = self.robot.pred_mean_map_info.cell_size
        grid_msg.info.origin.position.x = self.robot.pred_mean_map_info.map_origin_x
        grid_msg.info.origin.position.y = self.robot.pred_mean_map_info.map_origin_y
        grid_msg.info.origin.orientation.w = 1.0
        pred_occ = np.clip(pred, 0, 100).astype(np.int8)
        grid_msg.data = pred_occ.flatten().tolist()
        return grid_msg

    def get_map_callback(self, msg):
        delta = float(msg.info.resolution)
        parameter.CELL_SIZE = delta
        parameter.FRONTIER_CELL_SIZE = self.frontier_downsample_factor * parameter.CELL_SIZE
        self.sync_parameter_modules()

        ros_map = np.array(msg.data).reshape(msg.info.height, msg.info.width).astype(np.int8)

        pad_size = int(parameter.NODE_RESOLUTION // parameter.CELL_SIZE + 1)
        processed_map = np.pad(
            ros_map,
            ((pad_size, pad_size), (pad_size, pad_size)),
            'constant',
            constant_values=parameter.UNKNOWN,
        )
        origin_x = msg.info.origin.position.x - delta * pad_size
        origin_y = msg.info.origin.position.y - delta * pad_size
        self.map_info = MapInfo(processed_map, origin_x, origin_y, delta)

        now = time.time()
        if now - self._last_map_stats_log_time > 2.0:
            values, counts = np.unique(ros_map, return_counts=True)
            self.get_logger().info(
                f"Map stats: size={msg.info.width}x{msg.info.height}, resolution={delta:.3f}, "
                f"values={ {int(v): int(c) for v, c in zip(values, counts)} }")
            self._last_map_stats_log_time = now

    def get_loc_callback(self, msg):
        self.robot_location = np.around(
            np.array([msg.pose.pose.position.x, msg.pose.pose.position.y]), 1)

    def get_planning_location(self):
        if self.map_info is None or self.robot_location is None:
            return None

        if self.start is None:
            x = np.array([
                (self.robot_location[0] // parameter.NODE_RESOLUTION) * parameter.NODE_RESOLUTION,
                (self.robot_location[0] // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION,
            ])
            y = np.array([
                (self.robot_location[1] // parameter.NODE_RESOLUTION) * parameter.NODE_RESOLUTION,
                (self.robot_location[1] // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION,
            ])
            mx, my = np.meshgrid(x, y)
            candidates = np.vstack([mx.T.ravel(), my.T.ravel()]).T
            candidates = candidates[np.argsort(np.linalg.norm(candidates - self.robot_location, axis=1))]
            for candidate in candidates:
                if is_free(candidate, self.map_info):
                    self.start = np.around(candidate, 1)
                    self.robot.node_manager = NodeManager(start=self.start)
                    self.get_logger().info(f"Initialized CogniPlan graph at {self.start.tolist()}")
                    return self.start
            free_coords = get_free_area_coords(self.map_info)
            if free_coords.size > 0:
                free_coords = np.array(free_coords).reshape(-1, 2)
                nearest_free = free_coords[int(np.argmin(np.linalg.norm(free_coords - self.robot_location, axis=1)))]
                self.start = np.around(nearest_free, 1)
                self.robot.node_manager = NodeManager(start=self.start)
                self.get_logger().warning(
                    f"No free graph grid start near robot; using nearest free cell {self.start.tolist()}")
                return self.start
            self.get_logger().warning("No free graph start found near robot yet")
            return None

        # find nearest node to the robot
        robot_node_location = self.robot_location
        if not (self.robot_location[0] == self.start[0] and self.robot_location[1] == self.start[1]):
            if len(self.robot.node_manager.nodes_dict) == 0:
                robot_node_location = self.start
            else:
                nearest = self.robot.node_manager.nodes_dict.nearest_neighbors(self.robot_location.tolist(), 1)
                if nearest:
                    robot_node_location = nearest[0].data.coords
        return np.around(robot_node_location, 1)

    def init_agent(self):
        policy_net = PolicyNet(parameter.NODE_INPUT_DIM, parameter.EMBEDDING_DIM).to(self.device)
        policy_checkpoint = self.get_parameter('policy_checkpoint').value
        if not os.path.exists(policy_checkpoint):
            raise FileNotFoundError(
                f"CogniPlan policy checkpoint not found: {policy_checkpoint}. "
                "Set the policy_checkpoint launch parameter after downloading CogniPlan checkpoints.")
        checkpoint = torch.load(policy_checkpoint, map_location=self.device)
        policy_net.load_state_dict(checkpoint['policy_model'])
        policy_net.eval()

        generator_config = self.get_parameter('generator_config').value
        generator_checkpoint = self.get_parameter('generator_checkpoint').value
        if not os.path.exists(generator_config):
            raise FileNotFoundError(f"CogniPlan generator config not found: {generator_config}")
        if not os.path.exists(generator_checkpoint):
            raise FileNotFoundError(
                f"CogniPlan generator checkpoint not found: {generator_checkpoint}")

        with open(generator_config, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.SafeLoader)
        use_cuda = self.device.type == 'cuda'
        generator = Generator(config['netG'], use_cuda)
        generator.load_state_dict(torch.load(generator_checkpoint, map_location=self.device))
        predictor = Evaluator(config, generator, use_cuda, parameter.N_GEN_SAMPLE)
        self.get_logger().info(f"Loaded policy from {policy_checkpoint}")
        self.get_logger().info(f"Loaded map predictor from {generator_checkpoint}")
        return Agent(policy_net, predictor, self.device, self.publish_graph)

    def publish_waypoint(self, loc, runtime):
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = float(loc[0])
        msg.point.y = float(loc[1])
        self.waypoint_pub.publish(msg)
        self.publish_runtime(runtime)

    def publish_runtime(self, runtime):
        msg = Float32()
        msg.data = float(runtime)
        self.run_time_pub.publish(msg)

    def visualize_graph(self):
        if self.robot.pred_node_manager is None:
            return

        edges = Marker()
        edges.header.frame_id = 'map'
        edges.header.stamp = self.get_clock().now().to_msg()
        edges.type = Marker.LINE_LIST
        edges.scale.x = 0.1
        edges.color.r = 0.0
        edges.color.g = 0.6
        edges.color.b = 0.0
        edges.color.a = 1.0
        edges.pose.orientation.w = 1.0

        for coords in self.robot.node_coords:
            node = self.robot.pred_node_manager.nodes_dict.find(coords.tolist())
            if node is None:
                continue
            for neighbor_coords in node.data.neighbor_set:
                start = Point()
                start.x = float(coords[0])
                start.y = float(coords[1])
                end_coords = (np.array(neighbor_coords) - coords) / 2 + coords
                end = Point()
                end.x = float(end_coords[0])
                end.y = float(end_coords[1])
                edges.points.append(start)
                edges.points.append(end)
        self.edge_pub.publish(edges)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        nodes = [
            (float(coords[0]), float(coords[1]), 0.0, float(max(0.0, utility)))
            for coords, utility in zip(self.robot.node_coords, self.robot.utility)
        ]
        self.node_pub.publish(point_cloud2.create_cloud(header, fields, nodes))

        frontier_fields = fields[:3]
        frontiers = [(float(p[0]), float(p[1]), 0.0) for p in self.robot.frontier]
        self.frontier_pub.publish(point_cloud2.create_cloud(header, frontier_fields, frontiers))

        updatebox = Marker()
        updatebox.header.frame_id = 'map'
        updatebox.header.stamp = self.get_clock().now().to_msg()
        updatebox.type = Marker.LINE_STRIP
        updatebox.scale.x = 0.15
        updatebox.color.r = 0.0
        updatebox.color.g = 0.0
        updatebox.color.b = 1.0
        updatebox.color.a = 0.8
        updatebox.pose.orientation.w = 1.0
        half_size = parameter.UPDATING_MAP_SIZE / 2.0
        rx, ry = self.robot_location[0], self.robot_location[1]
        corners = [
            Point(x=rx - half_size, y=ry - half_size, z=0.0),
            Point(x=rx + half_size, y=ry - half_size, z=0.0),
            Point(x=rx + half_size, y=ry + half_size, z=0.0),
            Point(x=rx - half_size, y=ry + half_size, z=0.0),
            Point(x=rx - half_size, y=ry - half_size, z=0.0),
        ]
        updatebox.points = corners
        self.updatebox_pub.publish(updatebox)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CogniPlanPlanner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Exception in CogniPlan planner: {exc}")
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
