"""Learned prediction backend (ExPro): an FPUnet fills the frontier windows.

self_adapt.SelfAdaptNode extends this node with online fine-tuning (AdExPro).
"""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from ament_index_python.packages import get_package_share_directory

from .grid_encoding import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE
from .map_prediction_model import MapPredictionModel
from .prediction_node_base import PredictionNodeBase


class MapPredictionNode(PredictionNodeBase):
    #: Build the predictor with an eager, differentiable net (SelfAdaptNode).
    TRAINABLE = False

    def __init__(self):
        super().__init__("map_prediction_node")
        self.predictor = self._create_predictor()

        from .map_prediction_model import TRTInferenceEngine
        if self.TRAINABLE:
            backend = "Eager (trainable)"
        elif isinstance(self.predictor.net, TRTInferenceEngine):
            backend = "TensorRT"
        else:
            backend = "TorchScript"
        self.get_logger().info(
            f"Prediction device: {self.predictor.device}  backend: {backend}"
        )

    def _declare_parameters(self):
        super()._declare_parameters()
        self.declare_parameter("model_path", "models/best_fpunet_256_8.pth")
        self.declare_parameter("device", "auto")
        self.declare_parameter("inference_batch_size", 16)
        self.declare_parameter("enable_uncertainty", False)
        self.declare_parameter("uncertainty_scale", 2.0)
        self.declare_parameter("correct_prediction", True)
        self.declare_parameter("wrong_prediction_confidence", 0.3)
        self.declare_parameter("wrong_prediction_free_flip_ratio", 1.0 / 6.0)
        self.declare_parameter("wrong_prediction_occupied_flip_ratio", 1.0 / 6.0)
        self.declare_parameter("use_trt", False)
        self.declare_parameter("trt_engine_path", "")

    def _load_parameters(self):
        super()._load_parameters()
        self.model_path                           = self.get_parameter("model_path").value
        self.device                               = self.get_parameter("device").value
        self.inference_batch_size                 = self.get_parameter("inference_batch_size").value
        self.enable_uncertainty                   = self.get_parameter("enable_uncertainty").value
        self.uncertainty_scale                    = self.get_parameter("uncertainty_scale").value
        self.correct_prediction                   = self.get_parameter("correct_prediction").value
        self.wrong_prediction_confidence          = self.get_parameter("wrong_prediction_confidence").value
        self.wrong_prediction_free_flip_ratio     = self.get_parameter("wrong_prediction_free_flip_ratio").value
        self.wrong_prediction_occupied_flip_ratio = self.get_parameter("wrong_prediction_occupied_flip_ratio").value
        self.use_trt                              = self.get_parameter("use_trt").value
        self.trt_engine_path                      = self.get_parameter("trt_engine_path").value or None

    def _create_predictor(self):
        package_share = get_package_share_directory("map_prediction")
        model_path = f"{package_share}/{self.model_path}"
        return MapPredictionModel(
            model_path=model_path,
            enable_uncertainty=self.enable_uncertainty,
            model_input_size=self.model_input_size,
            inference_batch_size=self.inference_batch_size,
            uncertainty_scale=self.uncertainty_scale,
            device=self.device,
            use_trt=self.use_trt,
            trt_engine_path=self.trt_engine_path,
            trainable=self.TRAINABLE,
        )

    def _build_prediction_outputs(self, visible_map, clustered_frontiers, capture):
        # capture (a list from a subclass's _begin_cycle, else None) collects the
        # input windows fed to the net, for SelfAdaptNode's training.
        if not self.enable_prediction:
            return self._build_passthrough_outputs(visible_map)

        visible_map_np = self.occupancy_grid_to_numpy(visible_map)
        exp = self.prediction_expansion_cells
        expanded_map_np, expanded_info = self._expand_map(visible_map_np, exp, visible_map)

        # Shift frontier grid coords into the expanded coordinate system
        raw_frontier_points = self.world_to_grid(clustered_frontiers, visible_map)
        expanded_frontier_points = [(x + exp, y + exp) for x, y in raw_frontier_points]

        # Snapshot the exact input window fed to the net, tagged with the frontier's
        # world coords, so the target can be re-cropped from a later (more complete)
        # map at training time. World coords are resolution/expansion-independent.
        if capture is not None:
            for (world_x, world_y, *_), point in zip(clustered_frontiers, expanded_frontier_points):
                window = MapPredictionModel.get_local_map(
                    expanded_map_np, point, self.model_input_size
                )
                capture.append((world_x, world_y, window))

        predicted_map_np, predicted_confidence_np = self.predictor.predict_map(
            expanded_map_np, expanded_frontier_points
        )

        if not self.correct_prediction:
            predicted_map_np, predicted_confidence_np = self.apply_wrong_prediction(
                predicted_map_np, predicted_confidence_np
            )

        # Overwrite the known region with ground-truth visible data
        h, w = visible_map_np.shape
        known_mask = visible_map_np != UNKNOWN_VALUE
        known_region = predicted_map_np[exp:exp + h, exp:exp + w]
        known_region[known_mask] = visible_map_np[known_mask]
        known_confidence = predicted_confidence_np[exp:exp + h, exp:exp + w]
        known_confidence[known_mask] = 1.0

        return (
            self.numpy_to_occupancy_grid(predicted_map_np, expanded_info),
            self.numpy_to_confidence_grid(predicted_confidence_np, expanded_info),
        )

    def apply_wrong_prediction(self, predicted_map, predicted_confidence):
        flipped_map = predicted_map.copy()
        self._flip_cells(flipped_map, FREE_VALUE, OCCUPIED_VALUE, self.wrong_prediction_free_flip_ratio)
        self._flip_cells(
            flipped_map,
            OCCUPIED_VALUE,
            FREE_VALUE,
            self.wrong_prediction_occupied_flip_ratio,
        )

        flipped_confidence = np.full_like(
            predicted_confidence, self.wrong_prediction_confidence
        )
        self.get_logger().info(
            "Using wrong-prediction mode with "
            f"free flip ratio={self.wrong_prediction_free_flip_ratio:.3f}, "
            f"occupied flip ratio={self.wrong_prediction_occupied_flip_ratio:.3f}, "
            f"confidence={self.wrong_prediction_confidence:.3f}"
        )
        return flipped_map, flipped_confidence

    def _flip_cells(self, grid, from_value, to_value, flip_ratio):
        if flip_ratio <= 0.0:
            return

        row_indices, col_indices = np.where(grid == from_value)
        cell_count = len(row_indices)
        if cell_count == 0:
            return

        flip_count = min(cell_count, int(round(cell_count * flip_ratio)))
        if flip_count == 0:
            return

        flip_indices = np.random.choice(cell_count, flip_count, replace=False)
        grid[row_indices[flip_indices], col_indices[flip_indices]] = to_value


def main(args=None):
    rclpy.init(args=args)
    node = MapPredictionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
