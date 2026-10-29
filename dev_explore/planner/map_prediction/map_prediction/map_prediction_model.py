import io
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F

from .fpunet import FPUNet
# Re-exported: the grid encoding is shared with the (torch-free) node base.
from .grid_encoding import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE


class TRTInferenceEngine:
    """Thin wrapper around a TensorRT ICudaEngine that matches the torch.nn.Module
    call signature used by the rest of the pipeline.

    Compatible with TRT 8.5+ (named-tensor I/O API) and TRT 8.0-8.4 (binding-index
    API). The appropriate code path is selected at construction time.
    """

    def __init__(self, engine, input_name: str = "input", output_name: str = "output"):
        self._engine = engine
        self._context = engine.create_execution_context()
        self._input_name = input_name
        self._output_name = output_name
        self.max_batch_size = self._get_max_batch_size()
        # Detect which execution API is available.
        self._use_named_api = hasattr(self._context, "execute_async_v3")
        self._stream = torch.cuda.Stream()

    def _get_max_batch_size(self):
        try:
            if hasattr(self._engine, "get_tensor_profile_shape"):
                return int(self._engine.get_tensor_profile_shape(self._input_name, 0)[2][0])
            if hasattr(self._engine, "get_binding_index"):
                binding_index = self._engine.get_binding_index(self._input_name)
                return int(self._engine.get_profile_shape(0, binding_index)[2][0])
        except Exception:
            return None
        return None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run TRT inference. x must be a contiguous CUDA float32 tensor."""
        caller_stream = torch.cuda.current_stream()
        self._stream.wait_stream(caller_stream)

        with torch.cuda.stream(self._stream):
            x = x.contiguous()
            stream = self._stream.cuda_stream

            if self._use_named_api:
                # TRT 8.5+ / TRT 10.x: named-tensor API.
                if not self._context.set_input_shape(self._input_name, tuple(x.shape)):
                    raise RuntimeError(f"TensorRT rejected input shape {tuple(x.shape)}")
                # After set_input_shape the output shape is fully determined.
                raw_out_shape = tuple(self._context.get_tensor_shape(self._output_name))
                # Replace the (possibly still -1) batch dim with the actual batch.
                out_shape = (x.shape[0],) + raw_out_shape[1:]
                output = torch.empty(out_shape, dtype=torch.float32, device=x.device)
                self._context.set_tensor_address(self._input_name, x.data_ptr())
                self._context.set_tensor_address(self._output_name, output.data_ptr())
                self._context.execute_async_v3(stream)
            else:
                # TRT 8.0-8.4: binding-index API.
                if not self._context.set_binding_shape(0, tuple(x.shape)):
                    raise RuntimeError(f"TensorRT rejected input shape {tuple(x.shape)}")
                raw_out_shape = tuple(self._context.get_binding_shape(1))
                out_shape = (x.shape[0],) + raw_out_shape[1:]
                output = torch.empty(out_shape, dtype=torch.float32, device=x.device)
                bindings = [x.data_ptr(), output.data_ptr()]
                self._context.execute_async_v2(bindings, stream)

        caller_stream.wait_stream(self._stream)
        return output


class MapPredictionModel:
    def __init__(
        self,
        model_path,
        enable_uncertainty,
        model_input_size=128,
        inference_batch_size=16,
        uncertainty_scale=2.0,
        device="auto",
        use_trt=True,
        trt_engine_path=None,
        trainable=False,
    ) -> None:
        self.device = self.resolve_device(device)
        self.enable_uncertainty = enable_uncertainty
        self.model_input_size = model_input_size
        self.inference_batch_size = inference_batch_size
        self.uncertainty_scale = uncertainty_scale
        self.clear_cuda_cache_after_inference = False

        net = FPUNet(in_channels=3, n_classes=3, feature_scale=2).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        net.load_state_dict(state_dict)
        net.eval()

        # The eager module is always retained: online training needs a
        # differentiable graph, which TensorRT / TorchScript trace cannot give.
        self._eager_net = net
        if trainable:
            # Inference uses a SEPARATE copy of the weights from the one being
            # trained (OnlineTrainer installs it), so prediction and training run
            # in parallel without blocking; weights sync only between rounds.
            self.net = net
        else:
            self.net = self._build_inference_backend(net, model_path, use_trt, trt_engine_path)

        # Guards the inference net during a between-rounds weight sync so a
        # forward never reads half-written weights. The (long) backward pass runs
        # on a different module and is NOT under this lock → truly parallel.
        self._weights_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _build_inference_backend(self, net, model_path, use_trt, trt_engine_path):
        """Return TRT engine if possible, otherwise a traced TorchScript model."""
        if use_trt and str(self.device).startswith("cuda"):
            engine = self._try_trt(net, model_path, trt_engine_path)
            if engine is not None:
                return engine
            print("[map_prediction] Falling back to TorchScript inference")

        return self._trace(net)

    def _trace(self, net):
        h = w = 2 * self.model_input_size
        example = torch.zeros(1, 3, h, w, device=self.device)
        return torch.jit.trace(net, example)

    def _try_trt(self, net, model_path, trt_engine_path):
        try:
            import tensorrt as trt  # noqa: F401
        except ImportError:
            print("[map_prediction] tensorrt package not found; skipping TRT")
            return None

        if trt_engine_path is None or trt_engine_path == "":
            trt_engine_path = os.path.splitext(model_path)[0] + ".trt.engine"

        return self._load_or_build_engine(net, trt_engine_path)

    def _load_or_build_engine(self, net, engine_path):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)

        if os.path.exists(engine_path):
            print(f"[map_prediction] Loading cached TRT engine: {engine_path}")
            runtime = trt.Runtime(logger)
            with open(engine_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is not None:
                return TRTInferenceEngine(engine)
            print("[map_prediction] Cached engine failed to deserialize; rebuilding")

        print(
            f"[map_prediction] Building TRT engine → {engine_path}  "
            "(first build takes 1–5 min, result is cached)"
        )
        try:
            engine = self._build_engine(net, engine_path, logger)
            return TRTInferenceEngine(engine)
        except Exception as exc:
            print(f"[map_prediction] TRT build failed: {exc}")
            return None

    def _build_engine(self, net, engine_path, logger):
        import tensorrt as trt

        h = w = 2 * self.model_input_size
        max_batch = max(self.inference_batch_size, 1)
        opt_batch = max(1, max_batch // 2)

        # --- Export model to ONNX in memory ---
        from torch.export import Dim

        onnx_buf = io.BytesIO()
        dummy = torch.zeros(1, 3, h, w, device=self.device)
        batch_dim = Dim("batch", min=1, max=max_batch)
        with torch.no_grad():
            torch.onnx.export(
                net,
                (dummy,),
                onnx_buf,
                input_names=["input"],
                output_names=["output"],
                # Positional form: torch>=2.6 exports ONNX through torch.export,
                # where a dict's keys must be forward()'s parameter names — not
                # the ONNX input_names above. FPUNet.forward takes `inputs`, so
                # {"input": ...} raised "top-level keys must be the arg names
                # ['inputs']" and dropped the whole TRT path to TorchScript. The
                # tuple matches args positionally and survives a rename.
                dynamic_shapes=({0: batch_dim},),
                opset_version=18,
            )
        onnx_buf.seek(0)

        # --- Build TRT engine ---
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        onnx_bytes = onnx_buf.read()
        if not parser.parse(onnx_bytes):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX→TRT parse errors: {'; '.join(errs)}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

        # Enable FP16 if the GPU supports it (2–4× throughput vs FP32).
        fp16_ok = getattr(builder, "platform_has_fast_fp16", False) or getattr(
            builder, "platform_has_fp16", False
        )
        if fp16_ok:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[map_prediction] TRT FP16 enabled")

        # Dynamic batch profile: 1 → opt_batch → max_batch.
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            min=(1, 3, h, w),
            opt=(opt_batch, 3, h, w),
            max=(max_batch, 3, h, w),
        )
        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("builder.build_serialized_network returned None")

        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"[map_prediction] TRT engine saved: {engine_path}")

        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(engine_bytes)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_device(device_name):
        if device_name == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_name

    def predict_map(self, input_map, frontier_points):
        current_map = np.array(input_map, copy=True)
        known_mask = current_map != UNKNOWN_VALUE
        predicted_map = np.array(current_map, copy=True)
        predicted_confidence = np.zeros(current_map.shape, dtype=np.float32)
        predicted_confidence[known_mask] = 1.0

        if not frontier_points:
            return predicted_map, predicted_confidence

        batch_labels = []
        batch_confidences = []
        batch_size = max(1, self.inference_batch_size)
        backend_max_batch = getattr(self.net, "max_batch_size", None)
        if backend_max_batch is not None:
            batch_size = min(batch_size, backend_max_batch)
        # Hold the weights lock for the whole inference pass so a between-rounds
        # weight sync can't mutate the net mid-prediction (uncontended except for
        # the brief sync; the training backward runs lock-free in parallel).
        with self._weights_lock, torch.no_grad():
            for start in range(0, len(frontier_points), batch_size):
                frontier_batch = frontier_points[start:start + batch_size]
                batch_tensor = self.build_model_input(current_map, frontier_batch)
                model_output = self.net(batch_tensor)
                batch_labels.append(self.probability_to_label(model_output).cpu().numpy())
                batch_confidences.append(
                    self.probability_to_confidence(model_output).cpu().numpy()
                )

        batch_labels = np.concatenate(batch_labels, axis=0)
        batch_confidence = np.concatenate(batch_confidences, axis=0)
        self.merge_predictions(
            predicted_map,
            predicted_confidence,
            current_map,
            frontier_points,
            batch_labels,
            batch_confidence,
            known_mask,
        )

        if self.clear_cuda_cache_after_inference:
            self.clear_cuda_cache()
        return predicted_map, predicted_confidence

    # ------------------------------------------------------------------
    # Online fine-tuning handoff
    #
    # The training logic itself lives in map_prediction_trainer.OnlineTrainer.
    # This model only owns the inference net + the lock that guards it, and
    # exposes the eager (differentiable) net plus two atomic mutators the trainer
    # uses to publish freshly trained weights.
    # ------------------------------------------------------------------

    @property
    def eager_net(self):
        """The differentiable module OnlineTrainer fine-tunes. Always built (even
        when not training) since TensorRT / TorchScript traces are not
        differentiable."""
        return self._eager_net

    def install_inference_net(self, net):
        """Atomically swap the net that predict_map() reads. OnlineTrainer calls
        this once, to publish the dedicated frozen inference copy."""
        with self._weights_lock:
            self.net = net

    def load_inference_weights(self, state_dict):
        """Atomically copy freshly trained weights into the current inference net.
        OnlineTrainer calls this once per round — the only place, besides
        install_inference_net, that the inference net changes."""
        with self._weights_lock:
            self.net.load_state_dict(state_dict)

    def build_model_input(self, current_map, frontier_batch):
        size = 2 * self.model_input_size
        local_maps = np.empty((len(frontier_batch), size, size), dtype=np.uint8)
        for batch_index, frontier_point in enumerate(frontier_batch):
            self.fill_local_map(
                current_map,
                frontier_point,
                self.model_input_size,
                local_maps[batch_index],
            )
        batch_tensor = torch.as_tensor(local_maps, device=self.device).unsqueeze(1)
        return self.label_to_one_hot(batch_tensor)

    def merge_predictions(
        self,
        predicted_map,
        predicted_confidence,
        current_map,
        frontier_batch,
        batch_labels,
        batch_confidence,
        known_mask,
    ):
        for batch_index, frontier_point in enumerate(frontier_batch):
            (
                map_slice_y,
                map_slice_x,
                local_slice_y,
                local_slice_x,
            ) = self.get_local_to_global_slices(
                batch_labels[batch_index].shape,
                frontier_point,
                current_map.shape,
            )
            local_map = batch_labels[batch_index][local_slice_y, local_slice_x]
            local_confidence = batch_confidence[batch_index][local_slice_y, local_slice_x]
            local_known_mask = known_mask[map_slice_y, map_slice_x]

            # On overlap, prefer the higher-confidence prediction. predicted_map
            # starts UNKNOWN with confidence=0, so the first frontier to touch a
            # cell always wins; later frontiers only overwrite when strictly more
            # confident. ~local_known_mask is redundant with the confidence test
            # (known cells start at conf=1.0) but kept as an explicit guard.
            predicted_map_slice = predicted_map[map_slice_y, map_slice_x]
            predicted_confidence_slice = predicted_confidence[map_slice_y, map_slice_x]
            update_mask = (
                (local_map != UNKNOWN_VALUE)
                & ~local_known_mask
                & (local_confidence > predicted_confidence_slice)
            )
            predicted_map_slice[update_mask] = local_map[update_mask]
            predicted_confidence_slice[update_mask] = local_confidence[update_mask]

        predicted_map[known_mask] = current_map[known_mask]
        predicted_confidence[known_mask] = 1.0

    def clear_cuda_cache(self):
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def probability_to_label(self, logits):
        labels = torch.argmax(logits, dim=1).to(dtype=torch.uint8)
        result = torch.full_like(labels, UNKNOWN_VALUE)
        result[labels == 0] = FREE_VALUE
        result[labels == 1] = OCCUPIED_VALUE
        result[labels == 2] = UNKNOWN_VALUE
        return result

    def probability_to_confidence(self, logits):
        scaled_logits = logits * self.uncertainty_scale if self.enable_uncertainty else logits
        probabilities = F.softmax(scaled_logits, dim=1)
        confidence, _ = torch.max(probabilities, dim=1)
        return confidence

    @staticmethod
    def label_to_one_hot(data):
        one_hot = torch.empty(
            (data.shape[0], 3, data.shape[2], data.shape[3]),
            dtype=torch.float32,
            device=data.device,
        )
        one_hot[:, 0] = (data[:, 0] > 200).to(dtype=torch.float32)
        one_hot[:, 1] = (data[:, 0] < 10).to(dtype=torch.float32)
        one_hot[:, 2] = ((data[:, 0] > 90) & (data[:, 0] < 110)).to(
            dtype=torch.float32
        )
        return one_hot

    def to_original_frame(self, local_prediction, frontier_point, map_shape, destination):
        map_slice_y, map_slice_x, local_slice_y, local_slice_x = (
            self.get_local_to_global_slices(
                local_prediction.shape,
                frontier_point,
                map_shape,
            )
        )
        destination[map_slice_y, map_slice_x] = local_prediction[
            local_slice_y,
            local_slice_x,
        ]
        return destination

    @staticmethod
    def get_local_to_global_slices(local_shape, frontier_point, map_shape):
        local_half_size = local_shape[0] // 2
        goal_x, goal_y = frontier_point
        map_height, map_width = map_shape

        start_y = max(0, goal_y - local_half_size)
        end_y = min(map_height, goal_y + local_half_size)
        start_x = max(0, goal_x - local_half_size)
        end_x = min(map_width, goal_x + local_half_size)

        offset_y_start = max(0, local_half_size - goal_y)
        offset_y_end = min(2 * local_half_size, map_height - goal_y + local_half_size)
        offset_x_start = max(0, local_half_size - goal_x)
        offset_x_end = min(2 * local_half_size, map_width - goal_x + local_half_size)

        return (
            slice(start_y, end_y),
            slice(start_x, end_x),
            slice(offset_y_start, offset_y_end),
            slice(offset_x_start, offset_x_end),
        )

    @staticmethod
    def get_local_map(global_map, frontier_point, local_map_size, default_value=FREE_VALUE):
        local_map = np.empty(
            (2 * local_map_size, 2 * local_map_size),
            dtype=np.uint8,
        )
        MapPredictionModel.fill_local_map(
            global_map,
            frontier_point,
            local_map_size,
            local_map,
            default_value,
        )
        return local_map

    @staticmethod
    def fill_local_map(
        global_map,
        frontier_point,
        local_map_size,
        destination,
        default_value=FREE_VALUE,
    ):
        goal_x, goal_y = frontier_point
        map_height, map_width = global_map.shape

        start_y = max(0, goal_y - local_map_size)
        end_y = min(map_height, goal_y + local_map_size)
        start_x = max(0, goal_x - local_map_size)
        end_x = min(map_width, goal_x + local_map_size)

        destination.fill(default_value)

        offset_y_start = max(0, local_map_size - goal_y)
        offset_y_end = min(2 * local_map_size, map_height - goal_y + local_map_size)
        offset_x_start = max(0, local_map_size - goal_x)
        offset_x_end = min(2 * local_map_size, map_width - goal_x + local_map_size)

        destination[offset_y_start:offset_y_end, offset_x_start:offset_x_end] = global_map[
            start_y:end_y,
            start_x:end_x,
        ]
