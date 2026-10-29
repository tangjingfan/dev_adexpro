"""Self-adapting prediction backend (AdExPro): the FPUnet predictor of
map_prediction_node, fine-tuned online from the robot's own observations.

A trainer thread runs beside inference. Each step it asks the configured adapt
policies for (input window, target window) pairs, takes one OnlineTrainer step
on them, and at the end of every round syncs the trained weights into the
inference net. An adapt policy decides only *what to train on*; the loss lives
in OnlineTrainer.
"""

import math
import os
import random
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException

from .grid_encoding import FREE_VALUE, OCCUPIED_VALUE, UNKNOWN_VALUE
from .map_prediction_model import MapPredictionModel
from .map_prediction_node import MapPredictionNode
from .map_prediction_trainer import OnlineTrainer


# ----------------------------------------------------------------------
# Adapt policies
# ----------------------------------------------------------------------

class AdaptPolicy:
    """Where training pairs come from. Subclass it and register the subclass in
    ADAPT_POLICIES to make it selectable from the adapt_policies parameter.

    observe() runs on the inference thread; begin_round() and sample() run on
    the trainer thread.
    """

    def observe(self, captures):
        """This cycle's (world_x, world_y, input_window) prediction captures."""

    def ready(self):
        """True once the policy can supply training pairs."""
        return True

    def begin_round(self):
        """Called once at the start of each training round."""

    def sample(self, n, current_map_np, map_msg, node):
        """Up to n (input_window, target_window, (world_x, world_y)) triples.

        current_map_np is the latest visible map in model encoding and map_msg
        the OccupancyGrid it came from; node gives access to its grid helpers.
        """
        raise NotImplementedError

    def size(self):
        """How many stored samples the policy holds (for the training log)."""
        return 0

    def describe(self):
        """Short status for the per-step log line."""
        return ""


class SpatialReplayBuffer:
    """Replay buffer that caps how many windows any one place may contribute.

    Captures are taken once per prediction cycle for every frontier, so a
    frontier the robot lingers near re-enters the buffer every cycle for as
    long as it lingers. Measured over one indoor run: 201 distinct places
    produced 4689 training samples, the hottest place appeared 436 times, and
    the hottest 10% of places accounted for 69% of all samples. Sampling that
    uniformly weights the gradient by dwell time, which has nothing to do with
    how much there is to learn anywhere.

    Bucketing captures by world position and keeping only the newest few per
    bucket removes that weighting. Newest rather than first because the target
    is re-cropped from the current map at training time, so a later capture of
    the same place carries a more complete target.

    Iterating yields the flattened contents, and len() is the total, so this
    drops in where a deque was used.
    """

    def __init__(self, bucket_size_m, per_bucket, capacity):
        self._bucket_size = float(bucket_size_m)
        self._per_bucket = max(1, int(per_bucket))
        self._capacity = max(1, int(capacity))
        self._buckets = OrderedDict()   # bucket key -> deque of captures
        self._count = 0

    def _key(self, world_x, world_y):
        return (
            math.floor(world_x / self._bucket_size),
            math.floor(world_y / self._bucket_size),
        )

    def append(self, capture):
        world_x, world_y, _ = capture
        key = self._key(world_x, world_y)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = deque(maxlen=self._per_bucket)
            self._buckets[key] = bucket
        else:
            # Touch: eviction below drops whole least-recently-updated buckets,
            # so a place still being observed is never the one thrown away.
            self._buckets.move_to_end(key)
        if len(bucket) == bucket.maxlen:
            self._count -= 1            # this bucket's oldest is about to fall out
        bucket.append(capture)
        self._count += 1

        while self._count > self._capacity and len(self._buckets) > 1:
            _, evicted = self._buckets.popitem(last=False)
            self._count -= len(evicted)

    def extend(self, captures):
        for capture in captures:
            self.append(capture)

    def __iter__(self):
        for bucket in self._buckets.values():
            yield from bucket

    def __len__(self):
        return self._count

    @property
    def bucket_count(self):
        return len(self._buckets)


class FrontierReplayPolicy(AdaptPolicy):
    """Replay the frontier windows the predictor was actually fed.

    Every prediction cycle stores each frontier's input window with its world
    position. At training time the target is re-cropped at that position from
    the current, more complete map, so the supervision is what the robot has
    since observed there.
    """

    def __init__(self, bucket_size_m, per_bucket, capacity, warmup):
        self._buffer = (
            SpatialReplayBuffer(bucket_size_m, per_bucket, capacity)
            if bucket_size_m > 0.0
            else deque(maxlen=capacity)
        )
        self._lock = threading.Lock()
        self._warmup = warmup
        self._population = []

    def observe(self, captures):
        if captures:
            with self._lock:
                self._buffer.extend(captures)

    def ready(self):
        return self.size() >= self._warmup

    def begin_round(self):
        # One snapshot per round, so a round trains on a fixed population.
        with self._lock:
            self._population = list(self._buffer)

    def sample(self, n, current_map_np, map_msg, node):
        batch = random.sample(self._population, min(n, len(self._population)))
        out = []
        for world_x, world_y, window in batch:
            grid = node.world_to_grid([(world_x, world_y, 0.0)], map_msg)
            target = MapPredictionModel.get_local_map(
                current_map_np, grid[0], node.model_input_size,
                default_value=UNKNOWN_VALUE,
            )
            out.append((window, target, (world_x, world_y)))
        return out

    def size(self):
        with self._lock:
            return len(self._buffer)

    def describe(self):
        with self._lock:
            n = len(self._buffer)
            buckets = getattr(self._buffer, "bucket_count", None)
        return f"{n}" if buckets is None else f"{n}/{buckets}places"


class ExploredReplayPolicy(AdaptPolicy):
    """Re-stage exploration inside space the robot has already explored.

    Frontier replay only supervises what the robot happened to see right past
    each frontier, so its labels hug the frontier. Here a training pair is
    synthesised from the current map instead:

      1. pick a known free cell as a virtual viewpoint;
      2. "observe" the free space within a random geodesic radius of it (plus
         the walls bounding that space) and mask everything else to UNKNOWN —
         an exploration state the robot could have been in;
      3. centre the window on one of that state's frontiers whose far side is
         known in the current map, as inference would;
      4. input = the masked window, target = the same window of the current map,
         cut back to within label_max_m of the observed area (0 = whole window).

    The supervision then reaches label_max_m into the unknown, not just the band
    the robot saw after the fact. Without the cap the target would be the whole
    explored part of the window, ~60 m beyond a ~10 m observed patch, which is
    guessing rather than prediction.
    """

    MAX_ATTEMPTS = 8   # per requested pair, before giving up on it

    def __init__(self, radius_min_m, radius_max_m, label_max_m, half_window):
        self._radius_min_m = float(radius_min_m)
        self._radius_max_m = max(float(radius_max_m), self._radius_min_m)
        self._label_max_m = float(label_max_m)
        self._half = int(half_window)
        self._rng = np.random.default_rng()
        self._kernel = np.ones((3, 3), np.uint8)

    def sample(self, n, current_map_np, map_msg, node):
        free = np.flatnonzero(current_map_np.ravel() == FREE_VALUE)
        if free.size == 0:
            return []
        res = map_msg.info.resolution
        origin = map_msg.info.origin.position
        width = current_map_np.shape[1]
        out = []
        for _ in range(n * self.MAX_ATTEMPTS):
            if len(out) == n:
                break
            pair = self._stage(current_map_np, free, width, res)
            if pair is None:
                continue
            window, target, (row, col) = pair
            out.append((window, target, (origin.x + (col + 0.5) * res,
                                         origin.y + (row + 0.5) * res)))
        return out

    def _stage(self, current_map_np, free, width, res):
        """One (input, target, frontier (row, col)) pair, or None."""
        radius = int(round(self._rng.uniform(self._radius_min_m, self._radius_max_m) / res))
        row, col = divmod(int(self._rng.choice(free)), width)

        # Work in a crop around the viewpoint wide enough to hold any window
        # centred within `radius` of it; outside the map counts as UNKNOWN.
        half = self._half + radius
        crop = MapPredictionModel.get_local_map(
            current_map_np, (col, row), half, default_value=UNKNOWN_VALUE)
        crop_free = (crop == FREE_VALUE).astype(np.uint8)

        # Geodesic (8-connected) ball through free space.
        seen = np.zeros_like(crop_free)
        seen[half, half] = 1
        for _ in range(radius):
            grown = cv2.dilate(seen, self._kernel) & crop_free
            if np.array_equal(grown, seen):
                break
            seen = grown
        # The walls bounding the observed free space are observed too.
        observed = seen.astype(bool) | (
            cv2.dilate(seen, self._kernel).astype(bool) & (crop == OCCUPIED_VALUE))

        # Frontiers of the staged state that open onto space the current map
        # knows — otherwise the window would have nothing new to supervise.
        hidden_known = ~observed & (crop != UNKNOWN_VALUE)
        frontier = seen.astype(bool) & cv2.dilate(
            hidden_known.astype(np.uint8), self._kernel).astype(bool)
        fr, fc = np.nonzero(frontier)
        if fr.size == 0:
            return None
        k = int(self._rng.integers(fr.size))
        fr, fc = int(fr[k]), int(fc[k])

        staged = np.where(observed, crop, UNKNOWN_VALUE).astype(np.uint8)
        h = self._half
        window = staged[fr - h:fr + h, fc - h:fc + h].copy()
        target = crop[fr - h:fr + h, fc - h:fc + h].copy()
        if self._label_max_m > 0:
            # UNKNOWN in the target = not supervised (CE covers known cells only).
            unobserved = (window == UNKNOWN_VALUE).astype(np.uint8)
            dist_m = cv2.distanceTransform(unobserved, cv2.DIST_L2, 3) * res
            target[dist_m > self._label_max_m] = UNKNOWN_VALUE
        return window, target, (row + fr - half, col + fc - half)


# ----------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------

class SelfAdaptNode(MapPredictionNode):
    TRAINABLE = True

    def __init__(self):
        super().__init__()
        # The inference worker (producer) and the trainer thread (consumer) run
        # in parallel; the policies' stores + the latest map are the only shared
        # state.
        self._round_count = 0
        self._latest_map_msg = None
        self._stop_event = threading.Event()
        self._online_log = None
        self._sample_log = None
        self.policies, self.policy_weights = self._create_policies()
        self.trainer = OnlineTrainer(
            self.predictor,
            lr=self.learning_rate,
            l2sp_lambda=self.l2sp_lambda,
            class_weights=tuple(self.class_weights),
            freeze_encoder=self.freeze_encoder,
            distill_weight=self.distill_weight,
            distill_temperature=self.distill_temperature,
            hard_example_weight=self.hard_example_weight,
            entropy_weight=self.entropy_weight,
            entropy_far_cells=self.entropy_far_cells,
        )
        self._open_online_log()
        self.get_logger().info(
            f"Self-adapt ENABLED: {self.trainer.trainable_param_count} trainable params, "
            f"lr={self.learning_rate}, l2sp={self.l2sp_lambda}, "
            f"freeze_encoder={self.freeze_encoder}, "
            f"round={self.train_steps_per_round} steps / "
            f"{self.train_round_interval_s}s, policies="
            + ", ".join(f"{name}:{w:g}" for name, w in
                        zip(self.adapt_policies, self.policy_weights))
        )
        self._trainer_thread = threading.Thread(target=self._trainer_loop, daemon=True)
        self._trainer_thread.start()

    def _declare_parameters(self):
        super()._declare_parameters()
        self.declare_parameter("learning_rate", 5e-5)
        self.declare_parameter("l2sp_lambda", 1e-2)
        self.declare_parameter("train_batch_size", 16)
        # A training "round" = this many gradient steps; inference weights are
        # synced only at the end of each round (between rounds the predictor keeps
        # using the previous round's weights). Rounds repeat every interval.
        self.declare_parameter("train_steps_per_round", 8)
        self.declare_parameter("train_round_interval_s", 2.0)
        # Which adapt policies supply training pairs, and the share of each batch
        # each one fills (names: see ADAPT_POLICIES).
        self.declare_parameter("adapt_policies", ["frontier_replay"])
        self.declare_parameter("adapt_policy_weights", [1.0])
        # Minimum *newly observed* cells (UNKNOWN in input window, observed in
        # target) required to accept a (input, target) pair into training. Pairs
        # whose target adds little new info compared to the input give no
        # meaningful gradient (CE supervises only on new-obs cells) and would
        # otherwise just inflate buffer hits for stationary regions.
        self.declare_parameter("label_min_new_cells", 200)
        # frontier_replay: buffer size and spatial de-duplication — bucket
        # captures by world position and keep at most replay_per_bucket of them
        # per bucket, so a place the robot lingers near stops dominating the
        # gradient. Bucket edge in metres; 0 disables and restores a plain deque.
        self.declare_parameter("replay_buffer_size", 2000)
        self.declare_parameter("replay_bucket_size_m", 2.0)
        self.declare_parameter("replay_per_bucket", 2)
        # explored_replay: the staged exploration state "observes" free space
        # within a geodesic radius drawn uniformly from this range (m) of a
        # random explored viewpoint; the rest of the window is supervised.
        self.declare_parameter("explored_radius_min_m", 5.0)
        self.declare_parameter("explored_radius_max_m", 20.0)
        # ...and supervises target cells only within this distance (m) of that
        # observed area (0 = the whole window).
        self.declare_parameter("explored_label_max_m", 20.0)
        # Don't start training until the buffer holds this many captures, so we
        # don't overfit a handful of correlated frontier windows from cycle one.
        # Keep modest: scenes often yield only a few frontiers per cycle, so a
        # large value would delay training for a very long time (or forever).
        self.declare_parameter("train_warmup_samples", 40)
        self.declare_parameter("freeze_encoder", True)
        # Uniform weights for online fine-tuning: P2Explore's [0.2, 0.6, 0.2]
        # up-weights occupied 3x, which biases the model to predict obstacles
        # under uncertainty — fine with full-floorplan targets, but online (with
        # partial, obstacle-biased supervision) it collapses to "occupied
        # everywhere" in the unsupervised extrapolation region.
        self.declare_parameter("class_weights", [0.2, 0.6, 0.2])
        # Distillation anchor: keep unsupervised (unknown) cells close to the
        # frozen pretrained prediction. This is the main guard against the
        # "obstacles everywhere" collapse in the rendered extrapolation region.
        self.declare_parameter("distill_weight", 1.0)
        self.declare_parameter("distill_temperature", 1.0)
        # Online hard-example mining: extra CE weight on supervised cells the
        # model got wrong (the high-error near-frontier band). 0 = uniform,
        # the prior behaviour; ~1-3 concentrates the step on the mistakes.
        self.declare_parameter("hard_example_weight", 0.0)
        # Distance-weighted entropy on the unsupervised (unknown) far field, as an
        # alternative to distillation: push deep-unknown cells toward low
        # confidence instead of pinning them to the (phantom) prior. Weight ramps
        # 0 -> 1 over entropy_far_cells from the observed footprint. 0 disables;
        # pair with distill_weight=0 to replace the anchor rather than stack it.
        self.declare_parameter("entropy_weight", 0.0)
        self.declare_parameter("entropy_far_cells", 30.0)
        self.declare_parameter("online_log_dir", "")

    def _load_parameters(self):
        super()._load_parameters()
        gp = self.get_parameter
        self.learning_rate          = gp("learning_rate").value
        self.l2sp_lambda            = gp("l2sp_lambda").value
        self.train_batch_size       = gp("train_batch_size").value
        self.train_steps_per_round  = gp("train_steps_per_round").value
        self.train_round_interval_s = gp("train_round_interval_s").value
        self.adapt_policies         = list(gp("adapt_policies").value)
        self.adapt_policy_weights   = [float(w) for w in gp("adapt_policy_weights").value]
        self.label_min_new_cells    = gp("label_min_new_cells").value
        self.replay_buffer_size     = gp("replay_buffer_size").value
        self.replay_bucket_size_m   = gp("replay_bucket_size_m").value
        self.replay_per_bucket      = gp("replay_per_bucket").value
        self.explored_radius_min_m  = gp("explored_radius_min_m").value
        self.explored_radius_max_m  = gp("explored_radius_max_m").value
        self.explored_label_max_m   = gp("explored_label_max_m").value
        self.train_warmup_samples   = gp("train_warmup_samples").value
        self.freeze_encoder         = gp("freeze_encoder").value
        self.class_weights          = list(gp("class_weights").value)
        self.distill_weight         = gp("distill_weight").value
        self.distill_temperature    = gp("distill_temperature").value
        self.hard_example_weight    = gp("hard_example_weight").value
        self.entropy_weight         = gp("entropy_weight").value
        self.entropy_far_cells      = gp("entropy_far_cells").value
        self.online_log_dir         = gp("online_log_dir").value or ""
        # Training needs the eager (differentiable) module, which the
        # TRT/TorchScript backend is not.
        self.use_trt = False

    def _create_policies(self):
        """The configured AdaptPolicy instances and their normalised weights."""
        if len(self.adapt_policies) != len(self.adapt_policy_weights):
            raise ValueError("adapt_policies and adapt_policy_weights differ in length")
        unknown = [name for name in self.adapt_policies if name not in ADAPT_POLICIES]
        if unknown:
            raise ValueError(f"Unknown adapt policies {unknown}; known: {sorted(ADAPT_POLICIES)}")
        total = sum(self.adapt_policy_weights)
        if total <= 0:
            raise ValueError("adapt_policy_weights must sum to > 0")
        policies = [ADAPT_POLICIES[name](self) for name in self.adapt_policies]
        return policies, [w / total for w in self.adapt_policy_weights]

    def _batch_shares(self, n):
        """Split a batch of n across the policies by weight (largest remainder)."""
        raw = [n * w for w in self.policy_weights]
        shares = [int(r) for r in raw]
        by_remainder = sorted(range(len(raw)), key=lambda i: raw[i] - shares[i], reverse=True)
        for i in by_remainder[:n - sum(shares)]:
            shares[i] += 1
        return shares

    # ------------------------------------------------------------------
    # Inference-side hooks
    # ------------------------------------------------------------------

    def _begin_cycle(self):
        # Capture this cycle's input windows for self-supervised training.
        return []

    def _end_cycle(self, visible_map, capture):
        # Hand fresh captures + the latest map to the (parallel) trainer thread.
        # No training happens here, so the inference cycle is never blocked by it.
        self._latest_map_msg = visible_map
        for policy in self.policies:
            policy.observe(capture)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _trainer_loop(self):
        """Runs in its own thread, in parallel with prediction. Each round does a
        fixed number of gradient steps on the trainer's own copy of the weights,
        then syncs them into the inference net (the only point weights change for
        prediction). Inference is never blocked by the backward pass."""
        last_warmup_log = 0.0
        while not self._stop_event.is_set():
            map_msg = self._latest_map_msg
            if map_msg is None or not all(p.ready() for p in self.policies):
                now = time.time()
                if now - last_warmup_log > 5.0:
                    last_warmup_log = now
                    status = ", ".join(f"{name} {p.describe()}".strip() for name, p in
                                       zip(self.adapt_policies, self.policies))
                    self.get_logger().info(
                        f"Trainer warming up: {status}"
                        f"{' (waiting for map)' if map_msg is None else ''}"
                    )
                self._stop_event.wait(0.5)
                continue

            # Latest observation snapshot; targets are re-cropped from it so they
            # reflect the most complete map (pairs keep improving over time).
            current_map_np = self.occupancy_grid_to_numpy(map_msg)
            if current_map_np is None:
                self._stop_event.wait(0.5)
                continue

            for policy in self.policies:
                policy.begin_round()
            self._round_count += 1
            round_t0 = time.time()
            steps_done = 0
            for _ in range(self.train_steps_per_round):
                if self._stop_event.is_set():
                    break
                inputs, targets, coords = self._build_training_batch(current_map_np, map_msg)
                if not inputs:
                    continue
                metrics = self.trainer.train_step(np.stack(inputs), np.stack(targets))
                self._log_train(metrics, len(inputs), coords)
                steps_done += 1
            train_ms = (time.time() - round_t0) * 1000.0

            # Publish the round's weights to the predictor (atomic, brief).
            sync_t0 = time.time()
            self.trainer.sync_inference_weights()
            sync_ms = (time.time() - sync_t0) * 1000.0

            # How long a round actually takes vs the interval it repeats on: if
            # train_ms approaches train_round_interval_s the loop is compute-bound
            # and raising the interval (or lowering steps) is pointless; if it is
            # far below, rounds are paced by the interval, not the GPU.
            per_step = train_ms / steps_done if steps_done else float("nan")
            self.get_logger().info(
                f"round {self._round_count}: {steps_done} steps in {train_ms:.0f} ms "
                f"({per_step:.0f} ms/step), sync {sync_ms:.0f} ms; "
                f"interval {self.train_round_interval_s:.1f}s"
            )

            self._stop_event.wait(self.train_round_interval_s)

    def _build_training_batch(self, current_map_np, map_msg):
        """One minibatch, each policy filling its share. A pair is only useful if
        the target reveals cells that were UNKNOWN in the input window — those new
        observations are the only thing CE supervises on (see
        OnlineTrainer.train_step), so drop pairs below label_min_new_cells."""
        inputs, targets, coords = [], [], []
        for policy, n in zip(self.policies, self._batch_shares(self.train_batch_size)):
            if n == 0:
                continue
            for window, target, xy in policy.sample(n, current_map_np, map_msg, self):
                new_obs = (target != UNKNOWN_VALUE) & (window == UNKNOWN_VALUE)
                if int(new_obs.sum()) < self.label_min_new_cells:
                    continue
                inputs.append(window)
                targets.append(target)
                coords.append(xy)
        return inputs, targets, coords

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def _open_online_log(self):
        if not self.online_log_dir:
            return
        log_dir, train_csv, sample_csv = self._resolve_online_paths()
        os.makedirs(log_dir, exist_ok=True)
        self._online_log = open(train_csv, "w", buffering=1)
        self._online_log.write(
            "wall_t,round,batch,loss,loss_ce,loss_distill,loss_l2sp,loss_entropy,"
            "masked_f1,masked_iou,masked_acc,n_cells,buffer\n"
        )
        # Per-sample companion log. The batch means above can't answer "what
        # fraction of windows does the model actually get wrong" — that is the
        # measurement the self-induced-bias work starts from. One row per
        # training sample: its world coords, supervised-cell count, and
        # pre-update masked acc/F1 on exactly those cells.
        self._sample_log = open(sample_csv, "w", buffering=1)
        self._sample_log.write("wall_t,round,x,y,n_cells,masked_acc,masked_f1\n")
        self.get_logger().info(
            f"Online training log: {train_csv} (+ per-sample {sample_csv})")

    def _resolve_online_paths(self):
        """(log_dir, train_csv, sample_csv) for this run's online logs.

        Launched by the experiment runner, EXPRO_RUN_ID co-locates the trainer's
        output with the recorder's under one self-describing run folder
        (<EXPRO_EXPERIMENTS_DIR>/runs/<id>/online_train/), with fixed filenames —
        no per-node stamp, because the run folder already carries the identity.
        A hand-run node (no EXPRO_RUN_ID) keeps the old self-stamped layout so a
        manual launch is unchanged.
        """
        run_id = os.environ.get("EXPRO_RUN_ID", "").strip()
        if run_id:
            exp = os.environ.get("EXPRO_EXPERIMENTS_DIR", "").strip() or \
                os.path.join(os.getcwd(), "experiments")
            log_dir = os.path.join(exp, "runs", run_id, "online_train")
            return (log_dir,
                    os.path.join(log_dir, "online_train.csv"),
                    os.path.join(log_dir, "online_samples.csv"))
        # A relative online_log_dir is taken under the experiments dir (set by
        # the start scripts and the runner), so the output lands in
        # dev_results/experiments/ wherever the node was launched from.
        log_dir = self.online_log_dir
        if not os.path.isabs(log_dir):
            exp = os.environ.get("EXPRO_EXPERIMENTS_DIR", "").strip() or \
                os.path.join(os.getcwd(), "experiments")
            log_dir = os.path.join(exp, log_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (log_dir,
                os.path.join(log_dir, f"online_train_{stamp}.csv"),
                os.path.join(log_dir, f"online_samples_{stamp}.csv"))

    def _log_train(self, m, batch_size, coords=None):
        buf = sum(p.size() for p in self.policies)
        buf_desc = " ".join(d for d in (p.describe() for p in self.policies) if d)
        if self._sample_log is not None and coords is not None:
            now_s = time.time()
            for (world_x, world_y), s in zip(coords, m.get("samples", [])):
                self._sample_log.write(
                    f"{now_s:.3f},{self._round_count},{world_x:.2f},{world_y:.2f},"
                    f"{s['n_cells']},{s['acc']:.4f},{s['f1']:.4f}\n"
                )
        self.get_logger().info(
            f"train round {self._round_count}: loss={m['loss']:.4f} "
            f"ce={m['loss_ce']:.4f} distill={m['loss_distill']:.4f} "
            f"l2sp={m['loss_l2sp']:.4f} entropy={m.get('loss_entropy', 0.0):.4f} "
            f"masked_f1={m['masked_f1']:.3f} "
            f"iou={m['masked_iou']:.3f} acc={m['masked_acc']:.3f} "
            f"n={m['n_cells']} buf={buf_desc}"
            + (f" hard={m['hard_frac']:.2f}" if m.get("hard_frac", 0.0) > 0 else "")
        )
        if self._online_log is not None:
            self._online_log.write(
                f"{time.time():.3f},{self._round_count},{batch_size},"
                f"{m['loss']:.6f},{m['loss_ce']:.6f},{m['loss_distill']:.6f},"
                f"{m['loss_l2sp']:.6f},{m.get('loss_entropy', 0.0):.6f},"
                f"{m['masked_f1']:.6f},{m['masked_iou']:.6f},{m['masked_acc']:.6f},"
                f"{m['n_cells']},{buf}\n"
            )

    def destroy_node(self):
        # Stop the trainer thread and flush the logs before teardown.
        self._stop_event.set()
        self._trainer_thread.join(timeout=5.0)
        for log in (self._online_log, self._sample_log):
            if log is not None:
                log.close()
        self._online_log = self._sample_log = None
        super().destroy_node()


# Name (as used in the adapt_policies parameter) -> factory taking the node.
ADAPT_POLICIES = {
    "frontier_replay": lambda node: FrontierReplayPolicy(
        node.replay_bucket_size_m,
        node.replay_per_bucket,
        node.replay_buffer_size,
        warmup=max(node.train_batch_size, node.train_warmup_samples),
    ),
    "explored_replay": lambda node: ExploredReplayPolicy(
        node.explored_radius_min_m,
        node.explored_radius_max_m,
        node.explored_label_max_m,
        half_window=node.model_input_size,
    ),
}


def main(args=None):
    rclpy.init(args=args)
    node = SelfAdaptNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
