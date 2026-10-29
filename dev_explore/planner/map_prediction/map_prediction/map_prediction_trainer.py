import copy

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .map_prediction_model import (
    FREE_VALUE,
    OCCUPIED_VALUE,
    UNKNOWN_VALUE,
    MapPredictionModel,
)


class OnlineTrainer:
    """Self-supervised online fine-tuning for a MapPredictionModel.

    Owns the training-only state (optimizer, masked loss, L2-SP snapshot,
    distillation reference net) and fine-tunes the model's eager, differentiable
    copy of the weights. The net that prediction actually reads is a SEPARATE
    frozen copy living on the model; sync_inference_weights() is the only point
    it is updated, so the (long) backward pass on this thread never blocks a
    forward pass. Constructing the trainer configures everything and installs the
    frozen inference net into the model — after that, run train_step() on a
    background thread and sync_inference_weights() once per round.
    """

    # Encoder blocks frozen during fine-tuning (everything else — the decoder
    # up_concat* columns and final_* heads — stays trainable).
    ENCODER_MODULES = ("conv00", "conv10", "conv20", "conv30", "conv40")

    def __init__(
        self,
        model,
        lr=5e-5,
        l2sp_lambda=1e-2,
        class_weights=(1.0, 1.0, 1.0),
        freeze_encoder=True,
        distill_weight=1.0,
        distill_temperature=1.0,
        hard_example_weight=0.0,
        entropy_weight=0.0,
        entropy_far_cells=30.0,
    ):
        """Set up the optimizer, the frozen pretrained reference for L2-SP +
        output distillation, and the masked loss, then install a dedicated frozen
        inference net into the model. The eager net stays in eval() so BatchNorm
        running stats never drift — inference and training then behave identically
        (and eval() does not disable autograd)."""
        self.model = model
        self.device = model.device
        net = model.eager_net
        net.eval()

        if freeze_encoder:
            for name in self.ENCODER_MODULES:
                module = getattr(net, name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(False)

        trainable_named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
        self._optimizer = torch.optim.Adam([p for _, p in trainable_named], lr=lr)
        # Detached snapshot of the pretrained trainable weights for L2-SP.
        self._theta0 = {n: p.detach().clone() for n, p in trainable_named}
        self._l2sp_lambda = float(l2sp_lambda)
        class_weight_tensor = torch.tensor(
            class_weights, dtype=torch.float32, device=self.device
        )
        self._ce = torch.nn.CrossEntropyLoss(weight=class_weight_tensor, reduction="none")

        # Online hard-example mining. The supervised cells (new_obs: UNKNOWN at
        # capture, observed in the target) are exactly the near-frontier band the
        # robot just walked into — the same band that carries the highest error
        # rate. Uniform weighting dilutes those hard cells' gradient among many
        # easy ones and lets L2-SP/distillation pull it back; upweighting the
        # cells the model got wrong by (1 + hard_example_weight) concentrates the
        # step there. 0 disables (uniform new_obs weighting, the prior behaviour).
        # Direction-neutral by construction: it upweights any wrong cell, so it
        # does not by itself trade invented walls for invented free — watch the
        # optimistic/pessimistic split (prediction_error_analysis) to be sure.
        self._hard_weight = max(0.0, float(hard_example_weight))

        # Distance-weighted entropy on the unsupervised (still-unknown) cells: an
        # alternative to distillation for the far field. CE never reaches these
        # cells (no target), so the model's far-field prediction is unconstrained
        # by data — distillation pins it to the (over-confident, phantom) prior,
        # which is exactly what we do NOT want. Instead push those cells toward
        # LOW confidence (high softmax entropy), scaled by distance from the
        # observed footprint: near the frontier the extrapolation keeps its shape
        # (weight ~0), deep in the unknown where there is no evidence it is driven
        # toward "I don't know" (weight -> 1). The planner already discounts
        # low-confidence predicted frontiers, so this attacks phantom without any
        # planner change. 0 disables. Typically paired with distill_weight=0.
        self._entropy_weight = max(0.0, float(entropy_weight))
        self._entropy_far_cells = max(1.0, float(entropy_far_cells))

        # Frozen pretrained copy: anchors predictions in the *unsupervised* region
        # (cells with no ground truth) toward the trustworthy prior, so fine-tuning
        # the observed near-field doesn't corrupt the far-field extrapolation that
        # is actually rendered/planned on.
        self._distill_weight = float(distill_weight)
        self._distill_temperature = max(1e-3, float(distill_temperature))
        if self._distill_weight > 0:
            self._ref_net = copy.deepcopy(net).eval()
            for param in self._ref_net.parameters():
                param.requires_grad_(False)
        else:
            self._ref_net = None

        # Dedicated inference net: a frozen-graph copy that prediction reads while
        # the eager net is trained on this thread. sync_inference_weights() copies
        # the trained weights over between rounds.
        inference_net = copy.deepcopy(net).eval()
        for param in inference_net.parameters():
            param.requires_grad_(False)
        model.install_inference_net(inference_net)
        self.trainable_param_count = sum(p.numel() for _, p in trainable_named)

    def sync_inference_weights(self):
        """Copy the trained weights into the model's inference net. Called once per
        round; brief and the only place the inference net is mutated after setup."""
        self.model.load_inference_weights(self.model.eager_net.state_dict())

    def train_step(self, input_windows, target_windows):
        """One masked gradient step on a batch of (input, later-observed) windows.

        input_windows / target_windows: np.uint8 arrays (B, H, W) in model values
        (FREE/OCCUPIED/UNKNOWN). CE is supervised only on cells that were UNKNOWN
        in the input window but became observed in the target (the real "the robot
        walked over and saw it" signal); cells already-observed-at-capture would
        otherwise teach the net to copy its own input. Returns prequential metrics
        computed on the same new-obs mask.
        """
        net = self.model.eager_net
        device = self.device

        inp = torch.as_tensor(input_windows, device=device).unsqueeze(1)  # (B,1,H,W)
        one_hot = MapPredictionModel.label_to_one_hot(inp)
        tgt = torch.as_tensor(target_windows, device=device).long()       # (B,H,W)

        target_idx = torch.full_like(tgt, 2)                  # default UNKNOWN class
        target_idx[tgt == int(FREE_VALUE)] = 0
        target_idx[tgt == int(OCCUPIED_VALUE)] = 1
        known = tgt != int(UNKNOWN_VALUE)                     # any GT in target
        new_obs = known & (inp[:, 0] == int(UNKNOWN_VALUE))   # GT *and* was unknown
        weight = new_obs.to(dtype=torch.float32)

        logits = net(one_hot)                                 # (B,3,H,W), grad on

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            tp = ((pred == 1) & (target_idx == 1) & new_obs).sum().float()
            fp = ((pred == 1) & (target_idx != 1) & new_obs).sum().float()
            fn = ((pred != 1) & (target_idx == 1) & new_obs).sum().float()
            denom_f1 = 2 * tp + fp + fn
            denom_iou = tp + fp + fn
            masked_f1 = (2 * tp / denom_f1).item() if denom_f1 > 0 else float("nan")
            masked_iou = (tp / denom_iou).item() if denom_iou > 0 else float("nan")
            masked_acc = (
                ((pred == target_idx) & new_obs).sum().float()
                / new_obs.sum().clamp(min=1)
            ).item()

            # Per-sample copies of the same masked metrics. The batch means above
            # cannot answer "what fraction of individual windows does the model
            # get wrong" — a few bad windows vanish inside a good batch mean —
            # and that fraction is exactly what the self-induced-bias analysis
            # needs. Same new-obs mask, same prequential timing (pre-update).
            batch_size = new_obs.shape[0]
            flat_obs = new_obs.view(batch_size, -1)
            flat_correct = ((pred == target_idx).view(batch_size, -1) & flat_obs)
            n_per = flat_obs.sum(dim=1)
            acc_per = flat_correct.sum(dim=1).float() / n_per.clamp(min=1)
            tp_per = ((pred == 1) & (target_idx == 1) & new_obs).view(batch_size, -1).sum(dim=1).float()
            fp_per = ((pred == 1) & (target_idx != 1) & new_obs).view(batch_size, -1).sum(dim=1).float()
            fn_per = ((pred != 1) & (target_idx == 1) & new_obs).view(batch_size, -1).sum(dim=1).float()
            denom_per = 2 * tp_per + fp_per + fn_per
            f1_per = torch.where(
                denom_per > 0, 2 * tp_per / denom_per.clamp(min=1),
                torch.full_like(denom_per, float("nan")),
            )
            sample_metrics = [
                {
                    "acc": acc_per[i].item(),
                    "f1": f1_per[i].item(),
                    "n_cells": int(n_per[i].item()),
                }
                for i in range(batch_size)
            ]

        # Hard-example mining: upweight the supervised cells the model got wrong.
        # `pred` is detached (argmax under no_grad), so scaling the loss weight by
        # a function of it introduces no gradient path — it only reallocates how
        # much each cell's CE counts.
        hard_frac = 0.0
        if self._hard_weight > 0.0:
            wrong = (pred != target_idx) & new_obs
            weight = weight * (1.0 + self._hard_weight * wrong.to(weight.dtype))
            total = weight.sum().clamp(min=1.0)
            hard_frac = ((self._hard_weight * wrong.to(weight.dtype)).sum()
                         / total).item()

        ce = self._ce(logits, target_idx)                     # (B,H,W)
        loss_ce = (ce * weight).sum() / weight.sum().clamp(min=1.0)

        # Distillation anchor on cells STILL UNKNOWN in target: keep the prediction
        # close to the frozen pretrained prior so fine-tuning the observed region
        # doesn't collapse the far-field extrapolation. (Cells that were already
        # known at capture time are neither supervised nor distilled — input is
        # one-hot=known there, so the model trivially reproduces them; L2-SP keeps
        # the decoder from drifting on those identity cells.)
        loss_distill = logits.new_zeros(())
        if self._ref_net is not None and self._distill_weight > 0:
            unsup = ~known
            n_unsup = unsup.sum()
            if n_unsup > 0:
                temp = self._distill_temperature
                with torch.no_grad():
                    ref_p = F.softmax(self._ref_net(one_hot) / temp, dim=1)
                log_p = F.log_softmax(logits / temp, dim=1)
                # KL(ref || current), per cell, summed over classes.
                kl = (ref_p * (ref_p.clamp_min(1e-8).log() - log_p)).sum(dim=1)
                loss_distill = (
                    self._distill_weight * (temp ** 2)
                    * (kl * unsup.to(dtype=torch.float32)).sum()
                    / n_unsup.to(dtype=torch.float32)
                )

        # Distance-weighted entropy: drive far unknown cells toward low
        # confidence. Uses the same unsupervised mask as distillation; the two
        # are alternatives (run one or the other). `pred` above is detached, so
        # this is the only far-field gradient path when distillation is off.
        loss_entropy = logits.new_zeros(())
        if self._entropy_weight > 0:
            unsup = ~known
            n_unsup = unsup.sum()
            if n_unsup > 0:
                # Distance (in cells) from the observed footprint in the INPUT
                # window; ramp 0 -> 1 over entropy_far_cells.
                obsv = (input_windows != int(UNKNOWN_VALUE)).astype(np.uint8)
                w = np.empty(input_windows.shape, dtype=np.float32)
                for b in range(obsv.shape[0]):
                    dt = cv2.distanceTransform(1 - obsv[b], cv2.DIST_L2, 3)
                    np.clip(dt / self._entropy_far_cells, 0.0, 1.0, out=w[b])
                w_t = torch.as_tensor(w, device=device)
                logp = F.log_softmax(logits, dim=1)
                entropy = -(logp.exp() * logp).sum(dim=1)      # (B,H,W)
                mask = unsup.to(dtype=torch.float32) * w_t
                # Maximise entropy on far unknown cells => minimise -entropy.
                loss_entropy = (
                    -self._entropy_weight
                    * (entropy * mask).sum() / mask.sum().clamp(min=1.0)
                )

        loss_l2sp = logits.new_zeros(())
        if self._l2sp_lambda > 0 and self._theta0:
            for name, param in net.named_parameters():
                ref = self._theta0.get(name)
                if ref is not None:
                    loss_l2sp = loss_l2sp + ((param - ref) ** 2).sum()
            loss_l2sp = 0.5 * self._l2sp_lambda * loss_l2sp

        loss = loss_ce + loss_distill + loss_l2sp + loss_entropy
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()

        return {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_distill": float(loss_distill.item()),
            "loss_l2sp": float(loss_l2sp.item()),
            "loss_entropy": float(loss_entropy.item()),
            "masked_f1": masked_f1,
            "masked_iou": masked_iou,
            "masked_acc": masked_acc,
            "n_cells": int(new_obs.sum().item()),
            "hard_frac": hard_frac,
            "samples": sample_metrics,
        }
