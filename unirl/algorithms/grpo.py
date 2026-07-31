"""Stage-driven ``GRPO`` over a ``TextSegment``.

Implements :class:`StageAlgorithm` and shares the module-level
``_grpo_clip_loss`` / ``_resolve_clip_range_from_schedule`` helpers (in
:mod:`unirl.algorithms.base`) with :class:`FlowGRPO` so their loss
math stays identical. The teacher-forced forward and per-token log-prob
recompute are owned by ``stage.replay(...)``; the algorithm is ~20 lines of
ratio-clip math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

from unirl.config.require import require
from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
    rollout_replay_logp_absdiff,
    typed_conditions,
)


@dataclass
class GRPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    clip_range: float = 1e-4
    clip_schedule: str = "constant"
    old_logp_source: str = "rollout"


class GRPO(StageAlgorithm):
    """GRPO over an AR ``TextSegment`` via ``ARStage.replay``.

    The teacher-forced forward and per-token log-prob recompute is owned by
    :meth:`ARStage.replay`; this class expands per-sample advantages to per-
    token via ``cu_seqlens`` and runs the same PPO clip math.

    Args:
        stage: The :class:`ARStage` whose ``replay`` produces packed-varlen
            new log-probs aligned with ``segment.log_probs``.
        clip_range: PPO clip range epsilon.
        clip_schedule: ``"constant"``, ``"linear_decay"``, or
            ``"cosine_decay"``.
        conditions_cls: Stage-typed conditions container with
            ``from_dict(Mapping[str, Condition])``.
        old_logp_source: ``"rollout"`` (default) trusts the rollout engine's
            emitted ``segment.log_probs``; ``"replay"`` recomputes it via
            ``stage.replay`` at pre-update weights. See :meth:`prepare_segment`.
        sampling_temperature: AR rollout temperature, applied as a
            ``logits / T`` scaling inside :meth:`ARStage.replay` so
            replay's log-softmax matches SGLang's sampling distribution
            (``log_softmax(logits / T)``). Injected at construction time
            from the rollout engine config; falls back to
            :class:`ARSamplingParams` default when no engine is configured.
    """

    # old_logp is frozen on the segment and does NOT change across mini-batch
    # updates, so reusing it across num_updates_per_batch>1 keeps the ratio
    # anchored. Under the default old_logp_source='rollout' the anchor is the
    # rollout (SGLang) log-prob — the deliberate rollout-anchored PPO ratio
    # (verl bypass_mode=True parity), matching DRPO — and the ratio absorbs the
    # rollout-vs-train engine gap on later mini-batches (accepted for parity).
    # Under 'replay' prepare_segment overwrites it with a train-side anchor.
    supports_multi_update = True
    anchor_fields = ("log_probs",)

    def recomputes_anchor(self) -> bool:
        # Only ``replay`` re-derives log_probs; ``rollout`` keeps the engine's emission.
        return self.old_logp_source == "replay"

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        clip_range_high: Optional[float] = None,
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        conditions_cls: Optional[Type[Any]] = None,
        old_logp_source: str = "rollout",
        sampling_temperature: Optional[float] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("GRPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.old_logp_source = str(old_logp_source).strip().lower()
        require(
            self.old_logp_source in ("rollout", "replay"),
            f"GRPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}",
        )
        self.clip_range = float(clip_range)
        self.clip_range_high = None if clip_range_high is None else float(clip_range_high)
        self.clip_schedule = str(clip_schedule)
        self.loss_agg_mode = str(loss_agg_mode)
        self.horizon = int(horizon)
        self.conditions_cls = conditions_cls
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> None:
        """Freeze the π_old anchor (``segment.log_probs``) before the
        ``num_updates_per_batch`` loop, per ``old_logp_source``.

        - ``"rollout"`` (default): keep the rollout engine's emitted
          ``segment.log_probs`` as the anchor for ALL N updates (verl
          bypass-mode parity; the ratio then also carries the rollout-vs-train
          engine gap).
        - ``"replay"``: recompute π_old via a ``torch.no_grad`` ``stage.replay``
          at the **pre-update** weights and **overwrite** ``segment.log_probs``.
          For stages whose rollout decode is numerically far enough from
          teacher-forced replay that a nominally on-policy ratio lands outside
          a narrow clip range — cached bf16 decode vs full-sequence attention,
          amplified by CFG on AR image tokens — this removes the engine gap
          from the ratio. :meth:`recomputes_anchor` is True in this mode, so
          the train stack drives the hook per micro-slice at exactly the
          geometry training will replay at, making mini-batch 1's ratio 1
          rather than approximately 1.
        """
        if self.old_logp_source != "replay":
            return
        if segment.tokens is None or int(segment.tokens.shape[0]) == 0:
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            frozen = self.stage.replay(typed_conds, segment=segment, temperature=self.sampling_temperature)
        # Keep the replay's native (fp32) precision — do NOT downcast to whatever
        # dtype the engine emitted, so the anchor stays as close as possible to
        # new_logp's fp32 replay (mirrors FlowGRPO / DRPO).
        segment.log_probs = frozen.detach().cpu()

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        if segment.tokens is None or segment.lengths is None or segment.log_probs is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        new_logp = self.stage.replay(
            typed_conds, segment=segment, temperature=self.sampling_temperature
        )  # [total_tokens]
        # old_logp = the frozen π_old anchor established by prepare_segment:
        # the rollout log-prob by default, or a train-side replay under
        # old_logp_source='replay'. Either way it stays frozen across all
        # num_updates_per_batch steps (see the supports_multi_update comment).
        old_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)
        adv_per_token = self._expand_advantages_to_tokens(
            advantages, segment.lengths, dtype=new_logp.dtype, device=new_logp.device
        )

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        clip_high = (
            None
            if self.clip_range_high is None
            else _resolve_clip_range_from_schedule(self.clip_range_high, self.clip_schedule, training_progress)
        )
        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_per_token,
            clip_range=clip_range,
            clip_range_high=clip_high,
        )

        # Loss aggregation (match DRPO / verl loss_agg_mode):
        #  - "seq-mean-token-sum-norm" (Dr.GRPO/DAPO): per-seq token-SUM / horizon,
        #    then mean over sequences (length-UNbiased).
        #  - "seq-mean-token-mean" (ORIGINAL GRPO): per-seq token-MEAN, then mean
        #    over sequences (length-normalized, the standard-GRPO length bias).
        #  - "token-mean" (default): flat mean over all tokens.
        if self.loss_agg_mode in ("seq-mean-token-sum-norm", "seq-mean-token-mean") and segment.lengths is not None:
            parts = torch.split(loss_per_elem, segment.lengths.tolist())
            if self.loss_agg_mode == "seq-mean-token-sum-norm":
                loss = torch.stack([p.sum() for p in parts]).mean() / float(self.horizon)
            else:  # seq-mean-token-mean — guard 0-length responses (mean of empty = NaN)
                loss = torch.stack([p.mean() if p.numel() else p.new_zeros(()) for p in parts]).mean()
        else:
            loss = loss_per_elem.mean()
        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            **rollout_replay_logp_absdiff(new_logp, old_logp),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )

    @staticmethod
    def _expand_advantages_to_tokens(
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand per-sample ``advantages [B]`` to per-token ``[total_tokens]``.

        Each sample's advantage is repeated across its ``lengths``-defined
        token span so that token positions in segment ``k`` all see
        ``advantages[k]``. ``lengths`` comes from
        :attr:`Batch.lengths` on the segment (derived from the framework-
        managed cu_seqlens).
        """
        bs = int(advantages.shape[0])
        if int(lengths.shape[0]) != bs:
            raise ValueError(f"GRPO advantage expansion: advantages batch={bs} != lengths={int(lengths.shape[0])}")
        chunks: List[torch.Tensor] = []
        adv_cast = advantages.detach().to(dtype=dtype, device=device)
        for k in range(bs):
            n = int(lengths[k].item())
            if n > 0:
                chunks.append(adv_cast[k].expand(n))
        if not chunks:
            return torch.zeros(0, dtype=dtype, device=device)
        return torch.cat(chunks, dim=0)


__all__ = ["GRPO", "GRPOConfig"]
