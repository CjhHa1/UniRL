"""T2AV composite reward — weighted blend of video + audio scorers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse


def _require_prompt_video_term(weights: Dict[str, float], scorers: Dict[str, RewardBackend]) -> None:
    covering = [
        name for name, weight in weights.items() if float(weight) != 0.0 and scorers[name].covers_prompt_video()
    ]
    if covering:
        return
    details = ", ".join(
        f"{name}(weight={weights[name]}, covers_prompt_video={scorers[name].covers_prompt_video()})" for name in weights
    )
    raise ValueError(
        "T2AVCompositeScorer: no positive-weight inner scorer relates the prompt to the video. "
        f"Current mix: {details}. "
        "Add videopickscore / videoalign / videoclipdelta, or nest imagebind with mode "
        "'text_video' or 'all'. clap and imagebind mode='audio_video' do not cover this."
    )


class T2AVCompositeScorer(RewardBackend):
    """Weighted blend of Hydra-instantiated inner reward scorers for T2AV."""

    input_kind = "video"

    def __init__(self, *, config: "T2AVCompositeSpec", base_device: str) -> None:
        super().__init__(model_name="t2av_composite", batch_size=config.batch_size)
        del base_device
        self.weights: Dict[str, float] = dict(config.weights or {})
        if not config.scorers:
            raise ValueError(
                "T2AVCompositeScorer requires a non-empty `scorers` mapping (name -> RewardBackend), "
                "the same nested `_target_` pattern as PerDomainRewardScorer."
            )
        if not self.weights:
            raise ValueError("T2AVCompositeScorer requires a non-empty `weights` dict (scorer_name -> weight).")

        for name, scorer in config.scorers.items():
            if not isinstance(scorer, RewardBackend):
                raise TypeError(
                    f"T2AVCompositeScorer: scorers[{name!r}] is {type(scorer).__name__}, not a "
                    "RewardBackend — each entry needs its own `_target_` for Hydra to instantiate."
                )

        weight_names = set(self.weights)
        scorer_names = set(config.scorers)
        if weight_names != scorer_names:
            missing = sorted(weight_names - scorer_names)
            extra = sorted(scorer_names - weight_names)
            raise ValueError(
                "T2AVCompositeScorer: `weights` and `scorers` must use the same keys "
                f"(weights missing scorers={missing}, scorers missing weights={extra})."
            )

        self._scorers: Dict[str, RewardBackend] = dict(config.scorers)
        _require_prompt_video_term(self.weights, self._scorers)

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        bs = request.batch_size
        try:
            import torch

            component_rewards: Dict[str, List[float]] = {}
            total = torch.zeros(bs, dtype=torch.float32)
            for name, scorer in self._scorers.items():
                resp = scorer.compute_rewards(request)
                comp = torch.tensor(list(resp.rewards), dtype=torch.float32)
                if comp.numel() != bs:
                    raise RuntimeError(
                        f"T2AVCompositeScorer: inner scorer {name!r} returned {comp.numel()} rewards "
                        f"for a batch of {bs}."
                    )
                component_rewards[name] = comp.tolist()
                total = total + float(self.weights[name]) * comp

            return RewardResponse(
                rewards=total.tolist(),
                component_rewards=component_rewards,
                successes=[True] * bs,
                errors=[None] * bs,
                compute_time=time.time() - start,
            )
        except Exception as e:
            return RewardResponse(
                rewards=[0.0] * bs,
                successes=[False] * bs,
                errors=[str(e)] * bs,
                compute_time=time.time() - start,
            )

    @property
    def preferred_input_kind(self) -> str:
        return self.input_kind

    def covers_prompt_video(self) -> bool:
        return True

    def is_available(self) -> bool:
        return all(s.is_available() for s in self._scorers.values())

    def offload(self) -> None:
        for s in self._scorers.values():
            s.offload()

    def onload(self) -> None:
        for s in self._scorers.values():
            s.onload()

    def dispose(self) -> None:
        for s in self._scorers.values():
            s.dispose()


@dataclass
class T2AVCompositeSpec(BaseRewardComponentSpec):
    """Typed config for the T2AV composite reward."""

    batch_size: int = 8
    device: str = "auto"
    weights: Dict[str, float] = field(default_factory=dict)
    # Hydra-instantiated inner backends (same nested `_target_` pattern as PerDomainSpec).
    # Inner fields such as mode / frame_selection / model_id live on those specs.
    scorers: Dict[str, RewardBackend] = field(default_factory=dict)


__all__ = ["T2AVCompositeScorer", "T2AVCompositeSpec"]
