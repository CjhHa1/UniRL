"""T2AV composite reward — weighted blend of video + audio scorers."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse

from .registry import resolve_builtin_reward_scorer_class, resolve_builtin_reward_spec_class

_SKIP_INNER_OVERRIDE = frozenset({"weights", "scorers"})


def _plain_mapping(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return dict(value)


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
        "Add videopickscore / videoalign / videoclipdelta, or set scorers.imagebind.mode to "
        "'text_video' or 'all'. clap and imagebind mode='audio_video' do not cover this."
    )


class T2AVCompositeScorer(RewardBackend):
    """Weighted blend of inner reward scorers for T2AV (video + audio)."""

    input_kind = "video"

    def __init__(self, *, config: "T2AVCompositeSpec", base_device: str) -> None:
        super().__init__(model_name="t2av_composite", batch_size=config.batch_size)
        self.weights: Dict[str, float] = dict(config.weights or {})
        if not self.weights:
            raise ValueError("T2AVCompositeScorer requires a non-empty `weights` dict (scorer_name -> weight).")

        named_overrides = _plain_mapping(config.scorers)
        extra = sorted(set(named_overrides) - set(self.weights))
        if extra:
            raise ValueError(f"T2AVCompositeScorer: scorers keys {extra} are not in weights {sorted(self.weights)}.")

        self._scorers: Dict[str, RewardBackend] = {}
        for name in self.weights:
            inner_cls = resolve_builtin_reward_scorer_class(name)
            inner_spec_cls = resolve_builtin_reward_spec_class(name)
            inner_spec = inner_spec_cls()
            overrides: Dict[str, Any] = {}
            for f in dataclasses.fields(config):
                if f.name in _SKIP_INNER_OVERRIDE or not hasattr(inner_spec, f.name):
                    continue
                overrides[f.name] = getattr(config, f.name)
            per_scorer = _plain_mapping(named_overrides.get(name))
            unknown = sorted(k for k in per_scorer if not hasattr(inner_spec, k))
            if unknown:
                raise ValueError(
                    f"T2AVCompositeScorer: scorers[{name!r}] has fields {unknown} not on {inner_spec_cls.__name__}."
                )
            overrides.update(per_scorer)
            if overrides:
                inner_spec = dataclasses.replace(inner_spec, **overrides)
            self._scorers[name] = inner_cls(config=inner_spec, base_device=base_device)

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
    # Copied onto inner specs that declare it (videopickscore). "first" keeps
    # historical behaviour; "middle" avoids scoring a blank opening frame.
    frame_selection: str = "first"
    weights: Dict[str, float] = field(default_factory=lambda: {"videopickscore": 0.5, "clap": 0.5})
    # Optional per-name inner-spec overrides (e.g. imagebind: {mode: all}).
    # Unknown keys are rejected against that inner spec; not an allow-list.
    scorers: Dict[str, Dict[str, Any]] = field(default_factory=dict)


__all__ = ["T2AVCompositeScorer", "T2AVCompositeSpec"]
