from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from unirl.types.sample import Sample

RolloutFilter = Callable[[List["Sample"], int], List["Sample"]]


def _is_incomplete(sample: "Sample") -> bool:
    if not sample.parts:
        return True
    return sample.parts[-1].harness_status not in ("completed", "failed")


def identity(samples: List["Sample"], current_version: int) -> List["Sample"]:
    del current_version
    return samples


def chain(*filters: RolloutFilter) -> RolloutFilter:
    def apply(samples: List["Sample"], current_version: int) -> List["Sample"]:
        kept = samples
        for filter_fn in filters:
            kept = filter_fn(kept, current_version)
            if not kept:
                break
        return kept

    return apply


def drop_incomplete(samples: List["Sample"], current_version: int) -> List["Sample"]:
    del current_version
    return [] if any(_is_incomplete(sample) for sample in samples) else samples


def keep_within_lag(max_lag: int) -> RolloutFilter:
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError(f"max_lag must be non-negative; got {max_lag}")

    def apply(samples: List["Sample"], current_version: int) -> List["Sample"]:
        versions = []
        for sample in samples:
            for part in sample.gen_parts():
                if part.weight_version is None:
                    raise RuntimeError("generated rollout has no weight version")
                versions.append(int(part.weight_version))
        if versions and max(versions) > current_version:
            raise RuntimeError("rollout has a future weight version")
        if versions and current_version - min(versions) > max_lag:
            return []
        return samples

    return apply


__all__ = ["RolloutFilter", "chain", "drop_incomplete", "identity", "keep_within_lag"]
