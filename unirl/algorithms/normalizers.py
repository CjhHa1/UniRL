"""Pure tensor math helpers for advantage normalization."""

from typing import Any, Dict, List, Optional

import torch


def _finite_stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.numel() == 0:
        return values.new_zeros(()), values.new_ones(())
    mean = values.mean()
    std = values.std(unbiased=False) if values.numel() > 1 else values.new_ones(())
    return mean, std


def normalize_grouped(
    rewards: torch.Tensor,
    group_indices: List[List[int]],
    epsilon: float = 1e-8,
    clip_max: Optional[float] = None,
    trim_outliers_ratio: float = 0.0,
    use_global_std: bool = False,
) -> torch.Tensor:
    """Unified grouped normalization - single implementation for all group-based strategies.

    Normalizes finite rewards within specified groups, subtracting the group mean
    and dividing by the group (or global) standard deviation. Non-finite rewards
    receive zero advantage.

    Args:
        rewards: Reward values [N]
        group_indices: List of index lists, each defining a group
        epsilon: Small value for numerical stability
        clip_max: If set, clip advantages to [-clip_max, clip_max]
        trim_outliers_ratio: Ratio of samples to trim from each end when computing stats
        use_global_std: If True, use global std for all groups

    Returns:
        Advantages tensor [N] with per-group normalization
    """
    advantages = torch.zeros_like(rewards)
    finite_rewards = rewards[torch.isfinite(rewards)]
    _, batch_std = _finite_stats(finite_rewards) if use_global_std else (None, None)

    for indices in group_indices:
        if not indices:
            continue

        group_rewards = rewards[indices]
        group_finite = torch.isfinite(group_rewards)
        finite_group_rewards = group_rewards[group_finite]
        if finite_group_rewards.numel() == 0:
            continue

        if trim_outliers_ratio > 0 and finite_group_rewards.numel() > 2:
            sorted_rewards, _ = torch.sort(finite_group_rewards)
            trim_size = int(len(sorted_rewards) * trim_outliers_ratio)
            if trim_size > 0 and trim_size * 2 < len(sorted_rewards):
                trimmed = sorted_rewards[trim_size:-trim_size]
            else:
                trimmed = sorted_rewards
            group_mean, group_std = _finite_stats(trimmed)
        else:
            group_mean, group_std = _finite_stats(finite_group_rewards)

        if use_global_std:
            group_std = batch_std

        advantages[indices] = torch.where(
            group_finite,
            (group_rewards - group_mean) / (group_std + epsilon),
            torch.zeros_like(group_rewards),
        )

    if clip_max is not None:
        advantages = torch.clamp(advantages, -clip_max, clip_max)

    return advantages


def normalize_global(
    rewards: torch.Tensor,
    epsilon: float = 1e-8,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """Global normalization across all finite samples.

    Args:
        rewards: Reward values [N]
        epsilon: Small value for numerical stability
        clip_max: If set, clip advantages to [-clip_max, clip_max]

    Returns:
        Advantages tensor [N] with global normalization; non-finite rewards receive
        zero advantage.
    """
    finite = torch.isfinite(rewards)
    mean, std = _finite_stats(rewards[finite])
    advantages = torch.where(
        finite,
        (rewards - mean) / (std + epsilon),
        torch.zeros_like(rewards),
    )

    if clip_max is not None:
        advantages = torch.clamp(advantages, -clip_max, clip_max)

    return advantages


def _normalize_group_id(group_id: Any) -> Optional[str]:
    if group_id is None:
        return None
    text = str(group_id).strip()
    return text if text else None


def require_valid_group_ids(group_ids: List[str]) -> List[str]:
    normalized: List[str] = []
    for sample_idx, raw_group_id in enumerate(group_ids):
        gid = _normalize_group_id(raw_group_id)
        if gid is None:
            raise ValueError(
                "adv_normalization_scope='group' requires a non-empty group_id "
                f"for every sample. Invalid group_id at sample_idx={sample_idx}."
            )
        normalized.append(gid)
    return normalized


def build_group_index_map(group_ids: List[str]) -> Dict[str, List[int]]:
    ordered_groups: Dict[str, List[int]] = {}
    for sample_idx, raw_group_id in enumerate(group_ids):
        gid = _normalize_group_id(raw_group_id)
        if gid is None:
            continue
        ordered_groups.setdefault(gid, []).append(sample_idx)
    return ordered_groups


def require_expected_group_sizes(
    group_index_map: Dict[str, List[int]],
    samples_per_prompt: int,
) -> List[List[int]]:
    expected = max(1, int(samples_per_prompt))
    invalid = [(gid, len(idxs)) for gid, idxs in group_index_map.items() if len(idxs) != expected]
    if invalid:
        formatted = ", ".join(f"{gid!r}:{size}" for gid, size in invalid[:5])
        if len(invalid) > 5:
            formatted = f"{formatted}, ..."
        raise ValueError(
            "adv_normalization_scope='group' requires every sample group to contain exactly "
            f"samples_per_prompt={expected} samples. Invalid group sizes: {formatted}."
        )
    return list(group_index_map.values())
