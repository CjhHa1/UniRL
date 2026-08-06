"""Group-id helpers for advantage normalization scopes."""

from typing import Any, Dict, List, Optional


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
