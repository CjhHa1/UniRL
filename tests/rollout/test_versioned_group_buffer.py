from __future__ import annotations

import torch

from unirl.rollout.async_runtime import VersionedGroupBuffer
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.segments.text import TextSegment


def _group(group_id: str, rewards: tuple[float, ...] = (0.0, 1.0)) -> RolloutResp:
    sample_ids = [f"{group_id}/s{i}" for i in range(len(rewards))]
    return RolloutResp(
        tracks={
            "ar": RolloutTrack(
                sample_ids=sample_ids,
                parent_ids=[group_id] * len(sample_ids),
                rewards=torch.tensor(rewards),
            )
        }
    )


def test_drain_freshest_orders_by_generation_id() -> None:
    buffer = VersionedGroupBuffer()
    buffer.put(_group("g0"), weight_version=0, gen_id=0)
    buffer.put(_group("g2"), weight_version=0, gen_id=2)
    buffer.put(_group("g1"), weight_version=0, gen_id=1)

    picked = buffer.drain_freshest(2)

    assert picked is not None
    assert [item.gen_id for item in picked] == [2, 1]
    assert buffer.size() == 1


def test_insufficient_drain_does_not_consume_groups() -> None:
    buffer = VersionedGroupBuffer()
    buffer.put(_group("g0"), weight_version=0, gen_id=0)

    assert buffer.drain_freshest(2) is None
    assert buffer.size() == 1


def test_stale_groups_are_evicted_before_drain() -> None:
    buffer = VersionedGroupBuffer()
    buffer.put(_group("old"), weight_version=0, gen_id=0)
    buffer.put(_group("fresh"), weight_version=2, gen_id=1)

    picked = buffer.drain_freshest(
        1,
        current_version=2,
        max_staleness=0,
    )

    assert picked is not None
    assert [item.resp.tracks["ar"].parent_ids[0] for item in picked] == ["fresh"]
    assert buffer.size() == 0


def test_signal_filter_is_inert_by_default_and_opt_in() -> None:
    zero_signal = _group("flat", rewards=(1.0, 1.0))

    default_buffer = VersionedGroupBuffer()
    default_buffer.put(zero_signal, weight_version=0, gen_id=0)
    assert default_buffer.drain_freshest(1) is not None

    filtered_buffer = VersionedGroupBuffer()
    filtered_buffer.put(zero_signal, weight_version=0, gen_id=0)
    picked = filtered_buffer.drain_freshest(
        1,
        has_signal=lambda resp: bool(resp.tracks["ar"].rewards.std(unbiased=False).item() > 0),
    )
    assert picked is None
    assert filtered_buffer.size() == 0


def test_rollout_resp_split_concat_preserves_track_tree() -> None:
    root_ids = ["g0/s0", "g0/s1", "g1/s0", "g1/s1"]
    resp = RolloutResp(
        tracks={
            "think": RolloutTrack(
                sample_ids=root_ids,
                parent_ids=["g0", "g0", "g1", "g1"],
                rewards=torch.tensor([0.0, 1.0, 2.0, 3.0]),
            ),
            "image": RolloutTrack(
                sample_ids=[f"{sample_id}/c0" for sample_id in root_ids],
                parent_ids=root_ids,
                parent_track="think",
                rewards=torch.tensor([10.0, 11.0, 12.0, 13.0]),
            ),
        }
    )

    shards = resp.split()
    restored = RolloutResp.concat(shards)

    assert len(shards) == 2
    assert set(restored.tracks) == {"think", "image"}
    assert restored.tracks["think"].sample_ids == root_ids
    assert restored.tracks["image"].parent_ids == root_ids
    assert restored.tracks["image"].parent_track == "think"
    assert torch.equal(
        restored.tracks["think"].rewards,
        resp.tracks["think"].rewards,
    )
    assert torch.equal(
        restored.tracks["image"].rewards,
        resp.tracks["image"].rewards,
    )


def test_split_concat_preserves_packed_text_segment() -> None:
    tokens = [
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([3], dtype=torch.long),
        torch.tensor([4, 5, 6], dtype=torch.long),
        torch.tensor([7, 8], dtype=torch.long),
    ]
    log_probs = [
        torch.tensor([-0.1, -0.2]),
        torch.tensor([-0.3]),
        torch.tensor([-0.4, -0.5, -0.6]),
        torch.tensor([-0.7, -0.8]),
    ]
    loss_mask = [
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0, 0.0, 1.0]),
        torch.tensor([1.0, 1.0]),
    ]
    segment = TextSegment.pack(
        tokens=tokens,
        log_probs=log_probs,
        loss_mask=loss_mask,
    )
    resp = RolloutResp(
        tracks={
            "ar": RolloutTrack(
                sample_ids=["g0/s0", "g0/s1", "g1/s0", "g1/s1"],
                parent_ids=["g0", "g0", "g1", "g1"],
                segment=segment,
            )
        }
    )

    restored = RolloutResp.concat(resp.split())
    restored_segment = restored.tracks["ar"].segment

    assert isinstance(restored_segment, TextSegment)
    assert torch.equal(restored_segment.tokens, segment.tokens)
    assert torch.equal(restored_segment.log_probs, segment.log_probs)
    assert torch.equal(restored_segment.loss_mask, segment.loss_mask)
    assert torch.equal(restored_segment.cu_seqlens, segment.cu_seqlens)
