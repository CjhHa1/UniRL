from __future__ import annotations

import json
from pathlib import Path

from unirl.tools.solrl_rank_metrics import compare


def _write_trace(path: Path, rewards: list[float]) -> None:
    path.mkdir()
    candidates = []
    for index, reward in enumerate(rewards):
        selection = "top" if index == 3 else ("bottom" if index == 0 else None)
        candidates.append(
            {
                "sample_id": f"p0/{index}",
                "reward": reward,
                "selection": selection,
            }
        )
    (path / "rollout_000000.json").write_text(
        json.dumps(
            {
                "rollout_id": 0,
                "groups": [{"group_id": "p0", "candidates": candidates}],
            }
        ),
        encoding="utf-8",
    )


def test_identical_traces_have_perfect_rank_metrics(tmp_path: Path) -> None:
    proxy, oracle = tmp_path / "proxy", tmp_path / "oracle"
    _write_trace(proxy, [0.0, 0.2, 0.4, 0.6])
    _write_trace(oracle, [0.0, 0.2, 0.4, 0.6])
    metrics = compare(proxy, oracle)
    assert metrics["spearman_mean"] == 1.0
    assert metrics["kendall_tau_b_mean"] == 1.0
    assert metrics["top_overlap_mean"] == 1.0
    assert metrics["bottom_overlap_mean"] == 1.0
    assert metrics["top_selected_true_reward_gap"] == 0.0
