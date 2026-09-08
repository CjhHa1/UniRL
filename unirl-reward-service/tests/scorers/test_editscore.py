from __future__ import annotations

import math
import sys
import types
from collections.abc import Callable

import pytest
from PIL import Image
from reward_service.scorers.base import ScoreItem
from reward_service.scorers.editscore import EditScoreScorer, _engine_kwargs


def _install_parser(monkeypatch: pytest.MonkeyPatch, parser: Callable) -> None:
    package = types.ModuleType("editscore")
    package.__path__ = []
    utils = types.ModuleType("editscore.utils")
    utils.mllm_output_to_dict = parser
    monkeypatch.setitem(sys.modules, "editscore", package)
    monkeypatch.setitem(sys.modules, "editscore.utils", utils)


def _bare_scorer(model, *, batched: bool = True, num_pass: int = 1) -> EditScoreScorer:
    scorer = object.__new__(EditScoreScorer)
    scorer.supports_offload = False
    scorer._max_image_side = None
    scorer._use_batch_inference = batched
    scorer.es = types.SimpleNamespace(
        model=model,
        SC_prompt="SC <instruction>",
        PQ_prompt="PQ",
        num_pass=num_pass,
        score_range=25,
        seed=42,
    )
    return scorer


class _MessageModel:
    @staticmethod
    def prepare_input(images, prompt):
        return ("sc" if isinstance(images, list) else "pq", prompt)


def _items(count: int = 1) -> list[ScoreItem]:
    image = Image.new("RGB", (8, 8))
    return [ScoreItem(history=[(f"edit-{index}", image), (f"edit-{index}", image)]) for index in range(count)]


def _valid_parser(text, **_kwargs):
    return {"score": [20, 25] if text == "sc-good" else [20, 16]}


def test_constructor_rejects_invalid_numeric_config_before_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNIRL_SCORER_BOOT_OFFLOADED", "1")
    with pytest.raises(ValueError, match="score_range"):
        EditScoreScorer(model_name_or_path="fake", score_range=0)


def test_boot_offloaded_requires_sleep_before_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNIRL_SCORER_BOOT_OFFLOADED", "1")
    with pytest.raises(ValueError, match="enable_sleep_mode"):
        EditScoreScorer(model_name_or_path="fake")


def test_sleep_mode_forces_vllm_caches_off(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def original_llm(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace()

    package = types.ModuleType("editscore")
    package.__path__ = []
    tools = types.ModuleType("editscore.mllm_tools")
    tools.__path__ = []
    backbone = types.ModuleType("editscore.mllm_tools.qwen3vl_vllm")
    backbone.LLM = original_llm

    class FakeEditScore:
        def __init__(self, **kwargs):
            self.model = types.SimpleNamespace(
                model=backbone.LLM(
                    enable_prefix_caching=True,
                    mm_processor_cache_gb=4,
                )
            )
            self.num_pass = kwargs["num_pass"]
            self.score_range = kwargs["score_range"]
            self.seed = kwargs["seed"]

    package.EditScore = FakeEditScore
    monkeypatch.setitem(sys.modules, "editscore", package)
    monkeypatch.setitem(sys.modules, "editscore.mllm_tools", tools)
    monkeypatch.setitem(sys.modules, "editscore.mllm_tools.qwen3vl_vllm", backbone)

    scorer = EditScoreScorer(
        model_name_or_path="fake",
        enable_sleep_mode=True,
    )

    assert captured == {
        "enable_prefix_caching": False,
        "enable_sleep_mode": True,
        "mm_processor_cache_gb": 0,
    }
    assert backbone.LLM is original_llm
    assert scorer.supports_offload is True


def test_extra_kwargs_cannot_duplicate_lifecycle_options() -> None:
    with pytest.raises(ValueError, match="enable_sleep_mode"):
        EditScoreScorer(
            model_name_or_path="fake",
            extra_llm_kwargs={"enable_sleep_mode": True},
        )


def test_extra_gpu_utilization_is_allowed_without_dedicated_value() -> None:
    assert _engine_kwargs(
        extra_llm_kwargs={"gpu_memory_utilization": 0.4},
        gpu_memory_utilization=None,
        enable_sleep_mode=False,
    ) == {"gpu_memory_utilization": 0.4}


def test_batch_retries_only_invalid_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, int]] = []
    sc_attempt = 0

    class Model(_MessageModel):
        def batch_inference(self, messages, seed):
            nonlocal sc_attempt
            kind = messages[0][0]
            calls.append((kind, len(messages), seed))
            if kind == "sc":
                sc_attempt += 1
                return ["bad", "sc-good"] if sc_attempt == 1 else ["sc-good"] * len(messages)
            return ["pq-good"] * len(messages)

    def parser(text, **kwargs):
        assert kwargs["give_up_parsing"] is False
        return False if text == "bad" else _valid_parser(text)

    _install_parser(monkeypatch, parser)
    scorer = _bare_scorer(Model())

    results = scorer.score(_items(2))

    assert calls == [
        ("sc", 2, 42),
        ("pq", 2, 42),
        ("sc", 1, 43),
        ("pq", 1, 43),
    ]
    assert results[0]["overall"] == pytest.approx(math.sqrt(8.0 * 6.4))
    assert results[0]["prompt_following"] == pytest.approx(8.0)
    assert results[0]["consistency"] == pytest.approx(10.0)
    assert results[0]["perceptual_quality"] == pytest.approx(6.4)


@pytest.mark.parametrize(
    "parsed",
    [
        False,
        {"score": []},
        {"score": [-5, -5]},
        {"score": [float("inf"), 1]},
        {"score": [True, 1]},
        {"score": [26, 1]},
    ],
)
def test_invalid_judge_output_becomes_nan(
    monkeypatch: pytest.MonkeyPatch,
    parsed,
) -> None:
    class Model(_MessageModel):
        @staticmethod
        def batch_inference(messages, seed):
            return ["invalid"] * len(messages)

    _install_parser(monkeypatch, lambda *_args, **_kwargs: parsed)
    scorer = _bare_scorer(Model())

    [result] = scorer.score(_items())

    assert all(math.isnan(value) for value in result.values())


@pytest.mark.parametrize("batched", [True, False])
def test_engine_failure_propagates(monkeypatch: pytest.MonkeyPatch, batched: bool) -> None:
    class EngineDeadError(RuntimeError):
        pass

    class Model(_MessageModel):
        @staticmethod
        def batch_inference(messages, seed):
            raise EngineDeadError("engine died")

        @staticmethod
        def inference(message, seed):
            raise EngineDeadError("engine died")

    _install_parser(monkeypatch, _valid_parser)
    scorer = _bare_scorer(Model(), batched=batched)

    with pytest.raises(EngineDeadError, match="engine died"):
        scorer.score(_items())


def test_sequential_mode_uses_same_parser_and_metric_order(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model(_MessageModel):
        @staticmethod
        def inference(message, seed):
            return "sc-good" if message[0] == "sc" else "pq-good"

    _install_parser(monkeypatch, _valid_parser)
    scorer = _bare_scorer(Model(), batched=False)

    [result] = scorer.score(_items())

    assert list(result) == ["overall", "prompt_following", "consistency", "perceptual_quality"]
    assert result["overall"] == pytest.approx(math.sqrt(8.0 * 6.4))
