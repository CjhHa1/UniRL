"""Unit tests for BaseTrainer.run_eval (issue #182) — no GPU / Ray."""

from __future__ import annotations

from typing import Optional

from unirl.trainer.base import BaseTrainer


class _StubTrainer(BaseTrainer):
    """Minimal subclass that records the run_eval call sequence."""

    def __init__(self) -> None:
        # Bypass BaseTrainer.__init__ (needs full Hydra/Ray wiring).
        self.calls: list[str] = []
        self._load_dir: Optional[str] = None
        self._eval_step: Optional[int] = None

    def maybe_load_checkpoint(self, load_dir: Optional[str], *, num_rollouts=None) -> int:  # type: ignore[override]
        self.calls.append("maybe_load_checkpoint")
        self._load_dir = load_dir
        return 0

    def _init_wandb(self, *, num_rollouts=None, extra=None) -> None:  # type: ignore[override]
        self.calls.append(f"_init_wandb:{num_rollouts}")

    def _finish_wandb(self) -> None:  # type: ignore[override]
        self.calls.append("_finish_wandb")

    def evaluate(self, step: int) -> float:
        self.calls.append("evaluate")
        self._eval_step = step
        return 0.42


def test_run_eval_calls_evaluate_and_finishes_wandb() -> None:
    trainer = _StubTrainer()
    score = trainer.run_eval(load_dir="/tmp/ckpt", step=7)
    assert score == 0.42
    assert trainer._load_dir == "/tmp/ckpt"
    assert trainer._eval_step == 7
    assert trainer.calls == [
        "maybe_load_checkpoint",
        "_init_wandb:0",
        "evaluate",
        "_finish_wandb",
    ]


def test_run_eval_finishes_wandb_even_on_evaluate_error() -> None:
    class _Boom(_StubTrainer):
        def evaluate(self, step: int) -> float:
            self.calls.append("evaluate")
            raise RuntimeError("boom")

    trainer = _Boom()
    try:
        trainer.run_eval()
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    assert trainer.calls[-1] == "_finish_wandb"
