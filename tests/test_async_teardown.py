from unittest.mock import Mock

import pytest

from unirl.trainer.base import BaseTrainer


def _trainer() -> BaseTrainer:
    trainer = object.__new__(BaseTrainer)
    trainer._finish_wandb = Mock()
    return trainer


def test_finish_after_drain_runs_drain_then_finish() -> None:
    trainer = _trainer()
    drain = Mock()

    trainer._finish_after_drain(drain)

    drain.assert_called_once_with()
    trainer._finish_wandb.assert_called_once_with()


def test_finish_after_drain_propagates_clean_exit_failure() -> None:
    trainer = _trainer()
    drain_error = RuntimeError("drain failed")

    with pytest.raises(RuntimeError) as raised:
        trainer._finish_after_drain(Mock(side_effect=drain_error))

    assert raised.value is drain_error
    trainer._finish_wandb.assert_called_once_with()


def test_finish_after_drain_preserves_primary_failure() -> None:
    trainer = _trainer()
    primary_error = ValueError("training failed")
    drain_error = RuntimeError("drain failed")

    with pytest.raises(ValueError) as raised:
        try:
            raise primary_error
        except ValueError:
            trainer._finish_after_drain(Mock(side_effect=drain_error))
            raise

    assert raised.value is primary_error
    trainer._finish_wandb.assert_called_once_with()


def test_finish_after_drain_does_not_swallow_keyboard_interrupt() -> None:
    trainer = _trainer()

    with pytest.raises(KeyboardInterrupt):
        trainer._finish_after_drain(Mock(side_effect=KeyboardInterrupt))

    trainer._finish_wandb.assert_called_once_with()
