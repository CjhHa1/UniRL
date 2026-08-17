from __future__ import annotations

from contextlib import contextmanager

from unirl.algorithms.diffusionnft import DiffusionNFT


class _Stage:
    pass


class _EMA:
    @contextmanager
    def use_shadow(self):
        yield


def test_positive_kl_requires_backend_reference_model() -> None:
    try:
        DiffusionNFT(
            params=object(),
            stage=_Stage(),
            nft_lora_policy=_EMA(),
            kl_coef=1.0e-4,
        )
    except TypeError as exc:
        assert "backend.model" in str(exc)
    else:
        raise AssertionError("expected KL-enabled DiffusionNFT to require a reference model")
