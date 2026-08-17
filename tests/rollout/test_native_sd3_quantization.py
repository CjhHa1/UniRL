from __future__ import annotations

import torch
from torch import nn

from unirl.rollout.engine.native_sd3.config import NativeSD3EngineConfig
from unirl.rollout.engine.native_sd3.quantization import (
    FP8Controller,
    RoutedTransformer,
    convert_transformer_for_fp8,
)


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.register_buffer("position_table", torch.arange(4))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.proj(values)


def test_disabled_fp8_path_preserves_parameter_names_and_bf16_forward() -> None:
    config = NativeSD3EngineConfig(fp8_enabled=False, compile_model=False)
    controller = FP8Controller(config)
    model = _TinyTransformer().to(torch.bfloat16)
    expected = model(torch.ones(2, 4, dtype=torch.bfloat16))

    targets, report = convert_transformer_for_fp8(model, config=config, controller=controller)
    routed = RoutedTransformer(model, controller)
    with controller.rollout(mode="bf16", total_steps=1):
        actual = routed(torch.ones(2, 4, dtype=torch.bfloat16))

    assert report.replaced == ()
    assert set(targets) == {"proj.weight", "proj.bias", "position_table"}
    assert targets["position_table"] is model.position_table
    assert torch.equal(actual, expected)


def test_disabled_engine_rejects_fp8_request() -> None:
    controller = FP8Controller(NativeSD3EngineConfig(fp8_enabled=False, compile_model=False))
    try:
        with controller.rollout(mode="fp8", total_steps=1):
            pass
    except RuntimeError as exc:
        assert "fp8_enabled=false" in str(exc)
    else:
        raise AssertionError("expected disabled FP8 controller to reject an FP8 request")
