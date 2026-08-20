from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

_RUNTIME_MODULE = "unirl.rollout.engine.vllm_omni.patches.runtime"


def _package(name: str) -> ModuleType:
    package = ModuleType(name)
    package.__path__ = []
    return package


@pytest.fixture
def runtime(monkeypatch):
    class LoRALayerWeights:
        def __init__(
            self,
            module_name: str,
            rank: int,
            lora_alpha: int,
            lora_a: torch.Tensor,
            lora_b: torch.Tensor,
            scaling: float = 1.0,
        ) -> None:
            self.module_name = module_name
            self.rank = rank
            self.lora_alpha = lora_alpha
            self.lora_a = lora_a
            self.lora_b = lora_b
            self.scaling = scaling

    class PackedLoRALayerWeights(LoRALayerWeights):
        def __init__(self, module_name, rank, lora_alphas, lora_a, lora_b, scaling):
            self.module_name = module_name
            self.rank = rank
            self.lora_alphas = lora_alphas
            self.lora_a = lora_a
            self.lora_b = lora_b
            self.scaling = scaling

    class DiffusionLoRAManager:
        def _get_lora_weights(self, _lora_model, full_module_name):
            return getattr(self, "stock_weights", {}).get(full_module_name)

    class HunyuanImage3Pipeline:
        pass

    class LoRAModel:
        pass

    class PEFTHelper:
        pass

    class OmniLoRARequest:
        pass

    modules = {
        "vllm": _package("vllm"),
        "vllm.lora": _package("vllm.lora"),
        "vllm.lora.lora_model": ModuleType("vllm.lora.lora_model"),
        "vllm.lora.lora_weights": ModuleType("vllm.lora.lora_weights"),
        "vllm.lora.peft_helper": ModuleType("vllm.lora.peft_helper"),
        "vllm.lora.utils": ModuleType("vllm.lora.utils"),
        "vllm_omni": _package("vllm_omni"),
        "vllm_omni.diffusion": _package("vllm_omni.diffusion"),
        "vllm_omni.diffusion.lora": _package("vllm_omni.diffusion.lora"),
        "vllm_omni.diffusion.lora.manager": ModuleType("vllm_omni.diffusion.lora.manager"),
        "vllm_omni.diffusion.models": _package("vllm_omni.diffusion.models"),
        "vllm_omni.diffusion.models.hunyuan_image3": _package("vllm_omni.diffusion.models.hunyuan_image3"),
        "vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3": ModuleType(
            "vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3"
        ),
        "vllm_omni.lora": _package("vllm_omni.lora"),
        "vllm_omni.lora.request": ModuleType("vllm_omni.lora.request"),
    }
    modules["vllm.lora.lora_model"].LoRAModel = LoRAModel
    modules["vllm.lora.lora_weights"].LoRALayerWeights = LoRALayerWeights
    modules["vllm.lora.lora_weights"].PackedLoRALayerWeights = PackedLoRALayerWeights
    modules["vllm.lora.peft_helper"].PEFTHelper = PEFTHelper
    modules["vllm.lora.utils"].get_adapter_absolute_path = lambda path: path
    modules["vllm_omni.diffusion.lora.manager"].DiffusionLoRAManager = DiffusionLoRAManager
    modules["vllm_omni.diffusion.lora.manager"].logger = SimpleNamespace(
        debug=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    modules[
        "vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3"
    ].HunyuanImage3Pipeline = HunyuanImage3Pipeline
    modules["vllm_omni.lora.request"].LoRARequest = OmniLoRARequest

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, _RUNTIME_MODULE, raising=False)

    imported = importlib.import_module(_RUNTIME_MODULE)
    yield SimpleNamespace(
        module=imported,
        manager_cls=DiffusionLoRAManager,
        hi3_pipeline_cls=HunyuanImage3Pipeline,
        lora_cls=LoRALayerWeights,
        packed_cls=PackedLoRALayerWeights,
    )
    sys.modules.pop(_RUNTIME_MODULE, None)


def _lora_model(weights_by_name):
    return SimpleNamespace(get_lora=weights_by_name.get)


def test_hi3_alias_deinterleaves_fused_qkv(runtime):
    full_name = "transformer.layers.0.self_attn.qkv_proj"
    alias = "model.layers.0.self_attn.qkv_proj"
    lora_a = torch.tensor([[10.0]])
    weights = runtime.lora_cls(
        module_name=alias,
        rank=1,
        lora_alpha=1,
        lora_a=lora_a,
        lora_b=torch.arange(8, dtype=torch.float32).reshape(8, 1),
    )
    base_layer = SimpleNamespace(output_sizes=[4, 2, 2], head_size=1, total_num_kv_heads=2)
    manager = runtime.manager_cls()
    manager.pipeline = runtime.hi3_pipeline_cls()
    manager._lora_modules = {full_name: SimpleNamespace(base_layer=base_layer)}

    runtime.module.patch_dit_hi3_lora_weights()
    result = manager._get_lora_weights(_lora_model({alias: weights}), full_name)

    assert isinstance(result, runtime.packed_cls)
    assert [part.flatten().tolist() for part in result.lora_b] == [[0.0, 1.0, 4.0, 5.0], [2.0, 6.0], [3.0, 7.0]]
    assert result.lora_a == [lora_a, lora_a, lora_a]


def test_hi3_alias_leaves_non_qkv_weight_plain(runtime):
    full_name = "transformer.layers.0.self_attn.o_proj"
    alias = "model.layers.0.self_attn.o_proj"
    weights = runtime.lora_cls(alias, 1, 1, torch.ones(1, 1), torch.ones(1, 1))
    manager = runtime.manager_cls()
    manager.pipeline = runtime.hi3_pipeline_cls()
    manager._lora_modules = {}

    runtime.module.patch_dit_hi3_lora_weights()

    assert manager._get_lora_weights(_lora_model({alias: weights}), full_name) is weights


def test_non_hi3_qkv_keeps_stock_layout(runtime):
    full_name = "transformer.layers.0.self_attn.qkv_proj"
    weights = runtime.lora_cls(full_name, 1, 1, torch.ones(1, 1), torch.arange(8).reshape(8, 1))
    manager = runtime.manager_cls()
    manager.pipeline = object()
    manager.stock_weights = {full_name: weights}

    runtime.module.patch_dit_hi3_lora_weights()

    assert manager._get_lora_weights(_lora_model({}), full_name) is weights


def test_hi3_qkv_fails_closed_when_layout_is_unknown(runtime):
    full_name = "transformer.layers.0.self_attn.qkv_proj"
    alias = "model.layers.0.self_attn.qkv_proj"
    weights = runtime.lora_cls(alias, 1, 1, torch.ones(1, 1), torch.arange(8).reshape(8, 1))
    manager = runtime.manager_cls()
    manager.pipeline = runtime.hi3_pipeline_cls()
    manager._lora_modules = {}

    runtime.module.patch_dit_hi3_lora_weights()

    with pytest.raises(RuntimeError, match="Refusing to install HI3 fused-qkv LoRA"):
        manager._get_lora_weights(_lora_model({alias: weights}), full_name)


def test_hi3_patch_is_idempotent(runtime):
    runtime.module.patch_dit_hi3_lora_weights()
    wrapped = runtime.manager_cls._get_lora_weights

    runtime.module.patch_dit_hi3_lora_weights()

    assert runtime.manager_cls._get_lora_weights is wrapped
