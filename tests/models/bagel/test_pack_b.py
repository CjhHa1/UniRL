import sys
from types import ModuleType, SimpleNamespace

import torch

from unirl.models.bagel.ar import BagelARStage, BagelARStep
from unirl.models.bagel.conditions import BagelARConditions, BagelDiffusionConditions
from unirl.models.bagel.diffusion import BagelDiffusionStage, BagelDiffusionStep
from unirl.sde.kernels import FlowSDEStrategy
from unirl.types.sampling import ARSamplingParams


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {index: None for index in range(num_layers)}
        self.value_cache = {index: None for index in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)


class _FakeLMModel(torch.nn.Module):
    def embed_tokens(self, token_ids):
        return token_ids.to(torch.float32).unsqueeze(-1)


class _FakeLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeLMModel()

    def forward_inference(
        self,
        *,
        packed_query_sequence,
        query_lens,
        packed_query_position_ids,
        packed_query_indexes,
        past_key_values,
        key_values_lens,
        packed_key_value_indexes,
        update_past_key_values,
        **_,
    ):
        batch_size = int(key_values_lens.numel())
        expected_queries = torch.cumsum(key_values_lens, dim=0) + torch.arange(batch_size, dtype=key_values_lens.dtype)
        assert torch.equal(packed_query_indexes.cpu(), expected_queries.cpu())
        source_blocks = torch.arange(int(key_values_lens.sum())).split(key_values_lens.tolist())
        expected_keys = torch.cat([block + row for row, block in enumerate(source_blocks)])
        assert torch.equal(packed_key_value_indexes.cpu(), expected_keys.cpu())
        assert torch.equal(query_lens, torch.ones_like(query_lens))
        del packed_query_position_ids
        old_tokens = past_key_values.key_cache[0].reshape(-1)
        old_blocks = list(old_tokens.split(key_values_lens.tolist()))
        query_tokens = packed_query_sequence.reshape(-1)
        hidden = torch.stack([query_tokens[row] + old_blocks[row].sum() for row in range(len(old_blocks))]).unsqueeze(
            -1
        )
        if update_past_key_values:
            merged = torch.cat(
                [torch.cat([block, query_tokens[row : row + 1]]) for row, block in enumerate(old_blocks)]
            )
            past_key_values.key_cache[0] = merged.reshape(-1, 1, 1)
            past_key_values.value_cache[0] = merged.reshape(-1, 1, 1)
        return SimpleNamespace(packed_query_sequence=hidden, past_key_values=past_key_values)

    def lm_head(self, hidden):
        vocab = 17
        targets = hidden.reshape(-1).long().remainder(vocab)
        columns = torch.arange(vocab, dtype=torch.float32, device=hidden.device)
        return -(columns.unsqueeze(0) - targets.unsqueeze(1)).abs()


class _FakeBagel:
    def __init__(self):
        self.language_model = _FakeLanguageModel()
        self.config = SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=1, freeze_und=False))

    def forward_cache_update_text(
        self,
        past_key_values,
        *,
        text_token_lens,
        packed_text_ids,
        key_values_lens,
        **_,
    ):
        new_blocks = list(packed_text_ids.to(torch.float32).split(text_token_lens.tolist()))
        if past_key_values.key_cache[0] is None:
            old_blocks = [packed_text_ids.new_zeros(0, dtype=torch.float32) for _ in new_blocks]
        else:
            old_blocks = list(past_key_values.key_cache[0].reshape(-1).split(key_values_lens.tolist()))
        merged = torch.cat([torch.cat([old, new]) for old, new in zip(old_blocks, new_blocks)])
        past_key_values.key_cache[0] = merged.reshape(-1, 1, 1)
        past_key_values.value_cache[0] = merged.reshape(-1, 1, 1)
        return past_key_values


class _FakeBundle:
    def __init__(self):
        self.model = _FakeBagel()
        self.device = torch.device("cpu")
        self.new_token_ids = {"bos_token_id": 0, "eos_token_id": 16}
        self.transformer = self.model.language_model


def _ar_conditions(second_prompt=(7, 8)):
    return BagelARConditions(
        prompt_splits=[
            [
                {"kind": "text", "ids": torch.tensor([1, 2])},
                {"kind": "text", "ids": torch.tensor([3])},
            ],
            [{"kind": "text", "ids": torch.tensor(second_prompt)}],
        ]
    )


def _ar_stage(forward_batch_size):
    bundle = _FakeBundle()
    bundle.transformer.eval()
    return BagelARStage(model=bundle, forward_batch_size=forward_batch_size)


def test_ar_bs1_and_packed_match_tokens_logprobs_and_isolate_samples():
    params = ARSamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=5,
    )
    serial = _ar_stage(1).autoregress(_ar_conditions(), sampling_params=params)
    packed = _ar_stage(2).autoregress(_ar_conditions(), sampling_params=params)

    assert torch.equal(serial.tokens, packed.tokens)
    assert torch.equal(serial.lengths, packed.lengths)
    assert torch.equal(serial.cu_seqlens, packed.cu_seqlens)
    torch.testing.assert_close(serial.log_probs, packed.log_probs, rtol=0, atol=0)

    changed_peer = _ar_stage(2).autoregress(_ar_conditions(second_prompt=(12, 13, 14)), sampling_params=params)
    first_end = int(packed.cu_seqlens[1])
    assert torch.equal(packed.tokens[:first_end], changed_peer.tokens[:first_end])
    torch.testing.assert_close(
        packed.log_probs[:first_end],
        changed_peer.log_probs[:first_end],
        rtol=0,
        atol=0,
    )


def test_ar_fbs1_uses_legacy_global_rng_and_packed_uses_generators(monkeypatch):
    params = ARSamplingParams(
        temperature=0.8,
        top_p=0.95,
        top_k=8,
        max_new_tokens=5,
    )
    calls = []
    original_step = BagelARStep.step

    def recording_step(self, logits, *, generators=None):
        calls.append(generators)
        return original_step(self, logits, generators=generators)

    monkeypatch.setattr(BagelARStep, "step", recording_step)
    torch.manual_seed(1234)
    _ar_stage(1).autoregress(_ar_conditions(), sampling_params=params)
    assert calls and all(generators is None for generators in calls)

    calls.clear()
    torch.manual_seed(1234)
    _ar_stage(2).autoregress(_ar_conditions(), sampling_params=params)
    assert calls and all(generators is not None for generators in calls)


def test_ar_vit_input_declines_packing():
    assert BagelARStage._batched_text_ids([[{"kind": "vit", "image": torch.zeros(3, 2, 2)}]]) is None


class _FakeStrategy:
    def __init__(self):
        self.denoise_calls = 0

    def denoise(
        self,
        *,
        noise_pred,
        sample,
        prev_sample,
        **_,
    ):
        self.denoise_calls += 1
        mean = sample - 0.25 * noise_pred
        output = mean if prev_sample is None else prev_sample
        log_prob = -((output - mean) ** 2).mean(dim=(1, 2))
        return output, log_prob, mean


def test_diffusion_single_sample_uses_legacy_denoise_path():
    strategy = _FakeStrategy()
    step = BagelDiffusionStep()
    step.denoise(
        strategy,
        v_t=torch.ones(3, 4),
        x_t=torch.zeros(3, 4),
        sigma=torch.tensor(1.0),
        sigma_next=torch.tensor(0.5),
        sigma_max=torch.tensor(1.0),
        eta=0.8,
    )
    assert strategy.denoise_calls == 1


def test_diffusion_packed_reduction_matches_bs1_and_isolates_rows():
    step = BagelDiffusionStep()
    strategy = _FakeStrategy()
    sample = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    velocity = torch.arange(24, dtype=torch.float32).flip(0).reshape(6, 4)
    kwargs = dict(
        sigma=torch.tensor(1.0),
        sigma_next=torch.tensor(0.5),
        sigma_max=torch.tensor(1.0),
        eta=0.8,
    )

    packed = step.denoise(
        strategy,
        v_t=velocity,
        x_t=sample,
        n_samples=2,
        **kwargs,
    )
    serial = [
        step.denoise(
            strategy,
            v_t=velocity[index * 3 : (index + 1) * 3],
            x_t=sample[index * 3 : (index + 1) * 3],
            **kwargs,
        )
        for index in range(2)
    ]

    torch.testing.assert_close(packed[0], torch.cat([value[0] for value in serial]))
    torch.testing.assert_close(packed[1], torch.stack([value[1] for value in serial]))
    torch.testing.assert_close(packed[2], torch.cat([value[2] for value in serial]))

    changed_velocity = velocity.clone()
    changed_velocity[3:] += 1000
    changed = step.denoise(
        strategy,
        v_t=changed_velocity,
        x_t=sample,
        n_samples=2,
        **kwargs,
    )
    torch.testing.assert_close(packed[0][:3], changed[0][:3], rtol=0, atol=0)
    torch.testing.assert_close(packed[1][0], changed[1][0], rtol=0, atol=0)


def test_diffusion_fixed_per_sequence_rng_matches_serial_steps(monkeypatch):
    torch_utils = ModuleType("diffusers.utils.torch_utils")

    def randn_tensor(shape, *, generator, device, dtype):
        if isinstance(generator, list):
            return torch.cat(
                [
                    torch.randn((1, *shape[1:]), generator=row_generator, device=device, dtype=dtype)
                    for row_generator in generator
                ]
            )
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    torch_utils.randn_tensor = randn_tensor
    monkeypatch.setitem(sys.modules, "diffusers", ModuleType("diffusers"))
    monkeypatch.setitem(sys.modules, "diffusers.utils", ModuleType("diffusers.utils"))
    monkeypatch.setitem(sys.modules, "diffusers.utils.torch_utils", torch_utils)

    step = BagelDiffusionStep()
    packed_strategy = FlowSDEStrategy()
    serial_strategies = [FlowSDEStrategy(), FlowSDEStrategy()]
    packed_generators = [torch.Generator().manual_seed(seed) for seed in (11, 29)]
    serial_generators = [torch.Generator().manual_seed(seed) for seed in (11, 29)]
    packed_sample = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
    serial_samples = [packed_sample[:3].clone(), packed_sample[3:].clone()]
    schedule = [1.0, 0.8, 0.5, 0.2]

    for current, following in zip(schedule, schedule[1:]):
        packed_velocity = packed_sample * 0.1 + 0.25
        packed_sample, packed_logp, _ = step.denoise(
            packed_strategy,
            v_t=packed_velocity,
            x_t=packed_sample,
            sigma=torch.tensor(current),
            sigma_next=torch.tensor(following),
            sigma_max=torch.tensor(0.8),
            eta=0.8,
            n_samples=2,
            generators=packed_generators,
        )

        serial_logps = []
        for index in range(2):
            velocity = serial_samples[index] * 0.1 + 0.25
            serial_samples[index], logp, _ = step.denoise(
                serial_strategies[index],
                v_t=velocity,
                x_t=serial_samples[index],
                sigma=torch.tensor(current),
                sigma_next=torch.tensor(following),
                sigma_max=torch.tensor(0.8),
                eta=0.8,
                generators=[serial_generators[index]],
            )
            serial_logps.append(logp)

        torch.testing.assert_close(packed_sample, torch.cat(serial_samples), rtol=0, atol=0)
        torch.testing.assert_close(packed_logp, torch.stack(serial_logps), rtol=0, atol=0)


def _context(length, start=0):
    cache = NaiveCache(1)
    if length:
        values = torch.arange(start, start + length, dtype=torch.float32).reshape(-1, 1, 1)
        cache.key_cache[0] = values
        cache.value_cache[0] = values + 100
    return {"kv_lens": [length], "ropes": [length], "past_key_values": cache}


def test_diffusion_context_merge_and_pack_fallback_rules():
    merged = BagelDiffusionStage._merge_contexts([_context(2), _context(0), _context(1, 9)])
    assert merged["kv_lens"] == [2, 0, 1]
    assert merged["ropes"] == [2, 0, 1]
    torch.testing.assert_close(
        merged["past_key_values"].key_cache[0].reshape(-1),
        torch.tensor([0.0, 1.0, 9.0]),
    )

    same_shape = BagelDiffusionConditions(
        gen_contexts=[_context(1), _context(1)],
        cfg_text_contexts=[_context(0), _context(0)],
        cfg_img_contexts=[_context(1), _context(1)],
        prompts=["a", "b"],
        image_shapes=[(512, 512), (512, 512)],
    )
    mixed_shape = BagelDiffusionConditions(
        gen_contexts=same_shape.gen_contexts,
        cfg_text_contexts=same_shape.cfg_text_contexts,
        cfg_img_contexts=same_shape.cfg_img_contexts,
        prompts=same_shape.prompts,
        image_shapes=[(512, 512), (384, 640)],
    )
    deferred = BagelDiffusionConditions(
        prompts=["a", "b"],
        image_shapes=[(512, 512), (512, 512)],
    )

    assert BagelDiffusionStage._can_pack_conditions(same_shape)
    assert not BagelDiffusionStage._can_pack_conditions(mixed_shape)
    assert not BagelDiffusionStage._can_pack_conditions(deferred)
