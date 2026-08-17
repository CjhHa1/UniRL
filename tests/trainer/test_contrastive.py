from __future__ import annotations

import torch

from unirl.trainer.contrastive import select_top_bottom_indices
from unirl.trainer.diffusion import DiffusionTrainer
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams


def _scored_part(rewards: list[float], *, groups: int, group_size: int) -> Part:
    assert len(rewards) == groups * group_size
    return Part(
        sample_ids=[f"p{group}/candidate/{row}" for group in range(groups) for row in range(group_size)],
        rewards=torch.tensor(rewards, dtype=torch.float32),
    )


def test_top_bottom_selection_is_group_contiguous_and_stable() -> None:
    part = _scored_part(
        [0.1, 0.9, 0.3, 0.8, 0.2, 0.4, 0.4, 0.0],
        groups=2,
        group_size=4,
    )
    indices = select_top_bottom_indices(part, top_k=1, bottom_k=1)
    assert indices.tolist() == [1, 0, 5, 7]

    selected = part.select(indices)
    assert selected.group_ids == [
        "p0/candidate",
        "p0/candidate",
        "p1/candidate",
        "p1/candidate",
    ]


def test_selection_rejects_overlapping_top_and_bottom() -> None:
    part = _scored_part([0.0, 1.0], groups=1, group_size=2)
    try:
        select_top_bottom_indices(part, top_k=2, bottom_k=1)
    except ValueError as exc:
        assert "exceeds scout group size" in str(exc)
    else:
        raise AssertionError("expected selection overlap to fail")


def test_tied_rewards_never_duplicate_top_and_bottom_rows() -> None:
    part = _scored_part([1.0, 1.0, 1.0, 1.0], groups=1, group_size=4)
    indices = select_top_bottom_indices(part, top_k=1, bottom_k=1)
    assert indices.tolist() == [0, 1]


def test_regen_shell_preserves_exact_selected_initial_noise() -> None:
    root = Part.input(
        ["prompt"],
        primitives={"text": Texts(texts=["a prompt"])},
    )
    scout_params = DiffusionSamplingParams(
        samples_per_prompt=4,
        num_samples_per_prompt=4,
        num_inference_steps=6,
        seed=123,
        init_noise_latent_shape=[2, 3, 3],
        sde_indices=[],
    )
    scout = Sample.request(root).fork(4, sampling_params=scout_params)
    full_xt = NoiseRecipe.from_sample(scout).resolve()
    assert full_xt is not None

    selected = scout.parts[-1].select(torch.tensor([3, 0]))
    regen_params = DiffusionSamplingParams(
        samples_per_prompt=2,
        num_samples_per_prompt=2,
        num_inference_steps=10,
        seed=123,
        init_noise_latent_shape=[2, 3, 3],
        sde_indices=[],
    )
    shell = DiffusionTrainer._selected_regen_shell(selected, regen_params)
    regen = scout.replace_frontier(shell)
    regen_xt = NoiseRecipe.from_sample(regen).resolve()
    assert regen_xt is not None
    assert torch.equal(regen_xt, full_xt.index_select(0, torch.tensor([3, 0])))
    assert shell.segment is None
    assert shell.primitives == {}
    assert shell.sampling_params.sigmas is None
