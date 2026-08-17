from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from unirl.reward.local.hpsv2 import HPSv2RewardScorer


class _Model:
    def __call__(self, image_input: torch.Tensor, text_input: torch.Tensor):
        means = image_input.float().mean(dim=(1, 2, 3))
        image_features = torch.stack((means, torch.ones_like(means)), dim=-1)
        return {
            "image_features": image_features,
            "text_features": text_input.float(),
        }


def test_hpsv2_batches_rows_and_accepts_string_device() -> None:
    scorer = HPSv2RewardScorer.__new__(HPSv2RewardScorer)
    scorer.device = "cpu"
    scorer.batch_size = 2
    scorer.model = _Model()
    scorer._hpsv2_preprocess_val = lambda image: torch.tensor(np.asarray(image, dtype=np.float32) / 255.0).permute(
        2, 0, 1
    )
    scorer._hpsv2_tokenizer = lambda prompts: torch.tensor([[float(index + 1), 1.0] for index, _ in enumerate(prompts)])
    request = SimpleNamespace(
        images=[
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 255, dtype=np.uint8),
        ],
        prompts=["zero", "one"],
    )
    assert scorer._compute_model_rewards(request) == [1.0, 3.0]
