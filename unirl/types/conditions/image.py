"""Image conditioning types.

``ImageLatentCondition`` carries dense VAE latents (img2img, first-frame, etc.).
``RaggedImageLatentCondition`` carries per-sample spatial latents whose grids
may differ (Qwen Image Edit Plus mixed-aspect inputs).
``ImageEmbedCondition`` carries ViT-style patch embeddings (SigLIP / CLIP
vision tower output, AR-emitted-image-token re-embeddings). Other roles
(``ImageMaskedLatentCondition``, ``ImageTokenCondition``) remain deferred
to first consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional

import torch

from unirl.distributed.tensor.batch import FieldKind, concat_field, field
from unirl.types.conditions.base import Condition, Modality


@dataclass
class ImageLatentCondition(Condition):
    """Image conditioning carried as VAE latents (img2img, first-frame, etc.)."""

    modality: ClassVar[Modality] = Modality.IMAGE

    latents: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)

    def __post_init__(self) -> None:
        if isinstance(self.latents, (list, tuple)):
            raise TypeError("ImageLatentCondition.latents must be a dense tensor; use a ragged condition type")


@dataclass
class RaggedImageLatentCondition(Condition):
    """Per-sample VAE latents with heterogeneous spatial grids.

    ``latents[i]`` is one ``[C, H_i, W_i]`` tensor. The list is the sample
    axis, so Batch concat/select/slice retain alignment without padding or
    changing image geometry.
    """

    modality: ClassVar[Modality] = Modality.IMAGE

    latents: List[torch.Tensor] = concat_field(default_factory=list)

    def __len__(self) -> int:
        return len(self.latents)


@dataclass
class ImageEmbedCondition(Condition):
    """Image conditioning carried as ViT-style patch embeddings.

    First consumer is HunyuanImage 3.0 (SigLIP2 ViT for i2t/it2i comprehension,
    plus AR-emitted image-vocab token re-embeddings on the diffusion side).
    Same shape as ``TextEmbedCondition.embeds`` but tagged ``Modality.IMAGE``.

    HunyuanImage3 stores ``embeds``, ``attn_mask``, and ``spatial_shapes`` as
    per-sample lists because patch counts vary with native image geometry.
    Fixed-grid encoders may use dense tensors for the first two fields. All
    three remain CONCAT-aligned so DP slice/select never detach one sample's
    spatial metadata from its embeddings.
    """

    modality: ClassVar[Modality] = Modality.IMAGE

    embeds: Optional[Any] = field(kind=FieldKind.CONCAT, default=None)
    attn_mask: Optional[Any] = field(kind=FieldKind.CONCAT, default=None)
    spatial_shapes: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)


__all__ = ["ImageEmbedCondition", "ImageLatentCondition", "RaggedImageLatentCondition"]
