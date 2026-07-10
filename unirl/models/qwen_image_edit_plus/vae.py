"""QwenImageEditPlus VAE stages — source-image encoder + (reused) decoder.

``QwenImageEditPlusVAEEncodeStage`` is the one genuinely new stage: it
turns source ``NativeImages`` into a :class:`RaggedImageLatentCondition`
carrying one VAE-encoded spatial latent ``[16, H_i/8, W_i/8]`` per sample. The diffusion
step (:class:`QwenImageEditPlusDiffusionStep`) packs both the noise latent
and this image latent with the same 2×2 channel-pack before concatenating
along the token dimension — mirrors ``vde_editplus.py:232`` and the
FLUX.2-Klein image-edit pattern (``flux2_klein/vae.py:86-158``).

The decode side reuses :class:`unirl.models.qwen_image.QwenImageVAEDecodeStage`
unchanged (same VAE, same 5D un-normalization math) — re-exported here so
the Edit-Plus package is self-contained.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from unirl.models.types.codec import EncodeStage
from unirl.types.conditions import RaggedImageLatentCondition
from unirl.types.primitives import Images, NativeImages

from .bundle import QwenImageEditPlusBundle

# Upstream ``QwenImageEditPlusPipelineConfig`` (sglang/diffusers) resizes the
# source image to a fixed total pixel area of ``1024 * 1024`` while preserving
# aspect ratio (see ``VAE_IMAGE_SIZE`` in
# ``sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py``). The
# trainsite encoder MUST match this — using the generation grid (e.g. 384²)
# instead yields a different ``image_latent`` shape than the sglang/vllm_omni
# rollout engines, breaking the trainsite-vs-separate-engine parity contract
# the recipe YAMLs promise. Mirrors upstream ``calculate_dimensions``.
_VAE_IMAGE_AREA = 1024 * 1024
_VAE_SIZE_ALIGN = 32  # upstream rounds to 32-pixel multiples


def _vae_size_for_aspect(width: int, height: int) -> tuple[int, int]:
    """Aspect-preserving resize target matching upstream ``VAE_IMAGE_SIZE``.

    Returns ``(vae_width, vae_height)`` aligned to ``_VAE_SIZE_ALIGN`` with
    total area ≈ ``_VAE_IMAGE_AREA``. Mirrors
    ``sglang.multimodal_gen.utils.calculate_dimensions``.
    """
    ratio = float(width) / float(height)
    vae_width = math.sqrt(_VAE_IMAGE_AREA * ratio)
    vae_height = vae_width / ratio
    vae_width = round(vae_width / _VAE_SIZE_ALIGN) * _VAE_SIZE_ALIGN
    vae_height = round(vae_height / _VAE_SIZE_ALIGN) * _VAE_SIZE_ALIGN
    return int(vae_width), int(vae_height)


class QwenImageEditPlusVAEEncodeStage(EncodeStage[NativeImages, RaggedImageLatentCondition]):
    """Encode a source image into a VAE-latent condition for token concat.

    Pipeline:

    1. Resize source pixels to the upstream ``VAE_IMAGE_SIZE`` grid
       (≈1024², aspect-preserving, 32-aligned). The data source loads
       condition images at native resolution (arbitrary H×W), but
       upstream ``QwenImageEditPlusPipelineConfig.calculate_condition_image_size``
       resizes to ``1024*1024`` total area for VAE encoding — the
       trainsite MUST match so the emitted ``image_latent`` shape is
       byte-identical to the sglang/vllm_omni rollout engines (the
       recipe YAMLs promise fixed-seed parity). The generation grid
       (e.g. 384²) is the *output* canvas size, NOT the source-image
       VAE size; using it here was a parity-breaking bug.
    2. ``[0, 1] → [-1, 1]`` (VAE input convention).
    3. Lift to 5D ``[B, 3, 1, H, W]`` (Qwen-Image VAE is a video VAE —
       ``_encode`` unpacks ``_, _, num_frame, height, width = x.shape``;
       a 4D input crashes).
    4. ``vae.encode(x).latent_dist.mode()`` — deterministic (matches
       diffusers' ``retrieve_latents(sample_mode="argmax")`` so
       rollout/replay don't drift). Mirrors ``flux2_klein/vae.py:141``.
    5. Per-channel normalize: ``(latents - latents_mean) / latents_std``
       (mirrors upstream ``QwenImageEditPlusPipeline._encode_vae_image``
       at ``pipeline_qwen_image_edit_plus.py:489-499``; the decode side
       in :class:`QwenImageVAEDecodeStage` applies the inverse, so
       skipping this would put rollout/trainsite image latents on a
       different scale than the transformer was trained on).
    6. Return one spatial latent ``[16, H_i/8, W_i/8]`` per sample wrapped in
       :class:`RaggedImageLatentCondition`. **Do NOT** ``_pack_latents`` here —
       the diffusion step packs both noise and image latents together so
       they share the same 2×2 pack logic; the condition carries the
       spatial latent (mirrors ``Flux2KleinConditions.image_latent``).
    """

    def __init__(self, bundle: QwenImageEditPlusBundle) -> None:
        self.bundle = bundle

    @torch.no_grad()
    def encode(self, images: NativeImages | Images, *, height: int, width: int) -> RaggedImageLatentCondition:
        """Encode source pixels into a ragged latent condition.

        Args:
            images: source images with one native ``[3, H_i, W_i]`` tensor per
                sample in ``[0, 1]``.
            height, width: generation grid (must be divisible by 16: 8× VAE
                downsample + 2× patchify). Used only for the 16-alignment
                guard; the source image is resized to the upstream
                ``VAE_IMAGE_SIZE`` grid (≈1024²), NOT to the generation
                grid, matching sglang/vllm_omni rollout engines.

        Returns:
            :class:`RaggedImageLatentCondition` with one
            ``[16, H_vae_i/8, W_vae_i/8]`` tensor per sample.
        """
        if not isinstance(images, (NativeImages, Images)):
            raise TypeError(
                f"QwenImageEditPlusVAEEncodeStage.encode: expected NativeImages, got {type(images).__name__}"
            )
        pixels_list = images.pixels if isinstance(images, NativeImages) else list(images.pixels.unbind(0))
        if not pixels_list or any(pixels.ndim != 3 or pixels.shape[0] != 3 for pixels in pixels_list):
            raise ValueError(
                "QwenImageEditPlusVAEEncodeStage.encode: expected per-sample pixels "
                f"[3, H, W] in [0,1], got {[tuple(pixels.shape) for pixels in pixels_list]}"
            )
        if int(height) % 16 != 0 or int(width) % 16 != 0:
            raise ValueError(
                f"QwenImageEditPlusVAEEncodeStage.encode: height ({height}) and "
                f"width ({width}) must be divisible by 16 (8× VAE + 2× patchify)"
            )

        vae = self.bundle.vae
        device = self.bundle.device
        dtype = self.bundle.dtype
        vae_f32 = vae.to(torch.float32)

        # Stable shape buckets preserve input order while allowing each VAE
        # call to stay dense. Portrait and landscape samples never influence
        # one another's resize target.
        groups: Dict[tuple[int, int], List[int]] = {}
        for index, pixels in enumerate(pixels_list):
            src_h, src_w = int(pixels.shape[-2]), int(pixels.shape[-1])
            vae_w, vae_h = _vae_size_for_aspect(src_w, src_h)
            groups.setdefault((vae_h, vae_w), []).append(index)

        # Per-channel normalization mirrors the upstream Edit-Plus pipeline.
        z_dim = int(vae.config.z_dim)
        latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(1, z_dim, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(1, z_dim, 1, 1)

        by_index: Dict[int, torch.Tensor] = {}
        for (vae_h, vae_w), indices in groups.items():
            resized_items = []
            for index in indices:
                pixels = pixels_list[index].to(device=device, dtype=torch.float32).unsqueeze(0)
                if tuple(pixels.shape[-2:]) != (vae_h, vae_w):
                    pixels = torch.nn.functional.interpolate(
                        pixels,
                        size=(vae_h, vae_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                resized_items.append(pixels)
            pixels_batch = torch.cat(resized_items, dim=0)
            scaled_5d = (pixels_batch * 2.0 - 1.0).unsqueeze(2)
            image_latents = vae_f32.encode(scaled_5d).latent_dist.mode().squeeze(2)
            mean = latents_mean.to(image_latents.device, image_latents.dtype)
            std = latents_std.to(image_latents.device, image_latents.dtype)
            image_latents = ((image_latents - mean) / std).to(dtype=dtype)
            for local_index, source_index in enumerate(indices):
                by_index[source_index] = image_latents[local_index]

        return RaggedImageLatentCondition(latents=[by_index[index] for index in range(len(pixels_list))])


__all__ = ["QwenImageEditPlusVAEEncodeStage"]
