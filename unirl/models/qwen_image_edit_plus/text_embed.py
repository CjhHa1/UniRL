"""QwenImageEditPlusTextEmbedStage — condition-image + text → TextEmbedCondition.

Unlike the base :class:`unirl.models.qwen_image.QwenImageTextEmbedStage`
(text-only), Qwen-Image-Edit-Plus feeds the **source image into the
Qwen2.5-VL text encoder** so the prompt embeddings carry visual context
about the image being edited. This mirrors upstream diffusers
``QwenImageEditPlusPipeline._get_qwen_prompt_embeds`` /
``pipeline_qwenimage_edit_plus.py`` (and the SGLang rollout path, whose
captured ``prompt_embeds`` include image-placeholder tokens):

- The user content is prefixed with an image tag
  ``"Picture 1: <|vision_start|><|image_pad|><|vision_end|>"`` and wrapped
  in the **edit** chat template (different system prompt than base
  Qwen-Image; ``prompt_template_encode_start_idx = 64`` vs 34).
- A :class:`transformers.Qwen2VLProcessor` builds ``input_ids`` +
  ``pixel_values`` + ``image_grid_thw`` (the source image resized to the
  ``CONDITION_IMAGE_SIZE`` ≈ 384² grid, aspect-preserving, 32-aligned —
  the *condition* size for the text encoder, distinct from the ≈1024²
  *VAE* size the latent-concat path uses).
- The Qwen2.5-VL forward runs with the pixel values; the last hidden
  state is split per sample, the 64-token system prefix is dropped, and
  the remainder (image-placeholder tokens followed by the prompt text
  tokens) is padded to the batch max with a parallel attention mask.

Set ``QwenImageEditPlusPipelineConfig.use_condition_image_prompt=False`` to
fall back to the base text-only :class:`QwenImageTextEmbedStage` (reproduces
the pre-image-conditioning behavior).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from unirl.models.types.embedding import EmbedStage
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Images, Texts

from .bundle import QwenImageEditPlusBundle

# Edit chat template + drop index (upstream diffusers
# ``QwenImageEditPlusPipeline``: ``prompt_template_encode`` /
# ``prompt_template_encode_start_idx``). The system prompt differs from base
# Qwen-Image, so the fixed prefix is 64 tokens (not 34).
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, "
    "background), then explain how the user's text instruction should alter or modify the image. Generate a new "
    "image that meets the user's requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_START_IDX = 64
IMG_PROMPT_TEMPLATE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
TOKENIZER_MAX_LENGTH = 1024

# Condition-image grid fed to the text encoder (upstream ``CONDITION_IMAGE_SIZE``).
# NOTE: this is the low-res condition size for the *text encoder*; the VAE
# latent-concat path resizes the same source image to a separate ≈1024² grid.
CONDITION_IMAGE_AREA = 384 * 384
_SIZE_ALIGN = 32


def _condition_size_for_aspect(width: int, height: int) -> Tuple[int, int]:
    """Aspect-preserving ``(w, h)`` at ≈``CONDITION_IMAGE_AREA``, 32-aligned.

    Mirrors upstream ``calculate_dimensions(CONDITION_IMAGE_SIZE, w / h)``.
    """
    ratio = float(width) / float(height)
    cond_w = math.sqrt(CONDITION_IMAGE_AREA * ratio)
    cond_h = cond_w / ratio
    cond_w = round(cond_w / _SIZE_ALIGN) * _SIZE_ALIGN
    cond_h = round(cond_h / _SIZE_ALIGN) * _SIZE_ALIGN
    return int(cond_w), int(cond_h)


class QwenImageEditPlusTextEmbedStage(EmbedStage[Texts, TextEmbedCondition]):
    """Condition-image + text → ``TextEmbedCondition`` via Qwen2.5-VL.

    The ``embed`` signature takes the source ``Images`` alongside the
    prompts (both positive and negative branches pass the *same* source
    images, matching upstream ``encode_prompt(image=...)``).
    """

    def __init__(
        self,
        bundle: QwenImageEditPlusBundle,
        *,
        max_sequence_length: int = 512,
        processor_path: Optional[str] = None,
        processor_subfolder: str = "processor",
    ) -> None:
        if max_sequence_length > TOKENIZER_MAX_LENGTH:
            raise ValueError(
                f"QwenImageEditPlusTextEmbedStage.max_sequence_length cannot exceed "
                f"{TOKENIZER_MAX_LENGTH} (tokenizer cap) but got {max_sequence_length}"
            )
        self.bundle = bundle
        self.max_sequence_length = int(max_sequence_length)
        # The processor (tokenizer merges + image processor) must come from the
        # same place as the text encoder / tokenizer: honor a text-encoder
        # override (``config.text_encoder_ckpt_path``) threaded in as
        # ``processor_path``, else fall back to the main checkpoint.
        self.processor = self._load_processor(processor_path or bundle.pretrained_path, processor_subfolder)

    @staticmethod
    def _load_processor(path: str, subfolder: str):
        """Load the Qwen2.5-VL processor (image processor + tokenizer merges).

        The Edit-Plus checkpoint ships the processor under a ``processor/``
        subfolder (diffusers ``register_modules(processor=...)`` layout). No
        ``min_pixels`` / ``max_pixels`` override — the source image is
        pre-resized to the condition grid before the processor's own
        smart-resize, matching upstream.
        """
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(path, subfolder=subfolder)

    def embed(self, p: Texts, images: Images) -> TextEmbedCondition:  # type: ignore[override]
        """Encode prompts conditioned on the source images."""
        prompt_embeds, prompt_embeds_mask = self._encode(list(p.texts), images)
        return TextEmbedCondition(
            embeds=prompt_embeds,
            attn_mask=prompt_embeds_mask,
            pooled=None,
        )

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        """Split a padded ``[B, T, D]`` tensor into ``B`` variable-length
        ``[t_i, D]`` slices using a ``[B, T]`` 0/1 mask."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return list(torch.split(selected, valid_lengths.tolist(), dim=0))

    def _condition_pils(self, images: Images):
        """Convert source ``Images`` to per-sample PILs resized to the
        condition grid (≈384², aspect-preserving), mirroring upstream's
        ``image_processor.resize`` before the VL processor."""
        import PIL.Image

        pils = images.to_pils()
        resized = []
        for pil in pils:
            cond_w, cond_h = _condition_size_for_aspect(pil.width, pil.height)
            if pil.width != cond_w or pil.height != cond_h:
                pil = pil.resize((cond_w, cond_h), PIL.Image.LANCZOS)
            resized.append(pil)
        return resized

    def _encode(self, prompts: List[str], images: Images) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        device = bundle.device
        dtype = next(bundle.text_encoder.parameters()).dtype

        condition_pils = self._condition_pils(images)
        if len(condition_pils) != len(prompts):
            raise ValueError(
                f"QwenImageEditPlusTextEmbedStage._encode: image count {len(condition_pils)} "
                f"!= prompt count {len(prompts)}"
            )

        # One image placeholder per prompt (V1: single source image per prompt).
        base_img_prompt = IMG_PROMPT_TEMPLATE.format(1)
        txt = [PROMPT_TEMPLATE.format(base_img_prompt + e) for e in prompts]

        model_inputs = self.processor(
            text=txt,
            images=condition_pils,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            encoder_out = bundle.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values.to(dtype=dtype),
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )
        hidden_states = encoder_out.hidden_states[-1]

        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        # Strip the 64-token edit-template system prefix; the remainder is
        # [image-placeholder tokens][prompt text tokens].
        split_hidden_states = [item[PROMPT_TEMPLATE_START_IDX:] for item in split_hidden_states]
        attn_mask_list = [
            torch.ones(item.size(0), dtype=torch.long, device=item.device) for item in split_hidden_states
        ]
        max_seq_len = max(item.size(0) for item in split_hidden_states)

        prompt_embeds = torch.stack(
            [
                torch.cat([item, item.new_zeros(max_seq_len - item.size(0), item.size(1))])
                for item in split_hidden_states
            ]
        )
        prompt_embeds_mask = torch.stack(
            [torch.cat([item, item.new_zeros(max_seq_len - item.size(0))]) for item in attn_mask_list]
        )

        # Final slice to the configured budget (image-placeholder tokens are at
        # the front, so a short prompt keeps its full visual context).
        prompt_embeds = prompt_embeds[:, : self.max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, : self.max_sequence_length]
        return prompt_embeds.to(device=device, dtype=dtype), prompt_embeds_mask


__all__ = ["QwenImageEditPlusTextEmbedStage"]
