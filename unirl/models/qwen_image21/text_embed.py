"""QwenImage21TextEmbedStage - Qwen3-VL raw-template prompt -> pre-final-norm ``TextEmbedCondition`` (see README)."""

from __future__ import annotations

import torch

from unirl.models.qwen_image.text_embed import extract_masked_hidden
from unirl.models.types.embedding import EmbedStage
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .bundle import QwenImage21Bundle

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
)


class QwenImage21TextEmbedStage(EmbedStage[Texts, TextEmbedCondition]):
    """Qwen3-VL text-to-image prompt -> right-padded ``TextEmbedCondition`` ``[B, L, 4096]``."""

    def __init__(self, bundle: QwenImage21Bundle) -> None:
        self.bundle = bundle
        sys_message = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        sys_tokens = bundle.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
        self.drop_idx = len(sys_tokens[0])

    def embed(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts; the system-prefix tokens are dropped and the rest right-padded."""
        bundle = self.bundle
        prompts = [PROMPT_TEMPLATE.format(text or " ") for text in p.texts]
        inputs = bundle.processor(text=prompts, padding=True, padding_side="left", return_tensors="pt").to(
            bundle.device
        )
        forward_kwargs = {k: inputs[k] for k in ("input_ids", "attention_mask", "mm_token_type_ids") if k in inputs}

        encoder = bundle.text_encoder.model
        handle = encoder.language_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            with torch.no_grad():
                hidden = encoder(**forward_kwargs).last_hidden_state
        finally:
            handle.remove()

        split = [h[self.drop_idx :] for h in extract_masked_hidden(hidden, inputs.attention_mask)]
        max_len = max(h.shape[0] for h in split)
        embeds = torch.stack([torch.cat([h, h.new_zeros(max_len - h.shape[0], h.shape[1])]) for h in split])
        attn_mask = torch.stack(
            [
                torch.cat(
                    [h.new_ones(h.shape[0], dtype=torch.long), h.new_zeros(max_len - h.shape[0], dtype=torch.long)]
                )
                for h in split
            ]
        )
        return TextEmbedCondition(embeds=embeds, attn_mask=attn_mask, pooled=None)


__all__ = ["QwenImage21TextEmbedStage"]
