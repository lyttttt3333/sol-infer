import os
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _dump_ltx2_contexts_if_requested(
    *,
    prompt_embeds: torch.Tensor | None,
    prompt_attention_mask: torch.Tensor | None,
    neg_prompt_embeds: torch.Tensor | None,
    neg_prompt_attention_mask: torch.Tensor | None,
    pos_embeds: torch.Tensor,
    pos_audio_embeds: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_embeds: torch.Tensor | None = None,
    neg_audio_embeds: torch.Tensor | None = None,
    neg_mask: torch.Tensor | None = None,
) -> None:
    dump_dir = os.environ.get("SGLANG_LTX2_DUMP_CONTEXT_DIR")
    if not dump_dir:
        return
    out = Path(dump_dir)
    out.mkdir(parents=True, exist_ok=True)

    def cpu(x):
        return None if x is None else x.detach().cpu()

    torch.save(
        {
            "raw_prompt_embeds": cpu(prompt_embeds),
            "raw_prompt_attention_mask": cpu(prompt_attention_mask),
            "raw_negative_prompt_embeds": cpu(neg_prompt_embeds),
            "raw_negative_attention_mask": cpu(neg_prompt_attention_mask),
            "video_context_pos": cpu(pos_embeds),
            "audio_context_pos": cpu(pos_audio_embeds),
            "connector_mask_pos": cpu(pos_mask),
            "video_context_neg": cpu(neg_embeds),
            "audio_context_neg": cpu(neg_audio_embeds),
            "connector_mask_neg": cpu(neg_mask),
        },
        out / "sglang_contexts.pt",
    )


def _load_ltx2_contexts_if_requested(
    *,
    pos_embeds: torch.Tensor,
    pos_audio_embeds: torch.Tensor,
    neg_embeds: torch.Tensor | None,
    neg_audio_embeds: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    load_dir = os.environ.get("SGLANG_LTX2_LOAD_CONTEXT_DIR")
    if not load_dir:
        return pos_embeds, pos_audio_embeds, neg_embeds, neg_audio_embeds

    root = Path(load_dir)
    path = root / "official_contexts.pt"
    if not path.exists():
        path = root / "sglang_contexts.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"SGLANG_LTX2_LOAD_CONTEXT_DIR={load_dir!r} does not contain "
            "official_contexts.pt or sglang_contexts.pt"
        )
    payload = torch.load(path, map_location="cpu")

    def load_tensor(key: str, reference: torch.Tensor | None) -> torch.Tensor | None:
        if reference is None:
            return None
        value = payload.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{path} does not contain tensor key {key!r}")
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"{path}:{key} has shape {tuple(value.shape)}, expected "
                f"{tuple(reference.shape)}"
            )
        return value.to(device=reference.device, dtype=reference.dtype)

    return (
        load_tensor("video_context_pos", pos_embeds),
        load_tensor("audio_context_pos", pos_audio_embeds),
        load_tensor("video_context_neg", neg_embeds),
        load_tensor("audio_context_neg", neg_audio_embeds),
    )


class LTX2TextConnectorStage(PipelineStage):
    """
    Stage for applying LTX-2 Text Connectors to split/transform text embeddings
    into video and audio contexts.
    """

    def __init__(self, connectors):
        super().__init__()
        self.connectors = connectors

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # Input: batch.prompt_embeds (from Gemma, [B, S, D])
        # Output: batch.prompt_embeds (Video Context), batch.audio_prompt_embeds (Audio Context)

        prompt_embeds = batch.prompt_embeds
        prompt_attention_mask = batch.prompt_attention_mask
        neg_prompt_embeds = batch.negative_prompt_embeds
        neg_prompt_attention_mask = batch.negative_attention_mask

        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0] if len(prompt_embeds) > 0 else None

        if isinstance(prompt_attention_mask, list):
            prompt_attention_mask = (
                prompt_attention_mask[0] if len(prompt_attention_mask) > 0 else None
            )

        if isinstance(neg_prompt_embeds, list):
            neg_prompt_embeds = (
                neg_prompt_embeds[0] if len(neg_prompt_embeds) > 0 else None
            )

        if isinstance(neg_prompt_attention_mask, list):
            neg_prompt_attention_mask = (
                neg_prompt_attention_mask[0]
                if len(neg_prompt_attention_mask) > 0
                else None
            )

        if prompt_embeds is None or prompt_attention_mask is None:
            raise ValueError(
                "LTX2TextConnectorStage requires prompt embeddings and "
                "attention mask."
            )

        if batch.do_classifier_free_guidance:
            if neg_prompt_embeds is None or neg_prompt_attention_mask is None:
                raise ValueError(
                    "LTX2TextConnectorStage requires negative prompt embeddings "
                    "and attention mask when classifier-free guidance is enabled."
                )

            # Official LTX-2.3 processes positive and negative prompts through
            # the connector independently; batching shifts output numerics.
            dtype = prompt_embeds.dtype
            pos_additive_mask = (prompt_attention_mask.to(torch.int64) - 1).to(
                dtype
            ) * torch.finfo(dtype).max
            neg_additive_mask = (neg_prompt_attention_mask.to(torch.int64) - 1).to(
                dtype
            ) * torch.finfo(dtype).max

            with set_forward_context(current_timestep=None, attn_metadata=None):
                pos_embeds, pos_audio_embeds, pos_mask = self.connectors(
                    prompt_embeds, pos_additive_mask, additive_mask=True
                )
                neg_embeds, neg_audio_embeds, neg_mask = self.connectors(
                    neg_prompt_embeds, neg_additive_mask, additive_mask=True
                )
            pos_embeds, pos_audio_embeds, neg_embeds, neg_audio_embeds = (
                _load_ltx2_contexts_if_requested(
                    pos_embeds=pos_embeds,
                    pos_audio_embeds=pos_audio_embeds,
                    neg_embeds=neg_embeds,
                    neg_audio_embeds=neg_audio_embeds,
                )
            )

            _dump_ltx2_contexts_if_requested(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                neg_prompt_embeds=neg_prompt_embeds,
                neg_prompt_attention_mask=neg_prompt_attention_mask,
                pos_embeds=pos_embeds,
                pos_audio_embeds=pos_audio_embeds,
                pos_mask=pos_mask,
                neg_embeds=neg_embeds,
                neg_audio_embeds=neg_audio_embeds,
                neg_mask=neg_mask,
            )

            batch.prompt_embeds = [pos_embeds]
            batch.audio_prompt_embeds = [pos_audio_embeds]
            batch.prompt_attention_mask = pos_mask
            batch.negative_prompt_embeds = [neg_embeds]
            batch.negative_audio_prompt_embeds = [neg_audio_embeds]
            batch.negative_attention_mask = neg_mask
        else:
            # Prepare additive mask for connectors (as per diffusers implementation)
            dtype = prompt_embeds.dtype
            additive_attention_mask = (prompt_attention_mask.to(torch.int64) - 1).to(
                dtype
            ) * torch.finfo(dtype).max

            with set_forward_context(current_timestep=None, attn_metadata=None):
                (
                    connector_prompt_embeds,
                    connector_audio_prompt_embeds,
                    connector_mask,
                ) = self.connectors(
                    prompt_embeds, additive_attention_mask, additive_mask=True
                )
            connector_prompt_embeds, connector_audio_prompt_embeds, _, _ = (
                _load_ltx2_contexts_if_requested(
                    pos_embeds=connector_prompt_embeds,
                    pos_audio_embeds=connector_audio_prompt_embeds,
                    neg_embeds=None,
                    neg_audio_embeds=None,
                )
            )

            _dump_ltx2_contexts_if_requested(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                neg_prompt_embeds=None,
                neg_prompt_attention_mask=None,
                pos_embeds=connector_prompt_embeds,
                pos_audio_embeds=connector_audio_prompt_embeds,
                pos_mask=connector_mask,
            )

            batch.prompt_embeds = [connector_prompt_embeds]
            batch.audio_prompt_embeds = [connector_audio_prompt_embeds]
            batch.prompt_attention_mask = connector_mask

        return batch
