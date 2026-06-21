"""
Shared helpers for the two SA3 export entry scripts (export_text_to_audio.py /
export_audio_to_audio.py).

The torch.export → MLX-lower → write-.pte step and the sidecar/tokenizer-bundle
emission are no longer hand-rolled here: both are now owned by the downstreamed
``neural_tilde.GenModule`` exporter (``GenModule.export_to_pte``). What remains is
the SA3-specific glue: the latent/sample-rate constants and locating the HuggingFace
tokenizer to hand to ``GenModule.register_tokenizer``.
"""

import json
import os

MAX_LENGTH = 256          # SA3 conditioner token length
SAMPLES_PER_LATENT = 4096
SAMPLE_RATE = 44100


def default_target_sec(t_lat: int) -> float:
    """Clip duration in seconds for T_lat (T_lat·4096 / 44.1 kHz), rounded to the nearest
    integer second. Returned as float so register_attribute keeps target_sec float32
    (an int default would be inferred as int64)."""
    return float(round(t_lat * SAMPLES_PER_LATENT / SAMPLE_RATE))


def register_sa3_tokenizer(gm, model_name: str, max_length: int = MAX_LENGTH) -> None:
    """Register the SA3 prompt tokenizer on a ``GenModule`` so its standalone bundle
    (``<stem>.tokenizer.json`` + ``<stem>.tokenizer.config.json``) is written next to
    the ``.pte`` on export.

    Locates the shared ``t5gemma-b-b-ul2`` fast tokenizer in the HF cache (reusing
    ``emit_sidecars.find_tokenizer_dir``) and forwards its pad settings. ``ids_key`` /
    ``mask_key`` are the model's token-condition input names, so ``neural.tokenizer``'s
    output dictionary keys line up with what ``neural.gen~`` feeds the model.
    """
    from emit_sidecars import find_tokenizer_dir, repo_for

    tok_dir = find_tokenizer_dir(repo_for(model_name))
    with open(os.path.join(tok_dir, "tokenizer_config.json")) as f:
        hf_cfg = json.load(f)
    gm.register_tokenizer(
        os.path.join(tok_dir, "tokenizer.json"),
        max_length=max_length,
        pad_token=hf_cfg.get("pad_token", "<pad>"),
        padding_side=hf_cfg.get("padding_side", "right"),
        ids_key="token_ids",        # = the model's token-ids condition input name
        mask_key="attention_mask",  # = the model's attention-mask condition input name
    )
