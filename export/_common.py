"""
Shared helpers for the two SA3 export entry scripts (export_text_to_audio.py /
export_audio_to_audio.py).

The torch.export → MLX-lower → write-.pte step and the sidecar/tokenizer-bundle
emission are no longer hand-rolled here: both are now owned by the downstreamed
``neural_tilde.GenModule`` exporter (``GenModule.export_to_pte``). What remains is
the SA3-specific glue: the latent/sample-rate constants, locating the HuggingFace
tokenizer to hand to ``GenModule.register_tokenizer``, and baking a LoRA into the
model's weights before it is handed to the pipeline builders.
"""

import json
import os
import re

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


# --- LoRA ---------------------------------------------------------------------

def _sanitize(s: str) -> str:
    """Filename-safe tag fragment: separators become dashes, everything else is dropped.

    Dropping rather than dashing keeps checkpoint names readable (``step=1500`` →
    ``step1500``, not ``step-1500``). The characters that actually matter are the ones
    Max tokenizes a message on (space, comma, semicolon) and path separators; dots are
    only dropped for tidiness, since GenModule always appends ``.pte`` and both its
    sidecar and tokenizer naming agree on stripping exactly that.
    """
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^A-Za-z0-9+-]+", "", s)
    return re.sub(r"-{2,}", "-", s).strip("-")


def lora_tag(lora_ckpt_paths, strength: float, lora_name: str = None) -> str:
    """Filename tag identifying the LoRA baked into an export ("" when there is none).

    The sidecar has no provenance field, so this tag is the *only* record of what was
    baked into a .pte. That is why the strength is part of it: it is frozen into the
    weights at merge time, so the same checkpoint at 2.0 and 4.0 are different models
    that would otherwise silently overwrite each other's ~3 GB artifact.

    An explicit `lora_name` replaces the whole tag, strength included. It is *required*
    when stacking, because a tag derived from one checkpoint name cannot identify a
    stack: [a, b] and [a, c] would collide, and for DoRA the order matters too.
    """
    if not lora_ckpt_paths:
        return ""
    if lora_name:
        return _sanitize(lora_name)
    if len(lora_ckpt_paths) > 1:
        raise SystemExit(
            f"error: pass --lora-name when stacking {len(lora_ckpt_paths)} LoRAs — a tag "
            f"derived from the first checkpoint alone cannot identify the stack.")
    stem = _sanitize(os.path.splitext(os.path.basename(lora_ckpt_paths[0]))[0])
    return f"{stem}-s{f'{strength:g}'.replace('.', '-')}"


def _domain(name: str):
    """The music/sfx axis of a model or base_model name, or None if it says neither.

    `MODELS` is only small-music / small-sfx, so this axis is the whole discriminator.
    The checkpoint's `base_model` is a free-form training-side string ("sa3-sm-music")
    with no canonical mapping to the CLI names, hence substring matching. Note this is
    a different axis from model_configs.base_models, which means "undistilled".
    """
    name = (name or "").lower()
    hits = {d for d in ("music", "sfx") if d in name}
    return hits.pop() if len(hits) == 1 else None


def _read_lora_configs(lora_ckpt_paths):
    """The `lora_config` metadata of each checkpoint, in order."""
    from stable_audio_3.models.lora import load_lora_checkpoint

    return [load_lora_checkpoint(p)[1] for p in lora_ckpt_paths]


def _check_base_model(lora_ckpt_paths, configs, model_name: str, force: bool) -> None:
    """Refuse to bake a LoRA into a model it wasn't trained on.

    Metadata is the only signal available: small-music and small-sfx are architecturally
    identical, so a music LoRA loads onto sfx with every tensor matching and is silently
    wrong. Shapes cannot catch this.
    """
    want = _domain(model_name)
    for path, cfg in zip(lora_ckpt_paths, configs):
        name = os.path.basename(path)
        base = cfg.get("base_model")
        got = _domain(base)

        if got is None:
            print(f"    NOTE: {name}: base_model={base!r} — can't tell what it was trained "
                  f"on, merging into {model_name!r} unchecked")
            continue

        # "sa3-sm-music-base" would pass a bare "music" test, but MODELS are all distilled.
        undistilled = bool(re.search(r"-base\b", (base or "").lower()))
        if want is not None and (got != want or undistilled):
            why = (f"was trained on the undistilled {base!r}" if got == want
                   else f"was trained on {base!r}")
            msg = (f"LoRA {name} {why} but you are exporting {model_name!r}. "
                   f"Merging it would bake in garbage.")
            if not force:
                raise SystemExit(f"error: {msg} Pass --force to merge anyway.")
            print(f"    WARNING: {msg} Merging anyway (--force).")


def _warn_on_stacking(lora_ckpt_paths, configs) -> None:
    """DoRA/BoRA do not stack cleanly — the last adapter's magnitude wins.

    Parametrizations compose as p1(p0(W)), and the DoRA forward *sets* row norms to its
    own `magnitude` rather than adding to them, so a second DoRA overwrites the first's.
    Measured on the two checkpoints in lora_checkpoints/: the stacked row-norms match the
    last adapter's magnitude to 2e-6 while differing from the first's by 6e-2, and
    reversing the order changes the result.
    """
    if len(lora_ckpt_paths) < 2:
        return
    kinds = [c.get("adapter_type", "lora") for c in configs]
    if any(k.startswith(("dora", "bora")) for k in kinds):
        print(f"    WARNING: stacking {len(lora_ckpt_paths)} adapters of type {kinds} — "
              f"DoRA/BoRA do not compose additively. The last adapter's magnitude "
              f"overwrites the earlier ones and the result depends on the order given.")


def _assert_lora_landed(sa3, lora_ckpt_paths) -> None:
    """Fail if the checkpoint(s) never actually reached the model.

    `load_and_apply_loras` calls `load_state_dict(..., strict=False)` and discards the
    result, so a checkpoint whose keys don't line up loads *nothing* and leaves every
    adapter at its zero init. A zero-init DoRA reconstructs the base weight, so the merge
    becomes a no-op and we would export the base model under a LoRA filename — with the
    filename being the only provenance the sidecar can carry, that artifact is
    indistinguishable from a real LoRA export. `lora_B` / `M_xs` are zero-initialised, so
    an all-zero adapter means nothing landed.
    """
    from stable_audio_3.models.lora import get_lora_layers

    layers = (get_lora_layers(sa3.model.model)
              + get_lora_layers(sa3.model.conditioner))
    names = ", ".join(os.path.basename(p) for p in lora_ckpt_paths)
    if not layers:
        raise SystemExit(f"error: LoRA {names} patched no layers at all — nothing to merge.")

    def _at_zero_init(p):
        for attr in ("lora_B", "M_xs"):
            t = getattr(p, attr, None)
            if t is not None:
                return bool((t == 0).all())
        return False

    dead = [n for n, p in layers if _at_zero_init(p)]
    if len(dead) == len(layers):
        raise SystemExit(
            f"error: LoRA {names} did not match {sa3.model_config.get('model_type', 'the model')} "
            f"— all {len(layers)} adapters are still at zero init, so merging would be a no-op "
            f"and the export would be the base model under a LoRA name.")
    if dead:
        print(f"    WARNING: {len(dead)}/{len(layers)} adapters are still at zero init — "
              f"those layers did not load from the checkpoint.")


def merge_lora_for_export(sa3, lora_ckpt_paths, strength: float, model_name: str,
                          force: bool = False) -> None:
    """Load LoRA checkpoint(s) onto `sa3` and bake them into the base weights, in place.

    Must run after ``StableAudioModel.from_pretrained`` and before any
    ``build_*_pipeline``: the builders take the DiT and the conditioner *by reference*,
    so merging here lands the weights inside the exported pipeline with no further
    plumbing — and they structurally rewrite the autoencoder, which must happen after
    the LoRA is resolved.

    ``merge_lora`` evaluates each parametrization once, writes the result into
    ``.weight`` and removes the hook, leaving plain ``nn.Linear`` for torch.export to
    trace. The merge is exact (bitwise-identical weights to the live parametrization).

    `strength` is baked in and frozen — the exported .pte cannot retune it. It is not a
    0..1 blend: for DoRA it is discontinuous at 0, where the forward early-outs to the
    base weights while any epsilon above already applies the trained magnitude.
    """
    from stable_audio_3.models.lora import enable_lora, has_lora, merge_lora

    if strength == 0:
        raise SystemExit(
            "error: --lora-strength 0 disables the LoRA entirely (the adapter forward "
            "early-outs to the base weights), which would export the base model under a "
            "LoRA filename. Omit --lora-ckpt-path to export the base model.")

    configs = _read_lora_configs(lora_ckpt_paths)
    _check_base_model(lora_ckpt_paths, configs, model_name, force)
    _warn_on_stacking(lora_ckpt_paths, configs)

    sa3.load_lora(lora_ckpt_paths)
    _assert_lora_landed(sa3, lora_ckpt_paths)

    dit, cond = sa3.model.model, sa3.model.conditioner

    # Defensive: adapters are enabled at construction and only the DiT's per-step sigma
    # gating disables them, which never runs here (we never call generate). Merging a
    # disabled adapter bakes an identity and silently drops the LoRA, so make it explicit.
    enable_lora(dit)
    enable_lora(cond)

    # Before the merge: the baked weight captures whatever strength is current.
    sa3.set_lora_strength(strength)

    # Scoped to the two subtrees load_lora patched; merge_lora removes *every*
    # parametrization on a module it visits, not just LoRA ones.
    merge_lora(dit)
    merge_lora(cond)

    residual = [n for n, m in sa3.model.named_modules() if hasattr(m, "parametrizations")]
    assert not has_lora(sa3.model) and not residual, (
        f"LoRA merge left {len(residual)} parametrized module(s) behind; torch.export "
        f"would trace the hooks instead of plain weights: {residual[:3]}")
