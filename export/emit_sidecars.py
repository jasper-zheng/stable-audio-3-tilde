"""
Emit the neural~ C++ runtime sidecars for an exported SA3 .pte.

Writes, beside the .pte in export/artifacts/ (model-stem = {model}_text_to_audio_T{t}
for text-to-audio, {model}_audio_to_audio_T{t} for audio-to-audio):
  {stem}.json                   model sidecar (generative profile, NO tokenizer block)
  {stem}.tokenizer.config.json  tokenizer config (neural.tokenizer loads this)
  {stem}.tokenizer.json         the HF fast tokenizer (copied from the HF cache)

Why the split: tokenization is owned by the standalone `neural.tokenizer` Max object, so
the tokenizer settings live in their own config file rather than inside the model sidecar.
The host discovers the config by the `<pte-stem>.tokenizer.config.json` convention. The
config's `ids_key`/`mask_key` are the model's token-input names, so the tokenizer's output
dictionary keys line up with what `neural.gen~` feeds the model.

Standalone (no torch/transformers needed — just reads the HF tokenizer cache):
    .venv/bin/python export/emit_sidecars.py --model small-music --t-lat 32          # text-to-audio
    .venv/bin/python export/emit_sidecars.py --model small-music --t-lat 32 --a2a    # audio-to-audio
    # optional: --cfg  --steps 8  --max-length 256  --tokenizer-dir <dir>  --out <artifacts>
Also importable from the exporters via emit_all(...) / emit_all_a2a(...).
"""

import argparse
import glob
import json
import os
import shutil

SUBFOLDER = "t5gemma-b-b-ul2"
ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")


def repo_for(model: str) -> str:
    """HF repo id for a model name (matches stable_audio_3/model_configs.py)."""
    return f"stabilityai/stable-audio-3-{model}"


def find_tokenizer_dir(repo_id: str = None) -> str:
    """Locate the `t5gemma-b-b-ul2` tokenizer in the HF cache.

    Prefers the given model's own repo; falls back to any cached SA3 small-* snapshot
    (the conditioner tokenizer is the shared `t5gemma-b-b-ul2` across the small models).
    """
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    pats = []
    if repo_id:
        pats.append(os.path.join(cache, f"models--{repo_id.replace('/', '--')}/snapshots/*/{SUBFOLDER}"))
    pats.append(os.path.join(cache, f"models--stabilityai--stable-audio-3-*/snapshots/*/{SUBFOLDER}"))
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    raise SystemExit(
        f"ERROR: tokenizer '{SUBFOLDER}' not found in HF cache (tried: {pats}).\n"
        f"Pass --tokenizer-dir explicitly."
    )


def _negative_prompt_inputs() -> list:
    """The negative-prompt condition inputs for the CFG variants. A second by-name
    `condition` set — the host tokenizes the negative prompt with the same tokenizer
    and routes it here. An all-pad negative prompt ⇒ pure unconditional (null) branch.
    """
    return [
        {"name": "neg_token_ids",      "role": "condition",
         "shape": [1, 256],                    "dtype": "int64",
         "description": "Negative-prompt token ids (same tokenizer as token_ids). "
                        "All-pad ⇒ pure unconditional guidance."},
        {"name": "neg_attention_mask", "role": "condition",
         "shape": [1, 256],                    "dtype": "int32"},
    ]


def _cfg_scale_input() -> dict:
    """The runtime cfg_scale attribute for the CFG variants."""
    return {"name": "cfg_scale", "role": "attribute",
            "shape": [1],                          "dtype": "float32",
            "min": 1.0, "max": 15.0, "default": 1.0,
            "description": "Classifier-free guidance scale (full APG). 1.0 = no guidance "
                           "(output matches the non-CFG model); try ~7.0 for stronger prompt "
                           "adherence. Off-label for distilled checkpoints — may degrade quality."}


def model_sidecar(t_lat: int, steps: int, cfg: bool = False) -> dict:
    """Generative-profile sidecar for text_to_audio_T{t_lat}. NO tokenizer block.

    cfg=True ⇒ the CFG variant: adds the negative-prompt condition set + a runtime
    cfg_scale attribute (classifier-free guidance).
    """
    audio_len = 4096 * t_lat
    inputs = [
        {"name": "token_ids",      "role": "condition",
         "shape": [1, 256],                    "dtype": "int64"},
        {"name": "attention_mask", "role": "condition",
         "shape": [1, 256],                    "dtype": "int32"},
    ]
    if cfg:
        inputs += _negative_prompt_inputs()
    inputs += [
        {"name": "target_sec",     "role": "attribute",
         "shape": [1],                         "dtype": "float32",
         "min": 0.0, "max": 384.0, "default": 3.0,
         "description": "Total length in seconds to condition the generation on."},
    ]
    if cfg:
        inputs.append(_cfg_scale_input())
    inputs += [
        {"name": "noise_x",        "role": "noise",
         "shape": [1, 256, t_lat],             "dtype": "float32"},
        {"name": "noise_steps",    "role": "noise",
         "shape": [steps - 1, 1, 256, t_lat],  "dtype": "float32"},
    ]
    return {
        "kind": "gen",
        "executorch_version": "1.3.1",
        "delegate": "mlx",
        "seed": 0,
        "methods": {
            "forward": {
                "inputs": inputs,
                "output": {
                    "name": "audio", "role": "buffer", "dtype": "float32",
                    "shape": [1, 2, audio_len],
                    "channels": 2, "length": audio_len,
                    "sample_rate": 44100,
                },
            }
        },
        "attributes": [],
    }


def model_sidecar_a2a(t_lat: int, steps: int, default_strength: float = 1.0,
                      cfg: bool = False) -> dict:
    """Generative-profile sidecar for audio_to_audio_T{t_lat} — audio-to-audio with an
    inpaint mask + init audio. The inputs are the superset of the init-audio modes; the
    mode is selected by the input values (defaults = plain text-to-audio):

      - audio_in:         `buffer` — init / clip to inpaint (host pre-resampled→44.1k stereo,
                          cropped/padded). Unset ⇒ host feeds silence (⇒ plain text-to-audio).
      - inpaint_mask:     `condition` per-latent-frame keep-mask (1=keep, 0=regenerate),
                          supplied by name as a flat list; unset ⇒ 0 ⇒ no preservation.
      - noise_level:      `attribute` denoising strength σmax. 1.0 (default) ignores audio_in
                          for the init mix; lower values track the init clip (audio-to-audio).

    Mode cheatsheet:  text-to-audio = defaults;  audio-to-audio = audio_in + noise_level<1;
                      inpaint = audio_in + inpaint_mask (noise_level=1.0).

    cfg=True ⇒ the CFG variant: adds the negative-prompt condition set + a runtime
    cfg_scale attribute (classifier-free guidance). The input order mirrors the exported
    AudioToAudioPipelineCFG.forward signature exactly (load-bearing for the host).
    """
    audio_len = 4096 * t_lat
    inputs = [
        {"name": "audio_in",       "role": "buffer",
         "shape": [1, 2, audio_len],           "dtype": "float32",
         "channels": 2, "length": audio_len,
         "sample_rate": 44100,
         "description": "Init audio (audio-to-audio) or clip to inpaint; host resamples to "
                        "44.1 kHz stereo and crops/pads to length. Unset ⇒ silence ⇒ pure "
                        "text-to-audio."},
        {"name": "token_ids",      "role": "condition",
         "shape": [1, 256],                    "dtype": "int64"},
        {"name": "attention_mask", "role": "condition",
         "shape": [1, 256],                    "dtype": "int32"},
    ]
    if cfg:
        inputs += _negative_prompt_inputs()
    inputs += [
        {"name": "inpaint_mask",   "role": "condition",
         "shape": [1, 1, t_lat],               "dtype": "float32",
         "description": "Per-latent-frame keep-mask: 1.0 = keep (preserve), 0.0 = regenerate. "
                        f"{t_lat} frames span the clip ({audio_len} samples @ 44.1 kHz). "
                        "Supplied by name as a flat list; unset ⇒ zero-filled ⇒ no preservation "
                        "(text-to-audio / audio-to-audio)."},
        {"name": "target_sec",     "role": "attribute",
         "shape": [1],                         "dtype": "float32",
         "min": 0.0, "max": 384.0, "default": 3.0,
         "description": "Total length in seconds to condition the generation on."},
        {"name": "noise_level",    "role": "attribute",
         "shape": [1],                         "dtype": "float32",
         "min": 0.0, "max": 1.0, "default": default_strength,
         "description": "Denoising strength (sigma_max). 1.0 ignores audio_in for the init mix "
                        "(text-to-audio / inpainting); lower values track the init clip more "
                        "closely (audio-to-audio)."},
    ]
    if cfg:
        inputs.append(_cfg_scale_input())
    inputs += [
        {"name": "noise_x",        "role": "noise",
         "shape": [1, 256, t_lat],             "dtype": "float32"},
        {"name": "noise_steps",    "role": "noise",
         "shape": [steps - 1, 1, 256, t_lat],  "dtype": "float32"},
    ]
    return {
        "kind": "gen",
        "executorch_version": "1.3.1",
        "delegate": "mlx",
        "seed": 0,
        "methods": {
            "forward": {
                "inputs": inputs,
                "output": {
                    "name": "audio", "role": "buffer", "dtype": "float32",
                    "shape": [1, 2, audio_len],
                    "channels": 2, "length": audio_len,
                    "sample_rate": 44100,
                },
            }
        },
        "attributes": [],
    }


def tokenizer_config(tok_filename: str, hf_cfg: dict, max_length: int) -> dict:
    """Standalone config consumed by neural.tokenizer."""
    return {
        "type": "tokenizers-cpp",
        "tokenizer_file": tok_filename,        # relative to this config file
        "max_length": max_length,
        "padding": "max_length",
        "pad_token": hf_cfg.get("pad_token", "<pad>"),
        "padding_side": hf_cfg.get("padding_side", "right"),
        "ids_key": "token_ids",                # = the model's token-ids condition input name
        "mask_key": "attention_mask",          # = the model's attention-mask condition input name
    }


def _emit_tokenizer(stem: str, out: str, tokenizer_dir: str, max_length: int,
                    repo_id: str = None) -> None:
    """Copy the HF fast tokenizer next to the .pte and emit its standalone config."""
    tok_dir = tokenizer_dir or find_tokenizer_dir(repo_id)
    with open(os.path.join(tok_dir, "tokenizer_config.json")) as f:
        hf_cfg = json.load(f)
    tok_filename = f"{stem}.tokenizer.json"
    shutil.copyfile(os.path.join(tok_dir, "tokenizer.json"),
                    os.path.join(out, tok_filename))
    print(f"  wrote {tok_filename}  (from {tok_dir})")
    with open(os.path.join(out, f"{stem}.tokenizer.config.json"), "w") as f:
        json.dump(tokenizer_config(tok_filename, hf_cfg, max_length), f, indent=2)
    print(f"  wrote {stem}.tokenizer.config.json")


def emit_all(model: str = "small-music", t_lat: int = 32, steps: int = 8,
             out: str = ARTIFACTS, tokenizer_dir: str = None, max_length: int = 256,
             cfg: bool = False) -> None:
    """Sidecar bundle for the text-to-audio model: {model}_text_to_audio[_cfg]_T{t_lat}."""
    stem = f"{model}_text_to_audio{'_cfg' if cfg else ''}_T{t_lat}"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{stem}.json"), "w") as f:
        json.dump(model_sidecar(t_lat, steps, cfg=cfg), f, indent=2)
    print(f"  wrote {stem}.json")
    _emit_tokenizer(stem, out, tokenizer_dir, max_length, repo_for(model))


def emit_all_a2a(model: str = "small-music", t_lat: int = 32, steps: int = 8,
                 out: str = ARTIFACTS, tokenizer_dir: str = None, max_length: int = 256,
                 default_strength: float = 1.0, cfg: bool = False) -> None:
    """Sidecar bundle for the audio-to-audio model: {model}_audio_to_audio[_cfg]_T{t_lat}."""
    stem = f"{model}_audio_to_audio{'_cfg' if cfg else ''}_T{t_lat}"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{stem}.json"), "w") as f:
        json.dump(model_sidecar_a2a(t_lat, steps, default_strength, cfg=cfg), f, indent=2)
    print(f"  wrote {stem}.json")
    _emit_tokenizer(stem, out, tokenizer_dir, max_length, repo_for(model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small-music",
                    help="model name (stem prefix + tokenizer repo); e.g. small-music, small-sfx")
    ap.add_argument("--a2a", action="store_true",
                    help="emit the audio-to-audio sidecar instead of text-to-audio")
    ap.add_argument("--cfg", action="store_true",
                    help="emit the CFG variant sidecar (negative prompt + cfg_scale)")
    ap.add_argument("--t-lat", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8,
                    help="pingpong steps (re-noise count = steps - 1)")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--out", default=ARTIFACTS)
    args = ap.parse_args()
    emit = emit_all_a2a if args.a2a else emit_all
    emit(model=args.model, t_lat=args.t_lat, steps=args.steps, out=args.out,
         tokenizer_dir=args.tokenizer_dir, max_length=args.max_length, cfg=args.cfg)


if __name__ == "__main__":
    main()
