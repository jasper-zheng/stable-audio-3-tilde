"""
SA3 audio-to-audio exporter (init audio + inpaint mask) → one ExecuTorch `.pte`
(via neural_tilde.GenModule), lowered to MLX (macOS) or CUDA (Windows x64 + NVIDIA).

Adds the SAME-S encoder + a runtime denoising-strength control + conditioning-based
inpainting on top of the text-to-audio pipeline (defined in `sa3_modules.py`). The
pipeline is described to the downstreamed `neural_tilde.GenModule` exporter (register
each input by name + role — including the `buffer` init waveform `neural.gen~` reads
from a Max `buffer~` — then one `forward` generation method) and lowered to a single
delegated `.pte` by `GenModule.export_to_pte`. One ExecuTorch call: an init/clip
waveform + pre-tokenized prompt + keep-mask + duration + strength + seeded noise →
stereo PCM. The mode falls out of the input values (mirrors model.generate): unset
audio_in + mask=0 + strength=1 ⇒ plain text-to-audio; audio_in + strength<1 ⇒
audio-to-audio; audio_in + mask ⇒ inpaint / continuation.

Forward contract (positional; order + dtype are load-bearing for the host):
  audio_in         (1, 2, T_lat*4096)  float32  init / clip @44.1k stereo (host pre-resampled/cropped); unset⇒silence  (buffer)
  token_ids        (1, 256)            int64    1=real token, 0=pad       (condition)
  attention_mask   (1, 256)            int32    1=real token, 0=pad       (condition)
  inpaint_mask     (1, 1, T_lat)       float32  per-latent-frame keep-mask (1=keep, 0=regenerate); unset⇒0  (condition)
  target_sec       (1,)                float32  duration conditioning [0, 384]   (attribute)
  noise_level      (1,)                float32  denoising strength σmax in (0,1]; 1.0 ignores audio_in for the mix  (attribute)
  noise_x          (1, 256, T_lat)     float32  initial Gaussian noise (mixed with init_latents in-graph)  (noise)
  noise_steps      (7, 1, 256, T_lat)  float32  per-step re-noise (steps 0-6)    (noise)
  → audio          (1, 2, T_lat*4096)  float32  PCM @ 44.1 kHz (T_lat=32 ⇒ ~2.97 s)

With --cfg the contract adds neg_token_ids/neg_attention_mask (after attention_mask)
and a runtime cfg_scale (after noise_level) — classifier-free guidance, ~2× DiT cost.

With --lora-ckpt-path the LoRA/DoRA is merged into the weights before tracing, so the
forward contract is unchanged — a LoRA export differs only in its weights and filename.
--lora-strength is baked in and frozen: retuning it means re-exporting.

--delegate picks the backend (default: mlx on macOS, cuda elsewhere). A CUDA export
needs an NVIDIA GPU + MSVC on the export machine, writes its weights to a `.ptd` beside
the `.pte`, and gets a `_cuda` filename suffix (it only runs on Windows/NVIDIA).

Run:
  python export/export_audio_to_audio.py --model small-music --t-lat 32 [--cfg]
       [--delegate mlx|cuda] [--init-noise-level 1.0]
       [--lora-ckpt-path lora.safetensors] [--lora-strength 2.0]

Produces, in export/artifacts/ (`{lora}` = the LoRA tag, absent without --lora-ckpt-path;
`{d}` = `_cuda` for the CUDA delegate, empty for MLX):
  {model}_audio_to_audio[_cfg]_T{t}[_{lora}]{d}.pte + .json + .tokenizer.config.json + .tokenizer.json
  (+ .ptd weights for CUDA)
"""

import argparse
import os
import sys

from neural_tilde import GenModule

sys.path.insert(0, os.path.dirname(__file__))
from sa3_modules import build_audio_to_audio_pipeline, build_audio_to_audio_pipeline_cfg, NUM_STEPS
from _common import (register_sa3_tokenizer, default_target_sec, merge_lora_for_export,
                     lora_tag, DELEGATES, default_delegate, artifact_suffix, export_kwargs,
                     locate_flatc, MAX_LENGTH, SAMPLES_PER_LATENT, SAMPLE_RATE)

from stable_audio_3.model import StableAudioModel

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

MODELS = ["small-music", "small-sfx"]  # distilled checkpoints only (see assert_distilled)
LATENT_CHANNELS = 256  # DiT latent channels (noise_x / noise_steps channel dim)


def build_gen_module(model, t_lat: int, default_strength: float, cfg: bool) -> GenModule:
    """Describe `model` to GenModule: register each forward input by name + role (the
    init waveform is a `buffer` read from a Max buffer~) and the single `forward`
    generation method, in the exact (load-bearing) forward order."""
    audio_len = t_lat * SAMPLES_PER_LATENT
    gm = GenModule(model)

    gm.register_buffer_input(
        "audio_in", channels=2, length=audio_len, sample_rate=SAMPLE_RATE,
        description="Init audio (audio-to-audio) or clip to inpaint; host resamples to "
                    "44.1 kHz stereo and crops/pads to length.")
    gm.register_condition("token_ids", [1, MAX_LENGTH], "int64")
    gm.register_condition("attention_mask", [1, MAX_LENGTH], "int32")
    if cfg:
        gm.register_condition(
            "neg_token_ids", [1, MAX_LENGTH], "int64",
            "Negative-prompt token ids (same tokenizer as token_ids). "
            "All-pad equal unconditional guidance.")
        gm.register_condition("neg_attention_mask", [1, MAX_LENGTH], "int32")
    gm.register_condition(
        "inpaint_mask", [1, 1, t_lat], "float32",
        "Per-latent-frame mask: 1.0 = keep (preserve), 0.0 = regenerate. "
        f"{t_lat} frames span the clip ({audio_len} samples at 44.1 kHz). "
        "Supplied by name as a flat list of floats in [0,1]. ")
    gm.register_attribute("target_sec", default_target_sec(t_lat), 0.0, 384.0,
                          "Total length in seconds to condition the generation on.")
    gm.register_attribute(
        "noise_level", default_strength, 0.0, 1.0,
        "Denoising strength (sigma_max). 1.0 ignores audio_in for the init mix; "
        "lower values track the init clip closer.")
    if cfg:
        gm.register_attribute(
            "cfg_scale", 1.0, 1.0, 15.0,
            "Classifier-free guidance scale (full APG). 1.0 = no guidance (output matches the non-CFG model)")
    gm.register_noise("noise_x", [1, LATENT_CHANNELS, t_lat])
    gm.register_noise("noise_steps", [NUM_STEPS - 1, 1, LATENT_CHANNELS, t_lat])

    inputs = ["audio_in", "token_ids", "attention_mask"]
    if cfg:
        inputs += ["neg_token_ids", "neg_attention_mask"]
    inputs += ["inpaint_mask", "target_sec", "noise_level"]
    if cfg:
        inputs += ["cfg_scale"]
    inputs += ["noise_x", "noise_steps"]

    gm.register_method("forward", inputs=inputs, out_channels=2,
                       out_length=audio_len, out_sample_rate=SAMPLE_RATE)
    return gm


def run_export(model_name, t_lat, init_noise_level, device="cpu", cfg=False, seed=0,
               *, loras=None, lora_strength=2.0, lora_name=None, force=False,
               delegate=None):
    assert t_lat % 2 == 0, f"T_lat must be even (got {t_lat})"
    assert 0.0 < init_noise_level <= 1.0, f"init_noise_level must be in (0, 1] (got {init_noise_level})"
    tag = lora_tag(loras, lora_strength, lora_name)
    delegate = delegate or default_delegate()
    stem = (f"{model_name}_audio_to_audio{'_cfg' if cfg else ''}_T{t_lat}"
            f"{'_' + tag if tag else ''}{artifact_suffix(delegate)}")
    print(f"SA3 audio-to-audio export  ({model_name}, T_lat={t_lat}{', +CFG' if cfg else ''}"
          f"{', +LoRA' if loras else ''}, {delegate})")

    print(f"  Loading {model_name} from HuggingFace...")
    sa3 = StableAudioModel.from_pretrained(model_name, device=device, model_half=False)

    # Before the builders: they take the DiT/conditioner by reference, so the merged
    # weights land in the pipeline on their own.
    if loras:
        print(f"  Merging LoRA at strength {lora_strength} (baked in, frozen)...")
        merge_lora_for_export(sa3, loras, lora_strength, model_name, force=force)

    print("  Building pipeline...")
    if cfg:
        model, _ = build_audio_to_audio_pipeline_cfg(sa3, t_lat, device)
    else:
        model, _ = build_audio_to_audio_pipeline(sa3, t_lat, device)

    print("  Describing model for neural.gen~ (GenModule)...")
    gm = build_gen_module(model, t_lat, init_noise_level, cfg)
    register_sa3_tokenizer(gm, model_name, MAX_LENGTH)
    gm.set_seed(seed)

    print(f"  Exporting + lowering to {delegate.upper()} (GenModule.export_to_pte)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    locate_flatc()
    pte_path = gm.export_to_pte(os.path.join(OUTPUT_DIR, stem),
                                delegate=delegate, **export_kwargs(delegate))
    return pte_path


def main():
    parser = argparse.ArgumentParser(
        description="Export the SA3 audio-to-audio (init audio + inpaint mask) pipeline to a "
                    "single ExecuTorch .pte via neural_tilde.GenModule")
    parser.add_argument("--model", default="small-music", choices=MODELS,
                        help="distilled checkpoint to export (base models are not supported)")
    parser.add_argument("--t-lat", type=int, default=32, help="T_lat (must be even)")
    parser.add_argument("--init-noise-level", type=float, default=0.4,
                        help="default denoising strength σmax in (0,1] (runtime input)")
    parser.add_argument("--cfg", action="store_true",
                        help="export the CFG variant (runtime cfg_scale + negative prompt; ~2x DiT cost)")
    parser.add_argument("--seed", type=int, default=0,
                        help="default RNG seed written to the sidecar (host may override at runtime)")
    parser.add_argument("--delegate", default=default_delegate(), choices=DELEGATES,
                        help="ExecuTorch backend: mlx (macOS / Apple Silicon) or cuda (Windows x64 + "
                             "NVIDIA). Defaults to the one for this platform.")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device when loading the model; doesn't affect the .pte backend device")

    # --- LoRA ---
    parser.add_argument("--lora-ckpt-path", action="append", dest="loras", metavar="PATH",
                        help="LoRA/DoRA checkpoint to bake into the exported weights. "
                             "Repeat to stack multiple LoRAs.")
    parser.add_argument("--lora-strength", type=float, default=2.0,
                        help="LoRA strength, baked in at export time (frozen; the .pte cannot "
                             "retune it, so a different strength means a different export)")
    parser.add_argument("--lora-name", default=None, metavar="TAG",
                        help="filename tag for the LoRA, replacing the one derived from the "
                             "checkpoint name + strength")
    parser.add_argument("--force", action="store_true",
                        help="merge the LoRA even if it was trained on a different base model")
    args = parser.parse_args()

    if not args.loras:
        if args.lora_name:
            parser.error("--lora-name requires --lora-ckpt-path")
        if args.force:
            parser.error("--force requires --lora-ckpt-path")

    pte_path = run_export(args.model, args.t_lat, args.init_noise_level,
                          args.device, args.cfg, args.seed,
                          loras=args.loras, lora_strength=args.lora_strength,
                          lora_name=args.lora_name, force=args.force,
                          delegate=args.delegate)
    print(f"\nExport complete: {pte_path}")


if __name__ == "__main__":
    main()
