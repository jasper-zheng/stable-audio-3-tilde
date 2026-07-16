"""
SA3 text-to-audio exporter → one ExecuTorch MLX `.pte` (via neural_tilde.GenModule).

Composes T5Gemma text conditioning + 8-step pingpong DiT + SAME-S decode (defined in
`sa3_modules.py`), describes it to the downstreamed `neural_tilde.GenModule` exporter
(register each input by name + role, then one `forward` generation method), and lets
`GenModule.export_to_pte` trace, MLX-lower and write the `.pte` + sidecars that the
`neural.gen~` Max external loads and runs. One ExecuTorch call: pre-tokenized prompt +
duration + seeded noise → stereo PCM audio.

Forward contract (positional; order + dtype are load-bearing for the host):
  token_ids       (1, 256)            int64    pre-tokenized prompt      (condition)
  attention_mask  (1, 256)            int32    1=real token, 0=pad       (condition)
  target_sec      (1,)                float32  duration conditioning     (attribute, [0, 384])
  noise_x         (1, 256, T_lat)     float32  initial Gaussian noise    (noise)
  noise_steps     (7, 1, 256, T_lat)  float32  per-step re-noise         (noise)
  → audio         (1, 2, T_lat*4096)  float32  PCM @ 44.1 kHz (T_lat=32 ⇒ ~2.97 s)

With --cfg the contract adds neg_token_ids/neg_attention_mask (after attention_mask)
and a runtime cfg_scale (after target_sec) — classifier-free guidance, ~2× DiT cost.

With --lora-ckpt-path the LoRA/DoRA is merged into the weights before tracing, so the
forward contract is unchanged — a LoRA export differs only in its weights and filename.
--lora-strength is baked in and frozen: retuning it means re-exporting.

Run:
  python export/export_text_to_audio.py --model small-music --t-lat 32 [--cfg]
       [--lora-ckpt-path lora.safetensors] [--lora-strength 2.0]

Produces, in export/artifacts/ (`{lora}` = the LoRA tag, absent without --lora-ckpt-path):
  {model}_text_to_audio[_cfg]_T{t}[_{lora}].pte                    the model
  {model}_text_to_audio[_cfg]_T{t}[_{lora}].json                   model sidecar (neural.gen~ profile)
  {model}_text_to_audio[_cfg]_T{t}[_{lora}].tokenizer.config.json  tokenizer config
  {model}_text_to_audio[_cfg]_T{t}[_{lora}].tokenizer.json         copied HF tokenizer
"""

import argparse
import os
import sys

from neural_tilde import GenModule

sys.path.insert(0, os.path.dirname(__file__))
from sa3_modules import build_full_pipeline, build_full_pipeline_cfg, NUM_STEPS
from _common import (register_sa3_tokenizer, default_target_sec, merge_lora_for_export,
                     lora_tag, MAX_LENGTH, SAMPLES_PER_LATENT, SAMPLE_RATE)

from stable_audio_3.model import StableAudioModel

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

MODELS = ["small-music", "small-sfx"]  # distilled checkpoints only (see assert_distilled)
LATENT_CHANNELS = 256  # DiT latent channels (noise_x / noise_steps channel dim)


def build_gen_module(model, t_lat: int, cfg: bool) -> GenModule:
    """Describe `model` to GenModule: register each forward input by name + role and the
    single `forward` generation method, in the exact (load-bearing) forward order."""
    audio_len = t_lat * SAMPLES_PER_LATENT
    gm = GenModule(model)

    gm.register_condition("token_ids", [1, MAX_LENGTH], "int64")
    gm.register_condition("attention_mask", [1, MAX_LENGTH], "int32")
    if cfg:
        gm.register_condition(
            "neg_token_ids", [1, MAX_LENGTH], "int64",
            "Negative-prompt token ids (same tokenizer as token_ids). "
            "All-pad equal unconditional guidance.")
        gm.register_condition("neg_attention_mask", [1, MAX_LENGTH], "int32")
    gm.register_attribute("target_sec", default_target_sec(t_lat), 0.0, 384.0,
                          "Total length in seconds to condition the generation on.")
    if cfg:
        gm.register_attribute(
            "cfg_scale", 1.0, 1.0, 15.0,
            "Classifier-free guidance scale (full APG). 1.0 = no guidance (output matches the non-CFG model)")
    gm.register_noise("noise_x", [1, LATENT_CHANNELS, t_lat])
    gm.register_noise("noise_steps", [NUM_STEPS - 1, 1, LATENT_CHANNELS, t_lat])

    inputs = ["token_ids", "attention_mask"]
    if cfg:
        inputs += ["neg_token_ids", "neg_attention_mask"]
    inputs += ["target_sec"]
    if cfg:
        inputs += ["cfg_scale"]
    inputs += ["noise_x", "noise_steps"]

    gm.register_method("forward", inputs=inputs, out_channels=2,
                       out_length=audio_len, out_sample_rate=SAMPLE_RATE)
    return gm


def run_export(model_name, t_lat, device="cpu", cfg=False, seed=0,
               *, loras=None, lora_strength=2.0, lora_name=None, force=False):
    assert t_lat % 2 == 0, f"T_lat must be even (got {t_lat})"
    tag = lora_tag(loras, lora_strength, lora_name)
    stem = f"{model_name}_text_to_audio{'_cfg' if cfg else ''}_T{t_lat}{'_' + tag if tag else ''}"
    print(f"SA3 text-to-audio export  ({model_name}, T_lat={t_lat}{', +CFG' if cfg else ''}"
          f"{', +LoRA' if loras else ''})")

    print(f"  Loading {model_name} from HuggingFace...")
    sa3 = StableAudioModel.from_pretrained(model_name, device=device, model_half=False)

    # Before the builders: they take the DiT/conditioner by reference, so the merged
    # weights land in the pipeline on their own.
    if loras:
        print(f"  Merging LoRA at strength {lora_strength} (baked in, frozen)...")
        merge_lora_for_export(sa3, loras, lora_strength, model_name, force=force)

    print("  Building pipeline...")
    if cfg:
        model, _ = build_full_pipeline_cfg(sa3, t_lat, device)
    else:
        model, _ = build_full_pipeline(sa3, t_lat, device)

    print("  Describing model for neural.gen~ (GenModule)...")
    gm = build_gen_module(model, t_lat, cfg)
    register_sa3_tokenizer(gm, model_name, MAX_LENGTH)
    gm.set_seed(seed)

    print("  Exporting + lowering to MLX (GenModule.export_to_pte)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pte_path = gm.export_to_pte(os.path.join(OUTPUT_DIR, stem),
                                delegate="mlx", decompose_conv=True)
    return pte_path


def main():
    parser = argparse.ArgumentParser(
        description="Export the SA3 text-to-audio pipeline to a single ExecuTorch MLX .pte "
                    "via neural_tilde.GenModule")
    parser.add_argument("--model", default="small-music", choices=MODELS,
                        help="distilled checkpoint to export (base models are not supported)")
    parser.add_argument("--t-lat", type=int, default=32, help="T_lat (must be even)")
    parser.add_argument("--cfg", action="store_true",
                        help="export the CFG variant (runtime cfg_scale + negative prompt; ~2x DiT cost)")
    parser.add_argument("--seed", type=int, default=0,
                        help="default RNG seed written to the sidecar (host may override at runtime)")
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

    pte_path = run_export(args.model, args.t_lat, args.device, args.cfg, args.seed,
                          loras=args.loras, lora_strength=args.lora_strength,
                          lora_name=args.lora_name, force=args.force)
    print(f"\nExport complete: {pte_path}")


if __name__ == "__main__":
    main()
