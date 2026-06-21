"""
SA3 export building blocks — the wrappers + helpers used to export the two
small-music pipelines to a single ExecuTorch MLX `.pte` each.

This module is the one place the export wrappers and graph helpers live;
`export_text_to_audio.py` and `export_audio_to_audio.py` are thin entry points over
`build_full_pipeline` / `build_audio_to_audio_pipeline`.

Two pipelines (each with an optional `*CFG` sibling, selected by the exporter's --cfg):

    FullPipeline                      (text-to-audio)
      ├─ TextConditioning             (T5Gemma encoder + NumberConditioner)
      │    └─ EncoderOnly             (T5GemmaEncoderModel → last_hidden_state)
      └─ AudioGenerator
           ├─ UnrolledDiT             (8-step pingpong DiT, loop unrolled)
           └─ DecoderExport           (SoftNormBottleneck → SAMEDecoder → PatchedPretransform)

    AudioToAudioPipeline              (audio-to-audio with inpaint mask + init audio)
      ├─ TextConditioning
      ├─ EncoderExport                (SAME-S encode: waveform → DiT latents)
      ├─ UnrolledDiTDynLocal          (runtime sigmas + inpaint local_add_cond)
      └─ DecoderExport

The base pipelines export with `cfg_scale=1.0` (no guidance). The `*CFG` siblings
(FullPipelineCFG / AudioToAudioPipelineCFG) add a runtime `cfg_scale` + negative prompt
(classifier-free guidance, full APG) at ~2× DiT cost per step.
"""

import torch
import torch.nn as nn
from torch.export import ExportedProgram

from stable_audio_3.models.dit import DiffusionTransformer
from stable_audio_3.inference.sampling import build_schedule

NUM_STEPS = 8  # pingpong denoising steps; re-noise count = NUM_STEPS - 1


# ===========================================================================
# Prep + graph helpers
# ===========================================================================

def disable_stochastic_ops(ae) -> None:
    """Zero the `torch.randn_like` noise sources in the autoencoder so the exported
    graph is deterministic.

    1. SoftNormBottleneck.decode (bottleneck.py) — eval-mode scale=1e-3 noise.
    2. TransformerResamplingBlock.forward (autoencoders.py) — mask_noise=0.01.

    Sweeps the whole `ae` (encoder + decoder) so the audio-to-audio encode path is
    deterministic too. For text-to-audio the encoder never runs, so zeroing its
    mask_noise is a harmless no-op there.
    """
    ae.bottleneck.noise_regularize = False
    from stable_audio_3.models.autoencoders import TransformerResamplingBlock
    for module in ae.modules():
        if isinstance(module, TransformerResamplingBlock):
            module.mask_noise = 0.0


def remove_weight_norm_all(module) -> None:
    """Recursively strip weight_norm from Conv1d/Conv2d layers.

    weight_norm hooks emit aten.convolution.default, which the MLX backend rejects
    (it only accepts aten.conv{1,2,3}d). Removing it materialises the weight so the
    Conv emits aten.conv1d.default instead.
    """
    from torch.nn.utils import remove_weight_norm
    for submodule in module.modules():
        if isinstance(submodule, (nn.Conv1d, nn.Conv2d)):
            try:
                remove_weight_norm(submodule)
            except ValueError:
                pass  # no weight_norm on this layer


def replace_same_padding(module) -> None:
    """Replace Conv1d(padding='same') with the integer-padding equivalent.

    to_edge_transform_and_lower turns the string-padding conv variant into
    aten.convolution.default, which the MLX backend rejects. For non-transposed
    stride-1 conv, same_padding = (kernel_size - 1) // 2.
    """
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv1d) and submodule.padding == "same":
            assert submodule.stride == (1,) and not submodule.transposed, \
                f"padding='same' replacement only supports stride=1 non-transposed conv, got {submodule}"
            submodule.padding = ((submodule.kernel_size[0] - 1) // 2,)


def decompose_convolution_nodes(exported: ExportedProgram) -> ExportedProgram:
    """Rewrite aten.convolution.default → aten.conv1d.default in the exported graph
    (safety net before the MLX partitioner, which only accepts aten.conv{1,2,3}d).

    aten.convolution: (input, weight, bias, stride, padding, dilation, transposed,
                       output_padding, groups)
    aten.conv1d:      (input, weight, bias, stride, padding, dilation, groups)
    """
    graph = exported.graph_module.graph
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target != torch.ops.aten.convolution.default:
            continue
        inp, weight, bias, stride, padding, dilation, transposed, output_padding, groups = node.args
        if transposed:
            continue  # leave transposed convolutions alone
        with graph.inserting_before(node):
            new_node = graph.call_function(
                torch.ops.aten.conv1d.default,
                args=(inp, weight, bias, stride, padding, dilation, groups),
            )
            new_node.meta = dict(node.meta)
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
    graph.lint()
    exported.graph_module.recompile()
    return exported


def assert_distilled(sa3) -> None:
    """Guard: only the distilled checkpoints fit the 8-step pingpong / cfg=1 export.

    Distilled models (`small-music`, `small-sfx`) use `diffusion_objective == "rf_denoiser"`
    (pingpong, 8 steps, cfg=1). The `-base` checkpoints are `rectified_flow` (cfg≈7, ~50
    steps, euler/dpmpp) — they need a CFG + many-step graph this exporter does not build.
    """
    objective = getattr(sa3.model, "diffusion_objective", None)
    if objective != "rf_denoiser":
        raise SystemExit(
            f"This exporter only supports distilled models (diffusion_objective == 'rf_denoiser', "
            f"8-step pingpong, cfg=1). Got diffusion_objective={objective!r} — looks like a `-base` "
            f"checkpoint, which needs cfg≠1 + ~50 steps + euler/dpmpp (not exported here)."
        )


def compute_sigmas(sa3_model, t_lat: int, device: str = "cpu") -> list[float]:
    """LogSNRShift 8-step schedule with endpoints forced to [1.0, ..., 0.0].

    PyTorch build_schedule only re-anchors the first point; MLX logsnr_shift clamps
    both ends, so we force the last sigma to exactly 0.0 to match.
    Returns 9 floats: [s0=1.0, s1, ..., s7, s8=0.0].
    """
    dist_shift = getattr(sa3_model.model, "sampling_dist_shift", None)
    sigmas = build_schedule(
        steps=NUM_STEPS,
        sigma_max=1.0,
        dist_shift=dist_shift,
        fallback_seq_len=t_lat,
        include_endpoint=True,
        device=device,
    )
    sigmas_list = sigmas.tolist()
    sigmas_list[-1] = 0.0
    return sigmas_list


# ===========================================================================
# Wrappers — text conditioning
# ===========================================================================

class EncoderOnly(nn.Module):
    """Extract last_hidden_state as a plain tensor.

    T5GemmaEncoderModel returns a ModelOutput dataclass; wrapping here ensures
    torch.export sees a plain tensor (avoids pytree-unpacking edge cases).
    """
    def __init__(self, t5_model: nn.Module):
        super().__init__()
        self.enc = t5_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class TextConditioning(nn.Module):
    """T5Gemma + NumberConditioner → the two tensors the DiT consumes.

    cross_attn_cond: (1, 257, 768) — 256 prompt tokens + 1 seconds_total token
    global_embed:    (1, 768)      — seconds_total for adaLN
    """
    def __init__(self, t5_cond, num_cond):
        super().__init__()
        self.encoder      = EncoderOnly(t5_cond.model)   # T5GemmaEncoderModel
        self.proj_out     = t5_cond.proj_out             # nn.Identity (768→768)
        self.padding_emb  = t5_cond.padding_embedding    # nn.Parameter (768,) learned pad
        self.sec_embedder = num_cond.embedder            # NumberEmbedder → (768,) per token
        self.sec_min = num_cond.min_val                  # 0.0
        self.sec_max = num_cond.max_val                  # 384.0

    def forward(
        self,
        input_ids: torch.Tensor,       # (1, 256) int64
        attention_mask: torch.Tensor,  # (1, 256) int32
        seconds_total: torch.Tensor,   # (1,) float32
    ):
        embeddings = self.encoder(input_ids, attention_mask)  # (1, 256, 768)
        embeddings = self.proj_out(embeddings)                # Identity, no-op

        # Learned padding: replace pad positions (mask=0) with the learned embedding.
        mask_bool  = attention_mask.unsqueeze(-1).bool()      # (1, 256, 1)
        pad_expand = self.padding_emb.unsqueeze(0).unsqueeze(0).expand_as(embeddings)
        embeddings = torch.where(mask_bool, embeddings, pad_expand)  # (1, 256, 768)

        # seconds_total via NumberEmbedder (ExpoFourierFeatures → Linear).
        normalized  = (seconds_total.clamp(self.sec_min, self.sec_max) - self.sec_min) / (self.sec_max - self.sec_min)
        seconds_emb = self.sec_embedder(normalized).unsqueeze(1)  # (1, 1, 768)

        cross_attn_cond = torch.cat([embeddings, seconds_emb], dim=1)  # (1, 257, 768)
        global_embed    = seconds_emb[:, 0, :]                          # (1, 768)
        return cross_attn_cond, global_embed


# ===========================================================================
# Wrappers — DiT denoising loop (text-to-audio)
# ===========================================================================

class UnrolledDiT(nn.Module):
    """8-step pingpong denoising loop, fully unrolled for torch.export.

    No CFG (cfg_scale=1.0), no inpainting (local_add_cond=None baked in).
    Per-step sigma values are Python float constants — fully static at trace time.
    """
    def __init__(self, dit: DiffusionTransformer, sigmas: list[float]):
        super().__init__()
        self.dit = dit
        self.sigmas = sigmas  # length 9: [s0, ..., s8]; Python floats baked as constants

    def _step(self, x, s_curr: float, s_next: float, noise, cross_attn_cond, global_embed):
        t = x.new_full((1,), s_curr)  # (1,) float32, constant at trace time
        v = self.dit._forward(
            x, t,
            cross_attn_cond=cross_attn_cond,
            global_embed=global_embed,
            local_add_cond=None,  # text-to-audio only, baked as None at trace time
        )
        denoised = x - s_curr * v
        return (1.0 - s_next) * denoised + s_next * noise

    def forward(
        self,
        x: torch.Tensor,                # (1, 256, T_lat)
        cross_attn_cond: torch.Tensor,  # (1, 257, 768)
        global_embed: torch.Tensor,     # (1, 768)
        noises: torch.Tensor,           # (7, 1, 256, T_lat)
    ) -> torch.Tensor:
        s = self.sigmas
        # Steps 0-6: denoise + re-noise (t_next > 0).
        x = self._step(x, s[0], s[1], noises[0], cross_attn_cond, global_embed)
        x = self._step(x, s[1], s[2], noises[1], cross_attn_cond, global_embed)
        x = self._step(x, s[2], s[3], noises[2], cross_attn_cond, global_embed)
        x = self._step(x, s[3], s[4], noises[3], cross_attn_cond, global_embed)
        x = self._step(x, s[4], s[5], noises[4], cross_attn_cond, global_embed)
        x = self._step(x, s[5], s[6], noises[5], cross_attn_cond, global_embed)
        x = self._step(x, s[6], s[7], noises[6], cross_attn_cond, global_embed)
        # Step 7 (final): t_next=0, pure denoised — no noise injection.
        t_final = x.new_full((1,), s[7])
        v = self.dit._forward(
            x, t_final,
            cross_attn_cond=cross_attn_cond,
            global_embed=global_embed,
            local_add_cond=None,
        )
        return x - s[7] * v


# ===========================================================================
# Wrappers — SAME-S decoder
# ===========================================================================

class DecoderExport(nn.Module):
    """Chains the three decode sub-modules directly.

    We do NOT call ae.decode() — that wrapper has iterate_batch branches and a
    torch.no_grad() context that interfere with export tracing.
    """
    def __init__(self, ae):
        super().__init__()
        self.bottleneck   = ae.bottleneck    # SoftNormBottleneck
        self.decoder      = ae.decoder        # SAMEDecoder
        self.pretransform = ae.pretransform   # PatchedPretransform

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck.decode(latents)   # (B, 256, T_lat)
        x = self.decoder(x)                   # (B, 512, T_lat*16)
        x = self.pretransform.decode(x)       # (B, 2, T_lat*4096)
        return x


# ===========================================================================
# Wrappers — composed audio generator + full (text-to-audio) pipeline
# ===========================================================================

class AudioGenerator(nn.Module):
    """8-step pingpong DiT denoising + SAME-S decode → PCM audio."""
    def __init__(self, dit, ae, sigmas: list[float]):
        super().__init__()
        self.denoiser = UnrolledDiT(dit, sigmas)
        self.decoder  = DecoderExport(ae)

    def forward(self, x, cross_attn_cond, global_embed, noises) -> torch.Tensor:
        latents = self.denoiser(x, cross_attn_cond, global_embed, noises)
        return self.decoder(latents)


class FullPipeline(nn.Module):
    """Text conditioning → 8-step DiT denoising → SAME-S decode → PCM audio."""
    def __init__(self, t5_cond, num_cond, dit, ae, sigmas: list[float]):
        super().__init__()
        self.text_cond = TextConditioning(t5_cond, num_cond)
        self.audio_gen = AudioGenerator(dit, ae, sigmas)

    def forward(
        self,
        token_ids: torch.Tensor,       # (1, 256) int64
        attention_mask: torch.Tensor,  # (1, 256) int32
        target_sec: torch.Tensor,      # (1,) float32
        noise_x: torch.Tensor,         # (1, 256, T_lat) float32
        noise_steps: torch.Tensor,     # (7, 1, 256, T_lat) float32
    ) -> torch.Tensor:                 # (1, 2, T_lat*4096)
        cross_attn_cond, global_embed = self.text_cond(token_ids, attention_mask, target_sec)
        return self.audio_gen(noise_x, cross_attn_cond, global_embed, noise_steps)


def build_full_pipeline(sa3, t_lat: int, device: str = "cpu"):
    """Build an eval'd FullPipeline (text-to-audio) from a loaded StableAudioModel.

    Extracts the sub-models, disables stochastic ops / weight_norm / same-padding on
    the autoencoder, computes the LogSNRShift schedule, and returns (model, t5_cond).
    `t5_cond` exposes the tokenizer for building reference inputs.
    """
    assert_distilled(sa3)
    t5_cond  = sa3.model.conditioner.conditioners["prompt"]         # T5GemmaConditioner
    num_cond = sa3.model.conditioner.conditioners["seconds_total"]  # NumberConditioner
    dit = sa3.dit.model    # DiffusionTransformer
    ae  = sa3.same.model   # AudioAutoencoder (sa3.same is an AutoencoderPretransform wrapper)
    t5_cond.model.float()  # no bf16 surprises
    dit.eval()
    ae.eval()

    disable_stochastic_ops(ae)
    remove_weight_norm_all(ae)
    replace_same_padding(ae)

    sigmas = compute_sigmas(sa3, t_lat, device)
    model = FullPipeline(t5_cond, num_cond, dit, ae, sigmas).eval()
    return model, t5_cond


# ===========================================================================
# Audio-to-audio (init audio + inpaint mask + denoising strength) building blocks
# ===========================================================================
#
# Adds the SAME-S *encoder* + a runtime denoising-strength control + conditioning-based
# inpainting on top of the text-to-audio pipeline. Mirrors
# `model.generate(init_audio=..., init_noise_level=σmax, inpaint_audio=..., inpaint_mask=...)`:
#   init_latents   = encode(audio_in)
#   x₀             = init_latents·(1−σmax) + noise_x·σmax      (noise_x is pure Gaussian noise)
#   schedule top   = σmax  (all 8 pingpong steps still run; strength reweights the schedule)
#   local_add_cond = cat([keep_mask, init_latents·keep_mask])  (fed every step)
#   → text-conditioned unrolled DiT denoise → SAME-S decode → audio
#
# σmax (noise_level) is a *runtime* (1,) input: the LogSNRShift schedule is recomputed
# in-graph (LogSNRShift.shift is pure elementwise tensor math), so one `.pte` covers every
# strength. SA3 inpainting is *conditioning-based* (config model_type
# "diffusion_cond_inpaint"): the DiT receives the keep-mask + masked init latents as
# `local_add_cond`; the sampler runs standard from pure noise (no per-step replacement).
# The modes fall out of the input values, no branch (mirrors model.py:280-296):
#   text-to-audio:  audio_in=silence, noise_level=1.0, inpaint_mask=0
#   audio-to-audio: audio_in=clip,    noise_level=σmax<1, inpaint_mask=0
#   inpaint:        audio_in=clip,    noise_level=1.0, inpaint_mask=keep-mask


class EncoderExport(nn.Module):
    """SAME-S encode: waveform → DiT latents. Mirror of DecoderExport.

    Chains the three encode sub-modules directly (we do NOT call ae.encode() / the
    AutoencoderPretransform wrapper — those have iterate_batch/chunked branches and a
    torch.no_grad() context that interfere with export tracing). Equivalent to
    `sa3.same.encode(audio)` when the outer AutoencoderPretransform.scale == 1.0
    (true for small-music; asserted in the factory).
    """
    def __init__(self, ae):
        super().__init__()
        self.pretransform = ae.pretransform   # PatchedPretransform (waveform → patches)
        self.encoder      = ae.encoder         # SAMEEncoder
        self.bottleneck   = ae.bottleneck      # SoftNormBottleneck

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.pretransform.encode(audio)              # (B, 512, T_lat*16)
        x = self.encoder(x)                              # (B, 256, T_lat)
        x = self.bottleneck.encode(x, return_info=False)  # (B, 256, T_lat)
        return x


def logsnr_sigmas(sigma_max: torch.Tensor, t_lat: int, dist_shift) -> torch.Tensor:
    """In-graph LogSNRShift schedule from a runtime sigma_max (= noise_level).

    Mirrors compute_sigmas / build_schedule but with sigma_max as a (1,) tensor so the
    whole schedule is recomputed inside the exported graph. `dist_shift.shift` is pure
    elementwise tensor math with a fixed (int) seq_len, so logsnr_start is a Python-float
    constant at trace time and the call traces cleanly.

    Returns a (NUM_STEPS+1,) = (9,) tensor: [sigma_max, s1, ..., s7, 0.0].
    """
    base = torch.linspace(1.0, 0.0, NUM_STEPS + 1, device=sigma_max.device, dtype=sigma_max.dtype)
    t = sigma_max.reshape(1) * base  # (9,)
    if dist_shift is not None:
        t = dist_shift.shift(t, t_lat)
    # Re-anchor endpoints exactly: shift() does not pin t[0]==sigma_max when sigma_max<1.
    return torch.cat([sigma_max.reshape(1), t[1:-1], t.new_zeros(1)])


def build_inpaint_local_cond(init_latents: torch.Tensor, inpaint_mask: torch.Tensor) -> torch.Tensor:
    """local_add_cond = cat([mask, init_latents·mask], dim=1) → (1, 257, T_lat).

    inpaint_mask: (1, 1, T_lat) keep-mask (1=keep, 0=regenerate). The masked input zeros
    the regenerate frames. Mirrors model.py:287-296 (inpaint_mask + inpaint_masked_input).
    """
    masked_input = init_latents * inpaint_mask              # (1,256,T_lat)·(1,1,T_lat)
    return torch.cat([inpaint_mask, masked_input], dim=1)   # (1, 257, T_lat)


class UnrolledDiTDynLocal(nn.Module):
    """8-step pingpong loop with *runtime* sigmas AND a constant inpaint local_add_cond.

    Reads each per-step sigma from a runtime (9,) tensor (so denoising strength varies at
    inference) and feeds the inpaint local conditioning (mask + masked init latents) to
    dit._forward every step. The local_add_cond is constant across all steps.
    """
    def __init__(self, dit: DiffusionTransformer):
        super().__init__()
        self.dit = dit

    def _step(self, x, s_curr, s_next, noise, cross_attn_cond, global_embed, local_add_cond):
        # s_curr, s_next: 0-dim tensors sliced from the runtime sigma schedule.
        v = self.dit._forward(
            x, s_curr.reshape(1),
            cross_attn_cond=cross_attn_cond,
            global_embed=global_embed,
            local_add_cond=local_add_cond,
        )
        denoised = x - s_curr * v
        return (1.0 - s_next) * denoised + s_next * noise

    def forward(self, x, cross_attn_cond, global_embed, sigmas, local_add_cond, noises) -> torch.Tensor:
        # sigmas: (9,) runtime tensor; local_add_cond: (1,257,T_lat); noises: (7,1,256,T_lat)
        for i in range(NUM_STEPS - 1):  # steps 0-6: denoise + re-noise
            x = self._step(x, sigmas[i], sigmas[i + 1], noises[i],
                           cross_attn_cond, global_embed, local_add_cond)
        # Final step: t_next = sigmas[8] = 0, pure denoised — no noise injection.
        s_last = sigmas[NUM_STEPS - 1]
        v = self.dit._forward(
            x, s_last.reshape(1),
            cross_attn_cond=cross_attn_cond,
            global_embed=global_embed,
            local_add_cond=local_add_cond,
        )
        return x - s_last * v


class AudioToAudioPipeline(nn.Module):
    """Audio-to-audio with inpaint mask + init audio (the superset of init-audio modes).

    forward(audio_in, token_ids, attention_mask, inpaint_mask, target_sec,
            noise_level, noise_x, noise_steps):
      init_latents   = encode(audio_in)                                  # (1,256,T)
      sigmas         = logsnr schedule with top = noise_level            (recomputed in-graph)
      x0             = init_latents·(1−σmax) + noise_x·σmax  (σmax=1 ⇒ x0=noise_x = pure noise)
      local_add_cond = cat([inpaint_mask, init_latents·inpaint_mask])     (fed every step)
      latents        = dynamic-sigma pingpong DiT denoise (text + locally conditioned)
      → audio        = decode(latents)

    Modes (selected by the input values; defaults = plain text-to-audio):
      text-to-audio:  audio_in ignored, noise_level=1.0, inpaint_mask=0
      audio-to-audio: audio_in=clip,    noise_level=σmax∈(0,1), inpaint_mask=0
      inpaint:        audio_in=clip,    noise_level=1.0, inpaint_mask=keep-mask
    Matches model.generate (model.py:280-296), which always feeds local_add_cond (mask=0
    for non-inpaint).
    """
    def __init__(self, t5_cond, num_cond, dit, ae, dist_shift, t_lat: int):
        super().__init__()
        self.text_cond  = TextConditioning(t5_cond, num_cond)
        self.encoder    = EncoderExport(ae)
        self.denoiser   = UnrolledDiTDynLocal(dit)
        self.decoder    = DecoderExport(ae)
        self.dist_shift = dist_shift   # plain LogSNRShift object; used to build the schedule
        self.t_lat      = t_lat

    def forward(
        self,
        audio_in: torch.Tensor,          # (1, 2, T_lat*4096) float32 — init / clip to inpaint
        token_ids: torch.Tensor,         # (1, 256) int64
        attention_mask: torch.Tensor,    # (1, 256) int32
        inpaint_mask: torch.Tensor,      # (1, 1, T_lat) float32 — 1=keep, 0=regenerate
        target_sec: torch.Tensor,        # (1,) float32
        noise_level: torch.Tensor,       # (1,) float32 in (0, 1]
        noise_x: torch.Tensor,           # (1, 256, T_lat) float32 — pure Gaussian noise
        noise_steps: torch.Tensor,       # (7, 1, 256, T_lat) float32
    ) -> torch.Tensor:                   # (1, 2, T_lat*4096)
        init_latents = self.encoder(audio_in)                              # (1, 256, T_lat)
        sigmas = logsnr_sigmas(noise_level, self.t_lat, self.dist_shift)
        smax = noise_level.reshape(1, 1, 1)                                # broadcast over (1,256,T_lat)
        x0 = init_latents * (1.0 - smax) + noise_x * smax
        local_add_cond = build_inpaint_local_cond(init_latents, inpaint_mask)
        cross_attn_cond, global_embed = self.text_cond(token_ids, attention_mask, target_sec)
        latents = self.denoiser(x0, cross_attn_cond, global_embed, sigmas, local_add_cond, noise_steps)
        return self.decoder(latents)


def build_audio_to_audio_pipeline(sa3, t_lat: int, device: str = "cpu"):
    """Build an eval'd AudioToAudioPipeline from a loaded StableAudioModel.

    Same prep as build_full_pipeline plus: asserts the AutoencoderPretransform scale is
    1.0 (EncoderExport/DecoderExport are scale-free) and grabs the LogSNRShift for the
    in-graph dynamic schedule.
    """
    assert_distilled(sa3)
    t5_cond  = sa3.model.conditioner.conditioners["prompt"]
    num_cond = sa3.model.conditioner.conditioners["seconds_total"]
    dit = sa3.dit.model
    ae  = sa3.same.model

    scale = float(getattr(sa3.same, "scale", 1.0))
    assert abs(scale - 1.0) < 1e-9, (
        f"AutoencoderPretransform.scale={scale} != 1.0; EncoderExport/DecoderExport assume "
        f"scale==1.0 (the verified scale-free decode). Add scale handling before exporting.")

    t5_cond.model.float()
    dit.eval()
    ae.eval()

    disable_stochastic_ops(ae)
    remove_weight_norm_all(ae)
    replace_same_padding(ae)

    dist_shift = getattr(sa3.model, "sampling_dist_shift", None)
    model = AudioToAudioPipeline(t5_cond, num_cond, dit, ae, dist_shift, t_lat).eval()
    return model, t5_cond


# ===========================================================================
# Classifier-free guidance (cfg_scale + negative_prompt) building blocks
# ===========================================================================
#
# Sibling wrappers that bake CFG into both pipelines. `cfg_scale` is a *runtime*
# (1,) input (like noise_level); the negative prompt is a second token set
# (neg_token_ids / neg_attention_mask). These reproduce DiffusionTransformer.forward
# (dit.py:479-625) at the generate() defaults: full APG (apg_scale=1.0),
# cfg_interval=(0,1) ⇒ CFG every step, no rescale (scale_phi=0), no norm-clamp.
#
# Per-step combine (rf_denoiser): run dit._forward once on a batch-2 [cond; uncond],
#   cond_den   = x − σ·cond_v ;  uncond_den = x − σ·uncond_v ;  diff = cond_den − uncond_den
#   cfg_diff   = orthogonal component of diff w.r.t. cond_den   (full APG, apg_scale=1.0)
#   denoised   = cond_den + (cfg_scale − 1)·cfg_diff            → feeds the pingpong renoise
# At cfg_scale=1.0 the (cfg_scale−1)=0 factor makes denoised == cond_den, so the graph
# reduces exactly to the non-CFG export. CFG always costs ~2× DiT per step — that is why
# these are sibling artifacts.


def _apg_orthogonal(diff: torch.Tensor, cond_den: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Orthogonal component of `diff` w.r.t. `cond_den` — the no-mask branch of
    DiffusionTransformer.apg_project (dit.py:311-343), implemented with explicit ops
    (no F.normalize over multiple dims) so the MLX backend traces it cleanly.

    Matches F.normalize(cond_den, dim=[-1,-2]) then diff − (diff·v1)·v1.
    """
    norm = cond_den.pow(2).sum(dim=(-1, -2), keepdim=True).sqrt().clamp(min=eps)
    v1 = cond_den / norm                                            # unit direction of cond_den
    parallel = (diff * v1).sum(dim=(-1, -2), keepdim=True) * v1
    return diff - parallel


def _cfg_combine(x, s_curr, cond_v, uncond_v, cfg_scale) -> torch.Tensor:
    """Full-APG CFG combine → the guided `denoised` latent (rf_denoiser objective).

    s_curr may be a Python float (static schedule) or a 0-dim tensor (runtime schedule);
    cfg_scale is a (1,) runtime tensor. apg_scale=1.0 ⇒ cfg_diff = orthogonal component.
    """
    cond_den   = x - s_curr * cond_v
    uncond_den = x - s_curr * uncond_v
    diff = cond_den - uncond_den
    cfg_diff = _apg_orthogonal(diff, cond_den)
    return cond_den + (cfg_scale - 1.0) * cfg_diff


class TextConditioningCFG(nn.Module):
    """T5Gemma + NumberConditioner → batch-2 [positive; negative] DiT conditioning.

    Batches the positive and negative token sets through the shared encoder in one
    forward. The positive row uses the learned padding embedding at pad positions
    (matching TextConditioning); the negative row uses zeros at pad positions (matching
    dit.forward's `where(neg_mask, neg_cond, 0)`), so an all-pad negative prompt becomes
    the pure null/unconditional branch. The seconds_total token + global adaLN embed are
    the positive ones, shared across both rows.

      cross_attn_cond: (2, 257, 768)   row0 = positive, row1 = negative/null
      global_embed:    (2, 768)        seconds_total, duplicated
    """
    def __init__(self, t5_cond, num_cond):
        super().__init__()
        self.encoder      = EncoderOnly(t5_cond.model)
        self.proj_out     = t5_cond.proj_out
        self.padding_emb  = t5_cond.padding_embedding
        self.sec_embedder = num_cond.embedder
        self.sec_min = num_cond.min_val
        self.sec_max = num_cond.max_val

    def forward(
        self,
        input_ids: torch.Tensor,           # (1, 256) int64  positive prompt
        attention_mask: torch.Tensor,      # (1, 256) int32
        neg_input_ids: torch.Tensor,       # (1, 256) int64  negative prompt (all-pad = unconditional)
        neg_attention_mask: torch.Tensor,  # (1, 256) int32
        seconds_total: torch.Tensor,       # (1,) float32
    ):
        ids  = torch.cat([input_ids, neg_input_ids], dim=0)            # (2, 256)
        mask = torch.cat([attention_mask, neg_attention_mask], dim=0)  # (2, 256)
        emb  = self.encoder(ids, mask)                                 # (2, 256, 768)
        emb  = self.proj_out(emb)                                      # Identity, no-op

        # Per-row pad fill: positive → learned padding embedding, negative → zeros (null).
        mask_bool = mask.unsqueeze(-1).bool()                          # (2, 256, 1)
        pad_pos = self.padding_emb.view(1, 1, -1).expand(1, emb.shape[1], emb.shape[2])
        pad_neg = torch.zeros_like(pad_pos)
        pad_val = torch.cat([pad_pos, pad_neg], dim=0)                 # (2, 256, 768)
        emb = torch.where(mask_bool, emb, pad_val)                     # (2, 256, 768)

        # seconds_total via NumberEmbedder; shared across both rows.
        normalized  = (seconds_total.clamp(self.sec_min, self.sec_max) - self.sec_min) / (self.sec_max - self.sec_min)
        seconds_emb = self.sec_embedder(normalized).unsqueeze(1)       # (1, 1, 768)
        seconds_2   = seconds_emb.expand(2, 1, seconds_emb.shape[-1])  # (2, 1, 768)

        cross_attn_cond = torch.cat([emb, seconds_2], dim=1)           # (2, 257, 768)
        global_embed    = seconds_emb[:, 0, :].expand(2, seconds_emb.shape[-1])  # (2, 768)
        return cross_attn_cond, global_embed


class UnrolledDiTCFG(nn.Module):
    """8-step pingpong denoising loop with CFG, unrolled, static sigmas (text-to-audio).

    Like UnrolledDiT but each step runs dit._forward once on a batch-2 [cond; uncond]
    and applies the full-APG CFG combine with a runtime cfg_scale. No inpainting.
    """
    def __init__(self, dit: DiffusionTransformer, sigmas: list[float]):
        super().__init__()
        self.dit = dit
        self.sigmas = sigmas  # length 9; Python floats baked as constants

    def _denoised(self, x, s_curr, cross_attn_cond, global_embed, cfg_scale):
        xb = torch.cat([x, x], dim=0)                  # (2, 256, T_lat)
        tb = x.new_full((2,), s_curr)                  # (2,)
        vb = self.dit._forward(xb, tb, cross_attn_cond=cross_attn_cond,
                               global_embed=global_embed, local_add_cond=None)
        cond_v, uncond_v = torch.chunk(vb, 2, dim=0)   # each (1, 256, T_lat)
        return _cfg_combine(x, s_curr, cond_v, uncond_v, cfg_scale)

    def forward(self, x, cross_attn_cond, global_embed, cfg_scale, noises) -> torch.Tensor:
        s = self.sigmas
        for i in range(NUM_STEPS - 1):  # steps 0-6: denoise + re-noise
            denoised = self._denoised(x, s[i], cross_attn_cond, global_embed, cfg_scale)
            x = (1.0 - s[i + 1]) * denoised + s[i + 1] * noises[i]
        # Final step (t_next = 0): pure denoised, no noise injection.
        return self._denoised(x, s[NUM_STEPS - 1], cross_attn_cond, global_embed, cfg_scale)


class AudioGeneratorCFG(nn.Module):
    """8-step pingpong CFG denoising + SAME-S decode → PCM audio."""
    def __init__(self, dit, ae, sigmas: list[float]):
        super().__init__()
        self.denoiser = UnrolledDiTCFG(dit, sigmas)
        self.decoder  = DecoderExport(ae)

    def forward(self, x, cross_attn_cond, global_embed, cfg_scale, noises) -> torch.Tensor:
        latents = self.denoiser(x, cross_attn_cond, global_embed, cfg_scale, noises)
        return self.decoder(latents)


class FullPipelineCFG(nn.Module):
    """Text-to-audio with classifier-free guidance.

    Sibling of FullPipeline with two extra runtime inputs: a negative prompt
    (neg_token_ids / neg_attention_mask) and a (1,) cfg_scale. cfg_scale=1.0 ⇒ output
    == FullPipeline; all-pad negative prompt ⇒ pure unconditional uncond branch.
    """
    def __init__(self, t5_cond, num_cond, dit, ae, sigmas: list[float]):
        super().__init__()
        self.text_cond = TextConditioningCFG(t5_cond, num_cond)
        self.audio_gen = AudioGeneratorCFG(dit, ae, sigmas)

    def forward(
        self,
        token_ids: torch.Tensor,            # (1, 256) int64
        attention_mask: torch.Tensor,       # (1, 256) int32
        neg_token_ids: torch.Tensor,        # (1, 256) int64
        neg_attention_mask: torch.Tensor,   # (1, 256) int32
        target_sec: torch.Tensor,           # (1,) float32
        cfg_scale: torch.Tensor,            # (1,) float32
        noise_x: torch.Tensor,              # (1, 256, T_lat) float32
        noise_steps: torch.Tensor,          # (7, 1, 256, T_lat) float32
    ) -> torch.Tensor:                      # (1, 2, T_lat*4096)
        cross_attn_cond, global_embed = self.text_cond(
            token_ids, attention_mask, neg_token_ids, neg_attention_mask, target_sec)
        return self.audio_gen(noise_x, cross_attn_cond, global_embed, cfg_scale, noise_steps)


def build_full_pipeline_cfg(sa3, t_lat: int, device: str = "cpu"):
    """Build an eval'd FullPipelineCFG (CFG sibling of build_full_pipeline)."""
    assert_distilled(sa3)
    t5_cond  = sa3.model.conditioner.conditioners["prompt"]
    num_cond = sa3.model.conditioner.conditioners["seconds_total"]
    dit = sa3.dit.model
    ae  = sa3.same.model
    t5_cond.model.float()
    dit.eval()
    ae.eval()

    disable_stochastic_ops(ae)
    remove_weight_norm_all(ae)
    replace_same_padding(ae)

    sigmas = compute_sigmas(sa3, t_lat, device)
    model = FullPipelineCFG(t5_cond, num_cond, dit, ae, sigmas).eval()
    return model, t5_cond


class UnrolledDiTDynLocalCFG(nn.Module):
    """8-step pingpong loop with *runtime* sigmas, a constant inpaint local_add_cond, AND CFG.

    The merge of UnrolledDiTDynLocal (runtime sigma schedule + local_add_cond fed every
    step) and the batch-2 full-APG CFG combine (runtime cfg_scale). Used by the
    audio-to-audio CFG pipeline. The local_add_cond is expanded to the batch-2 [cond;
    uncond] rows — inpaint conditioning is prompt-independent, so both rows share it.
    """
    def __init__(self, dit: DiffusionTransformer):
        super().__init__()
        self.dit = dit

    def _denoised(self, x, s_curr, cross_attn_cond, global_embed, local_add_cond, cfg_scale):
        xb = torch.cat([x, x], dim=0)                                  # (2, 256, T_lat)
        tb = s_curr.reshape(1).expand(2)                               # (2,) 0-dim sigma → batch-2
        local_b = local_add_cond.expand(2, local_add_cond.shape[1], local_add_cond.shape[2])
        vb = self.dit._forward(xb, tb, cross_attn_cond=cross_attn_cond,
                               global_embed=global_embed, local_add_cond=local_b)
        cond_v, uncond_v = torch.chunk(vb, 2, dim=0)
        return _cfg_combine(x, s_curr, cond_v, uncond_v, cfg_scale)

    def forward(self, x, cross_attn_cond, global_embed, sigmas, local_add_cond, cfg_scale, noises) -> torch.Tensor:
        for i in range(NUM_STEPS - 1):  # steps 0-6: denoise + re-noise
            denoised = self._denoised(x, sigmas[i], cross_attn_cond, global_embed, local_add_cond, cfg_scale)
            x = (1.0 - sigmas[i + 1]) * denoised + sigmas[i + 1] * noises[i]
        # Final step: t_next = sigmas[8] = 0, pure denoised — no noise injection.
        return self._denoised(x, sigmas[NUM_STEPS - 1], cross_attn_cond, global_embed, local_add_cond, cfg_scale)


class AudioToAudioPipelineCFG(nn.Module):
    """Audio-to-audio (init audio + inpaint mask + strength) with classifier-free guidance.

    Sibling of AudioToAudioPipeline with a negative prompt + runtime cfg_scale on top of
    the runtime noise_level and the inpaint local conditioning. At cfg_scale=1.0 the
    output equals the non-CFG pipeline.
    """
    def __init__(self, t5_cond, num_cond, dit, ae, dist_shift, t_lat: int):
        super().__init__()
        self.text_cond  = TextConditioningCFG(t5_cond, num_cond)
        self.encoder    = EncoderExport(ae)
        self.denoiser   = UnrolledDiTDynLocalCFG(dit)
        self.decoder    = DecoderExport(ae)
        self.dist_shift = dist_shift
        self.t_lat      = t_lat

    def forward(
        self,
        audio_in: torch.Tensor,             # (1, 2, T_lat*4096) float32 — init / clip to inpaint
        token_ids: torch.Tensor,            # (1, 256) int64
        attention_mask: torch.Tensor,       # (1, 256) int32
        neg_token_ids: torch.Tensor,        # (1, 256) int64
        neg_attention_mask: torch.Tensor,   # (1, 256) int32
        inpaint_mask: torch.Tensor,         # (1, 1, T_lat) float32 — 1=keep, 0=regenerate
        target_sec: torch.Tensor,           # (1,) float32
        noise_level: torch.Tensor,          # (1,) float32 in (0, 1]
        cfg_scale: torch.Tensor,            # (1,) float32
        noise_x: torch.Tensor,              # (1, 256, T_lat) float32 — pure Gaussian noise
        noise_steps: torch.Tensor,          # (7, 1, 256, T_lat) float32
    ) -> torch.Tensor:                      # (1, 2, T_lat*4096)
        init_latents = self.encoder(audio_in)
        sigmas = logsnr_sigmas(noise_level, self.t_lat, self.dist_shift)
        smax = noise_level.reshape(1, 1, 1)
        x0 = init_latents * (1.0 - smax) + noise_x * smax
        local_add_cond = build_inpaint_local_cond(init_latents, inpaint_mask)
        cross_attn_cond, global_embed = self.text_cond(
            token_ids, attention_mask, neg_token_ids, neg_attention_mask, target_sec)
        latents = self.denoiser(x0, cross_attn_cond, global_embed, sigmas, local_add_cond, cfg_scale, noise_steps)
        return self.decoder(latents)


def build_audio_to_audio_pipeline_cfg(sa3, t_lat: int, device: str = "cpu"):
    """Build an eval'd AudioToAudioPipelineCFG (CFG sibling of build_audio_to_audio_pipeline)."""
    assert_distilled(sa3)
    t5_cond  = sa3.model.conditioner.conditioners["prompt"]
    num_cond = sa3.model.conditioner.conditioners["seconds_total"]
    dit = sa3.dit.model
    ae  = sa3.same.model

    scale = float(getattr(sa3.same, "scale", 1.0))
    assert abs(scale - 1.0) < 1e-9, (
        f"AutoencoderPretransform.scale={scale} != 1.0; EncoderExport/DecoderExport assume "
        f"scale==1.0 (the verified scale-free decode). Add scale handling before exporting.")

    t5_cond.model.float()
    dit.eval()
    ae.eval()

    disable_stochastic_ops(ae)
    remove_weight_norm_all(ae)
    replace_same_padding(ae)

    dist_shift = getattr(sa3.model, "sampling_dist_shift", None)
    model = AudioToAudioPipelineCFG(t5_cond, num_cond, dit, ae, dist_shift, t_lat).eval()
    return model, t5_cond
