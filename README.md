# Integrating Stable Audio 3 in Max/MSP

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-shuoyang--zheng%2Fstable--audio--3--tilde-yellow)](https://huggingface.co/shuoyang-zheng/stable-audio-3-tilde)


**Export [Stable Audio 3](https://huggingface.co/collections/stabilityai/stable-audio-3) models for the
[`neural.gen~`](https://github.com/jasper-zheng/neural_tilde) package in Max/MSP**  
Models are exported with [ExecuTorch](https://github.com/pytorch/executorch) for hardware-accelerated local inference: MLX backend on macOS (Apple Silicon), or CUDA backend on Windows (x64 + NVIDIA GPU). Supoorts distilled checkpoints (`small-music` and `small-sfx`) and LoRA. 


Two pipelines, each with an optional classifier-free guidance (`--cfg`) variant:

| pipeline | script | inputs |
|---|---|---|
| **text-to-audio** | `export_text_to_audio.py` | prompt + noise |
| **audio-to-audio** (init audio + inpaint mask) | `export_audio_to_audio.py` | prompt + noise + noise strength + init audio + inpaint mask |

<!-- 
Audio-to-audio is the superset of the init-audio modes — the mode falls out of the input values
(no branch), mirroring `model.generate`:

| mode | `audio_in` | `noise_level` | `inpaint_mask` |
|---|---|---|---|
| text-to-audio | silence (unset) | 1.0 | 0 (unset) |
| audio-to-audio | clip | σmax ∈ (0,1) | 0 |
| inpaint / continuation | clip | 1.0 | keep-mask (1=keep, 0=regenerate) | -->

## Prerequisites / Installation

Some of the exported models for `neural.gen~` can be found at HuggingFace [stable-audio-3-tilde](https://huggingface.co/shuoyang-zheng/stable-audio-3-tilde).

If you would like to export your own, you need `neural_tilde` **v0.1.0 or newer** in Max, and
export on the platform the model will run on:

- **macOS (MLX):** an **Apple Silicon Mac** (the MLX backend is Metal-only).
- **Windows (CUDA):** Windows 10/11 x64 with an **NVIDIA GPU** (Turing / RTX 20xx or newer), the
  **CUDA Toolkit 13.0** (driver ≥ 580) and **Visual Studio 2022** with the "Desktop development
  with C++" workload, and a **Developer PowerShell for VS 2022** (`cl.exe` must be on `PATH`).

  **Note:** On Windows, exporting models on CUDA needs about **35 GB of RAM** at its peak,
  plus ~4.5 GB of GPU memory. Set a fixed page file of 32 GB (System Properties → Advanced →
  Performance Settings → Advanced → Virtual memory) and reboot.

<!-- > **Why the specific versions?** The exported `.pte` is run by the ExecuTorch runtime built into
> the `neural.gen~` external, which for `neural_tilde` v0.1.0 is **ExecuTorch 1.5.1**. So this
> repo pins `executorch==1.5.1` (its PyPI wheel ships the MLX backend), and the `torch==2.14.0`
> it was built against, since `torch.export` → ExecuTorch lowering must run on the same torch.
> `uv sync` handles this for you, so don't mix in a different torch or ExecuTorch. -->

**1. Clone this repo and install the Python environment** with
[`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/jasper-zheng/stable-audio-3-tilde.git
cd stable-audio-3-tilde
uv sync     # installs torch, ExecuTorch, neural-tilde and the rest
```

On Windows, `uv sync` installs the CUDA 13.0 build of torch and `triton-windows` automatically.

Verify the backend is importable:

```bash
# macOS
uv run python -c "from executorch.backends.mlx.partitioner import MLXPartitioner; print('MLX backend OK')"
# Windows
uv run python -c "import torch, triton; from executorch.backends.cuda.cuda_partitioner import CudaPartitioner; print('CUDA backend OK' if torch.cuda.is_available() else 'no CUDA GPU')"
```

**2. HuggingFace access** to the checkpoints
([`stabilityai/stable-audio-3-small-music`](https://huggingface.co/stabilityai/stable-audio-3-small-music)
and [`stabilityai/stable-audio-3-small-sfx`](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)) —
run `uv run huggingface-cli login` if you haven't authenticated.


## Export

### Important limitation:  
1. The generation length need to be fixed at export time. The exported model always generates in this length (set by `--t-lat`, total latents). The default is `--t-lat 32` (≈ 2.97 s). You need to **export multiple models if you want to support different generation lengths**.
2. `--t-lat` must be an even number, it sets the generation length (`4096 x T_lat` samples at 44.1 kHz).
3. LoRA is merged into the weights at export time, so `--lora-strength` is **frozen** — the exported model can't retune it. Export once per strength (see [LoRA](#lora)). The exported model is frozen and cannot be further trained.

### Commands:

The same commands work on both platforms. The backend follows the platform: MLX on macOS, CUDA on
Windows (or pick one with `--delegate mlx|cuda`). On Windows, run them from a **Developer PowerShell
for VS 2022**. A CUDA export takes much longer, because every kernel is autotuned on GPU.

#### text-to-audio 
```bash
# text-to-audio 
uv run python export/export_text_to_audio.py  --model small-music --t-lat 32
```

```bash
# add --cfg for the classifier-free-guidance variant with negative prompt
uv run python export/export_text_to_audio.py  --model small-music --t-lat 32 --cfg
```

#### audio-to-audio
```bash
# audio-to-audio (init audio + inpaint mask + strength
uv run python export/export_audio_to_audio.py --model small-music --t-lat 32
```

```bash
# add --cfg for the classifier-free-guidance variant with negative prompt
uv run python export/export_audio_to_audio.py --model small-music --t-lat 32 --cfg
```

### LoRA

Both exporters can bake a LoRA / DoRA checkpoint into the exported weights. The LoRA is merged before tracing, so the model's inputs are unchanged — a LoRA export differs only in its weights
and its filename.

```bash
uv run python export/export_text_to_audio.py --model small-music --t-lat 32 \
    --lora-ckpt-path lora_checkpoints/my-lora.safetensors --lora-strength 2.0
```

| flag | |
|---|---|
| `--lora-ckpt-path` | checkpoint to merge; repeat to stack several |
| `--lora-strength` | strength baked in at export (default 2.0) |
| `--lora-name` | filename |
| `--force` | merge despite a base-model mismatch |

Things to know:

- **Strength is frozen.** It is baked into the weights, so one `.pte` per strength. `neural.gen~` has no runtime LoRA-strength control.
- The exporter refuses a checkpoint whose base model disagrees with `--model` (pass `--force` to override), and errors if the checkpoint doesn't actually match the model — otherwise it would merge as a no-op and quietly ship the base weights under a LoRA filename.


### Exported files

Each run writes these files into `export/artifacts/`:

```
*.pte                    the ExecuTorch model (MLX: ~2.9-3.3 GB with the weights; CUDA: ~50 MB)
*.ptd                    [CUDA only:] the weights (~2.9 GB), found through `data_file` in the JSON
*.json                   sidecar (to inform `neural.gen~` about in/outputs, attributes, etc.)
*.tokenizer.config.json  tokenizer config
*.tokenizer.json         copied HF tokenizer
```

where the filename is `{model}_text_to_audio[_cfg]_T{t}` or `{model}_audio_to_audio[_cfg]_T{t}`, plus
a `_cuda` suffix for CUDA exports (e.g. `small-music_text_to_audio_T32_cuda.pte`). The
`neural.gen~` host loads this bundle; keep all the files together. A CUDA export runs only on
Windows with an NVIDIA GPU (Turing / RTX 20xx or newer), and an MLX export only on Apple Silicon.


## Model I/O

Positional; **order + shape + roles are recognized by neural.gen~**

**Text-to-audio** (`{model}_text_to_audio_T{t}.pte`):

```
**name**        **shape**           **role**    **description**
token_ids       (1, 256)            condition   pre-tokenized prompt
attention_mask  (1, 256)            condition   1=real token, 0=pad
target_sec      (1,)                attribute   duration conditioning [0, 384]
noise_x         (1, 256, T_lat)     noise       initial Gaussian noise
noise_steps     (7, 1, 256, T_lat)  noise       per-step re-noise (steps 0-6)
→ audio         (1, 2, T_lat*4096)  audio       44.1 kHz
```

**Audio-to-audio** (`{model}_audio_to_audio_T{t}.pte`):

```
**name**         **shape**           **role**    **description**
audio_in         (1, 2, T_lat*4096)  buffer      init / clip @44.1k stereo
token_ids        (1, 256)            condition   pre-tokenized prompt
attention_mask   (1, 256)            condition   1=real token, 0=pad
inpaint_mask     (1, 1, T_lat)       condition   per-latent-frame mask (1=keep, 0=regenerate)
target_sec       (1,)                attribute   duration conditioning [0, 384]
noise_level      (1,)                attribute   how much the init audio influences the output
noise_x          (1, 256, T_lat)     noise       initial Gaussian noise
noise_steps      (7, 1, 256, T_lat)  noise       per-step re-noise (steps 0-6)
→ audio          (1, 2, T_lat*4096)  audio       44.1 kHz
```

**CFG variants** (`--cfg`) adds: `neg_token_ids` (1, 256), `neg_attention_mask` (1, 256), and `cfg_scale` (1,).  

`cfg_scale` is an "attribute"-role input, 1.0 reproduces the non-CFG output; ~7.0
gives stronger prompt adherence. CFG is not very useful for distilled checkpoints and may degrade quality.

## License

See [LICENSE](LICENSE) (Stability AI Community License).
