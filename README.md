# Integrating Stable Audio 3 in Max/MSP

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-shuoyang--zheng%2Fstable--audio--3--tilde-yellow)](https://huggingface.co/shuoyang-zheng/stable-audio-3-tilde)


**Export [Stable Audio 3](https://huggingface.co/collections/stabilityai/stable-audio-3) models for the
[`neural.gen~`](https://github.com/jasper-zheng/neural_tilde) package in Max/MSP**  
Models are exported with [ExecuTorch](https://github.com/pytorch/executorch) in MLX backend (MacOS / Apple Silicon only) for hardware-accelerated local inference. Supoorts distilled checkpoints (`small-music` and `small-sfx`). 


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

If you would like to export
your own, you need an **Apple Silicon Mac** (the MLX backend is Metal-only) with **full Xcode**
installed (the Metal compiler ships with Xcode, *not* the standalone Command Line Tools).
<!-- 
> **Why the specific versions below?** The MLX backend only exists on ExecuTorch's `main` branch,
> which builds against `torch==2.12.0` and requires Python `<3.14`. `torch.export` → ExecuTorch
> lowering must run on the *same* torch ExecuTorch was built with, so this repo pins
> `torch==2.12.0` / `torchaudio==2.11.0` (from the PyTorch *test* channel, not PyPI). `uv sync`
> handles this for you — don't mix in a different torch. -->

**1. Check the Metal compiler is available** (install Xcode from the App Store if it errors):

```bash
xcrun -sdk macosx --find metal
```

**2. Clone this repo and install the Python environment** with
[`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/jasper-zheng/stable-audio-3-tilde.git
cd stable-audio-3-tilde
uv sync                            # installs torch + torchaudio + the rest
uv pip install pip wheel "cmake<4" # build tools ExecuTorch's installer needs inside the venv
uv pip install neural-tilde        # the neural.gen~ integration package
```

**3. Build ExecuTorch with the MLX backend into this same venv.** It is **not** on PyPI — build it
from source. From the repo root:

```bash
git clone https://github.com/pytorch/executorch.git
source .venv/bin/activate              # use this repo's venv so ExecuTorch installs into it
cd executorch
python install_executorch.py --minimal # builds the MLX backend 
cd ..
```

The build takes a while (it compiles native + Metal code). See the
[ExecuTorch MLX Getting Started](https://github.com/pytorch/executorch/tree/main/backends/mlx#getting-started)
guide for details and troubleshooting.

Verify the backend is importable:

```bash
python -c "import executorch.backends.mlx; print('MLX backend OK')"
```

**4. HuggingFace access** to the checkpoints
([`stabilityai/stable-audio-3-small-music`](https://huggingface.co/stabilityai/stable-audio-3-small-music)
and [`stabilityai/stable-audio-3-small-sfx`](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)) —
run `uv run huggingface-cli login` if you haven't authenticated.


## Export

### Important limitation:  
1. The generation length need to be fixed at export time. The exported model always generates in this length (set by `--t-lat`, total latents). The default is `--t-lat 32` (≈ 2.97 s). You need to **export multiple models if you want to support different generation lengths**.
2. `--t-lat` must be an even number, it sets the generation length (`4096 x T_lat` samples at 44.1 kHz).
3. LoRA / fine-tuning is not supported yet. The exported model is frozen and cannot be further trained.

### Commands:

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

### Exported files

Each run writes four files into `export/artifacts/`:

```
*.pte                    the ExecuTorch model (~2.9-3.3 GB)
*.json                   sidecar (to inform `neural.gen~` about in/outputs, attributes, etc.)
*.tokenizer.config.json  tokenizer config
*.tokenizer.json         copied HF tokenizer
```

where the filename is `{model}_text_to_audio[_cfg]_T{t}` or `{model}_audio_to_audio[_cfg]_T{t}`. The
`neural.gen~` host loads this bundle.


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
