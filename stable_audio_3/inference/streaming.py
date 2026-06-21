"""Stateful streaming wrapper for the SAME autoencoder.

The SAME encoder/decoder are non-causal transformers with chunked attention. When
audio is processed in independent buffers (as in ``demo.ipynb``), each buffer is
**zero-padded** to a chunk boundary at both edges, so the edge frames are
reconstructed against silence instead of the real neighbouring audio. Concatenating
those buffers produces a value/derivative jump at every seam — the audible click.

This module fixes that with **overlap-save** streaming: every model call is fed real
neighbouring context (left context + a small right-context lookahead) and only the
validated central region is emitted. Context is carried as state across calls, so the
result is sample-identical to a single-pass run of the whole signal (once the model's
inference-time noise is disabled — see ``StreamingAutoencoder``).

This is the streaming generalisation of the offline overlap-save already present in
:meth:`AudioAutoencoder.decode_audio` and the MLX port's ``decode_chunked``; it adds
nothing to the model weights (frozen-weights / inference-only).

Latency note: SAME-S compresses 4096 samples (~92.9 ms at 44.1 kHz) into one latent
frame, and folds attention in chunks of ``chunk_size // stride`` frames, so windows
align to that many frames. Algorithmic latency is roughly
``(hop + right_lookahead + 1) * frame_seconds`` plus compute. The only frozen-weight
levers are minimising ``right_lookahead`` and (experimentally) reducing ``chunk_size``.
"""

from __future__ import annotations

import typing as tp

import torch

from ..models.autoencoders import AudioAutoencoder, TransformerResamplingBlock


def _disable_inference_noise(ae: AudioAutoencoder) -> None:
    """Make encode/decode deterministic so overlapping windows splice exactly.

    Two sources of per-call randomness break overlap-save splicing:
      * the soft-norm bottleneck adds noise on every ``decode`` (``noise_regularize``);
      * decoder resampling blocks perturb their query tokens (``mask_noise``).
    Both are instance attributes read in ``forward`` — clearing them changes no weights.
    """
    if getattr(ae, "bottleneck", None) is not None and hasattr(
        ae.bottleneck, "noise_regularize"
    ):
        ae.bottleneck.noise_regularize = False
    for module in ae.modules():
        if isinstance(module, TransformerResamplingBlock):
            module.mask_noise = 0


class _OverlapSaveStage:
    """Generic stateful overlap-save engine over a 1-D (last-axis) stream.

    Frames are the natural unit of the stage: for encode, one input frame is
    ``in_unit`` audio samples and one output frame is ``out_unit = 1`` latent; for
    decode it is reversed. Each step feeds a window of
    ``left + hop + right`` frames of *real* data and keeps the central ``hop``
    output frames. ``left``/``hop``/``right`` and the window length must be multiples
    of ``align`` (the model's chunk-fold granularity) to avoid internal zero-padding.
    """

    def __init__(
        self,
        fn: tp.Callable[[torch.Tensor], torch.Tensor],
        *,
        in_unit: int,
        out_unit: int,
        left: int,
        hop: int,
        right: int,
        align: int,
    ):
        for name, v in (("left", left), ("hop", hop), ("right", right)):
            if v % align != 0:
                raise ValueError(f"{name}={v} must be a multiple of align={align}")
        if hop <= 0:
            raise ValueError("hop must be > 0")
        self.fn = fn
        self.in_unit = in_unit
        self.out_unit = out_unit
        self.left = left
        self.hop = hop
        self.right = right
        self.align = align
        self.reset()

    def reset(self) -> None:
        self._buf: tp.Optional[torch.Tensor] = None  # (B, C, n * in_unit)
        self._n_left = 0  # left-context frames currently at the front of _buf

    @property
    def lookahead_frames(self) -> int:
        return self.right

    def _frames(self) -> int:
        return 0 if self._buf is None else self._buf.shape[-1] // self.in_unit

    def _append(self, x: torch.Tensor) -> None:
        self._buf = x if self._buf is None else torch.cat([self._buf, x], dim=-1)

    def push(self, x: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """Feed new input frames; return any newly-finalised output, else ``None``."""
        self._append(x)
        outs = []
        while self._frames() >= self._n_left + self.hop + self.right:
            window = self._n_left + self.hop + self.right  # frames (multiple of align)
            win = self._buf[..., : window * self.in_unit]
            out = self.fn(win)  # (B, C, window * out_unit)
            start = self._n_left * self.out_unit
            stop = (self._n_left + self.hop) * self.out_unit
            outs.append(out[..., start:stop])
            drop = self._n_left + self.hop - self.left  # keep `left` frames as context
            self._buf = self._buf[..., drop * self.in_unit :]
            self._n_left = self.left
        if not outs:
            return None
        return torch.cat(outs, dim=-1)

    def flush(self) -> tp.Optional[torch.Tensor]:
        """Emit all buffered frames; the tail has no right context (stream end)."""
        # A trailing partial frame (< in_unit samples) is invisible to the frame
        # counter; zero-pad it up to a whole frame, matching offline preprocessing.
        if self._buf is not None:
            rem_sub = self._buf.shape[-1] % self.in_unit
            if rem_sub:
                pad_shape = (*self._buf.shape[:-1], self.in_unit - rem_sub)
                z = self._buf.new_zeros(pad_shape)
                self._buf = torch.cat([self._buf, z], dim=-1)
        remaining = self._frames() - self._n_left
        if remaining <= 0:
            self.reset()
            return None
        # Pad up to an `align` multiple with edge replication, then trim the extra.
        total = self._n_left + remaining
        pad = (self.align - total % self.align) % self.align
        buf = self._buf
        if pad:
            tail = buf[..., -self.in_unit :].repeat(1, 1, pad)
            buf = torch.cat([buf, tail], dim=-1)
        out = self.fn(buf)
        start = self._n_left * self.out_unit
        stop = (self._n_left + remaining) * self.out_unit
        result = out[..., start:stop]
        self.reset()
        return result


class StreamingAutoencoder:
    """Click-free streaming round-trip over a SAME autoencoder (frozen weights).

    Wrap an :class:`AutoencoderModel` (or a raw :class:`AudioAutoencoder`) and feed it
    audio incrementally::

        stream = StreamingAutoencoder(ae)              # ae = AutoencoderModel.from_pretrained("same-s")
        for block in audio_blocks:                     # block: (B, 2, n) @ model SR, stereo
            latents = stream.encode_push(block)        # newly-finalised latents or None
            if latents is not None:
                audio = stream.decode_push(latents)    # newly-finalised audio or None
                if audio is not None:
                    play(audio)
        # drain the buffered tail at end-of-stream:
        tail_lat = stream.encode_flush()
        if tail_lat is not None:
            audio = stream.decode_push(tail_lat)
        audio = stream.decode_flush()

    Input audio must already be at ``ae.sample_rate`` and have the model's channel
    count (stereo); no resampling/channel conversion is done per call. All sizes are in
    latent frames (1 frame = ``downsampling_ratio`` samples).
    """

    def __init__(
        self,
        ae,
        *,
        left_overlap: int = 2,
        right_lookahead: int = 2,
        hop: int = 2,
        chunk_size: tp.Optional[int] = None,
    ):
        # Accept either AutoencoderModel (has .autoencoder) or a raw AudioAutoencoder.
        autoencoder = getattr(ae, "autoencoder", ae)
        if not isinstance(autoencoder, AudioAutoencoder):
            raise TypeError(
                "ae must be an AutoencoderModel or AudioAutoencoder, got "
                f"{type(autoencoder).__name__}"
            )
        self.ae = autoencoder
        self.sample_rate = getattr(ae, "sample_rate", getattr(autoencoder, "sample_rate"))
        self.device = getattr(ae, "device", next(autoencoder.parameters()).device)
        self.spl = int(autoencoder.downsampling_ratio)  # samples per latent frame

        _disable_inference_noise(self.ae)
        if chunk_size is not None:
            self._set_chunk_size(chunk_size)

        align = self._fold_alignment()  # latent frames per chunk-fold
        common = dict(left=left_overlap, hop=hop, right=right_lookahead, align=align)

        self._enc = _OverlapSaveStage(
            self._encode_window, in_unit=self.spl, out_unit=1, **common
        )
        self._dec = _OverlapSaveStage(
            self._decode_window, in_unit=1, out_unit=self.spl, **common
        )

    # --- model-config introspection ---------------------------------------

    def _resampling_blocks(self) -> list[TransformerResamplingBlock]:
        return [
            m for m in self.ae.modules() if isinstance(m, TransformerResamplingBlock)
        ]

    def _set_chunk_size(self, chunk_size: int) -> None:
        for blk in self._resampling_blocks():
            # The midpoint-shift re-chunking tiles the sequence into blocks of
            # effective_chunk = chunk_size + chunk_size // stride and shifts by half
            # of that, so effective_chunk must be even — i.e. chunk_size // stride must
            # be even, i.e. chunk_size must be a multiple of 2 * stride. The trained
            # default (32 = 2 * 16) is therefore the *minimum*; chunk_size cannot be
            # lowered to reduce latency, only raised (coarser, more context).
            if chunk_size % (2 * blk.stride) != 0:
                raise ValueError(
                    f"chunk_size={chunk_size} must be a multiple of 2*stride="
                    f"{2 * blk.stride} for midpoint-shift chunking to tile evenly"
                )
            blk.chunk_size = chunk_size

    def _fold_alignment(self) -> int:
        """Latent-frame granularity windows must align to (= chunk_size // stride)."""
        aligns = {blk.chunk_size // blk.stride for blk in self._resampling_blocks()}
        if not aligns:
            return 1
        # Use the LCM so a single window length satisfies every block.
        import math

        align = 1
        for a in aligns:
            align = align * a // math.gcd(align, a)
        return align

    # --- per-window model calls (fed real, aligned context) ----------------

    @torch.inference_mode()
    def _encode_window(self, audio: torch.Tensor) -> torch.Tensor:
        return self.ae.encode(audio.to(self.device))

    @torch.inference_mode()
    def _decode_window(self, latents: torch.Tensor) -> torch.Tensor:
        return self.ae.decode(latents.to(self.device))

    # --- public streaming API ----------------------------------------------

    def encode_push(self, audio: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """Push audio ``(B, C, n)``; return newly-finalised latents or ``None``."""
        return self._enc.push(audio.to(self.device))

    def encode_flush(self) -> tp.Optional[torch.Tensor]:
        """Drain buffered audio at end-of-stream (tail lacks right context)."""
        return self._enc.flush()

    def decode_push(self, latents: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """Push latents ``(B, 256, n)``; return newly-finalised audio or ``None``."""
        return self._dec.push(latents.to(self.device))

    def decode_flush(self) -> tp.Optional[torch.Tensor]:
        """Drain buffered latents at end-of-stream (tail lacks right context)."""
        return self._dec.flush()

    def reset(self) -> None:
        self._enc.reset()
        self._dec.reset()

    # --- latency accounting -------------------------------------------------

    @property
    def frame_seconds(self) -> float:
        return self.spl / self.sample_rate

    @property
    def encode_latency_frames(self) -> int:
        """Latents that must be buffered before the first emit (warmup lookahead)."""
        return self._enc.hop + self._enc.right

    @property
    def roundtrip_latency_seconds(self) -> float:
        """Algorithmic encode+decode latency, excluding compute time."""
        added = self._enc.hop + self._enc.right + self._dec.hop + self._dec.right
        return (added + 1) * self.frame_seconds
