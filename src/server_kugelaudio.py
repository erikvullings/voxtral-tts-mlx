from __future__ import annotations

import os
import re
import threading
from typing import Any, Iterable, Optional, Protocol, cast

import mlx.core as mx
import numpy as np

from api_shared import OpenAISpeechRequest, VoxtralExtendedRequest, create_app
from mlx_utils import apply_logit_penalty, warmup_mlx_model
from server_mlx_audio import collect_generation_audio, write_audio_output

# Special token IDs (same as the upstream kugelaudio-open processor)
_SPEECH_START_ID = 151652
_SPEECH_DIFFUSION_ID = 151654
_SPEECH_END_ID = 151653


class _KugelAudioModel(Protocol):
    def generate(self, *, text: str, voice: str, **kwargs: Any) -> Iterable[Any]: ...


# ---------------------------------------------------------------------------
# Voice conditioning
# ---------------------------------------------------------------------------

class _FirstCallVoiceWrapper:
    """Wraps the MLX Qwen2 language model so that the very first __call__
    uses a pre-built voice-conditioned inputs_embeds instead of input_ids.

    Background
    ----------
    The MLX KugelAudio model's _generate_impl always starts with:

        hidden_states, cache = self.language_model(input_ids=prompt_ids)

    The upstream PyTorch model injects pre-encoded voice embeddings into that
    first forward pass.  Since we cannot easily modify mlx-audio, we wrap the
    language_model object: the first call (the full-prompt forward pass) uses
    our voice-conditioned inputs_embeds; every subsequent call (per-token KV
    cache steps) falls through to the real language model unchanged.
    """

    def __init__(self, real_lm: Any, voice_inputs_embeds: mx.array) -> None:
        self._real_lm = real_lm
        self._voice_inputs_embeds = voice_inputs_embeds
        self._intercepted = False

    def __call__(self, input_ids: Any = None, inputs_embeds: Any = None, **kwargs: Any) -> Any:
        if not self._intercepted:
            self._intercepted = True
            # Override with our voice-conditioned embeddings; ignore input_ids
            return self._real_lm(inputs_embeds=self._voice_inputs_embeds, **kwargs)
        return self._real_lm(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_lm, name)


def _load_voice_pt(pt_path: str) -> mx.array:
    """Load a .pt voice file and return acoustic_mean as an MLX array.

    The .pt file must contain {"acoustic_mean": tensor} where tensor has shape
    (1, T, vae_dim) or (T, vae_dim).  It is saved by the acoustic encoder
    (see scripts/encode_voice_kugelaudio.py).
    """
    import torch  # always available in this venv

    vc = torch.load(pt_path, weights_only=True, map_location="cpu")
    if "acoustic_mean" not in vc:
        raise ValueError(f"Voice file {pt_path!r} has no 'acoustic_mean' key. Keys: {list(vc.keys())}")

    mean_np = vc["acoustic_mean"].float().numpy()
    # Normalise to (T, vae_dim) — drop batch dim if present
    if mean_np.ndim == 3:
        mean_np = mean_np[0]  # (T, vae_dim)
    return mx.array(mean_np)


def _build_voice_inputs_embeds(model: Any, text: str, acoustic_mean: mx.array) -> mx.array:
    """Build the full voice-conditioned inputs_embeds for the initial LM pass.

    Mirrors what the upstream KugelAudioProcessor.__call__ + _process_speech_inputs
    do in the PyTorch implementation:

      system_prompt
      " Voice input:\\n Speaker 0:" + [SPEECH_DIFFUSION placeholder * T]
      "\\n Text input:\\n {text}\\n Speech output:\\n"
      SPEECH_START_ID

    The SPEECH_DIFFUSION placeholders are replaced with
        acoustic_connector( (acoustic_mean + bias) * scale )
    which injects the voice's acoustic characteristics into the LM context.
    """
    tokenizer = model.tokenizer
    lm = model.language_model

    # ── build token sequences ──────────────────────────────────────────────
    system_prompt = (
        " Transform the text provided by various speakers into speech output,"
        " utilizing the distinct voice of each respective speaker.\n"
    )
    formatted_text = text.strip()
    if not formatted_text.startswith("Speaker"):
        formatted_text = f"Speaker 0: {formatted_text}"

    sys_ids = tokenizer.encode(system_prompt, add_special_tokens=False)
    voice_hdr_ids = tokenizer.encode(" Voice input:\n", add_special_tokens=False)
    spk_prefix_ids = tokenizer.encode(" Speaker 0:", add_special_tokens=False)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    text_ids = tokenizer.encode(
        f" Text input:\n {formatted_text}\n Speech output:\n",
        add_special_tokens=False,
    )

    # acoustic_mean: (T, vae_dim) → number of voice placeholder tokens
    num_voice_tokens: int = int(acoustic_mean.shape[0])

    voice_start = len(sys_ids) + len(voice_hdr_ids) + len(spk_prefix_ids)
    voice_end = voice_start + num_voice_tokens

    all_ids = (
        sys_ids
        + voice_hdr_ids
        + spk_prefix_ids
        + [_SPEECH_DIFFUSION_ID] * num_voice_tokens
        + newline_ids
        + text_ids
        + [_SPEECH_START_ID]
    )

    # ── compute token embeddings for the whole sequence ───────────────────
    id_array = mx.array([all_ids], dtype=mx.int32)
    token_embeds = lm.embed_tokens(id_array)  # (1, L, H)
    mx.eval(token_embeds)

    # ── compute voice embeddings via acoustic_connector ───────────────────
    # acoustic_mean: (T, vae_dim) → (1, T, vae_dim)
    latents = mx.expand_dims(acoustic_mean, axis=0).astype(mx.float32)

    # Apply the same scaling as _process_speech_inputs in the PyTorch model
    bias = model.speech_bias_factor
    scale = model.speech_scaling_factor
    latents = (latents + bias) * scale

    voice_embed = model.acoustic_connector(latents)  # (1, T, H)
    mx.eval(voice_embed)

    # ── splice voice embeddings into the token embeddings ─────────────────
    inputs_embeds = mx.concatenate(
        [
            token_embeds[:, :voice_start, :],
            voice_embed,
            token_embeds[:, voice_end:, :],
        ],
        axis=1,
    )
    mx.eval(inputs_embeds)
    return inputs_embeds



class KugelAudioOpenAISpeechRequest(OpenAISpeechRequest):
    model: str = "kugelaudio/kugelaudio-0-open"
    voice: str = "default"
    cfg_scale: float = 3.0
    max_new_tokens: int | None = None
    do_sample: bool = False
    temperature: float = 1.0
    speaker_names: list[str] | None = None
    ref_audio: list[str] | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "kugelaudio/kugelaudio-0-open",
                "input": "Hallo, dit is een KugelAudio voice sample.",
                "voice": "warm",
                "language": "nl",
                "response_format": "mp3",
                "cfg_scale": 3.0,
                "max_new_tokens": 2048,
                "do_sample": False,
                "temperature": 1.0,
            }
        }
    }


class KugelAudioSpeechRequest(VoxtralExtendedRequest):
    voice_reference_path: Optional[str] = None
    cfg_scale: float = 3.0
    max_new_tokens: int | None = None
    do_sample: bool = False
    temperature: float = 1.0
    speaker_names: list[str] | None = None
    ref_audio: list[str] | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Hallo, dit is een KugelAudio sample.",
                "voice_reference_path": "warm",
                "language": "nl",
                "cfg_scale": 3.0,
                "max_new_tokens": 2048,
                "output_filename": "kugelaudio_lesson.mp3",
            }
        }
    }


class RealKugelAudioEngine:
    MODEL_ID = "kugelaudio/kugelaudio-0-open"
    PRESET_VOICES = ("default", "warm", "clear")

    # Fallback seeds used when no .pt voice file is found so that each preset
    # name at least produces consistent output across calls.
    _FALLBACK_SEEDS: dict[str, int] = {"default": 42, "warm": 7, "clear": 13}

    @staticmethod
    def _strip_speaker_labels(text: str) -> str:
        cleaned = re.sub(r"(?im)^\s*speaker\s*\d+\s*:\s*", "", text)
        return cleaned.strip()

    @staticmethod
    def _auto_max_new_tokens(text: str, requested: Any) -> int:
        if requested is not None:
            try:
                value = int(requested)
                return max(512, min(8192, value))
            except (TypeError, ValueError):
                pass

        words = len([w for w in text.replace("\n", " ").split(" ") if w.strip()])
        estimated = int(words * 7.5) + 1800
        if "\n" in text:
            estimated += 1200
        return max(2048, min(8192, estimated))

    def __init__(self):
        self._model: Optional[_KugelAudioModel] = None
        self._lock = threading.Lock()

    @staticmethod
    def _generate_with_fallback(model: _KugelAudioModel, kwargs: dict[str, Any]):
        attempts = [
            kwargs,
            {k: v for k, v in kwargs.items() if k not in {"speaker_names", "ref_audio"}},
            {k: v for k, v in kwargs.items() if k not in {"speaker_names", "ref_audio", "language"}},
        ]

        last_error: Optional[Exception] = None
        for attempt in attempts:
            try:
                return model.generate(**attempt)
            except TypeError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise RuntimeError(f"KugelAudio generate signature mismatch: {last_error}")
        raise RuntimeError("KugelAudio generation failed")

    @staticmethod
    def _warmup_model(model: _KugelAudioModel) -> None:
        """Run a minimal generation to trigger MLX Metal shader compilation."""
        def generate_warmup():
            return model.generate(
                text="Hi.",
                voice="default",
                cfg_scale=3.0,
                max_tokens=10,
                verbose=False,
            )

        warmup_mlx_model(model, generate_warmup)

    def _load_model(self) -> _KugelAudioModel:
        with self._lock:
            if self._model is None:
                print(f"📦 Loading KugelAudio MLX model ({self.MODEL_ID})...")
                from mlx_audio.tts.utils import load

                loaded_model = cast(_KugelAudioModel, load(self.MODEL_ID))
                print("✅ KugelAudio model loaded.")
                # Apply speech_end_id penalty to prevent premature generation cutoff
                loaded_model = apply_logit_penalty(loaded_model, _SPEECH_END_ID, penalty_strength=5.0)
                self._warmup_model(loaded_model)
                self._model = loaded_model

            model = self._model
            if model is None:
                raise RuntimeError("KugelAudio model failed to initialize")
            return model

    @staticmethod
    def _apply_speed(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
        """Pitch-preserving time-stretch via Rubber Band Library (pyrubberband)."""
        speed = max(0.25, min(4.0, speed))
        if abs(speed - 1.0) < 0.001 or audio.size == 0:
            return audio

        import pyrubberband

        # Prepend silence so the stretcher stabilizes before first speech frames.
        pad = int(0.1 * sample_rate)
        padded = np.concatenate([np.zeros(pad, dtype=audio.dtype), audio])
        stretched = pyrubberband.time_stretch(padded, sample_rate, speed).astype(audio.dtype)
        return stretched[round(pad / speed) :]

    @staticmethod
    def _trim_leading_silence(
        audio: np.ndarray,
        *,
        sample_rate: int,
        threshold: float = 0.01,
        lead_in_ms: float = 10.0,
    ) -> np.ndarray:
        if audio.size == 0:
            return audio

        onset_indices = np.flatnonzero(np.abs(audio) > threshold)
        if onset_indices.size == 0:
            return audio

        lead_in_samples = int(sample_rate * lead_in_ms / 1000.0)
        trim_idx = max(0, int(onset_indices[0]) - lead_in_samples)
        if trim_idx <= 0:
            return audio
        return audio[trim_idx:]

    @staticmethod
    def _resolve_pt_file(name: str) -> Optional[str]:
        """Return path to a .pt voice file for *name*, or None if not found.

        Search order:
          1. voices/<name>.pt
          2. <name>.pt  (absolute or relative path passed directly)
        """
        for candidate in (
            os.path.join("voices", f"{name}.pt"),
            f"{name}.pt",
            name if name.endswith(".pt") else None,
        ):
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    @staticmethod
    def list_voices() -> dict[str, list[str]]:
        voices_dir = "voices"
        wav_voices: list[str] = []
        pt_voices: list[str] = []
        if os.path.isdir(voices_dir):
            stems = {
                os.path.splitext(name)[0]
                for name in os.listdir(voices_dir)
            }
            wav_voices = sorted(
                s for s in stems
                if os.path.isfile(os.path.join(voices_dir, f"{s}.wav"))
            )
            pt_voices = sorted(
                s for s in stems
                if os.path.isfile(os.path.join(voices_dir, f"{s}.pt"))
            )
        return {
            "presets": list(RealKugelAudioEngine.PRESET_VOICES),
            "ptVoices": pt_voices,
            "wavFiles": wav_voices,
        }

    @staticmethod
    def _split_text_chunks(text: str, max_chars: int = 240) -> list[tuple[str, bool]]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        sentences = [
            chunk.strip()
            for chunk in re.split(r"(?<=[.!?])\s+", normalized)
            if chunk.strip()
        ]
        if not sentences:
            sentences = [normalized]

        def split_long_piece(piece: str) -> list[tuple[str, bool]]:
            if len(piece) <= max_chars:
                return [(piece, True)]

            # First fallback: split by clause punctuation to keep phrasing natural.
            clause_parts = [
                seg.strip()
                for seg in re.split(r"(?<=[,;:])\s+", piece)
                if seg.strip()
            ]
            if len(clause_parts) <= 1:
                clause_parts = [piece]

            out: list[str] = []
            for clause in clause_parts:
                if len(clause) <= max_chars:
                    out.append(clause)
                    continue

                # Final fallback: split on whitespace only (never inside a word).
                words = clause.split()
                if not words:
                    continue
                current = words[0]
                for word in words[1:]:
                    candidate = f"{current} {word}"
                    if len(candidate) <= max_chars:
                        current = candidate
                    else:
                        out.append(current)
                        current = word
                out.append(current)

            if not out:
                return []
            chunks_with_flags: list[tuple[str, bool]] = []
            for idx, chunk in enumerate(out):
                chunks_with_flags.append((chunk, idx == len(out) - 1))
            return chunks_with_flags

        sentence_units: list[tuple[str, bool]] = []
        for sentence in sentences:
            sentence_units.extend(split_long_piece(sentence))

        # Keep sentence boundaries explicit; do not merge neighboring sentences.
        return sentence_units

    def _synthesize_chunk(
        self,
        model: _KugelAudioModel,
        *,
        text: str,
        call_kwargs: dict[str, Any],
        pt_file: str | None,
        seed: int,
    ) -> tuple[np.ndarray, int]:
        chunk_kwargs = dict(call_kwargs)
        chunk_kwargs["text"] = text
        chunk_kwargs["max_new_tokens"] = self._auto_max_new_tokens(
            text,
            call_kwargs.get("max_new_tokens"),
        )

        if pt_file is not None:
            acoustic_mean = _load_voice_pt(pt_file)
            voice_inputs_embeds = _build_voice_inputs_embeds(model, text, acoustic_mean)

            original_lm = model.language_model  # type: ignore[attr-defined]
            model.language_model = _FirstCallVoiceWrapper(original_lm, voice_inputs_embeds)  # type: ignore[attr-defined]
            try:
                results = self._generate_with_fallback(model, chunk_kwargs)
                audio, sample_rate = collect_generation_audio(
                    results, default_sample_rate=getattr(model, "sample_rate", 24000)
                )
                return self._trim_leading_silence(
                    audio,
                    sample_rate=sample_rate,
                ), sample_rate
            finally:
                model.language_model = original_lm  # type: ignore[attr-defined]

        mx.random.seed(seed)
        results = self._generate_with_fallback(model, chunk_kwargs)
        audio, sample_rate = collect_generation_audio(
            results, default_sample_rate=getattr(model, "sample_rate", 24000)
        )
        return self._trim_leading_silence(
            audio,
            sample_rate=sample_rate,
        ), sample_rate

    def synthesize(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        **kwargs: Any,
    ) -> str:
        model = self._load_model()

        # Resolve voice name — accept preset names OR a bare stem that
        # resolves to a .pt file in voices/ (custom voices).
        voice_raw = (voice_path or kwargs.get("voice") or "default").strip().lower()
        is_preset = voice_raw in self.PRESET_VOICES
        pt_file = self._resolve_pt_file(voice_raw)

        if not is_preset and pt_file is None:
            raise ValueError(
                f"Unknown voice {voice_raw!r}. "
                "Use a preset (default/warm/clear) or place a <name>.pt file in voices/. "
                "Generate .pt files with: uv run python scripts/encode_voice_kugelaudio.py"
            )

        # For preset names that have no .pt yet, voice is preset-only.
        voice = voice_raw if is_preset else voice_raw

        language = kwargs.get("language")
        if language is not None:
            lang = str(language).strip().lower()
            if not re.fullmatch(r"[a-z]{2}", lang):
                raise ValueError(
                    "KugelAudio language must be a 2-letter code (for example 'en' or 'nl')."
                )
            language = lang

        normalized_text = self._strip_speaker_labels(text)
        if not normalized_text:
            raise ValueError("Text input cannot be empty")

        call_kwargs: dict[str, Any] = {
            "text": normalized_text,
            "voice": voice,
            "language": language,
            "cfg_scale": float(kwargs.get("cfg_scale", 3.0)),
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "do_sample": bool(kwargs.get("do_sample", False)),
            "temperature": float(kwargs.get("temperature", 1.0)),
        }

        chunk_chars = int(os.getenv("KUGELAUDIO_MAX_CHARS_PER_CHUNK", "240"))
        chunk_chars = max(80, min(600, chunk_chars))
        text_chunks = self._split_text_chunks(normalized_text, max_chars=chunk_chars)
        if not text_chunks:
            raise ValueError("Text input cannot be empty")

        # Use explicit per-boundary pauses for consistent pacing.
        stitch_gap_ms = int(os.getenv("KUGELAUDIO_STITCH_GAP_MS", "40"))
        stitch_gap_ms = max(0, min(500, stitch_gap_ms))
        sentence_gap_ms = int(os.getenv("KUGELAUDIO_SENTENCE_GAP_MS", "160"))
        sentence_gap_ms = max(0, min(5000, sentence_gap_ms))

        fallback_seed = self._FALLBACK_SEEDS.get(voice, 42)
        sample_rate = int(getattr(model, "sample_rate", 24000))
        stitched: list[np.ndarray] = []
        for idx, (chunk, ends_sentence) in enumerate(text_chunks):
            chunk_audio, chunk_sample_rate = self._synthesize_chunk(
                model,
                text=chunk,
                call_kwargs=call_kwargs,
                pt_file=pt_file,
                seed=fallback_seed + idx,
            )
            if idx > 0 and stitch_gap_ms > 0:
                stitched.append(
                    np.zeros(
                        int(chunk_sample_rate * stitch_gap_ms / 1000.0),
                        dtype=np.float32,
                    )
                )
            stitched.append(chunk_audio)
            if ends_sentence and sentence_gap_ms > 0:
                stitched.append(
                    np.zeros(
                        int(chunk_sample_rate * sentence_gap_ms / 1000.0),
                        dtype=np.float32,
                    )
                )
            sample_rate = chunk_sample_rate

        audio = np.concatenate(stitched) if stitched else np.zeros(0, dtype=np.float32)

        speed = float(kwargs.get("speed", 1.0))
        audio = self._apply_speed(audio, sample_rate, speed)

        return write_audio_output(output_path, audio, sample_rate=sample_rate)


kugelaudio_engine = RealKugelAudioEngine()
app = create_app(
    title="KugelAudio TTS Translation Layer",
    engine=kugelaudio_engine,
    voice_response_model=dict[str, list[str]],
    route_prefix="kugelaudio",
    backend_name="kugelaudio",
    openai_request_model=KugelAudioOpenAISpeechRequest,
    extended_request_model=KugelAudioSpeechRequest,
    backend_capabilities={
        "model": RealKugelAudioEngine.MODEL_ID,
        "voiceCloning": {
            "supported": True,
            "notes": (
                "Custom voices via .pt files in voices/. "
                "Generate with: uv run python scripts/encode_voice_kugelaudio.py --input voices/my.wav"
            ),
        },
        "voiceSelection": {
            "type": "preset+custom",
            "presets": list(RealKugelAudioEngine.PRESET_VOICES),
            "custom": "Place <name>.pt files in voices/ (created by encode_voice_kugelaudio.py)",
        },
        "ssmlProsody": {
            "tagParsing": False,
            "supportedTags": [],
            "notes": "No SSML tag parser is implemented in this adapter.",
        },
        "speakerControl": {
            "multiSpeaker": {
                "supported": False,
                "notes": (
                    "Current KugelAudio adapter is single-speaker. "
                    "Speaker labels are flattened into one voice."
                ),
            }
        },
        "languageConditioning": {
            "apiDefaultLanguage": None,
            "notes": "If provided, language must be an ISO-639-1 two-letter code.",
        },
        "generationBudgeting": {
            "maxNewTokens": {
                "clientConfigRequired": False,
                "auto": True,
                "defaultPolicy": "auto-by-input-length",
                "clampRange": [2048, 8192],
                "overrideField": "max_new_tokens",
            }
        },
        "longTextStability": {
            "chunking": {
                "enabled": True,
                "strategy": "sentence-first-then-clause-then-whitespace",
                "maxCharsPerChunkEnv": "KUGELAUDIO_MAX_CHARS_PER_CHUNK",
                "defaultMaxCharsPerChunk": 240,
            },
            "stitching": {
                "intraChunkGapMsEnv": "KUGELAUDIO_STITCH_GAP_MS",
                "defaultIntraChunkGapMs": 40,
                "sentenceGapMsEnv": "KUGELAUDIO_SENTENCE_GAP_MS",
                "defaultSentenceGapMs": 160,
                "maxSentenceGapMs": 5000,
            },
        },
        "timeScaling": {
            "supported": True,
            "requestField": "speed",
            "range": [0.25, 4.0],
            "implementation": "pitch-preserving-rubberband",
        },
    },
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_kugelaudio:app", host="0.0.0.0", port=8000, reload=True)
