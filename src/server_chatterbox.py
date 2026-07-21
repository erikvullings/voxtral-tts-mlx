import os
import re
import threading
from typing import Any, Optional, cast


def _normalize_mps_watermark_env() -> None:
    """Guard against invalid watermark combinations that crash torch/mps init."""
    low_raw = os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")
    high_raw = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")

    try:
        low = float(low_raw) if low_raw is not None else 1.4
    except ValueError:
        low = 1.4

    try:
        high = float(high_raw) if high_raw is not None else 1.7
    except ValueError:
        high = 1.7

    if low < 0:
        low = 0.0
    if high <= low:
        high = max(1.7, low + 0.1)

    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = f"{low:.3f}"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = f"{high:.3f}"


_normalize_mps_watermark_env()

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from api_shared import (
    _SENTENCE_SPLIT_RE,
    OpenAISpeechRequest,
    TranscriptDocument,
    VoiceOption,
    VoxtralExtendedRequest,
    VoxtralTranscriptRequest,
    VoxtralTranscriptResponse,
    create_app,
)


class ChatterboxOpenAISpeechRequest(OpenAISpeechRequest):
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    max_new_tokens: int | None = None


class ChatterboxVoxtralExtendedRequest(VoxtralExtendedRequest):
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    max_new_tokens: int | None = None


class RealChatterboxEngine:
    VOICE_ALIASES: dict[str, Optional[str]] = {
        "nl_female": "voices/nl_female.wav",
        "female": "voices/nl_female.wav",
        "default": "voices/nl_female.wav",
        "nl_male": "voices/jasper.wav",
        "male": "voices/jasper.wav",
        "jasper": "voices/jasper.wav",
    }
    FEMALE_ALIASES = {"nl_female", "female", "default"}

    def __init__(self):
        self._model: Optional[ChatterboxMultilingualTTS] = None
        self._lock = threading.Lock()

    @staticmethod
    def _detect_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> ChatterboxMultilingualTTS:
        with self._lock:
            if self._model is None:
                self._ensure_perth_watermarker()
                preferred_device = self._detect_device()
                devices = [preferred_device]
                if preferred_device != "cpu":
                    devices.append("cpu")

                last_error: Optional[Exception] = None
                for device in devices:
                    print(f"📦 Loading Chatterbox multilingual model on {device}...")
                    try:
                        try:
                            # Older releases accepted t3_model; newer releases select a default.
                            self._model = ChatterboxMultilingualTTS.from_pretrained(
                                device=device,
                                t3_model="v3",
                            )
                        except TypeError:
                            self._model = ChatterboxMultilingualTTS.from_pretrained(
                                device=device,
                            )
                        break
                    except RuntimeError as e:
                        last_error = e
                        if device != "cpu":
                            # Some torch/mps setups fail during model materialization.
                            print(f"⚠️ Failed on {device} ({e}); falling back to cpu.")
                            self._model = None
                            continue
                        raise

                if self._model is None and last_error is not None:
                    raise last_error

                print("✅ Chatterbox model loaded.")

            model = self._model
            if model is None:
                raise RuntimeError("Chatterbox model failed to initialize")
            return model

    @staticmethod
    def _ensure_perth_watermarker() -> None:
        """Work around perth builds where PerthImplicitWatermarker is exported as None."""
        try:
            import perth

            if getattr(perth, "PerthImplicitWatermarker", None) is None:
                perth.PerthImplicitWatermarker = perth.DummyWatermarker
        except Exception:
            # Keep runtime resilient: chatterbox may still work without watermarking.
            return

    @staticmethod
    def _resolve_audio_prompt_path(voice_path: Optional[str]) -> Optional[str]:
        if not voice_path:
            voice_path = "nl_female"

        key = voice_path.strip().lower()
        if key in RealChatterboxEngine.VOICE_ALIASES:
            aliased = RealChatterboxEngine.VOICE_ALIASES[key]
            if aliased is None:
                return None
            if os.path.isfile(aliased):
                return aliased
            # Allow optional female fallback filename when alias target is not present.
            if key in RealChatterboxEngine.FEMALE_ALIASES:
                for candidate in [
                    "voices/female.wav",
                    "voices/nl-female.wav",
                ]:
                    if os.path.isfile(candidate):
                        return candidate
                raise ValueError(
                    "Female voice requested, but no female reference clip was found. "
                    "Add voices/nl_female.wav (or voices/female.wav) and retry."
                )

        candidates = [
            voice_path,
            os.path.join("voices", voice_path),
            os.path.join("voices", f"{voice_path}.wav"),
        ]

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return None

    @staticmethod
    def list_voices() -> list[VoiceOption]:
        options = [
            VoiceOption(
                name="nl_female",
                gender="female",
                source="built-in",
                is_default=True,
            )
        ]

        for alias, source in RealChatterboxEngine.VOICE_ALIASES.items():
            if source is None:
                continue
            if os.path.isfile(source):
                gender = "male" if "male" in alias or alias == "jasper" else "custom"
                options.append(
                    VoiceOption(
                        name=alias,
                        gender=gender,
                        source=source,
                        is_default=False,
                    )
                )

        voices_dir = "voices"
        if os.path.isdir(voices_dir):
            for filename in sorted(os.listdir(voices_dir)):
                if not filename.lower().endswith(".wav"):
                    continue
                full_path = os.path.join(voices_dir, filename)
                stem = os.path.splitext(filename)[0].lower()
                known = any(option.name == stem for option in options)
                if known:
                    continue
                options.append(
                    VoiceOption(
                        name=stem,
                        gender="custom",
                        source=full_path,
                        is_default=False,
                    )
                )

        # Deduplicate aliases that point to the same visible option name.
        unique: dict[str, VoiceOption] = {}
        for option in options:
            unique[option.name] = option
        return list(unique.values())

    @staticmethod
    def _to_mono_float32(audio: Any) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            arr = audio.detach().float().cpu().numpy()
        else:
            arr = np.asarray(audio, dtype=np.float32)

        arr = np.squeeze(arr)
        if arr.ndim > 1:
            arr = arr[0]
        return arr.astype(np.float32)

    @staticmethod
    def _apply_speed(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
        speed = max(0.25, min(4.0, speed))
        # Chatterbox tends to sound fast; use a stronger slowdown curve for speed < 1.0.
        effective_speed = speed * speed if speed < 1.0 else speed
        if abs(effective_speed - 1.0) < 0.001 or audio.size == 0:
            return audio

        import pyrubberband

        pad = int(0.1 * sample_rate)
        padded = np.concatenate([np.zeros(pad, dtype=audio.dtype), audio])
        stretched = pyrubberband.time_stretch(
            padded, sample_rate, effective_speed
        ).astype(audio.dtype)
        return stretched[round(pad / effective_speed) :]

    @staticmethod
    def _apply_emphasis_markup(text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            body = match.group(1).strip()
            if not body:
                return ""
            # Emulate emphasis by punctuation and capitalization cues.
            return f", {body.upper()}!"

        return re.sub(
            r"<emphasis>(.*?)</emphasis>",
            repl,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

    @staticmethod
    def _split_markup_segments(text: str) -> list[tuple[str, float]]:
        """Split text on break tags into (segment_text, pause_seconds_after)."""
        normalized = RealChatterboxEngine._apply_emphasis_markup(text)
        break_pattern = re.compile(
            r"<break\s+time=\"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)\"\s*/?>",
            flags=re.IGNORECASE,
        )

        segments: list[tuple[str, float]] = []
        cursor = 0
        for match in break_pattern.finditer(normalized):
            piece = normalized[cursor : match.start()].strip()
            val = float(match.group("value"))
            unit = match.group("unit").lower()
            pause_s = val / 1000.0 if unit == "ms" else val
            pause_s = max(0.0, min(3.0, pause_s))
            segments.append((piece, pause_s))
            cursor = match.end()

        tail = normalized[cursor:].strip()
        if tail or not segments:
            segments.append((tail, 0.0))

        return segments

    @staticmethod
    def _split_text_chunks(text: str, max_chars: int = 220) -> list[str]:
        parts = [
            chunk.strip() for chunk in _SENTENCE_SPLIT_RE.split(text) if chunk.strip()
        ]
        if not parts:
            return [text.strip()] if text.strip() else []

        chunks: list[str] = []
        current = ""
        for part in parts:
            if not current:
                current = part
                continue
            candidate = f"{current} {part}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(current)
                current = part

        if current:
            chunks.append(current)

        return chunks

    @staticmethod
    def _auto_max_new_tokens(text: str, requested: Any) -> int:
        if requested is not None:
            try:
                value = int(requested)
                return max(120, min(700, value))
            except (TypeError, ValueError):
                pass

        words = len([w for w in text.split(" ") if w.strip()])
        estimated = int(words * 4.0) + 160
        return max(220, min(650, estimated))

    @staticmethod
    def _generate_chunk_with_cap(
        model: ChatterboxMultilingualTTS,
        *,
        text: str,
        language_id: str,
        temperature: float,
        max_new_tokens: int,
        cfg_weight: float,
        exaggeration: float,
        audio_prompt_path: Optional[str],
    ) -> torch.Tensor:
        # Use Chatterbox internals directly so max_new_tokens can be bounded.
        from chatterbox.mtl_tts import (
            SUPPORTED_LANGUAGES,
            T3Cond,
            drop_invalid_tokens,
            punc_norm,
        )

        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. Supported languages: {supported_langs}"
            )

        if audio_prompt_path:
            model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        elif model.conds is None:
            raise ValueError(
                "No voice conditionals available; provide voice_reference_path"
            )

        if model.conds is None:
            raise ValueError("Model conditionals are not initialized")

        # Keep generated prosody neutral/consistent across chunks.
        if float(model.conds.t3.emotion_adv[0, 0, 0].item()) != 0.5:
            _cond = model.conds.t3
            model.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=0.5 * torch.ones(1, 1, 1),
            ).to(device=model.device)

        clean = punc_norm(text)
        text_tokens = model.tokenizer.text_to_tokens(
            clean,
            language_id=language_id.lower() if language_id else None,
        ).to(model.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = model.t3.hp.start_text_token
        eot = model.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = model.t3.inference(
                t3_cond=model.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=2.0,
                min_p=0.05,
                top_p=1.0,
            )
            speech_tokens = speech_tokens[0]
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(model.device)

            wav, _ = model.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=model.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu()

        return wav

    def synthesize(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        **kwargs: Any,
    ) -> str:
        model = self._load_model()

        language = str(kwargs.get("language") or "").lower()
        speed = float(kwargs.get("speed", 1.0))
        temperature = float(kwargs.get("temperature", 0.8))
        max_new_tokens_override = kwargs.get("max_new_tokens")
        cfg_weight = float(kwargs.get("cfg_weight", 0.5))
        exaggeration = float(kwargs.get("exaggeration", 0.5))
        temperature = max(0.05, min(5.0, temperature))
        cfg_weight = max(0.0, min(1.0, cfg_weight))
        exaggeration = max(0.0, min(2.0, exaggeration))

        audio_prompt_path = self._resolve_audio_prompt_path(voice_path)

        segment_plan = self._split_markup_segments(text)
        if not segment_plan or all(not seg.strip() for seg, _ in segment_plan):
            raise ValueError("Text input cannot be empty")

        stitched: list[np.ndarray] = []
        inter_chunk_gap = np.zeros(int(model.sr * 0.1), dtype=np.float32)
        for seg_idx, (segment_text, pause_s) in enumerate(segment_plan):
            if segment_text:
                chunks = self._split_text_chunks(segment_text)
                for chunk_idx, chunk in enumerate(chunks):
                    max_new_tokens = self._auto_max_new_tokens(
                        chunk,
                        max_new_tokens_override,
                    )
                    wav = self._generate_chunk_with_cap(
                        model,
                        text=chunk,
                        language_id=language,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        cfg_weight=cfg_weight,
                        exaggeration=exaggeration,
                        audio_prompt_path=audio_prompt_path,
                    )
                    if seg_idx > 0 or chunk_idx > 0:
                        stitched.append(inter_chunk_gap)
                    stitched.append(self._to_mono_float32(wav))

            if pause_s > 0:
                stitched.append(np.zeros(int(model.sr * pause_s), dtype=np.float32))

        if not stitched:
            raise ValueError("Text input cannot be empty")

        audio = np.concatenate(stitched)

        audio = self._apply_speed(audio, sample_rate=model.sr, speed=speed)

        if output_path.endswith(".mp3"):
            if "MP3" in sf.available_formats():
                sf.write(output_path, audio, samplerate=model.sr, format="MP3")
                return output_path

            wav_path = output_path[:-4] + ".wav"
            sf.write(wav_path, audio, samplerate=model.sr, format="WAV")
            return wav_path

        sf.write(output_path, audio, samplerate=model.sr, format="WAV")
        return output_path

chatterbox_engine = RealChatterboxEngine()
app = create_app(
    title="Chatterbox TTS Translation Layer",
    engine=chatterbox_engine,
    voice_response_model=list[VoiceOption],
    openai_request_model=ChatterboxOpenAISpeechRequest,
    extended_request_model=ChatterboxVoxtralExtendedRequest,
    supports_ssml_emphasis=True,
    supports_ssml_breaks=True,
    backend_capabilities={
        "model": "chatterbox-tts",
        "voiceCloning": {
            "supported": True,
            "inputs": ["voice", "voice_reference_path"],
            "referenceAudio": {
                "required": False,
                "notes": "If omitted, default alias resolves to voices/nl_female.wav.",
            },
            "referenceText": {
                "supported": False,
                "notes": "This adapter does not use ref_text for cloning.",
            },
        },
        "ssmlProsody": {
            "tagParsing": True,
            "supportedTags": [
                "<break time=\"Xms\"/>",
                "<break time=\"Xs\"/>",
                "<emphasis>...</emphasis>",
            ],
            "notes": "Emphasis is rendered with expressive punctuation/casing in adapter logic.",
        },
        "languageConditioning": {
            "apiDefaultLanguage": None,
            "notes": "Language is forwarded to tokenizer when provided; otherwise model defaults apply.",
        },
        "generationBudgeting": {
            "maxNewTokens": {
                "clientConfigRequired": False,
                "auto": True,
                "defaultPolicy": "auto-per-chunk-by-input-length",
                "clampRange": [220, 650],
                "overrideField": "max_new_tokens",
            }
        },
    },
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_chatterbox:app", host="0.0.0.0", port=8001, reload=True)
