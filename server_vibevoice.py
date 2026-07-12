from __future__ import annotations

import os
import threading
from typing import Any, Optional, Protocol, cast

from pydantic import BaseModel

from api_shared import OpenAISpeechRequest, VoxtralExtendedRequest, create_app
from request_normalizer import build_stage_directions, normalize_request_text
from server_mlx_audio import coerce_audio_array, collect_generation_audio, write_audio_output


class _VibeVoiceModel(Protocol):
    def generate(self, **kwargs: Any): ...


class VibeVoiceSegment(BaseModel):
    text: str
    emotion: Optional[str] = None


class VibeVoiceOpenAISpeechRequest(OpenAISpeechRequest):
    model: str = "mlx-community/VibeVoice-ASR-bf16"
    voice: str = "narrator"
    emotion: Optional[str] = None
    segments: list[VibeVoiceSegment] | None = None
    use_stage_directions: bool = True
    voice_profile: Optional[str] = None
    speaker_names: list[str] | None = None


class VibeVoiceSpeechRequest(VoxtralExtendedRequest):
    voice_reference_path: Optional[str] = "narrator"
    emotion: Optional[str] = None
    segments: list[VibeVoiceSegment] | None = None
    use_stage_directions: bool = True
    voice_profile: Optional[str] = None
    speaker_names: list[str] | None = None


class RealVibeVoiceEngine:
    MODEL_ID = "mlx-community/VibeVoice-ASR-bf16"
    FALLBACK_MODEL_IDS = (
        "mlx-community/VibeVoice-bf16",
        "mlx-community/VibeVoice-TTS-bf16",
        "vibevoice-community/VibeVoice",
    )
    VARIANT_ORDER = ("calm", "normal", "excited", "shouting")
    EMOTION_MAP = {
        "neutral": "normal",
        "normal": "normal",
        "happy": "excited",
        "excited": "excited",
        "enthusiastic": "excited",
        "angry": "shouting",
        "shouting": "shouting",
        "sad": "calm",
        "serious": "calm",
        "calm": "calm",
    }
    PROFILE_LIBRARY: dict[str, dict[str, str]] = {
        "narrator": {
            "calm": "Narrator_Calm",
            "normal": "Narrator_Normal",
            "excited": "Narrator_Excited",
            "shouting": "Narrator_Shouting",
        },
        "erik": {
            "calm": "Erik_Calm",
            "normal": "Erik_Normal",
            "excited": "Erik_Excited",
            "shouting": "Erik_Shouting",
        },
        "sarah": {
            "calm": "Sarah_Calm",
            "normal": "Sarah_Normal",
            "excited": "Sarah_Excited",
            "shouting": "Sarah_Shouting",
        },
    }

    def __init__(self):
        self._model: Optional[_VibeVoiceModel] = None
        self._active_model_id: Optional[str] = None
        self._lock = threading.Lock()

    @classmethod
    def _candidate_model_ids(cls) -> list[str]:
        raw = os.getenv("VIBEVOICE_MODEL_FALLBACKS", "").strip()
        if raw:
            candidates = [part.strip() for part in raw.split(",") if part.strip()]
            return [cls.MODEL_ID, *candidates]
        return [cls.MODEL_ID, *cls.FALLBACK_MODEL_IDS]

    def _load_model(self) -> _VibeVoiceModel:
        with self._lock:
            if self._model is None:
                from mlx_audio.tts.utils import load

                load_errors: list[str] = []
                for candidate in self._candidate_model_ids():
                    print(f"📦 Loading VibeVoice MLX model ({candidate})...")
                    try:
                        self._model = cast(_VibeVoiceModel, load(candidate))
                        self._active_model_id = candidate
                        print(f"✅ VibeVoice model loaded ({candidate}).")
                        break
                    except Exception as exc:
                        load_errors.append(f"{candidate}: {exc}")

                if self._model is None:
                    joined = "\n".join(load_errors)
                    raise RuntimeError(
                        "Unable to load any VibeVoice model candidate. "
                        "Tried:\n"
                        f"{joined}"
                    )

            model = self._model
            if model is None:
                raise RuntimeError("VibeVoice model failed to initialize")
            return model

    @classmethod
    def _resolve_profile_variants(
        cls,
        *,
        voice_path: Optional[str],
        voice_profile: Optional[str],
        speaker_names: list[str] | None,
    ) -> list[str]:
        if speaker_names:
            return [str(name).strip() for name in speaker_names if str(name).strip()]

        profile_key = (voice_profile or voice_path or "narrator").strip().lower()
        if profile_key in cls.PROFILE_LIBRARY:
            profile = cls.PROFILE_LIBRARY[profile_key]
            return [profile[k] for k in cls.VARIANT_ORDER]

        base = (voice_path or voice_profile or "narrator").strip()
        if not base:
            base = "narrator"
        return [
            f"{base}_Calm",
            f"{base}_Normal",
            f"{base}_Excited",
            f"{base}_Shouting",
        ]

    @classmethod
    def _variant_for_emotion(cls, emotion: Optional[str]) -> str:
        if not emotion:
            return "normal"
        return cls.EMOTION_MAP.get(emotion.strip().lower(), "normal")

    @classmethod
    def _make_segments(
        cls,
        *,
        text: str,
        emotion: Optional[str],
        raw_segments: Any,
    ) -> list[VibeVoiceSegment]:
        if isinstance(raw_segments, list) and raw_segments:
            segments: list[VibeVoiceSegment] = []
            for item in raw_segments:
                if isinstance(item, VibeVoiceSegment):
                    segments.append(item)
                elif isinstance(item, dict):
                    segments.append(VibeVoiceSegment.model_validate(item))
            if segments:
                return segments

        clean = text.strip()
        if not clean:
            return []
        return [VibeVoiceSegment(text=clean, emotion=emotion)]

    @classmethod
    def _build_speaker_routed_prompt(
        cls,
        *,
        text: str,
        emotion: Optional[str],
        segments: Any,
        variant_speakers: list[str],
        use_stage_directions: bool,
    ) -> tuple[str, list[str]]:
        planned = cls._make_segments(text=text, emotion=emotion, raw_segments=segments)
        if not planned:
            raise ValueError("Text input cannot be empty")

        variant_to_index = {name: idx for idx, name in enumerate(cls.VARIANT_ORDER)}

        lines: list[str] = []
        for seg in planned:
            variant = cls._variant_for_emotion(seg.emotion)
            speaker_idx = variant_to_index.get(variant, 1)
            body = normalize_request_text(seg.text)
            if not body:
                continue
            if use_stage_directions and seg.emotion:
                body = normalize_request_text(
                    body,
                    stage_directions=build_stage_directions(seg.emotion),
                )
            lines.append(f"Speaker {speaker_idx}: {body}")

        if not lines:
            raise ValueError("Text input cannot be empty")

        return "\n".join(lines), variant_speakers

    @staticmethod
    def _extract_audio(results: Any, model: _VibeVoiceModel) -> tuple[Any, int]:
        default_sample_rate = int(getattr(model, "sample_rate", 24000))

        try:
            return collect_generation_audio(results, default_sample_rate=default_sample_rate)
        except Exception:
            pass

        if hasattr(results, "audio"):
            audio = coerce_audio_array(getattr(results, "audio"))
            sample_rate = int(
                getattr(results, "sample_rate", getattr(results, "sr", default_sample_rate))
            )
            return audio, sample_rate

        if isinstance(results, tuple) and len(results) == 2:
            return coerce_audio_array(results[0]), int(results[1])

        raise RuntimeError("VibeVoice returned an unsupported audio output format")

    @staticmethod
    def _generate_with_fallback(model: _VibeVoiceModel, call_kwargs: dict[str, Any]) -> Any:
        attempts = [
            call_kwargs,
            {
                **call_kwargs,
                "text": call_kwargs.get("transcript", call_kwargs.get("text")),
            },
            {
                **call_kwargs,
                "input_text": call_kwargs.get("transcript", call_kwargs.get("text")),
            },
        ]

        last_error: Optional[Exception] = None
        for kwargs in attempts:
            try:
                return model.generate(**kwargs)
            except TypeError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise RuntimeError(f"VibeVoice generate signature mismatch: {last_error}")
        raise RuntimeError("VibeVoice generation failed")

    def synthesize(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        **kwargs: Any,
    ) -> str:
        model = self._load_model()

        variant_speakers = self._resolve_profile_variants(
            voice_path=voice_path,
            voice_profile=kwargs.get("voice_profile"),
            speaker_names=kwargs.get("speaker_names"),
        )
        routed_prompt, speakers = self._build_speaker_routed_prompt(
            text=text,
            emotion=kwargs.get("emotion"),
            segments=kwargs.get("segments"),
            variant_speakers=variant_speakers,
            use_stage_directions=bool(kwargs.get("use_stage_directions", True)),
        )

        call_kwargs: dict[str, Any] = {
            "text": routed_prompt,
            "transcript": routed_prompt,
            "speaker_names": speakers,
        }

        language = kwargs.get("language")
        if language is not None:
            call_kwargs["language"] = str(language)

        speed = kwargs.get("speed")
        if speed is not None:
            call_kwargs["speed"] = float(speed)

        results = self._generate_with_fallback(model, call_kwargs)
        audio, sample_rate = self._extract_audio(results, model)
        return write_audio_output(output_path, audio, sample_rate=sample_rate)

    @classmethod
    def list_voices(cls) -> dict[str, Any]:
        return {
            "profiles": sorted(cls.PROFILE_LIBRARY.keys()),
            "variants": list(cls.VARIANT_ORDER),
            "emotionMap": dict(cls.EMOTION_MAP),
        }


vibevoice_engine = RealVibeVoiceEngine()
app = create_app(
    title="VibeVoice TTS Translation Layer",
    engine=vibevoice_engine,
    voice_response_model=dict[str, Any],
    route_prefix="vibevoice",
    openai_request_model=VibeVoiceOpenAISpeechRequest,
    extended_request_model=VibeVoiceSpeechRequest,
    supports_ssml_emphasis=False,
    supports_ssml_breaks=False,
    backend_capabilities={
        "model": RealVibeVoiceEngine.MODEL_ID,
        "modelFallbacks": list(RealVibeVoiceEngine.FALLBACK_MODEL_IDS),
        "voiceCloning": {
            "supported": True,
            "notes": (
                "Emotion control is routed through speaker variants. "
                "Use voice profiles and speaker routing instead of SSML emotion tags."
            ),
        },
        "speakerRouting": {
            "supported": True,
            "maxSpeakers": 4,
            "routingFormat": "Speaker N: ...",
            "variantOrder": list(RealVibeVoiceEngine.VARIANT_ORDER),
            "emotionToVariant": dict(RealVibeVoiceEngine.EMOTION_MAP),
            "secondaryStageDirections": True,
            "examples": [
                "Speaker 0: [calm] Welkom iedereen.",
                "Speaker 2: [excited] Dit is fantastisch nieuws!",
            ],
        },
        "ssmlProsody": {
            "tagParsing": False,
            "supportedTags": [],
            "notes": "Do not rely on SSML emotion tags; use speaker routing and segments.",
        },
        "languageConditioning": {
            "apiDefaultLanguage": None,
            "notes": "Language is optional and forwarded when provided.",
        },
    },
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_vibevoice:app", host="0.0.0.0", port=8000, reload=True)
