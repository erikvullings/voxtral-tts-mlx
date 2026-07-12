from __future__ import annotations

import os
import re
import threading
from typing import Any, Optional, Protocol, cast

from api_shared import OpenAISpeechRequest, VoxtralExtendedRequest, create_app
from server_mlx_audio import collect_generation_audio, write_audio_output


class _KugelAudioModel(Protocol):
    def generate(self, *, text: str, voice: str, **kwargs: Any): ...


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
    def _resolve_voice_file(name: str) -> Optional[str]:
        candidate = name.strip()
        if not candidate:
            return None
        if os.path.isfile(candidate):
            return candidate

        for option in (
            os.path.join("voices", candidate),
            os.path.join("voices", f"{candidate}.wav"),
        ):
            if os.path.isfile(option):
                return option
        return None

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

    def _load_model(self) -> _KugelAudioModel:
        with self._lock:
            if self._model is None:
                print(f"📦 Loading KugelAudio MLX model ({self.MODEL_ID})...")
                from mlx_audio.tts.utils import load

                self._model = cast(_KugelAudioModel, load(self.MODEL_ID))
                print("✅ KugelAudio model loaded.")

            model = self._model
            if model is None:
                raise RuntimeError("KugelAudio model failed to initialize")
            return model

    @staticmethod
    def list_voices() -> dict[str, list[str]]:
        voices_dir = "voices"
        folder_voices: list[str] = []
        if os.path.isdir(voices_dir):
            folder_voices = sorted(
                {
                    os.path.splitext(name)[0]
                    for name in os.listdir(voices_dir)
                    if name.lower().endswith(".wav")
                }
            )
        return {
            "presets": list(RealKugelAudioEngine.PRESET_VOICES),
            "voicesFolder": folder_voices,
        }

    def synthesize(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        **kwargs: Any,
    ) -> str:
        model = self._load_model()

        if voice_path and voice_path.strip().lower() not in self.PRESET_VOICES:
            raise ValueError(
                "KugelAudio supports preset voices only: default, warm, clear."
            )

        voice = (voice_path or kwargs.get("voice") or "default").strip().lower()
        if voice not in self.PRESET_VOICES:
            raise ValueError(
                "KugelAudio supports preset voices only: default, warm, clear."
            )

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

        max_new_tokens = self._auto_max_new_tokens(text, kwargs.get("max_new_tokens"))

        call_kwargs = {
            "text": normalized_text,
            "voice": voice,
            "language": language,
            "cfg_scale": float(kwargs.get("cfg_scale", 3.0)),
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(kwargs.get("do_sample", False)),
            "temperature": float(kwargs.get("temperature", 1.0)),
        }

        results = self._generate_with_fallback(model, call_kwargs)

        audio, sample_rate = collect_generation_audio(
            results, default_sample_rate=getattr(model, "sample_rate", 24000)
        )
        return write_audio_output(output_path, audio, sample_rate=sample_rate)


kugelaudio_engine = RealKugelAudioEngine()
app = create_app(
    title="KugelAudio TTS Translation Layer",
    engine=kugelaudio_engine,
    voice_response_model=dict[str, list[str]],
    route_prefix="kugelaudio",
    openai_request_model=KugelAudioOpenAISpeechRequest,
    extended_request_model=KugelAudioSpeechRequest,
    backend_capabilities={
        "model": RealKugelAudioEngine.MODEL_ID,
        "voiceCloning": {
            "supported": False,
            "notes": "Preset voices only in this adapter.",
        },
        "voiceSelection": {
            "type": "preset",
            "presets": list(RealKugelAudioEngine.PRESET_VOICES),
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
    },
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_kugelaudio:app", host="0.0.0.0", port=8000, reload=True)
