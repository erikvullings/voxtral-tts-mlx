from __future__ import annotations

import threading
from typing import Any, Optional, Protocol, cast

from api_shared import OpenAISpeechRequest, VoxtralExtendedRequest, create_app
from mlx_utils import warmup_mlx_model
from request_normalizer import build_stage_directions, normalize_request_text
from server_mlx_audio import (
    collect_generation_audio,
    resolve_reference_audio_path,
    write_audio_output,
)


class _MossTTSModel(Protocol):
    def generate(self, **kwargs: Any): ...


class MossOpenAISpeechRequest(OpenAISpeechRequest):
    model: str = "OpenMOSS-Team/MOSS-TTS-v1.5"
    voice: str = "default"
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    prompt_audio_codes: Any = None
    mode: str = "generation"
    max_tokens: int | None = None
    tokens: int | None = None
    instruction: str | None = None
    quality: str | None = None
    sound_event: str | None = None
    ambient_sound: str | None = None
    language: str | None = None
    scene: str | None = None
    stage_directions: str | list[str] | None = None
    paragraph_pause_seconds: float | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "OpenMOSS-Team/MOSS-TTS-v1.5",
                "input": "Hello, this model supports voice cloning and continuation.",
                "voice": "default",
                "language": "en",
                "response_format": "mp3",
                "ref_audio": "voices/sample.wav",
                "ref_text": "This is the reference transcript.",
                "max_tokens": 4096,
            }
        }
    }


class MossSpeechRequest(VoxtralExtendedRequest):
    voice_reference_path: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    prompt_audio_codes: Any = None
    mode: str = "generation"
    max_tokens: int | None = None
    tokens: int | None = None
    instruction: str | None = None
    quality: str | None = None
    sound_event: str | None = None
    ambient_sound: str | None = None
    language: str | None = None
    scene: str | None = None
    stage_directions: str | list[str] | None = None
    paragraph_pause_seconds: float | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "MOSS-TTS supports zero-shot voice cloning and continuation.",
                "voice_reference_path": "voices/sample.wav",
                "ref_text": "This is the reference transcript.",
                "language": "English",
                "mode": "generation",
                "max_tokens": 4096,
                "output_filename": "moss_lesson.mp3",
            }
        }
    }


class RealMossTTSEngine:
    MODEL_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"

    def __init__(self):
        self._model: Optional[_MossTTSModel] = None
        self._lock = threading.Lock()

    @staticmethod
    def _warmup_model(model: _MossTTSModel) -> None:
        """Run a minimal generation to trigger MLX Metal shader compilation."""
        def generate_warmup():
            return model.generate(
                text="Hi.",
                language="en",
                max_tokens=10,
            )

        warmup_mlx_model(model, generate_warmup)

    def _load_model(self) -> _MossTTSModel:
        with self._lock:
            if self._model is None:
                print(f"📦 Loading MOSS-TTS MLX model ({self.MODEL_ID})...")
                from mlx_audio.tts.utils import load

                self._model = cast(_MossTTSModel, load(self.MODEL_ID))
                print("✅ MOSS-TTS model loaded.")
                self._warmup_model(self._model)
            model = self._model
            if model is None:
                raise RuntimeError("MOSS-TTS model failed to initialize")
            return model

    @staticmethod
    def list_voices() -> list[str]:
        return []

    @staticmethod
    def _normalize_reference_audio(value: Any) -> Any:
        if isinstance(value, str):
            resolved = resolve_reference_audio_path(value)
            return resolved or value
        return value

    @staticmethod
    def _auto_max_tokens(text: str, requested: Any) -> int:
        if requested is not None:
            try:
                value = int(requested)
                return max(1024, min(8192, value))
            except (TypeError, ValueError):
                pass

        words = len([w for w in text.replace("\n", " ").split(" ") if w.strip()])
        estimated = int(words * 7.0) + 1800
        return max(2200, min(8192, estimated))

    def synthesize(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        **kwargs: Any,
    ) -> str:
        model = self._load_model()

        prepared_text = normalize_request_text(
            text,
            stage_directions=build_stage_directions(kwargs.get("stage_directions")),
            paragraph_pause_seconds=kwargs.get("paragraph_pause_seconds"),
            paragraph_pause_format="[pause {seconds:.1f}s]",
        )
        max_tokens = self._auto_max_tokens(prepared_text, kwargs.get("max_tokens"))

        ref_audio = kwargs.get("ref_audio")
        if ref_audio is not None:
            ref_audio = self._normalize_reference_audio(ref_audio)
        ref_text = kwargs.get("ref_text")
        prompt_audio_codes = kwargs.get("prompt_audio_codes")

        resolved_voice = resolve_reference_audio_path(voice_path)
        if ref_audio is None and prompt_audio_codes is None:
            if resolved_voice is not None:
                ref_audio = resolved_voice
            elif voice_path and voice_path.strip().lower() not in {"default", ""}:
                raise ValueError(
                    "MOSS-TTS voice cloning requires a local reference WAV file."
                )

        results = model.generate(
            text=prepared_text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            prompt_audio_codes=prompt_audio_codes,
            mode=str(kwargs.get("mode", "generation")),
            stream=False,
            max_tokens=max_tokens,
            tokens=kwargs.get("tokens"),
            instruction=kwargs.get("instruction"),
            quality=kwargs.get("quality"),
            sound_event=kwargs.get("sound_event"),
            ambient_sound=kwargs.get("ambient_sound"),
            language=kwargs.get("language"),
            scene=kwargs.get("scene"),
        )

        audio, sample_rate = collect_generation_audio(
            results, default_sample_rate=getattr(model, "sample_rate", 24000)
        )
        return write_audio_output(output_path, audio, sample_rate=sample_rate)


moss_engine = RealMossTTSEngine()
app = create_app(
    title="MOSS TTS Translation Layer",
    engine=moss_engine,
    voice_response_model=list[str],
    route_prefix="moss",
    openai_request_model=MossOpenAISpeechRequest,
    extended_request_model=MossSpeechRequest,
    backend_capabilities={
        "model": RealMossTTSEngine.MODEL_ID,
        "voiceCloning": {
            "supported": True,
            "inputs": ["voice_reference_path", "ref_audio", "prompt_audio_codes"],
            "referenceAudio": {
                "required": False,
                "notes": "Provide ref_audio or prompt_audio_codes for cloning/continuation workflows.",
            },
            "referenceText": {
                "supported": True,
                "required": False,
                "notes": "ref_text can improve cloning quality.",
            },
        },
        "ssmlProsody": {
            "tagParsing": False,
            "supportedTags": [],
            "notes": "No SSML XML tag parser is implemented in this adapter.",
        },
        "tokenControls": {
            "pauseToken": {
                "format": "[pause X.Ys]",
                "example": "Hallo [pause 0.8s] hoe gaat het?",
                "notes": "Explicit pause control is supported by model prompting.",
            }
        },
        "languageConditioning": {
            "apiDefaultLanguage": None,
            "notes": "Language tags are optional but recommended for stronger multilingual synthesis.",
        },
        "generationBudgeting": {
            "maxTokens": {
                "clientConfigRequired": False,
                "auto": True,
                "defaultPolicy": "auto-by-input-length",
                "clampRange": [2200, 8192],
                "overrideField": "max_tokens",
            }
        },
    },
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_moss:app", host="0.0.0.0", port=8000, reload=True)
