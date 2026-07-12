from __future__ import annotations

import threading
from typing import Any, Optional, Protocol, cast

from pydantic import BaseModel

from api_shared import OpenAISpeechRequest, VoxtralExtendedRequest, create_app
from request_normalizer import build_higgs_tokens, build_stage_directions, normalize_request_text
from server_mlx_audio import (
    collect_generation_audio,
    resolve_reference_audio_path,
    write_audio_output,
)


class _HiggsAudioModel(Protocol):
    def generate(self, **kwargs: Any): ...


class HiggsReference(BaseModel):
    audio_path: str
    text: Optional[str] = None


class HiggsOpenAISpeechRequest(OpenAISpeechRequest):
    model: str = "bosonai/higgs-audio-v3-tts-4b"
    voice: str = "default"
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    references: list[HiggsReference] | None = None
    max_new_tokens: int | None = None
    temperature: float = 1.0
    top_p: float | None = 0.95
    top_k: int | None = 50
    seed: int | None = None
    fade_in_ms: float = 30.0
    fade_out_ms: float = 15.0
    emotion: str | list[str] | None = None
    style: str | list[str] | None = None
    sfx: str | list[str] | None = None
    prosody: str | list[str] | None = None
    stage_directions: str | list[str] | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "bosonai/higgs-audio-v3-tts-4b",
                "input": "Hello, this model supports zero-shot voice cloning.",
                "voice": "default",
                "language": "en",
                "response_format": "mp3",
                "ref_audio": "voices/sample.wav",
                "ref_text": "This is the reference transcript.",
                "max_new_tokens": 2048,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 50,
            }
        }
    }


class HiggsSpeechRequest(VoxtralExtendedRequest):
    voice_reference_path: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    references: list[HiggsReference] | None = None
    max_new_tokens: int | None = None
    temperature: float = 1.0
    top_p: float | None = 0.95
    top_k: int | None = 50
    seed: int | None = None
    fade_in_ms: float = 30.0
    fade_out_ms: float = 15.0
    emotion: str | list[str] | None = None
    style: str | list[str] | None = None
    sfx: str | list[str] | None = None
    prosody: str | list[str] | None = None
    stage_directions: str | list[str] | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "This is a Higgs Audio reference-cloning sample.",
                "voice_reference_path": "voices/sample.wav",
                "ref_text": "This is the reference transcript.",
                "language": "en",
                "max_new_tokens": 2048,
                "output_filename": "higgs_lesson.mp3",
            }
        }
    }


class RealHiggsAudioEngine:
    MODEL_ID = "bosonai/higgs-audio-v3-tts-4b"

    def __init__(self):
        self._model: Optional[_HiggsAudioModel] = None
        self._lock = threading.Lock()

    def _load_model(self) -> _HiggsAudioModel:
        with self._lock:
            if self._model is None:
                print(f"📦 Loading Higgs Audio MLX model ({self.MODEL_ID})...")
                from mlx_audio.tts.utils import load

                self._model = cast(_HiggsAudioModel, load(self.MODEL_ID))
                print("✅ Higgs Audio model loaded.")

            model = self._model
            if model is None:
                raise RuntimeError("Higgs Audio model failed to initialize")
            return model

    @staticmethod
    def list_voices() -> list[str]:
        return []

    @staticmethod
    def _normalize_references(references: Any) -> list[dict[str, Any]]:
        if not references:
            return []

        normalized: list[dict[str, Any]] = []
        for item in references:
            if not isinstance(item, dict):
                continue
            audio = item.get("audio_path") or item.get("audio") or item.get("path")
            if audio is None:
                continue
            if isinstance(audio, str):
                resolved = resolve_reference_audio_path(audio)
                audio = resolved or audio
            normalized.append(
                {
                    "audio_path": audio,
                    "text": item.get("text") or item.get("ref_text"),
                }
            )
        return normalized

    @staticmethod
    def _auto_max_new_tokens(text: str, requested: Any, references: list[dict[str, Any]]) -> int:
        if requested is not None:
            try:
                value = int(requested)
                return max(512, min(8192, value))
            except (TypeError, ValueError):
                pass

        words = len([w for w in text.replace("\n", " ").split(" ") if w.strip()])
        estimated = int(words * 6.0) + 1200
        if references:
            estimated += 300
        return max(1400, min(8192, estimated))

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
            prefix_tokens=build_higgs_tokens(
                emotion=kwargs.get("emotion"),
                style=kwargs.get("style"),
                sfx=kwargs.get("sfx"),
                prosody=kwargs.get("prosody"),
            ),
            stage_directions=build_stage_directions(kwargs.get("stage_directions")),
        )

        references = self._normalize_references(kwargs.get("references"))
        ref_audio = kwargs.get("ref_audio")
        if isinstance(ref_audio, str):
            ref_audio = resolve_reference_audio_path(ref_audio) or ref_audio
        ref_text = kwargs.get("ref_text")

        resolved_voice = resolve_reference_audio_path(voice_path)
        if not references and ref_audio is None:
            if resolved_voice is not None:
                ref_audio = resolved_voice
            elif voice_path and voice_path.strip().lower() not in {"default", ""}:
                raise ValueError(
                    "Higgs Audio voice cloning requires a local reference WAV file or references[]."
                )

        max_new_tokens = self._auto_max_new_tokens(
            prepared_text,
            kwargs.get("max_new_tokens"),
            references,
        )

        results = model.generate(
            text=prepared_text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            references=references or None,
            max_new_tokens=max_new_tokens,
            max_new_frames=kwargs.get("max_new_frames"),
            max_tokens=kwargs.get("max_tokens"),
            temperature=float(kwargs.get("temperature", 1.0)),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 50),
            seed=kwargs.get("seed"),
            fade_in_ms=float(kwargs.get("fade_in_ms", 30.0)),
            fade_out_ms=float(kwargs.get("fade_out_ms", 15.0)),
        )

        audio, sample_rate = collect_generation_audio(
            results, default_sample_rate=getattr(model, "sample_rate", 24000)
        )
        return write_audio_output(output_path, audio, sample_rate=sample_rate)


higgs_engine = RealHiggsAudioEngine()
app = create_app(
    title="Higgs Audio TTS Translation Layer",
    engine=higgs_engine,
    voice_response_model=list[str],
    route_prefix="higgs",
    openai_request_model=HiggsOpenAISpeechRequest,
    extended_request_model=HiggsSpeechRequest,
    backend_capabilities={
        "model": RealHiggsAudioEngine.MODEL_ID,
        "voiceCloning": {
            "supported": True,
            "inputs": ["voice_reference_path", "ref_audio", "references"],
            "referenceAudio": {
                "required": True,
                "notes": "Provide ref_audio or references[] for zero-shot cloning.",
            },
            "referenceText": {
                "supported": True,
                "required": False,
                "notes": "ref_text improves identity/prosody stability.",
            },
        },
        "ssmlProsody": {
            "tagParsing": False,
            "supportedTags": [],
            "notes": "SSML XML tags are not parsed by this adapter.",
        },
        "tokenControls": {
            "format": "<|category:value|>",
            "emotion": [
                "elation",
                "amusement",
                "enthusiasm",
                "determination",
                "pride",
                "contentment",
                "affection",
                "relief",
                "contemplation",
                "confusion",
                "surprise",
                "awe",
                "longing",
                "arousal",
                "anger",
                "fear",
                "disgust",
                "bitterness",
                "sadness",
                "shame",
                "helplessness",
            ],
            "style": ["singing", "shouting", "whispering"],
            "sfx": [
                "cough",
                "laughter",
                "crying",
                "screaming",
                "burping",
                "humming",
                "sigh",
                "sniff",
                "sneeze",
            ],
            "prosody": [
                "speed_very_slow",
                "speed_slow",
                "speed_fast",
                "speed_very_fast",
                "pause",
                "long_pause",
                "pitch_low",
                "pitch_high",
                "expressive_high",
                "expressive_low",
            ],
            "examples": [
                "<|emotion:elation|>",
                "<|style:whispering|>",
                "<|sfx:laughter|> haha",
                "<|prosody:pause|>",
            ],
        },
        "languageConditioning": {
            "apiDefaultLanguage": None,
            "notes": "Language is optional in this adapter and not required for synthesis.",
        },
        "generationBudgeting": {
            "maxNewTokens": {
                "clientConfigRequired": False,
                "auto": True,
                "defaultPolicy": "auto-by-input-length-with-reference-bias",
                "clampRange": [1400, 8192],
                "overrideField": "max_new_tokens",
            }
        },
    },
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_higgs:app", host="0.0.0.0", port=8000, reload=True)
