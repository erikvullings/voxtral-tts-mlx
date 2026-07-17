from __future__ import annotations

import threading
from typing import Any, Optional, Protocol, cast

from api_shared import OpenAISpeechRequest, VoxtralExtendedRequest, create_app
from server_mlx_audio import (
    collect_generation_audio,
    resolve_reference_audio_path,
    write_audio_output,
)


class _OmniVoiceModel(Protocol):
    def generate(self, **kwargs: Any): ...


class OmniVoiceOpenAISpeechRequest(OpenAISpeechRequest):
    model: str = "mlx-community/OmniVoice-bf16"
    voice: str = "default"
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "mlx-community/OmniVoice-bf16",
                "input": "Welcome to the lesson. This voice can be cloned from a local reference clip.",
                "voice": "default",
                "language": "en",
                "response_format": "mp3",
                "ref_audio": "voices/sample.wav",
                "ref_text": "Hello, I am reading the sample reference text.",
                "speed": 1.0,
            }
        }
    }


class OmniVoiceSpeechRequest(VoxtralExtendedRequest):
    voice_reference_path: Optional[str] = None
    ref_text: Optional[str] = None
    duration_s: Optional[float] = None
    instruct: str = "None"
    ref_audio_max_duration_s: float = 10.0
    num_steps: int = 32
    guidance_scale: float = 2.0
    class_temperature: float = 0.0
    position_temperature: float = 5.0
    layer_penalty_factor: float = 5.0
    t_shift: float = 0.1

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Welkom bij de les. Deze stem kan worden gekloond met een lokale referentie-opname.",
                "voice_reference_path": "voices/sample.wav",
                "ref_text": "Hallo, ik ben de referentiestem.",
                "language": "nl",
                "duration_s": 5.0,
                "output_filename": "omnivoice_lesson.mp3",
            }
        }
    }


class RealOmniVoiceEngine:
    MODEL_ID = "mlx-community/OmniVoice-bf16"

    def __init__(self):
        self._model: Optional[_OmniVoiceModel] = None
        self._lock = threading.Lock()

    def _load_model(self) -> _OmniVoiceModel:
        with self._lock:
            if self._model is None:
                print(f"📦 Loading OmniVoice MLX model ({self.MODEL_ID})...")
                from mlx_audio.tts.utils import load

                self._model = cast(_OmniVoiceModel, load(self.MODEL_ID))
                print("✅ OmniVoice model loaded.")

            model = self._model
            if model is None:
                raise RuntimeError("OmniVoice model failed to initialize")
            return model

    @staticmethod
    def list_voices() -> list[str]:
        return []

    def synthesize(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        **kwargs: Any,
    ) -> str:
        model = self._load_model()

        ref_audio = kwargs.get("ref_audio")
        if ref_audio is None:
            ref_audio = resolve_reference_audio_path(voice_path)
        elif isinstance(ref_audio, str):
            ref_audio = resolve_reference_audio_path(ref_audio) or ref_audio

        if voice_path and ref_audio is None:
            raise ValueError(
                "OmniVoice voice cloning requires a local reference WAV file."
            )

        language = str(kwargs.get("language") or "None")
        ref_text = kwargs.get("ref_text")
        duration_s = kwargs.get("duration_s")
        instruct = str(kwargs.get("instruct") or "None")

        results = model.generate(
            text=text,
            duration_s=duration_s,
            language=language,
            lang_code=language,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            ref_audio_max_duration_s=float(kwargs.get("ref_audio_max_duration_s", 10.0)),
            num_steps=int(kwargs.get("num_steps", 32)),
            guidance_scale=float(kwargs.get("guidance_scale", 2.0)),
            class_temperature=float(kwargs.get("class_temperature", 0.0)),
            position_temperature=float(kwargs.get("position_temperature", 5.0)),
            layer_penalty_factor=float(kwargs.get("layer_penalty_factor", 5.0)),
            t_shift=float(kwargs.get("t_shift", 0.1)),
            tokenizer=kwargs.get("tokenizer"),
            text_tokenizer=kwargs.get("text_tokenizer"),
        )

        audio, sample_rate = collect_generation_audio(
            results, default_sample_rate=getattr(model, "sample_rate", 24000)
        )
        return write_audio_output(output_path, audio, sample_rate=sample_rate)


omnivoice_engine = RealOmniVoiceEngine()
app = create_app(
    title="OmniVoice TTS Translation Layer",
    engine=omnivoice_engine,
    voice_response_model=list[str],
    route_prefix="omnivoice",
    openai_request_model=OmniVoiceOpenAISpeechRequest,
    extended_request_model=OmniVoiceSpeechRequest,
    backend_capabilities={
        "model": RealOmniVoiceEngine.MODEL_ID,
        "voiceCloning": {
            "supported": True,
            "inputs": ["voice_reference_path", "ref_audio"],
            "referenceAudio": {
                "required": True,
                "notes": "Local WAV reference is required for cloning.",
            },
            "referenceText": {
                "supported": True,
                "required": False,
                "notes": "ref_text is optional but recommended for better transfer.",
            },
        },
        "ssmlProsody": {
            "tagParsing": False,
            "supportedTags": [],
            "notes": "No SSML tag parser is implemented in this adapter.",
        },
        "languageConditioning": {
            "apiDefaultLanguage": None,
            "notes": "Language/lang_code are forwarded when provided; model defaults apply otherwise.",
        },
    },
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_omnivoice:app", host="0.0.0.0", port=8000, reload=True)
