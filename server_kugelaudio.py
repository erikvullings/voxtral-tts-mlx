from __future__ import annotations

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
    max_new_tokens: int = 2048
    do_sample: bool = False
    temperature: float = 1.0

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
    max_new_tokens: int = 2048
    do_sample: bool = False
    temperature: float = 1.0

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

    def __init__(self):
        self._model: Optional[_KugelAudioModel] = None
        self._lock = threading.Lock()

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
    def list_voices() -> list[str]:
        return list(RealKugelAudioEngine.PRESET_VOICES)

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

        results = model.generate(
            text=text,
            voice=voice,
            cfg_scale=float(kwargs.get("cfg_scale", 3.0)),
            max_new_tokens=int(kwargs.get("max_new_tokens", 2048)),
            do_sample=bool(kwargs.get("do_sample", False)),
            temperature=float(kwargs.get("temperature", 1.0)),
        )

        audio, sample_rate = collect_generation_audio(
            results, default_sample_rate=getattr(model, "sample_rate", 24000)
        )
        return write_audio_output(output_path, audio, sample_rate=sample_rate)


kugelaudio_engine = RealKugelAudioEngine()
app = create_app(
    title="KugelAudio TTS Translation Layer",
    engine=kugelaudio_engine,
    voice_response_model=list[str],
    route_prefix="kugelaudio",
    openai_request_model=KugelAudioOpenAISpeechRequest,
    extended_request_model=KugelAudioSpeechRequest,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_kugelaudio:app", host="0.0.0.0", port=8000, reload=True)
