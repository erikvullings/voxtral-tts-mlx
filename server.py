# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "pydantic",
#     "requests",
#     "soundfile",
# ]
# ///

import os
import threading
from typing import Any, Iterable, Optional, Protocol, cast

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Voxtral TTS Translation Layer", version="1.0.0")

# --- Fixed Pydantic Schemas with Language Parameter & UI Examples ---


class OpenAISpeechRequest(BaseModel):
    model: str = "voxtral"
    input: str
    voice: str = "nl_female"
    language: str = "nl"  # Added language parameter
    response_format: str = "mp3"
    speed: float = 1.0

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "voxtral",
                "input": "Welkom bij de Nederlandse les. Vandaag gaan we grammatica oefenen.",
                "voice": "nl_female",
                "language": "nl",  # Prefilled as Dutch in Swagger UI
                "response_format": "mp3",
                "speed": 1.0,
            }
        }
    }


class VoxtralExtendedRequest(BaseModel):
    text: str
    voice_reference_path: Optional[str] = "nl_female"
    language: str = "nl"  # Added language parameter
    emotion: str = "neutral"
    nfe_steps: int = 16
    temperature: float = 0.7
    output_filename: str = "output.mp3"

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Dit is een voorbeeldzin in het Nederlands met geavanceerde parameters.",
                "voice_reference_path": "nl_female",
                "language": "nl",  # Prefilled as Dutch in Swagger UI
                "emotion": "neutral",
                "nfe_steps": 16,
                "temperature": 0.7,
                "output_filename": "output.mp3",
            }
        }
    }


class _TTSModel(Protocol):
    def generate(self, *, text: str, voice: str) -> Iterable[Any]: ...


class RealVoxtralEngine:
    MLX_MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
    PRESET_VOICES = {
        "casual_male",
        "casual_female",
        "cheerful_female",
        "neutral_male",
        "neutral_female",
        "fr_male",
        "fr_female",
        "es_male",
        "es_female",
        "de_male",
        "de_female",
        "it_male",
        "it_female",
        "pt_male",
        "pt_female",
        "nl_male",
        "nl_female",
        "ar_male",
        "hi_male",
        "hi_female",
    }

    def __init__(self):
        self._model: Optional[_TTSModel] = None
        self._lock = threading.Lock()

    def _load_model(self):
        with self._lock:
            if self._model is None:
                print(f"📦 Loading Voxtral MLX model ({self.MLX_MODEL_ID})...")
                from mlx_audio.tts.utils import load

                self._model = cast(_TTSModel, load(self.MLX_MODEL_ID))
                print("✅ Model loaded.")

    def _resolve_voice(self, voice: Optional[str]) -> str:
        """Map a preset name or legacy file path to a valid preset voice name."""
        if not voice:
            return "nl_female"
        if "/" in voice or "\\" in voice or voice.endswith(".wav"):
            return "nl_female"
        return voice if voice in self.PRESET_VOICES else "nl_female"

    def synthesize(
        self, text: str, voice_path: Optional[str], output_path: str, **kwargs
    ) -> str:
        self._load_model()
        model = self._model
        if model is None:
            raise RuntimeError("Voxtral model failed to initialize")
        voice = self._resolve_voice(voice_path)

        audio_chunks = []
        for result in model.generate(text=text, voice=voice):
            audio_chunks.append(np.array(result.audio))

        audio = (
            np.concatenate(audio_chunks)
            if audio_chunks
            else np.zeros(0, dtype=np.float32)
        )

        if output_path.endswith(".mp3"):
            # Python 3.14 removed audioop; avoid pydub and write MP3 via soundfile if available.
            if "MP3" in sf.available_formats():
                sf.write(output_path, audio, samplerate=24000, format="MP3")
                return output_path

            wav_path = output_path[:-4] + ".wav"
            sf.write(wav_path, audio, samplerate=24000, format="WAV")
            return wav_path

        sf.write(output_path, audio, samplerate=24000, format="WAV")
        return output_path


voxtral_engine = RealVoxtralEngine()


def get_engine():
    return voxtral_engine


# --- API Routes ---


@app.post("/v1/audio/speech")
async def openai_compatible_speech(
    request: OpenAISpeechRequest, engine=Depends(get_engine)
):
    requested_format = request.response_format.lower()
    output_ext = "mp3" if requested_format == "mp3" else "wav"
    output_path = f"generated_lessons/lesson_{hash(request.input)}.{output_ext}"
    os.makedirs("generated_lessons", exist_ok=True)
    voice_ref = request.voice

    try:
        actual_output_path = engine.synthesize(
            text=request.input,
            voice_path=voice_ref,
            output_path=output_path,
            speed=request.speed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response_name = os.path.basename(actual_output_path)
    media_type = "audio/mpeg" if response_name.endswith(".mp3") else "audio/wav"
    return FileResponse(
        actual_output_path, media_type=media_type, filename=response_name
    )


@app.post("/v1/voxtral/speech")
async def voxtral_dedicated_speech(
    request: VoxtralExtendedRequest, engine=Depends(get_engine)
):
    output_path = f"generated_lessons/{request.output_filename}"
    os.makedirs("generated_lessons", exist_ok=True)
    voice_ref = request.voice_reference_path

    try:
        actual_output_path = engine.synthesize(
            text=request.text,
            voice_path=voice_ref,
            output_path=output_path,
            emotion=request.emotion,
            nfe_steps=request.nfe_steps,
            temperature=request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response_name = os.path.basename(actual_output_path)
    media_type = "audio/mpeg" if response_name.endswith(".mp3") else "audio/wav"
    return FileResponse(
        actual_output_path, media_type=media_type, filename=response_name
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
