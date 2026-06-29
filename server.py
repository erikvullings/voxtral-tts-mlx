# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "pydantic",
#     "requests",
#     "soundfile",
#     "pyrubberband",
# ]
# ///

import os
import re
import threading
from typing import Any, Iterable, Optional, Protocol, cast

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

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
    speed: float = 1.0
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
                "speed": 1.0,
                "output_filename": "output.mp3",
            }
        }
    }


class VoxtralTranscriptRequest(BaseModel):
    audio_path: str
    text: str
    language: Optional[str] = "nl"
    lesson_id: Optional[str] = None
    transcript_filename: Optional[str] = None
    alignment_model_size: str = "small"
    beam_size: int = 5

    model_config = {
        "json_schema_extra": {
            "example": {
                "audio_path": "generated_lessons/nl_lesson_1.mp3",
                "text": "Welkom bij de Nederlandse les. Vandaag oefenen we uitspraak.",
                "language": "nl",
                "lesson_id": "nl_lesson_1",
                "transcript_filename": "nl_lesson_1.json",
                "alignment_model_size": "small",
                "beam_size": 5,
            }
        }
    }


class TranscriptWord(BaseModel):
    text: str = Field(description="Word text as spoken in the audio")
    start: float = Field(description="Word start time in seconds", ge=0)
    end: float = Field(description="Word end time in seconds", ge=0)


class TranscriptSentence(BaseModel):
    id: str = Field(description="Sentence identifier, for example s1")
    text: str = Field(description="Sentence text")
    start: float = Field(description="Sentence start time in seconds", ge=0)
    end: float = Field(description="Sentence end time in seconds", ge=0)
    words: list[TranscriptWord] = Field(
        description="Word-level timings for this sentence"
    )


class TranscriptDocument(BaseModel):
    lesson_id: str = Field(description="Lesson identifier")
    sentences: list[TranscriptSentence] = Field(
        description="Sentence and word timestamps"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "lesson_id": "nl_lesson_1",
                "sentences": [
                    {
                        "id": "s1",
                        "text": "Welkom bij de Nederlandse les.",
                        "start": 0.0,
                        "end": 2.15,
                        "words": [
                            {"text": "Welkom", "start": 0.0, "end": 0.45},
                            {"text": "bij", "start": 0.48, "end": 0.65},
                        ],
                    }
                ],
            }
        }
    }


class VoxtralTranscriptResponse(TranscriptDocument):
    audio_path: str = Field(description="Aligned source audio path")
    transcript_path: str = Field(description="Saved transcript JSON path")


class _TTSModel(Protocol):
    def generate(self, *, text: str, voice: str) -> Iterable[Any]: ...


class RealVoxtralEngine:
    MLX_MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit"
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
        self._is_warmed_up = False

    def _warm_up_model(self, model: _TTSModel) -> None:
        if self._is_warmed_up:
            return

        try:
            # One-time throwaway inference to stabilize first-token decoding noise.
            for _ in model.generate(text=".", voice="nl_female"):
                pass
            self._is_warmed_up = True
            print("🔥 Voxtral warm-up completed.")
        except Exception as e:
            # Warm-up should never block synthesis; keep serving with cleanup fallback.
            print(f"⚠️ Voxtral warm-up skipped: {e}")

    def _load_model(self):
        with self._lock:
            if self._model is None:
                print(f"📦 Loading Voxtral MLX model ({self.MLX_MODEL_ID})...")
                from mlx_audio.tts.utils import load

                self._model = cast(_TTSModel, load(self.MLX_MODEL_ID))
                print("✅ Model loaded.")

            model = self._model
            if model is not None and not self._is_warmed_up:
                self._warm_up_model(model)

    def _resolve_voice(self, voice: Optional[str]) -> str:
        """Map a preset name or legacy file path to a valid preset voice name."""
        if not voice:
            return "nl_female"
        if "/" in voice or "\\" in voice or voice.endswith(".wav"):
            return "nl_female"
        return voice if voice in self.PRESET_VOICES else "nl_female"

    def _cleanup_start_artifact(
        self, audio: np.ndarray, sample_rate: int = 24000
    ) -> np.ndarray:
        if audio.size == 0:
            return audio

        # Detect stable speech onset and skip low-level startup hiss.
        max_scan = min(int(0.35 * sample_rate), audio.size)
        window_samples = max(1, int(0.008 * sample_rate))
        squared = audio[:max_scan] ** 2
        kernel = np.ones(window_samples, dtype=np.float32) / window_samples
        moving_power = np.convolve(squared, kernel, mode="valid")
        moving_rms = np.sqrt(np.maximum(moving_power, 0.0))

        trim_idx = 0
        if moving_rms.size > 0:
            floor_region = moving_rms[: max(1, int(0.05 * sample_rate))]
            noise_floor = float(np.percentile(floor_region, 30))
            peak_rms = float(np.max(moving_rms))

            # Adaptive threshold: above local floor, but still catches soft speech.
            onset_threshold = max(0.0035, noise_floor * 3.5, peak_rms * 0.08)
            active = moving_rms > onset_threshold

            sustain_samples = max(1, int(0.012 * sample_rate))
            sustained = np.convolve(
                active.astype(np.int32),
                np.ones(sustain_samples, dtype=np.int32),
                mode="valid",
            )
            sustained_idx = np.flatnonzero(sustained >= sustain_samples)

            if sustained_idx.size > 0:
                # Keep a tiny lead-in so plosives are not clipped.
                lead_in = int(0.004 * sample_rate)
                trim_idx = max(0, int(sustained_idx[0]) - lead_in)

        if trim_idx > 0:
            audio = audio[trim_idx:]

        if audio.size == 0:
            return audio

        # Suppress quiet startup hiss in the first 40 ms.

        gate_samples = min(int(0.04 * sample_rate), audio.size)
        gate_threshold = 0.0012
        if gate_samples > 0:
            prefix = audio[:gate_samples]
            audio[:gate_samples] = np.where(
                np.abs(prefix) < gate_threshold, 0.0, prefix
            )

        # Force absolute silence for a tiny prefix to eliminate encoder boundary pops.
        hard_mute_samples = min(int(0.006 * sample_rate), audio.size)
        if hard_mute_samples > 0:
            audio[:hard_mute_samples] = 0.0

        # Smooth fade-in to avoid audible transition from gating to speech.
        fade_samples = min(int(0.03 * sample_rate), audio.size)
        if fade_samples > 1:
            audio[:fade_samples] *= np.linspace(
                0.0, 1.0, fade_samples, dtype=audio.dtype
            )

        return audio

    def _apply_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Pitch-preserving time-stretch via Rubber Band Library (pyrubberband)."""
        speed = max(0.25, min(4.0, speed))
        if abs(speed - 1.0) < 0.001 or audio.size == 0:
            return audio
        import pyrubberband
        return pyrubberband.time_stretch(audio, 24000, speed).astype(audio.dtype)

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
        audio = self._cleanup_start_artifact(audio, sample_rate=24000)

        speed = float(kwargs.get("speed", 1.0))
        audio = self._apply_speed(audio, speed)

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


class FasterWhisperAligner:
    def __init__(self):
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _load_model(self, model_size: str):
        with self._lock:
            if model_size in self._models:
                return self._models[model_size]

            from faster_whisper import WhisperModel

            # macOS: use CPU + int8 for better compatibility/perf balance.
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            self._models[model_size] = model
            return model

    @staticmethod
    def _normalize_word(token: str) -> str:
        return re.sub(r"^[^\w]+|[^\w]+$", "", token.strip().lower(), flags=re.UNICODE)

    @staticmethod
    def _split_source_sentences(text: str) -> list[list[str]]:
        sentence_texts = [
            chunk.strip()
            for chunk in re.split(r"(?<=[.!?])\s+", text.strip())
            if chunk.strip()
        ]
        if not sentence_texts and text.strip():
            sentence_texts = [text.strip()]

        sentence_tokens: list[list[str]] = []
        for sentence in sentence_texts:
            tokens = [tok for tok in sentence.split() if tok.strip()]
            if tokens:
                sentence_tokens.append(tokens)
        return sentence_tokens

    def align(
        self,
        audio_path: str,
        *,
        lesson_id: str,
        source_text: str,
        language: Optional[str],
        model_size: str,
        beam_size: int,
    ) -> dict[str, Any]:
        model = self._load_model(model_size)

        segments, _ = model.transcribe(
            audio_path,
            task="transcribe",
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
            initial_prompt=source_text,
        )

        timed_words: list[dict[str, Any]] = []
        for segment in segments:
            for word in segment.words or []:
                if word.start is None or word.end is None:
                    continue
                word_text = (word.word or "").strip()
                if not word_text:
                    continue
                timed_words.append(
                    {
                        "text": word_text,
                        "start": round(float(word.start), 3),
                        "end": round(float(word.end), 3),
                    }
                )

        # Build sentence buckets from source text and fill with aligned word times.
        sentence_tokens = self._split_source_sentences(source_text)
        result_sentences: list[dict[str, Any]] = []

        word_index = 0
        for idx, tokens in enumerate(sentence_tokens, start=1):
            sentence_words: list[dict[str, Any]] = []
            for token in tokens:
                if word_index >= len(timed_words):
                    break

                timed = timed_words[word_index]
                sentence_words.append(
                    {
                        "text": token,
                        "start": timed["start"],
                        "end": timed["end"],
                    }
                )
                word_index += 1

            if sentence_words:
                result_sentences.append(
                    {
                        "id": f"s{idx}",
                        "text": " ".join(tokens),
                        "start": sentence_words[0]["start"],
                        "end": sentence_words[-1]["end"],
                        "words": sentence_words,
                    }
                )

        # If source tokenization and aligned words diverge, fall back to grouped raw words.
        if not result_sentences and timed_words:
            result_sentences.append(
                {
                    "id": "s1",
                    "text": source_text.strip(),
                    "start": timed_words[0]["start"],
                    "end": timed_words[-1]["end"],
                    "words": timed_words,
                }
            )

        if word_index < len(timed_words):
            remaining = timed_words[word_index:]
            if result_sentences:
                result_sentences[-1]["words"].extend(remaining)
                result_sentences[-1]["end"] = remaining[-1]["end"]
                result_sentences[-1]["text"] = " ".join(
                    word["text"] for word in result_sentences[-1]["words"]
                )
            else:
                result_sentences.append(
                    {
                        "id": "s1",
                        "text": " ".join(word["text"] for word in remaining),
                        "start": remaining[0]["start"],
                        "end": remaining[-1]["end"],
                        "words": remaining,
                    }
                )

        return {"lesson_id": lesson_id, "sentences": result_sentences}


voxtral_engine = RealVoxtralEngine()
aligner = FasterWhisperAligner()


def get_engine():
    return voxtral_engine


def get_aligner():
    return aligner


def _find_transcript_by_lesson_id(lesson_id: str, transcripts_dir: str) -> str:
    """Resolve a lesson_id to a transcript JSON path in generated_lessons."""
    safe_lesson_id = lesson_id.strip()
    if not safe_lesson_id:
        raise HTTPException(status_code=400, detail="lesson_id cannot be empty")
    if "/" in safe_lesson_id or "\\" in safe_lesson_id:
        raise HTTPException(status_code=400, detail="invalid lesson_id")

    direct_path = os.path.join(transcripts_dir, f"{safe_lesson_id}.json")
    if os.path.exists(direct_path):
        return direct_path

    if not os.path.isdir(transcripts_dir):
        raise HTTPException(status_code=404, detail="No transcripts directory found")

    for file_name in os.listdir(transcripts_dir):
        if not file_name.endswith(".json"):
            continue

        transcript_path = os.path.join(transcripts_dir, file_name)
        try:
            import json

            with open(transcript_path, "r", encoding="utf-8") as fp:
                transcript = json.load(fp)
        except Exception:
            continue

        if transcript.get("lesson_id") == safe_lesson_id:
            return transcript_path

    raise HTTPException(
        status_code=404, detail=f"Transcript not found for lesson_id '{safe_lesson_id}'"
    )


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
            speed=request.speed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response_name = os.path.basename(actual_output_path)
    media_type = "audio/mpeg" if response_name.endswith(".mp3") else "audio/wav"
    return FileResponse(
        actual_output_path, media_type=media_type, filename=response_name
    )


@app.post("/v1/voxtral/transcript", response_model=VoxtralTranscriptResponse)
async def voxtral_generate_transcript(
    request: VoxtralTranscriptRequest,
    transcript_aligner=Depends(get_aligner),
):
    derived_lesson_id = (
        request.lesson_id or os.path.splitext(os.path.basename(request.audio_path))[0]
    )

    if not os.path.exists(request.audio_path):
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {request.audio_path}",
        )

    try:
        transcript = transcript_aligner.align(
            request.audio_path,
            lesson_id=derived_lesson_id,
            source_text=request.text,
            language=request.language,
            model_size=request.alignment_model_size,
            beam_size=request.beam_size,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if request.transcript_filename:
        os.makedirs("generated_lessons", exist_ok=True)
        transcript_path = os.path.join("generated_lessons", request.transcript_filename)
        if not transcript_path.endswith(".json"):
            transcript_path += ".json"
    else:
        base_name, _ = os.path.splitext(request.audio_path)
        transcript_path = f"{base_name}.json"

    with open(transcript_path, "w", encoding="utf-8") as fp:
        import json

        json.dump(transcript, fp, ensure_ascii=False, indent=2)

    return {
        "lesson_id": transcript["lesson_id"],
        "audio_path": request.audio_path,
        "transcript_path": transcript_path,
        "sentences": transcript["sentences"],
    }


@app.get("/v1/voxtral/transcript/{lesson_id}", response_model=TranscriptDocument)
async def voxtral_get_transcript(lesson_id: str):
    transcripts_dir = "generated_lessons"
    transcript_path = _find_transcript_by_lesson_id(lesson_id, transcripts_dir)

    try:
        import json

        with open(transcript_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            return TranscriptDocument.model_validate(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load transcript: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
