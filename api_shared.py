import os
import re
import threading
from typing import Any, Optional, Protocol

import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


class OpenAISpeechRequest(BaseModel):
    model: str = "voxtral"
    input: str
    voice: str = "nl_female"
    language: str = "nl"
    response_format: str = "mp3"
    speed: float = 1.0

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "voxtral",
                "input": "Welkom bij de Nederlandse les. Vandaag gaan we grammatica oefenen.",
                "voice": "nl_female",
                "language": "nl",
                "response_format": "mp3",
                "speed": 1.0,
            }
        }
    }


class VoxtralExtendedRequest(BaseModel):
    text: str
    voice_reference_path: Optional[str] = "nl_female"
    language: str = "nl"
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
                "language": "nl",
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


class VoiceOption(BaseModel):
    name: str
    gender: str
    source: str
    is_default: bool = False


class TTSBackend(Protocol):
    def synthesize(
        self, text: str, voice_path: Optional[str], output_path: str, **kwargs: Any
    ) -> str: ...

    def list_voices(self) -> Any: ...


class FasterWhisperAligner:
    def __init__(self):
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _load_model(self, model_size: str):
        with self._lock:
            if model_size in self._models:
                return self._models[model_size]

            from faster_whisper import WhisperModel

            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            self._models[model_size] = model
            return model

    @staticmethod
    def _audio_duration(audio_path: str) -> Optional[float]:
        try:
            info = sf.info(audio_path)
            if info.frames and info.samplerate:
                return float(info.frames) / float(info.samplerate)
        except Exception:
            return None
        return None

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

        sentence_tokens = self._split_source_sentences(source_text)
        result_sentences: list[dict[str, Any]] = []

        word_index = 0
        sentence_rows: list[list[dict[str, Any]]] = []
        for tokens in sentence_tokens:
            row: list[dict[str, Any]] = []
            for token in tokens:
                if word_index < len(timed_words):
                    timed = timed_words[word_index]
                    row.append(
                        {"text": token, "start": timed["start"], "end": timed["end"]}
                    )
                    word_index += 1
                else:
                    row.append({"text": token, "start": None, "end": None})
            sentence_rows.append(row)

        untimed = [
            word for row in sentence_rows for word in row if word["start"] is None
        ]
        if untimed:
            timed_flat = [
                word
                for row in sentence_rows
                for word in row
                if word["start"] is not None
            ]
            tail_start = timed_flat[-1]["end"] if timed_flat else 0.0
            duration = self._audio_duration(audio_path)
            tail_end = max(duration or 0.0, tail_start)
            weights = [max(len(word["text"]), 1) for word in untimed]
            total_weight = sum(weights)
            span = tail_end - tail_start
            cursor = tail_start
            for word, weight in zip(untimed, weights):
                word["start"] = round(cursor, 3)
                cursor += span * weight / total_weight
                word["end"] = round(cursor, 3)

        for idx, row in enumerate(sentence_rows, start=1):
            if row:
                result_sentences.append(
                    {
                        "id": f"s{idx}",
                        "text": " ".join(word["text"] for word in row),
                        "start": row[0]["start"],
                        "end": row[-1]["end"],
                        "words": row,
                    }
                )

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
                last_words = result_sentences[-1]["words"]
                last_words[-1]["end"] = max(last_words[-1]["end"], remaining[-1]["end"])
                result_sentences[-1]["end"] = last_words[-1]["end"]
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


def _find_transcript_by_lesson_id(lesson_id: str, transcripts_dir: str) -> str:
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


def _request_kwargs(request: BaseModel, *, excluded: set[str]) -> dict[str, Any]:
    payload = request.model_dump(exclude_none=True, mode="json")
    for key in excluded:
        payload.pop(key, None)
    return payload


def create_app(
    *,
    title: str,
    engine: TTSBackend,
    voice_response_model: Any,
    version: str = "1.0.0",
    route_prefix: str = "voxtral",
    openai_request_model: type[BaseModel] = OpenAISpeechRequest,
    extended_request_model: type[BaseModel] = VoxtralExtendedRequest,
) -> FastAPI:
    app = FastAPI(title=title, version=version)
    aligner = FasterWhisperAligner()
    route_root = f"/v1/{route_prefix}"

    def get_engine() -> TTSBackend:
        return engine

    def get_aligner() -> FasterWhisperAligner:
        return aligner

    @app.get(f"{route_root}/voices", response_model=voice_response_model)
    async def list_voices(engine=Depends(get_engine)):
        return engine.list_voices()

    @app.post("/v1/audio/speech")
    async def openai_compatible_speech(
        request: openai_request_model, engine=Depends(get_engine)
    ):
        requested_format = request.response_format.lower()
        output_ext = "mp3" if requested_format == "mp3" else "wav"
        output_path = f"generated_lessons/lesson_{hash(request.input)}.{output_ext}"
        os.makedirs("generated_lessons", exist_ok=True)

        try:
            actual_output_path = engine.synthesize(
                text=request.input,
                voice_path=request.voice,
                output_path=output_path,
                **_request_kwargs(
                    request,
                    excluded={"model", "input", "voice", "response_format"},
                ),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        response_name = os.path.basename(actual_output_path)
        media_type = "audio/mpeg" if response_name.endswith(".mp3") else "audio/wav"
        return FileResponse(
            actual_output_path, media_type=media_type, filename=response_name
        )

    @app.post(f"{route_root}/speech")
    async def voxtral_dedicated_speech(
        request: extended_request_model, engine=Depends(get_engine)
    ):
        output_path = f"generated_lessons/{request.output_filename}"
        os.makedirs("generated_lessons", exist_ok=True)

        try:
            actual_output_path = engine.synthesize(
                text=request.text,
                voice_path=request.voice_reference_path,
                output_path=output_path,
                **_request_kwargs(
                    request,
                    excluded={"text", "voice_reference_path", "output_filename"},
                ),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        response_name = os.path.basename(actual_output_path)
        media_type = "audio/mpeg" if response_name.endswith(".mp3") else "audio/wav"
        return FileResponse(
            actual_output_path, media_type=media_type, filename=response_name
        )

    @app.post(f"{route_root}/transcript", response_model=VoxtralTranscriptResponse)
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
            os.makedirs("generated_lessons", exist_ok=True)
            transcript_path = os.path.join("generated_lessons", f"{derived_lesson_id}.json")

        with open(transcript_path, "w", encoding="utf-8") as fp:
            import json

            json.dump(transcript, fp, ensure_ascii=False, indent=2)

        return {
            "lesson_id": transcript["lesson_id"],
            "audio_path": request.audio_path,
            "transcript_path": transcript_path,
            "sentences": transcript["sentences"],
        }

    @app.get(f"{route_root}/transcript/{{lesson_id}}", response_model=TranscriptDocument)
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

    return app