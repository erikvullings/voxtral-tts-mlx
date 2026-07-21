import difflib
import os
import re
import threading
from typing import Any, Optional, Protocol

import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


def _load_local_env_file() -> None:
    """Load simple KEY=VALUE pairs from .env into process env if unset."""
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.isfile(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Non-fatal: explicit shell env variables still work.
        return


_load_local_env_file()


# Splits on whitespace that follows sentence-ending punctuation, optionally
# followed by a closing quote mark (e.g. `jij?"` or `Sarah."`). A plain
# `(?<=[.!?])\s+` misses that quoted case, since the character right before
# the whitespace is the quote mark, not the punctuation -- it would merge a
# quoted sentence into whatever follows it.
_SENTENCE_SPLIT_RE = re.compile(r'(?:(?<=[.!?])|(?<=[.!?]["\'”’]))\s+')


class OpenAISpeechRequest(BaseModel):
    model: str = "voxtral"
    input: str
    voice: str = "nl_female"
    language: Optional[str] = None
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
    language: Optional[str] = None
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


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def _replace_markdown_emphasis(text: str, *, supports_ssml_emphasis: bool) -> str:
    def _bold_repl(match: re.Match[str]) -> str:
        content = match.group(2).strip()
        if not content:
            return ""
        if supports_ssml_emphasis:
            return f"<emphasis>{content}</emphasis>"
        return content

    def _italic_repl(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        if not content:
            return ""
        if supports_ssml_emphasis:
            return f"<emphasis>{content}</emphasis>"
        return content

    converted = re.sub(r"(\*\*|__)(.+?)\1", _bold_repl, text, flags=re.DOTALL)
    converted = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
        _italic_repl,
        converted,
        flags=re.DOTALL,
    )
    converted = re.sub(
        r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)",
        _italic_repl,
        converted,
        flags=re.DOTALL,
    )
    return converted


def _replace_newlines_with_breaks(
    text: str, *, line_ms: int, paragraph_ms: int, supports_ssml_breaks: bool
) -> str:
    if not supports_ssml_breaks:
        return text

    paragraph_tag = f' <break time="{paragraph_ms}ms"/> '
    line_tag = f' <break time="{line_ms}ms"/> '

    with_paragraph_breaks = re.sub(r"\n\s*\n+", paragraph_tag, text)
    with_line_breaks = re.sub(r"\n", line_tag, with_paragraph_breaks)
    return with_line_breaks


def _preprocess_tts_text(
    text: str,
    *,
    supports_ssml_emphasis: bool,
    supports_ssml_breaks: bool,
) -> str:
    if not _env_flag("TTS_MARKDOWN_PROSODY_ENABLED", True):
        return text

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    line_ms = _env_int("TTS_MARKDOWN_LINE_BREAK_MS", 350, min_value=50, max_value=5000)
    paragraph_ms = _env_int(
        "TTS_MARKDOWN_PARAGRAPH_BREAK_MS",
        900,
        min_value=100,
        max_value=8000,
    )

    out = _replace_markdown_emphasis(
        normalized, supports_ssml_emphasis=supports_ssml_emphasis
    )
    out = _replace_newlines_with_breaks(
        out,
        line_ms=line_ms,
        paragraph_ms=paragraph_ms,
        supports_ssml_breaks=supports_ssml_breaks,
    )
    return out


def _base_capabilities(
    *,
    backend: str,
    supports_ssml_emphasis: bool,
    supports_ssml_breaks: bool,
) -> dict[str, Any]:
    ssml_tags: list[str] = []
    if supports_ssml_breaks:
        ssml_tags.append("<break time=\"Xms\"/>")
    if supports_ssml_emphasis:
        ssml_tags.append("<emphasis>...</emphasis>")

    return {
        "backend": backend,
        "voiceCloning": {
            "supported": False,
            "notes": "Backend-specific; see endpoint payload overrides.",
        },
        "ssmlProsody": {
            "tagParsing": bool(ssml_tags),
            "supportedTags": ssml_tags,
            "notes": (
                "Adapter-level parsing is limited to listed tags. "
                "Some models also support token-based prosody controls."
            ),
        },
        "markdownProsody": {
            "enabled": _env_flag("TTS_MARKDOWN_PROSODY_ENABLED", True),
            "lineBreakMs": _env_int(
                "TTS_MARKDOWN_LINE_BREAK_MS", 350, min_value=50, max_value=5000
            ),
            "paragraphBreakMs": _env_int(
                "TTS_MARKDOWN_PARAGRAPH_BREAK_MS",
                900,
                min_value=100,
                max_value=8000,
            ),
            "conversions": {
                "boldItalicToEmphasis": supports_ssml_emphasis,
                "singleNewlineToBreak": supports_ssml_breaks,
                "paragraphToLongBreak": supports_ssml_breaks,
            },
            "environment": {
                "enabledFlag": "TTS_MARKDOWN_PROSODY_ENABLED",
                "lineBreakMs": "TTS_MARKDOWN_LINE_BREAK_MS",
                "paragraphBreakMs": "TTS_MARKDOWN_PARAGRAPH_BREAK_MS",
            },
        },
        "languageConditioning": {
            "apiDefaultLanguage": None,
            "notes": "Language is optional at API level. Backend behavior differs.",
        },
    }


def _build_capabilities_payload(
    *,
    route_prefix: str,
    backend_name: str = "voxtral",
    engine: TTSBackend,
    supports_ssml_emphasis: bool,
    supports_ssml_breaks: bool,
    backend_capabilities: dict[str, Any] | None,
) -> dict[str, Any]:
    capabilities = _base_capabilities(
        backend=route_prefix,
        supports_ssml_emphasis=supports_ssml_emphasis,
        supports_ssml_breaks=supports_ssml_breaks,
    )

    if backend_capabilities:
        capabilities = _deep_merge_dict(capabilities, backend_capabilities)

    voices: Any
    try:
        voices = engine.list_voices()
    except Exception as exc:
        voices = {
            "error": str(exc),
        }

    capabilities["voices"] = voices
    capabilities["backend"] = backend_name
    capabilities["routes"] = {
        "openaiSpeech": "/v1/audio/speech",
        "voices": "/v1/voices",
        "speech": "/v1/speech",
        "transcriptPost": "/v1/transcript",
        "transcriptGet": "/v1/transcript/{lesson_id}",
        "capabilities": "/v1/capabilities",
    }
    return capabilities


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
    def _normalize_token(text: str) -> str:
        return re.sub(r"[^\w]", "", text, flags=re.UNICODE).lower()

    @staticmethod
    def _split_source_sentences(text: str) -> list[list[str]]:
        sentence_texts = [
            chunk.strip()
            for chunk in _SENTENCE_SPLIT_RE.split(text.strip())
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

        # Align source words to whisper's word list by content, not position.
        # A naive positional zip breaks permanently the moment whisper's word
        # count diverges from the source's -- e.g. a single hallucinated or
        # dropped word (common at TTS chunk-stitch boundaries) would shift
        # every later timestamp by one slot for the rest of the transcript.
        flat_tokens: list[tuple[int, str]] = [
            (sentence_idx, token)
            for sentence_idx, tokens in enumerate(sentence_tokens)
            for token in tokens
        ]
        source_norm = [self._normalize_token(token) for _, token in flat_tokens]
        whisper_norm = [self._normalize_token(w["text"]) for w in timed_words]

        matcher = difflib.SequenceMatcher(None, source_norm, whisper_norm, autojunk=False)
        matched: dict[int, dict[str, float]] = {}
        last_whisper_idx = -1
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for offset in range(i2 - i1):
                    timed = timed_words[j1 + offset]
                    matched[i1 + offset] = {"start": timed["start"], "end": timed["end"]}
                last_whisper_idx = max(last_whisper_idx, j2 - 1)
            elif tag == "replace":
                # Pair up the overlapping run positionally so a plain word
                # substitution still gets a timestamp; any surplus on either
                # side is left unmatched rather than dragging later words
                # out of position.
                for offset in range(min(i2 - i1, j2 - j1)):
                    timed = timed_words[j1 + offset]
                    matched[i1 + offset] = {"start": timed["start"], "end": timed["end"]}
                last_whisper_idx = max(last_whisper_idx, j2 - 1)
            elif tag == "insert":
                # Whisper heard extra word(s) with no source counterpart
                # (hallucination/repetition) -- drop them instead of letting
                # them consume a slot meant for a later source word.
                last_whisper_idx = max(last_whisper_idx, j2 - 1)
            # tag == "delete": source word whisper produced no timestamp for;
            # left unmatched below, handled by the interpolation pass.

        word_index = last_whisper_idx + 1

        sentence_rows: list[list[dict[str, Any]]] = []
        flat_idx = 0
        for tokens in sentence_tokens:
            row: list[dict[str, Any]] = []
            for token in tokens:
                timed = matched.get(flat_idx)
                if timed is not None:
                    row.append(
                        {"text": token, "start": timed["start"], "end": timed["end"]}
                    )
                else:
                    row.append({"text": token, "start": None, "end": None})
                flat_idx += 1
            sentence_rows.append(row)

        # Fill in words the aligner couldn't match to a whisper word (ASR
        # dropped or garbled them). Each contiguous run of unmatched words is
        # interpolated between its own neighboring matched timestamps rather
        # than against the document's last matched word -- a mid-transcript
        # gap must not be stretched out to the end of the audio.
        flat_words = [word for row in sentence_rows for word in row]
        total = len(flat_words)
        cursor_idx = 0
        while cursor_idx < total:
            if flat_words[cursor_idx]["start"] is not None:
                cursor_idx += 1
                continue

            gap_start_idx = cursor_idx
            while cursor_idx < total and flat_words[cursor_idx]["start"] is None:
                cursor_idx += 1
            gap = flat_words[gap_start_idx:cursor_idx]

            span_start = (
                flat_words[gap_start_idx - 1]["end"] if gap_start_idx > 0 else 0.0
            )
            if cursor_idx < total:
                span_end = flat_words[cursor_idx]["start"]
            else:
                duration = self._audio_duration(audio_path)
                span_end = max(duration or 0.0, span_start)
            span_end = max(span_end, span_start)

            weights = [max(len(word["text"]), 1) for word in gap]
            total_weight = sum(weights)
            span = span_end - span_start
            cursor = span_start
            for word, weight in zip(gap, weights):
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
    backend_name: str = "voxtral",
    openai_request_model: type[BaseModel] = OpenAISpeechRequest,
    extended_request_model: type[BaseModel] = VoxtralExtendedRequest,
    supports_ssml_emphasis: bool = False,
    supports_ssml_breaks: bool = False,
    backend_capabilities: dict[str, Any] | None = None,
) -> FastAPI:
    app = FastAPI(title=title, version=version)
    aligner = FasterWhisperAligner()
    route_root = f"/v1/{route_prefix}"

    def get_engine() -> TTSBackend:
        return engine

    def get_aligner() -> FasterWhisperAligner:
        return aligner

    @app.get("/v1/capabilities")
    async def get_capabilities(engine=Depends(get_engine)):
        return _build_capabilities_payload(
            route_prefix=route_prefix,
            backend_name=backend_name,
            engine=engine,
            supports_ssml_emphasis=supports_ssml_emphasis,
            supports_ssml_breaks=supports_ssml_breaks,
            backend_capabilities=backend_capabilities,
        )

    @app.post("/v1/audio/speech")
    async def openai_compatible_speech(
        request: openai_request_model, engine=Depends(get_engine)
    ):
        requested_format = request.response_format.lower()
        output_ext = "mp3" if requested_format == "mp3" else "wav"
        output_path = f"generated_lessons/lesson_{hash(request.input)}.{output_ext}"
        os.makedirs("generated_lessons", exist_ok=True)

        prepared_text = _preprocess_tts_text(
            request.input,
            supports_ssml_emphasis=supports_ssml_emphasis,
            supports_ssml_breaks=supports_ssml_breaks,
        )

        try:
            actual_output_path = engine.synthesize(
                text=prepared_text,
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

    # ── Generic routes ──────────────────────────────────────────────────────────
    @app.get("/v1/voices", response_model=voice_response_model)
    async def list_voices(engine=Depends(get_engine)):
        return engine.list_voices()

    @app.post("/v1/speech")
    async def speech_synthesis(
        request: extended_request_model, engine=Depends(get_engine)
    ):
        output_path = f"generated_lessons/{request.output_filename}"
        os.makedirs("generated_lessons", exist_ok=True)

        prepared_text = _preprocess_tts_text(
            request.text,
            supports_ssml_emphasis=supports_ssml_emphasis,
            supports_ssml_breaks=supports_ssml_breaks,
        )

        try:
            actual_output_path = engine.synthesize(
                text=prepared_text,
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

    @app.post("/v1/transcript", response_model=VoxtralTranscriptResponse)
    async def generate_transcript(
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

    @app.get("/v1/transcript/{lesson_id}", response_model=TranscriptDocument)
    async def get_transcript(lesson_id: str):
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