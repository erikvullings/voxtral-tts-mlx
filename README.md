# VOXTRAL TTS API

FastAPI wrapper around Voxtral TTS (MLX backend) with OpenAI-compatible endpoints.

Open the browser at [http://localhost:8000/docs](http://localhost:8000/docs).

```bash
source .venv/bin/activate
uv sync
uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## Notes

- Python `>=3.14` is supported.
- The server uses `mlx-audio` for inference.
- Audio export uses `soundfile` directly (no `pydub`).
- If MP3 encoding is not available in the local `libsndfile` build, the API falls back to WAV output.

## Optional Word/Sentence Timestamp Endpoint

Alignment is an optional second step.

1. Generate audio with `POST /v1/voxtral/speech`.
2. Generate timestamps with `POST /v1/voxtral/transcript`.

Example request body:

```json
{
  "audio_path": "generated_lessons/nl_lesson_1.mp3",
  "text": "Welkom bij de Nederlandse les. Vandaag oefenen we uitspraak.",
  "language": "nl",
  "lesson_id": "nl_lesson_1",
  "transcript_filename": "nl_lesson_1.json",
  "alignment_model_size": "small",
  "beam_size": 5
}
```

Response includes:

- `audio_path`: source audio file path that was aligned
- `transcript_path`: generated JSON file path
- `sentences`: sentence list with per-word `start`/`end` timestamps

Fetch a transcript later by lesson id:

- `GET /v1/voxtral/transcript/{lesson_id}`

Example:

```bash
curl http://127.0.0.1:8000/v1/voxtral/transcript/nl_lesson_1
```

Notes:

- Alignment uses local `faster-whisper` with `word_timestamps=true`.
- The first call may take longer because Whisper model weights are downloaded and cached.
