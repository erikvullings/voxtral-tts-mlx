# VOXTRAL TTS API

FastAPI wrapper around Voxtral TTS (MLX backend) with OpenAI-compatible endpoints.

Open the browser at [http://localhost:8000/docs](http://localhost:8000/docs).

```bash
source .venv/bin/activate
uv sync
uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## Mac-First Installer

Use the installer to bootstrap a local macOS environment quickly:

```bash
bash scripts/install-mac.sh
```

Optional flags:

- `--run`: start the API server after setup
- `--port <port>`: set server port (default: `8000`)

Example:

```bash
bash scripts/install-mac.sh --run --port 8001
```

The script performs:

- macOS + Apple Silicon checks
- `uv` installation (if missing)
- dependency sync with `uv sync`
- optional server start

## Notes

- This project is currently macOS-focused (Apple Silicon) because TTS inference uses `mlx-audio` / MLX.
- Linux and Windows are not supported by this repository as-is.
- Python `>=3.14` is supported.
- The server uses `mlx-audio` for inference.
- Audio export uses `soundfile` directly (no `pydub`).
- If MP3 encoding is not available in the local `libsndfile` build, the API falls back to WAV output.

## Platform Support

- Supported now: macOS on Apple Silicon (M-series), with MLX backend.
- Not supported as-is: Linux and Windows.
- Transcript alignment (`faster-whisper`) itself is cross-platform, but speech synthesis in this repo depends on MLX (`mlx-audio`).
- To support Linux/Windows, you would need to swap the TTS backend from MLX to a cross-platform engine and keep the same FastAPI endpoints.

## CI/CD (macOS)

GitHub Actions workflow: `.github/workflows/macos-ci.yml`

It runs on macOS and does the following automatically:

- validates dependencies (`uv sync`)
- runs a Python syntax check (`py_compile`)
- packages the repository as a source archive (`voxtral-api-macos-source.tar.gz`)
- uploads build artifacts to Actions

On version tags (`v*`), the workflow also publishes a GitHub Release with the packaged archive and SHA256 checksum.

## Quickstart (Copy/Paste)

Generate lesson audio:

```bash
curl -X POST http://127.0.0.1:8000/v1/voxtral/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welkom bij de Nederlandse les. Vandaag oefenen we uitspraak.",
    "voice_reference_path": "nl_female",
    "language": "nl",
    "output_filename": "nl_lesson_1.mp3"
  }' \
  --output generated_lessons/nl_lesson_1.mp3
```

Generate transcript with timings:

```bash
curl -X POST http://127.0.0.1:8000/v1/voxtral/transcript \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "generated_lessons/nl_lesson_1.mp3",
    "text": "Welkom bij de Nederlandse les. Vandaag oefenen we uitspraak.",
    "language": "nl",
    "lesson_id": "nl_lesson_1",
    "transcript_filename": "nl_lesson_1.json",
    "alignment_model_size": "small",
    "beam_size": 5
  }'
```

## Recommended Lesson Workflow

Best practice is a two-step pipeline:

1. Generate lesson audio with Voxtral TTS.
2. Generate transcript timings from that audio (optional, only when needed).

This keeps TTS generation fast and lets you run alignment separately.

### Step 1: Generate Lesson Audio

Use `POST /v1/voxtral/speech`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/voxtral/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welkom bij de Nederlandse les. Vandaag oefenen we uitspraak.",
    "voice_reference_path": "nl_female",
    "language": "nl",
    "output_filename": "nl_lesson_1.mp3"
  }' \
  --output generated_lessons/nl_lesson_1.mp3
```

Notes:

- Default voice is `nl_female`.
- `nl_male` is also supported.
- If MP3 encoding is unavailable locally, the API may return WAV.

### Step 2: Generate Transcript with Sentence + Word Timings

Use `POST /v1/voxtral/transcript` with the exact audio path and original lesson text.

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

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/voxtral/transcript \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "generated_lessons/nl_lesson_1.mp3",
    "text": "Welkom bij de Nederlandse les. Vandaag oefenen we uitspraak.",
    "language": "nl",
    "lesson_id": "nl_lesson_1",
    "transcript_filename": "nl_lesson_1.json",
    "alignment_model_size": "small",
    "beam_size": 5
  }'
```

The response includes:

- `lesson_id`
- `audio_path`
- `transcript_path`
- `sentences[]` with per-word `start` and `end` timings

The JSON written to disk is ready for frontend active highlighting.

## Production Defaults

Recommended defaults for production lesson generation:

- `voice_reference_path`: `nl_female` (use `nl_male` if preferred)
- `language`: `nl`
- `output_filename`: stable, deterministic name per lesson (for example `nl_lesson_1.mp3`)

Recommended defaults for transcript timing generation:

- `alignment_model_size`: `small` (good balance of quality and speed)
- `beam_size`: `5` (current API default)
- `lesson_id`: stable lesson identifier, reused by frontend
- `transcript_filename`: same base name as audio (for example `nl_lesson_1.json`)

If you need faster alignment at lower accuracy, try:

- `alignment_model_size`: `tiny`
- `beam_size`: `3`

Notes:

- Alignment uses local `faster-whisper` with `word_timestamps=true`.
- The first alignment call may take longer because Whisper model weights are downloaded and cached.

## Retrieve Transcript by Lesson ID

After alignment, you can fetch transcript JSON by lesson id:

- `GET /v1/voxtral/transcript/{lesson_id}`

Example:

```bash
curl http://127.0.0.1:8000/v1/voxtral/transcript/nl_lesson_1
```
