# MLX TTS API

FastAPI wrapper around MLX-based (Apple) TTS backends with OpenAI-compatible endpoints.

Current state:

- `server_voxtral.py` provides the Voxtral-backed entrypoint.
- `server_chatterbox.py` provides the Chatterbox-backed entrypoint.
- `api_shared.py` now owns the shared request/response models, transcript alignment, and HTTP route factory.

This is the first step in refactoring the project from a Voxtral-specific API into a more generic multi-model MLX TTS API.

Open the browser at [http://localhost:8000/docs](http://localhost:8000/docs).

```bash
source .venv/bin/activate
uv sync
uv run tts --host 0.0.0.0 --port 8000 --reload
```

Chatterbox entrypoint:

```bash
uv run tts --backend chatterbox --host 0.0.0.0 --port 8001 --reload
```

CLI options:

- `--backend voxtral|chatterbox`
- `--host <host>`
- `--port <port>`
- `--reload`

## Mac-First Installer

Use the unified installer to bootstrap a local macOS environment quickly.
It supports multiple backends and asks which one to install when `--backend` is omitted.

```bash
scripts/install-mac.sh
```

Optional flags:

- `--backend voxtral|chatterbox`: select backend explicitly (otherwise interactive prompt)
- `--run`: start the API server after setup
- `--port <port>`: set server port (default: `8000`)

Example:

```bash
scripts/install-mac.sh --backend voxtral --run --port 8001
```

## Chatterbox Alternative (Drop-In API)

You can run a Chatterbox-backed API that keeps the same endpoint contract as the Voxtral server (`/v1/audio/speech`, `/v1/voxtral/speech`, `/v1/voxtral/transcript`).

```bash
scripts/install-mac.sh --backend chatterbox --run --port 8001
```

This starts:

```bash
.venv-chatterbox/bin/tts --backend chatterbox --host 0.0.0.0 --port 8001 --reload
```

Compatibility notes:

- Request/response schema is shared through `api_shared.py`.
- Voice selection works through `voice` (`/v1/audio/speech`) or `voice_reference_path` (`/v1/voxtral/speech`).
- Built-in aliases: `nl_female` (default), `female`, `default`, and `nl_male`/`male` (mapped to `voices/jasper.wav` if present).
- You can also pass a custom local WAV path or filename in `voices/`.
- List available voices at runtime with `GET /v1/voxtral/voices`.
- Speed is supported via `speed` on both speech endpoints. For Chatterbox, values below `1.0` now apply a stronger slowdown curve (for example `0.9` is noticeably slower).
- Transcript endpoints remain unchanged and still use `faster-whisper` alignment.
- Chatterbox runs in an isolated `.venv-chatterbox` so Voxtral dependencies stay unchanged.

### Chatterbox Prosody Markup

Chatterbox does not expose full SSML support in this wrapper, but the API supports two lightweight tags in `text`:

- Pause tag: `<break time="500ms"/>` or `<break time="1.2s"/>` (clamped to max 3 seconds)
- Emphasis tag: `<emphasis>belangrijk</emphasis>`

Example:

```json
{
  "text": "Welkom. <break time=\"700ms\"/> Dit is <emphasis>heel belangrijk</emphasis> voor de uitspraak.",
  "voice_reference_path": "nl_female",
  "language": "nl",
  "speed": 0.9,
  "output_filename": "lesson_with_pauses.wav"
}
```

The script performs:

- macOS + Apple Silicon checks
- `uv` installation (if missing)
- dependency sync with `uv sync`
- optional server start

## System Dependencies

`pyrubberband` (used for pitch-preserving speed adjustment) wraps the native Rubber Band Library. Install it before `uv sync`:

```bash
brew install rubberband
```

## Notes

- This project is currently macOS-focused (Apple Silicon) because TTS inference uses `mlx-audio` / MLX.
- Linux and Windows are not supported by this repository as-is.
- Python `>=3.14` is supported.
- The project is being refactored toward a generic MLX TTS API with multiple backend adapters.
- The current shared seam is `api_shared.py`; backend-specific synthesis remains in `server_voxtral.py` and `server_chatterbox.py`.
- Audio export uses `soundfile` directly (no `pydub`).
- If MP3 encoding is not available in the local `libsndfile` build, the API falls back to WAV output.
- Speed adjustment uses `pyrubberband` (Rubber Band Library) for pitch-preserving time-stretch.

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
