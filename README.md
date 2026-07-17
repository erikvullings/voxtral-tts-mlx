# MLX TTS API

FastAPI wrapper around MLX-based (Apple) TTS backends with OpenAI-compatible endpoints.

Current state:

- `src/server_voxtral.py` provides the Voxtral-backed entrypoint.
- `src/server_chatterbox.py` provides the Chatterbox-backed entrypoint.
- `src/server_omnivoice.py`, `src/server_kugelaudio.py`, `src/server_higgs.py`, `src/server_moss.py`, and VibeVoice-MLX adapters provide additional backends.
- `src/api_shared.py` owns the shared request/response models, transcript alignment, and HTTP route factory.

This is a multi-model MLX TTS API with one shared HTTP layer and backend-specific synthesis adapters.

## Supported Backends

- `voxtral`: preset voices only, no raw voice cloning
- `chatterbox`: voice cloning from local reference audio
- `omnivoice`: zero-shot voice cloning from local reference audio
- `kugelaudio`: preset voices only (`default`, `warm`, `clear`)
- `higgs`: zero-shot voice cloning via `ref_audio` or `references[]`
- `moss`: voice cloning and continuation via `ref_audio`, `ref_text`, or `prompt_audio_codes`
- `vibevoice`: alias for `vibevoice-7b+coreml`
- `vibevoice-1.5b+coreml`: VibeVoice MLX 1.5B with CoreML semantic encoder
- `vibevoice-1.5b-coreml`: VibeVoice MLX 1.5B with default MLX semantic encoder
- `vibevoice-7b+coreml`: VibeVoice MLX 7B with CoreML semantic encoder (default)
- `vibevoice-7b-coreml`: VibeVoice MLX 7B with default MLX semantic encoder

Open the browser at [http://localhost:8000/docs](http://localhost:8000/docs).

```bash
cp .env.example .env
source .venv/bin/activate
uv sync
tts --host 0.0.0.0 --port 8000 --reload
```

Chatterbox entrypoint:

```bash
.venv-chatterbox/bin/python -m uvicorn server_chatterbox:app --host 0.0.0.0 --port 8001 --reload
```

Other mlx-audio backends:

```bash
tts --backend omnivoice --host 0.0.0.0 --port 8002 --reload
tts --backend kugelaudio --host 0.0.0.0 --port 8003 --reload
tts --backend higgs --host 0.0.0.0 --port 8004 --reload
tts --backend moss --host 0.0.0.0 --port 8005 --reload
tts --backend vibevoice-7b+coreml --host 0.0.0.0 --port 8006 --reload
```

CLI options:

- `--backend voxtral|chatterbox|omnivoice|kugelaudio|higgs|moss|vibevoice|vibevoice-1.5b+coreml|vibevoice-1.5b-coreml|vibevoice-7b+coreml|vibevoice-7b-coreml`
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

- `--backend voxtral|chatterbox|omnivoice|kugelaudio|higgs|moss`: select backend explicitly (otherwise interactive prompt)
- `--run`: start the API server after setup
- `--port <port>`: set server port (default: `8000`)

Example:

```bash
scripts/install-mac.sh --backend voxtral --run --port 8001
```

## Backend Compatibility Notes

All backends share the same OpenAI-compatible endpoint (`POST /v1/audio/speech`) and a backend-specific route set (`/v1/<backend>/speech`, `/v1/<backend>/transcript`, `/v1/<backend>/voices`).

General behavior:

- `voice` (OpenAI route) and `voice_reference_path` (dedicated route) are the primary voice controls.
- `language` is optional at the API layer (not forced to Dutch anymore).
- `speed` is supported on all routes where the backend supports time scaling.
- transcript generation remains shared through `faster-whisper` alignment.

Capability discovery endpoint:

- `GET /v1/capabilities`
- `GET /v1/<backend>/capabilities`

The capabilities payload includes:

- voice cloning support and required/optional inputs
- SSML/prosody support including exact supported tags when parser support exists
- token-based prosody controls where applicable
- language-conditioning behavior
- current runtime voices (`engine.list_voices()` output)

Unified request-normalizer pipeline:

- Higgs, MOSS, and VibeVoice now pass request text through one shared normalizer utility (`request_normalizer.py`).
- This keeps base text cleanup, token-prefix composition, stage-direction injection, and optional pause token formatting consistent across those backends.

### MLX Backend Utilities

Generic utility functions in `src/mlx_utils.py` support all MLX-based backends:

#### Warmup Function
`warmup_mlx_model(model, generate_fn, **generate_kwargs)`

Runs a minimal generation to trigger Metal shader compilation on Apple Silicon.
Without warmup, the first real synthesis call incurs extra latency and noise during compilation.

Example:

```python
from mlx_utils import warmup_mlx_model

def warmup_kugelaud(m):
    warmup_mlx_model(
        m,
        lambda **kw: (t for t in m.generate(**kw)),
        text="Hi.",
        voice="default",
        max_tokens=10,
    )
```

#### Logit Penalty Function
`apply_logit_penalty(model, token_id, penalty_strength=5.0)`

Penalizes a specific token ID during generation to prevent early stopping or over-production.
Commonly used to prevent speech cutoff at sentence ends.

Example (KugelAudio prevents `speech_end_id=151653` cutoff):

```python
from mlx_utils import apply_logit_penalty

# After loading model
model = apply_logit_penalty(model, token_id=151653, penalty_strength=5.0)
```

VibeVoice-MLX setup:

```bash
uv tool install --from git+https://github.com/gafiatulin/vibevoice-mlx vibevoice-mlx
```

No-install option:

- You can vendor/copy the `vibevoice_mlx/` package into this repository root.
- The adapter will automatically prefer `python -m vibevoice_mlx.e2e_pipeline` from local code.
- It also supports using a local clone at `/tmp/vibevoice-mlx` via `uv run --directory`.

The VibeVoice-MLX adapter uses CLI invocation under the hood and supports:

- `gafiatulin/vibevoice-1.5b-mlx`
- `gafiatulin/vibevoice-7b-mlx`
- `--coreml-semantic` (non-CoreML variants use default MLX semantic feedback)
- multi-speaker routing via `Speaker N:` transcript lines + `ref_audio`/`speaker_names`

Default adapter tuning for noise diagnostics and quality:

- `use_stage_directions` defaults to `false`
- `diffusion_steps` defaults to `20`
- `cfg_scale` defaults to `1.3`
- `seed` defaults to `42`
- `solver` defaults to `dpm`
- `quantize_diffusion` defaults to `false`
- `--silence-detection` and `--trim-trailing-silence` are enabled by default

### VibeVoice Diagnostic Matrix Script

Use the dedicated script to generate comparable multi-speaker and single-speaker
diagnostic files across seeds, CFG values, diffusion steps, and quantization.

```bash
uv run python scripts/diagnose_vibevoice_noise.py
```

Common variants:

```bash
# Only non-CoreML backend (default MLX semantic encoder)
uv run python scripts/diagnose_vibevoice_noise.py --backends vibevoice-7b-coreml

# Skip per-voice isolation tests
uv run python scripts/diagnose_vibevoice_noise.py --skip-single-speaker
```

Outputs are written to:

- `audio_tests/diagnostics/vibevoice/<backend>/*.mp3`
- `audio_tests/diagnostics/vibevoice/<backend>/single_speaker/*.mp3`
- `audio_tests/diagnostics/vibevoice/manifest.json`

### Voice Reference Preprocessing

The VibeVoice upstream loader does not perform loudness normalization or denoising,
so reference cleanup strongly affects cloning quality.

Use the preprocessing helper (ffmpeg required):

```bash
# Process every voices/*.wav into voices/clean/*-clean.wav
uv run python scripts/preprocess_voice_refs.py --all

# Process one file
uv run python scripts/preprocess_voice_refs.py --input voices/new_voice.wav

# Overwrite source files (careful)
uv run python scripts/preprocess_voice_refs.py --all --in-place
```

Default ffmpeg filter chain:

```text
highpass=f=70,lowpass=f=11000,adeclick,loudnorm=I=-20:TP=-2:LRA=7
```

Chatterbox runs in an isolated environment because its dependency graph differs from the mlx-audio stack:

```bash
scripts/install-mac.sh --backend chatterbox --run --port 8001
```

This starts:

```bash
.venv-chatterbox/bin/python -m uvicorn server_chatterbox:app --host 0.0.0.0 --port 8001 --reload
```

## Additional MLX-Audio Backends

All of these backends use the same shared HTTP layer in `api_shared.py`, but they expose their own route prefixes:

- `omnivoice` exposes `/v1/omnivoice/voices`, `/v1/omnivoice/speech`, and `/v1/omnivoice/transcript`
- `kugelaudio` exposes `/v1/kugelaudio/voices`, `/v1/kugelaudio/speech`, and `/v1/kugelaudio/transcript`
- `higgs` exposes `/v1/higgs/voices`, `/v1/higgs/speech`, and `/v1/higgs/transcript`
- `moss` exposes `/v1/moss/voices`, `/v1/moss/speech`, and `/v1/moss/transcript`
- `vibevoice` exposes `/v1/vibevoice/voices`, `/v1/vibevoice/speech`, and `/v1/vibevoice/transcript`

Voice support:

- `omnivoice`: zero-shot cloning from local WAV reference audio via `voice_reference_path` or `ref_audio`
- `kugelaudio`: preset voices (`default`, `warm`, `clear`) and custom voices via encoded .pt embeddings
- `higgs`: zero-shot cloning from `ref_audio` or `references[]`; `ref_text` improves fidelity
- `moss`: zero-shot cloning from `ref_audio` / `ref_text`, plus continuation via `prompt_audio_codes`
- `vibevoice`: speaker-routing with 4 speaker slots and emotion-to-variant mapping

## Voice Reference Guide (All Backends)

Use your own reference voice clip like this:

- put a WAV file in `voices/` (for example `voices/my_voice.wav`)
- pass it as `voice_reference_path` on dedicated routes, or as `voice`/`ref_audio` on OpenAI-compatible routes (backend dependent)

Reference usage by backend:

- `voxtral`: no local reference cloning (preset speaker IDs only)
- `kugelaudio`: no local reference cloning (preset voices only)
- `chatterbox`: local reference WAV supported; `ref_text` is not required
- `omnivoice`: local reference WAV supported; `ref_text` is optional but recommended for higher fidelity
- `higgs`: local reference WAV and `references[]` supported; `ref_text` is optional but recommended
- `moss`: local reference WAV supported; `ref_text` optional; continuation also supports `prompt_audio_codes`
- `vibevoice`: use `voice_profile` + `segments[]` with `emotion` to route text to speaker slots; SSML emotion tags are not used

Language parameter behavior:

- `chatterbox`: `language` is used by the tokenizer when provided; if omitted, model defaults are used
- `omnivoice`: `language` is forwarded when provided; if omitted, model defaults are used
- `moss`: `language` is forwarded when provided; language tags are recommended for multilingual stability
- `kugelaudio`: optional, but when provided it must be a 2-letter code such as `nl` or `en`
- `voxtral`, `higgs`: current adapters do not require `language` for synthesis
- `vibevoice`: language is optional and forwarded when provided

This means Dutch is not globally assumed by the API. Set `language` explicitly when you want deterministic language conditioning on backends that use it.

Quick examples:

```bash
curl -X POST http://127.0.0.1:8000/v1/omnivoice/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hallo, dit is een OmniVoice sample.",
    "voice_reference_path": "voices/sample.wav",
    "ref_text": "Hallo, ik ben de referentiestem.",
    "language": "nl",
    "output_filename": "omnivoice_sample.mp3"
  }'
```

```bash
# With preset voice
curl -X POST http://127.0.0.1:8003/v1/kugelaudio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hallo, dit is een KugelAudio sample.",
    "voice_reference_path": "warm",
    "language": "nl",
    "output_filename": "kugelaudio_sample.mp3"
  }' \
  --output kugelaudio_sample.mp3

# With custom voice (requires prior encoding)
curl -X POST http://127.0.0.1:8003/v1/kugelaudio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welkom bij de Nederlandse les.",
    "voice_reference_path": "anouk",
    "language": "nl",
    "output_filename": "kugelaudio_custom.mp3"
  }' \
  --output kugelaudio_custom.mp3
```

```bash
curl -X POST http://127.0.0.1:8000/v1/higgs/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a Higgs Audio sample.",
    "voice_reference_path": "voices/sample.wav",
    "ref_text": "This is the reference transcript.",
    "language": "en",
    "output_filename": "higgs_sample.mp3"
  }' \
  --output higgs_sample.mp3
```

```bash
curl -X POST http://127.0.0.1:8000/v1/moss/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "MOSS-TTS supports cloning from a local reference clip.",
    "voice_reference_path": "voices/sample.wav",
    "ref_text": "This is the reference transcript.",
    "language": "English",
    "output_filename": "moss_sample.mp3"
  }'
```

### KugelAudio Voice Encoding

KugelAudio supports custom voice embeddings. To encode your own voice reference WAV files into speaker embeddings:

**Step 1: Prepare voice reference clips**

Place one or more `.wav` files in a folder (e.g., `my_voices/`). Each file should be:
- 0.5 to 3 seconds of clean, single-speaker audio
- Any sample rate (resampling happens automatically)
- Minimal background noise for best results

```bash
# Example: organize reference files
mkdir -p my_voices
cp ~/Downloads/speaker1.wav my_voices/
cp ~/Downloads/speaker2.wav my_voices/
```

**Step 2: Encode voices to `.pt` embeddings**

Use the provided encoder script:

```bash
uv run python scripts/encode_voices_kugelaudio.py --input my_voices --output voices
```

This reads all `.wav` files, processes them through KugelAudioProcessor, extracts acoustic embeddings, and saves them as `.pt` files in the `voices/` directory. A `voices.json` manifest is also created.

Output:
- `voices/<speaker_name>.pt` — PyTorch archive with speaker embedding
- `voices/voices.json` — metadata manifest

First run will auto-download the KugelAudio model (~17GB, cached in `~/.cache/huggingface/hub/`).

**Step 3: Use custom voices with the API**

Once encoded, reference the voice by name (the filename stem without `.pt`):

```bash
curl -X POST http://127.0.0.1:8003/v1/kugelaudio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Custom voice synthesis.",
    "voice_reference_path": "speaker1",
    "language": "nl",
    "output_filename": "output.mp3"
  }' \
  --output output.mp3
```

**Discovering available voices**

Query the capabilities endpoint to list all available voices (presets + encoded):

```bash
curl http://127.0.0.1:8003/v1/kugelaudio/capabilities | jq .voices
```

## Prosody / SSML Support Matrix

- `chatterbox`: supports `<break .../>` and `<emphasis>...</emphasis>`
- `voxtral`: supports `<break .../>` (no emphasis tag handling)
- `omnivoice`: no SSML tag parsing in this adapter
- `kugelaudio`: no SSML tag parsing in this adapter; adapter forwards `speaker_names`/`ref_audio` when provided
- `higgs`: no SSML XML parser in adapter, but token-based controls are supported in prompt text
- `moss`: no SSML XML parser in adapter, but explicit pause token control is supported in prompt text
- `vibevoice`: no SSML tag parser in adapter; use speaker-routing and emotion variants

VibeVoice control model:

- primary control is speaker routing, not SSML emotion tags
- `segments[]` + `emotion` are mapped to 4 variant speaker slots
- generated prompt format is `Speaker N: ...`
- optional stage directions (for example `[excited]`) are prepended as a secondary cue
- if the primary VibeVoice checkpoint fails to load for TTS, the backend automatically tries fallback model IDs

Optional fallback override:

```dotenv
VIBEVOICE_MODEL_FALLBACKS=mlx-community/VibeVoice-bf16,mlx-community/VibeVoice-TTS-bf16
```

Example payload:

```json
{
  "text": "Welkom iedereen",
  "voice_profile": "sarah",
  "segments": [
    {"text": "Welkom iedereen", "emotion": "calm"},
    {"text": "Dit is fantastisch nieuws!", "emotion": "excited"}
  ],
  "use_stage_directions": true,
  "language": "nl",
  "output_filename": "vibevoice_sample.mp3"
}
```

Higgs token controls (prompt-level):

- format: `<|category:value|>`
- emotion: `elation`, `amusement`, `enthusiasm`, `determination`, `pride`, `contentment`, `affection`, `relief`, `contemplation`, `confusion`, `surprise`, `awe`, `longing`, `arousal`, `anger`, `fear`, `disgust`, `bitterness`, `sadness`, `shame`, `helplessness`
- style: `singing`, `shouting`, `whispering`
- sound effects: `cough`, `laughter`, `crying`, `screaming`, `burping`, `humming`, `sigh`, `sniff`, `sneeze`
- prosody: `speed_very_slow`, `speed_slow`, `speed_fast`, `speed_very_fast`, `pause`, `long_pause`, `pitch_low`, `pitch_high`, `expressive_high`, `expressive_low`

Examples:

- `<|emotion:elation|>`
- `<|style:whispering|>`
- `<|sfx:laughter|> haha`
- `<|prosody:pause|>`

MOSS pause control (prompt-level):

- format: `[pause X.Ys]`
- example: `Hallo [pause 0.8s] hoe gaat het?`

## Markdown-to-Prosody Conversion

When enabled, request text is preprocessed before synthesis on SSML-capable adapters:

- `**bold**` and `*italic*` (also `__bold__` and `_italic_`) become `<emphasis>...</emphasis>` on backends with emphasis support
- single newline becomes `<break time="...ms"/>`
- blank line (paragraph break) becomes a longer `<break time="...ms"/>`

Defaults were tuned for natural pacing:

- single newline: `350ms`
- paragraph break: `900ms`

You can override these via `.env` (or shell env vars):

```dotenv
TTS_MARKDOWN_PROSODY_ENABLED=1
TTS_MARKDOWN_LINE_BREAK_MS=350
TTS_MARKDOWN_PARAGRAPH_BREAK_MS=900
```

An example template is provided in `.env.example`.

Example (works best on `chatterbox`):

```json
{
  "text": "Welkom.\\nDit is **heel belangrijk**.\\n\\nVolgende alinea.",
  "voice_reference_path": "voices/my_voice.wav",
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
- The current shared seam is `src/api_shared.py`; backend-specific synthesis lives in `src/server_voxtral.py`, `src/server_chatterbox.py`, `src/server_omnivoice.py`, `src/server_kugelaudio.py`, `src/server_higgs.py`, and `src/server_moss.py`.
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
