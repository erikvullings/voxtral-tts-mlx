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
