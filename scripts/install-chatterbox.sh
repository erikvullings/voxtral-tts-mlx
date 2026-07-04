#!/usr/bin/env bash
set -euo pipefail

PORT="8000"
RUN_SERVER="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_SERVER="true"
      shift
      ;;
    --port)
      if [[ $# -lt 2 ]]; then
        echo "Error: --port requires a value"
        exit 1
      fi
      PORT="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/install-chatterbox.sh [--run] [--port <port>]

Options:
  --run          Start the Chatterbox API server after setup
  --port <port>  Port for uvicorn when --run is used (default: 8000)
  -h, --help     Show this help message
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is for macOS only."
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Warning: non-Apple Silicon machine detected."
  echo "This setup is optimized for macOS Apple Silicon and MPS acceleration."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  if [[ -f "$HOME/.local/bin/env" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/.local/bin/env"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv installation did not succeed."
  echo "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install it from https://brew.sh and re-run this script."
  exit 1
fi

if ! brew list rubberband >/dev/null 2>&1; then
  echo "Installing rubberband (required by pyrubberband for speed adjustment)..."
  brew install rubberband
fi

echo "Creating isolated Chatterbox environment (.venv-chatterbox)..."
uv venv --no-project --clear --python 3.11 .venv-chatterbox

echo "Installing Chatterbox-compatible dependencies..."
uv pip install --python .venv-chatterbox/bin/python \
  chatterbox-tts \
  fastapi \
  faster-whisper \
  numpy \
  pydantic \
  pyrubberband \
  requests \
  soundfile \
  uvicorn

echo "Setup complete."
echo "Run server with: .venv-chatterbox/bin/python -m uvicorn server_chatterbox:app --host 0.0.0.0 --port 8000 --reload"

if [[ "$RUN_SERVER" == "true" ]]; then
  echo "Starting Chatterbox server on port ${PORT}..."
  export PYTORCH_MPS_LOW_WATERMARK_RATIO="1.4"
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO="1.7"
  exec .venv-chatterbox/bin/python -m uvicorn server_chatterbox:app --host 0.0.0.0 --port "$PORT" --reload
fi
