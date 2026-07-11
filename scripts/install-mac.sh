#!/usr/bin/env bash
set -euo pipefail

PORT="8000"
RUN_SERVER="false"
BACKEND=""

BACKENDS=("voxtral" "chatterbox")

backend_exists() {
  local needle="$1"
  for candidate in "${BACKENDS[@]}"; do
    if [[ "$candidate" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

prompt_backend() {
  echo "Select backend to install:"
  local idx=1
  for candidate in "${BACKENDS[@]}"; do
    echo "  ${idx}) ${candidate}"
    idx=$((idx + 1))
  done

  while true; do
    printf "Enter choice [1-%d] (default 1): " "${#BACKENDS[@]}"
    read -r choice
    if [[ -z "$choice" ]]; then
      BACKEND="${BACKENDS[0]}"
      break
    fi
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#BACKENDS[@]} )); then
      BACKEND="${BACKENDS[$((choice - 1))]}"
      break
    fi
    echo "Invalid choice: $choice"
  done
}

install_chatterbox_backend() {
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
}

run_backend() {
  local backend="$1"
  local port="$2"

  if [[ "$backend" == "chatterbox" ]]; then
    export PYTORCH_MPS_LOW_WATERMARK_RATIO="1.4"
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO="1.7"
    exec .venv-chatterbox/bin/python -m uvicorn server_chatterbox:app --host 0.0.0.0 --port "$port" --reload
  fi

  exec uv run tts --backend "$backend" --host 0.0.0.0 --port "$port" --reload
}

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
    --backend)
      if [[ $# -lt 2 ]]; then
        echo "Error: --backend requires a value"
        exit 1
      fi
      BACKEND="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/install-mac.sh [--backend <name>] [--run] [--port <port>]

Options:
  --backend <name>  Backend to install/start (voxtral|chatterbox). If omitted,
                    the script asks interactively.
  --run             Start the selected API backend after setup
  --port <port>     Port for --run (default: 8000)
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
  echo "This project is optimized for Apple Silicon MLX and may not work as expected."
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

if [[ -z "$BACKEND" ]]; then
  prompt_backend
fi

BACKEND="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
if ! backend_exists "$BACKEND"; then
  echo "Unsupported backend: $BACKEND"
  echo "Supported backends: ${BACKENDS[*]}"
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install it from https://brew.sh and re-run this script."
  exit 1
fi

if ! brew list rubberband >/dev/null 2>&1; then
  echo "Installing rubberband (required by pyrubberband for speed adjustment)..."
  brew install rubberband
fi

echo "Syncing dependencies with uv..."
uv sync

if [[ "$BACKEND" == "chatterbox" ]]; then
  install_chatterbox_backend
fi

echo "Setup complete."
if [[ "$BACKEND" == "chatterbox" ]]; then
  echo "Run server with: .venv-chatterbox/bin/python -m uvicorn server_chatterbox:app --host 0.0.0.0 --port 8000 --reload"
else
  echo "Run server with: uv run tts --backend $BACKEND --host 0.0.0.0 --port 8000 --reload"
fi

if [[ "$RUN_SERVER" == "true" ]]; then
  echo "Starting $BACKEND backend on port ${PORT}..."
  run_backend "$BACKEND" "$PORT"
fi
