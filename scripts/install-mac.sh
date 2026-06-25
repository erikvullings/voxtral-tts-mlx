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
Usage: bash scripts/install-mac.sh [--run] [--port <port>]

Options:
  --run          Start the API server after setup
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

echo "Syncing dependencies with uv..."
uv sync

echo "Setup complete."
echo "Run server with: uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload"

if [[ "$RUN_SERVER" == "true" ]]; then
  echo "Starting server on port ${PORT}..."
  exec uv run uvicorn server:app --host 0.0.0.0 --port "$PORT" --reload
fi
