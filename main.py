from __future__ import annotations

import argparse

import uvicorn


BACKEND_APPS = {
    "voxtral": "server_voxtral:app",
    "chatterbox": "server_chatterbox:app",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts",
        description="Start the MLX TTS API with the selected backend.",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_APPS),
        default="voxtral",
        help="Backend entrypoint to serve",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    uvicorn.run(
        BACKEND_APPS[args.backend],
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
