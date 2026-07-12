#!/usr/bin/env python3
"""Generate Dutch audio samples across all supported TTS backends.

This script starts each backend one-by-one, calls /v1/audio/speech,
and writes MP3 outputs to audio_tests/<backend>_text.mp3.

Usage:
    uv run python scripts/generate_backend_audio_samples.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_TEXT = (
    'Het is morgen. De zon schijnt in de straat. Een man loopt buiten. '
    'Hij ziet een vrouw. Zij is nieuw in de buurt. De man zegt: "Goedemorgen!"\n\n'
    'De vrouw lacht. Zij groet de man ook. De man stelt zich voor. '
    '"Mijn naam is Thomas. Hoe heet jij?" De vrouw zegt: "Ik ben Sarah."\n\n'
    'Thomas vraagt hoe het gaat. Sarah zegt dat alles goed gaat. '
    '"Kijk eens naar die mooie bloemen in je tuin," zegt Thomas.\n\n'
    'Nu loopt Sarah verder. Zij zegt: "Tot ziens!" Thomas antwoordt ook met een groet. '
    '"Tot ziens, Sarah". De buren zijn vriendelijk.'
)

MULTI_SPEAKER_TEXT = (
    "Speaker 1: Goedemorgen Sarah, welkom in de buurt. Ik ben Thomas.\n"
    "Speaker 2: Dank je Thomas, wat een fijne straat is dit.\n"
    "Speaker 1: Kijk, de zon schijnt en de bloemen staan mooi in bloei vandaag.\n"
    "Speaker 2: Ja, ik voel me meteen thuis. Tot straks, buurman!"
)


class BackendConfig(TypedDict):
    voice: str
    default_language: str | None
    extra_payload: dict[str, object]
    multi_speaker_extra_payload: dict[str, object]


BACKENDS: dict[str, BackendConfig] = {
    "voxtral": {
        "voice": "nl_female",
        "default_language": "nl",
        "extra_payload": {},
        "multi_speaker_extra_payload": {},
    },
    "chatterbox": {
        "voice": "nl_female",
        "default_language": "nl",
        "extra_payload": {},
        "multi_speaker_extra_payload": {},
    },
    # Zero-shot cloning backends: pass local female reference audio.
    "omnivoice": {
        "voice": "voices/nl_female.wav",
        "default_language": "nl",
        "extra_payload": {
            "ref_text": "Goedemorgen, ik ben Sarah en ik ben nieuw in de buurt.",
        },
        "multi_speaker_extra_payload": {
            "ref_text": "Goedemorgen, ik ben Sarah en ik ben nieuw in de buurt.",
        },
    },
    "higgs": {
        "voice": "voices/nl_female.wav",
        "default_language": "nl",
        "extra_payload": {
            "ref_text": "Goedemorgen, ik ben Sarah en ik ben nieuw in de buurt.",
        },
        "multi_speaker_extra_payload": {
            "ref_text": "Goedemorgen, ik ben Sarah en ik ben nieuw in de buurt.",
        },
    },
    "moss": {
        "voice": "voices/nl_female.wav",
        "default_language": "Dutch",
        "extra_payload": {
            "ref_text": "Goedemorgen, ik ben Sarah en ik ben nieuw in de buurt.",
        },
        "multi_speaker_extra_payload": {
            "ref_text": "Goedemorgen, ik ben Sarah en ik ben nieuw in de buurt.",
        },
    },
    # KugelAudio only supports preset voices (default/warm/clear).
    "kugelaudio": {
        "voice": "warm",
        "default_language": "nl",
        "extra_payload": {},
        "multi_speaker_extra_payload": {
            "do_sample": False,
        },
    },
    "vibevoice": {
        "voice": "default",
        "default_language": None,
        "extra_payload": {
            "ref_audio": ["voices/bart.wav", "voices/anouk.wav"],
        },
        "multi_speaker_extra_payload": {
            "ref_audio": ["voices/bart.wav", "voices/anouk.wav"],
        },
    },
    "vibevoice-1.5b+coreml": {
        "voice": "default",
        "default_language": None,
        "extra_payload": {
            "ref_audio": ["voices/bart.wav", "voices/anouk.wav"],
        },
        "multi_speaker_extra_payload": {
            "ref_audio": ["voices/bart.wav", "voices/anouk.wav"],
        },
    },
    "vibevoice-1.5b-coreml": {
        "voice": "default",
        "default_language": None,
        "extra_payload": {
            "ref_audio": ["voices/bart.wav", "voices/anouk.wav"],
        },
        "multi_speaker_extra_payload": {
            "ref_audio": ["voices/bart.wav", "voices/anouk.wav"],
        },
    },
    "vibevoice-7b+coreml": {
        "voice": "default",
        "default_language": None,
        "extra_payload": {
            "ref_audio": ["voices/bart.wav", "voices/anouk.wav"],
            "quantize": 8,
        },
        "multi_speaker_extra_payload": {
            "ref_audio": ["voices/bart.wav", "voices/anouk.wav"],
            "quantize": 8,
        },
    },
    "vibevoice-7b-coreml": {
        "voice": "default",
        "default_language": None,
        "extra_payload": {
            "ref_audio": ["voices/bart.wav", "voices/anouk.wav"],
            "quantize": 8,
        },
        "multi_speaker_extra_payload": {
            "ref_audio": ["voices/bart.wav", "voices/anouk.wav"],
            "quantize": 8,
        },
    },
}


def _http_get_json(url: str, timeout: float) -> dict:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def _http_post_json_bytes(url: str, payload: dict, timeout: float) -> tuple[bytes, str]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        details = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore")
            details = body.strip()
        except Exception:
            details = ""
        if details:
            raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
        raise RuntimeError(f"HTTP {exc.code}: {exc.reason}") from exc
    return raw, content_type


def _is_mp3(raw: bytes, content_type: str) -> bool:
    lower_type = content_type.lower()
    if "audio/mpeg" in lower_type or "audio/mp3" in lower_type:
        return True
    if raw.startswith(b"ID3"):
        return True
    return raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")


def _is_wav(raw: bytes, content_type: str) -> bool:
    lower_type = content_type.lower()
    if "audio/wav" in lower_type or "audio/x-wav" in lower_type:
        return True
    return raw.startswith(b"RIFF") and raw[8:12] == b"WAVE"


def _convert_wav_bytes_to_mp3(raw_wav: bytes, out_mp3: Path) -> None:
    temp_wav = out_mp3.with_suffix(".tmp.wav")
    temp_wav.write_bytes(raw_wav)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", str(temp_wav), str(out_mp3)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        temp_wav.unlink(missing_ok=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr.strip()}")
        return

    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover - env-specific fallback
        temp_wav.unlink(missing_ok=True)
        raise RuntimeError(
            "Response was WAV and ffmpeg is unavailable; cannot convert to mp3. "
            "Install ffmpeg (brew install ffmpeg)."
        ) from exc

    formats = sf.available_formats()
    if "MP3" not in formats:
        temp_wav.unlink(missing_ok=True)
        raise RuntimeError(
            "Response was WAV and this soundfile build cannot write MP3. "
            "Install ffmpeg (brew install ffmpeg)."
        )

    audio, sample_rate = sf.read(str(temp_wav), dtype="float32")
    sf.write(str(out_mp3), audio, sample_rate, format="MP3")
    temp_wav.unlink(missing_ok=True)


def _start_backend(repo_root: Path, backend: str, port: int) -> subprocess.Popen[str]:
    if backend == "chatterbox":
        chatterbox_python = repo_root / ".venv-chatterbox" / "bin" / "python"
        if not chatterbox_python.exists():
            raise RuntimeError(
                "Chatterbox backend requires .venv-chatterbox. "
                "Run: bash scripts/install-mac.sh --backend chatterbox"
            )
        cmd = [
            str(chatterbox_python),
            "-m",
            "uvicorn",
            "server_chatterbox:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    else:
        cmd = [
            "uv",
            "run",
            "tts",
            "--backend",
            backend,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str((repo_root / ".uv-cache").resolve()))
    env.setdefault("HF_HOME", str((repo_root / ".hf-cache").resolve()))
    env.setdefault(
        "TRANSFORMERS_CACHE",
        str((repo_root / ".hf-cache" / "transformers").resolve()),
    )
    env.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "1.4")
    env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.7")

    return subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _scrub_ansi(text: str) -> str:
    ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    return ansi_re.sub("", text)


def _read_process_output(process: subprocess.Popen[str]) -> str:
    if process.stdout is None:
        return ""
    try:
        return process.stdout.read() or ""
    except Exception:
        return ""


def _wait_until_ready(
    base_url: str, timeout_s: float, process: subprocess.Popen[str], backend: str
) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raw_output = _read_process_output(process)
            clean_output = _scrub_ansi(raw_output).strip()
            if clean_output:
                raise RuntimeError(
                    f"{backend} exited before readiness (code={process.returncode}). "
                    f"Startup log:\n{clean_output}"
                )
            raise RuntimeError(
                f"{backend} exited before readiness (code={process.returncode})"
            )
        try:
            _http_get_json(f"{base_url}/openapi.json", timeout=5)
            return
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(1.0)
    raise TimeoutError(
        f"Backend {backend} did not become ready in {timeout_s:.0f}s: {last_error}"
    )


def _stop_backend(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            # Kill the full process group to stop uvicorn workers too.
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()
        process.wait(timeout=5)


def _render_target_path(out_dir: Path, backend: str, text_variant: str) -> Path:
    suffix = "multispeaker_text" if text_variant == "multi-speaker" else "text"
    return out_dir / f"{backend}_{suffix}.mp3"


def generate_for_backend(
    *,
    repo_root: Path,
    backend: str,
    config: BackendConfig,
    text: str,
    port: int,
    out_dir: Path,
    startup_timeout_s: float,
    request_timeout_s: float,
    language_override: str | None,
    text_variant: str,
) -> Path:
    base_url = f"http://127.0.0.1:{port}"
    process = _start_backend(repo_root, backend, port)

    try:
        _wait_until_ready(base_url, timeout_s=startup_timeout_s, process=process, backend=backend)

        effective_language = language_override
        if effective_language is None:
            effective_language = config.get("default_language")

        payload = {
            "model": backend,
            "input": text,
            "voice": config["voice"],
            "response_format": "mp3",
            "speed": 1.0,
        }
        if effective_language:
            payload["language"] = effective_language
        payload.update(config.get("extra_payload", {}))
        if text_variant == "multi-speaker":
            payload.update(config.get("multi_speaker_extra_payload", {}))
        raw, content_type = _http_post_json_bytes(
            f"{base_url}/v1/audio/speech",
            payload,
            timeout=request_timeout_s,
        )

        out_mp3 = _render_target_path(out_dir, backend, text_variant)
        if _is_mp3(raw, content_type):
            out_mp3.write_bytes(raw)
        elif _is_wav(raw, content_type):
            _convert_wav_bytes_to_mp3(raw, out_mp3)
        else:
            snippet = raw[:120].decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Unexpected response type '{content_type}' for {backend}: {snippet}"
            )

        return out_mp3
    finally:
        _stop_backend(process)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparable Dutch TTS samples across all backends."
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to synthesize (default: built-in Dutch sample)",
    )
    parser.add_argument(
        "--out-dir",
        default="audio_tests",
        help="Output directory for generated mp3 files",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=sorted(BACKENDS),
        default=sorted(BACKENDS),
        help="Subset of backends to run",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="Base port; script increments per backend",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for backend startup",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for TTS response per backend",
    )
    parser.add_argument(
        "--language",
        default=None,
        help=(
            "Force one language value for all backends (examples: nl, en, Dutch). "
            "If omitted, backend-specific defaults are used."
        ),
    )
    parser.add_argument(
        "--no-language",
        action="store_true",
        help="Do not send a language field in the request payload.",
    )
    parser.add_argument(
        "--text-variant",
        choices=["default", "multi-speaker"],
        default="default",
        help="Choose single-speaker default text or multi-speaker dialogue text.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_text = MULTI_SPEAKER_TEXT if args.text_variant == "multi-speaker" else args.text

    selected_backends = list(args.backends)
    language_override = None if args.no_language else args.language
    failures: list[str] = []

    for index, backend in enumerate(selected_backends):
        port = args.port + index
        print(f"[start] backend={backend} port={port}")
        try:
            out_path = generate_for_backend(
                repo_root=repo_root,
                backend=backend,
                config=BACKENDS[backend],
                text=selected_text,
                port=port,
                out_dir=out_dir,
                startup_timeout_s=args.startup_timeout,
                request_timeout_s=args.request_timeout,
                language_override=language_override,
                text_variant=args.text_variant,
            )
            print(f"[ok] backend={backend} file={out_path}")
        except Exception as exc:
            message = f"{backend}: {exc}"
            failures.append(message)
            print(f"[fail] {message}", file=sys.stderr)

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"\nDone. Generated {len(selected_backends)} file(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
