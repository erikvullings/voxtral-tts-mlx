#!/usr/bin/env python3
"""Generate VibeVoice diagnostic audio sets for noise/hallucination analysis.

The script runs one or more VibeVoice API backends, synthesizes a fixed test matrix,
and stores all outputs under audio_tests/diagnostics/vibevoice/.

Usage examples:
    uv run python scripts/diagnose_vibevoice_noise.py
    uv run python scripts/diagnose_vibevoice_noise.py --backends vibevoice-7b-coreml
    uv run python scripts/diagnose_vibevoice_noise.py --skip-single-speaker
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_DIALOGUE = (
    "Speaker 1: Welkom bij onze uitzending. Vandaag bespreken we taal en technologie.\\n"
    "Speaker 2: Dank je wel. Ik kijk ernaar uit om hierover te praten.\\n"
    "Speaker 1: Laten we beginnen met de eerste vraag.\\n"
    "Speaker 2: Een heldere uitspraak maakt luisteren prettig en natuurlijk."
)

DEFAULT_SINGLE_SPEAKER_TEXT = (
    "Speaker 1: Dit is een test met meerdere zinnen. "
    "De stem moet in elke zin hetzelfde blijven. "
    "We controleren ook tempo, klank en uitspraak."
)


@dataclass(frozen=True)
class DiagnosticCase:
    name: str
    backend: str
    diffusion_steps: int
    cfg_scale: float
    seed: int
    solver: str
    quantize: int | None = None
    quantize_diffusion: bool = False


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
            details = exc.read().decode("utf-8", errors="ignore").strip()
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


def _start_backend(repo_root: Path, backend: str, port: int) -> subprocess.Popen[str]:
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
    env.setdefault("TRANSFORMERS_CACHE", str((repo_root / ".hf-cache" / "transformers").resolve()))
    env.pop("PYTORCH_MPS_LOW_WATERMARK_RATIO", None)
    env.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    return subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _wait_until_ready(base_url: str, process: subprocess.Popen[str], timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            output = ""
            if process.stdout is not None:
                try:
                    output = process.stdout.read() or ""
                except Exception:
                    output = ""
            raise RuntimeError(
                f"Backend exited before readiness (code={process.returncode}).\\n{output.strip()}"
            )

        try:
            req = Request(f"{base_url}/openapi.json", method="GET")
            with urlopen(req, timeout=5):
                return
        except (URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1.0)

    raise TimeoutError(f"Backend did not become ready in {timeout_s:.0f}s")


def _stop_backend(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()
        process.wait(timeout=5)


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def _synthesize_case(
    *,
    base_url: str,
    backend: str,
    ref_audio: list[str],
    text: str,
    case: DiagnosticCase,
    out_path: Path,
    request_timeout_s: float,
) -> None:
    payload: dict[str, object] = {
        "model": backend,
        "input": text,
        "voice": "default",
        "response_format": "mp3",
        "use_stage_directions": False,
        "ref_audio": ref_audio,
        "diffusion_steps": case.diffusion_steps,
        "cfg_scale": case.cfg_scale,
        "seed": case.seed,
        "solver": case.solver,
        "quantize_diffusion": case.quantize_diffusion,
    }
    if case.quantize is not None:
        payload["quantize"] = case.quantize

    raw, content_type = _http_post_json_bytes(
        f"{base_url}/v1/audio/speech",
        payload,
        timeout=request_timeout_s,
    )
    if not _is_mp3(raw, content_type):
        snippet = raw[:200].decode("utf-8", errors="ignore")
        raise RuntimeError(f"Unexpected response type '{content_type}': {snippet}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)


def build_cases(
    backends: list[str], seeds: list[int], *, quick: bool = False
) -> list[DiagnosticCase]:
    cases: list[DiagnosticCase] = []

    for backend in backends:
        cases.append(
            DiagnosticCase(
                name=(
                    f"{backend}_quality_steps20_cfg13_seed42"
                    if quick
                    else f"{backend}_baseline_steps10_cfg13_seed42"
                ),
                backend=backend,
                diffusion_steps=20 if quick else 10,
                cfg_scale=1.3,
                seed=42,
                solver="dpm",
                quantize=None,
                quantize_diffusion=False,
            )
        )

        if quick:
            continue

        cases.append(
            DiagnosticCase(
                name=f"{backend}_quality_steps20_cfg13_seed42",
                backend=backend,
                diffusion_steps=20,
                cfg_scale=1.3,
                seed=42,
                solver="dpm",
                quantize=None,
                quantize_diffusion=False,
            )
        )

        for cfg in (1.15, 1.5):
            cases.append(
                DiagnosticCase(
                    name=f"{backend}_steps20_cfg{str(cfg).replace('.', '')}_seed42",
                    backend=backend,
                    diffusion_steps=20,
                    cfg_scale=cfg,
                    seed=42,
                    solver="dpm",
                    quantize=None,
                    quantize_diffusion=False,
                )
            )

        for seed in seeds:
            cases.append(
                DiagnosticCase(
                    name=f"{backend}_seed_sweep_steps20_cfg13_seed{seed}",
                    backend=backend,
                    diffusion_steps=20,
                    cfg_scale=1.3,
                    seed=seed,
                    solver="dpm",
                    quantize=None,
                    quantize_diffusion=False,
                )
            )

    return cases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VibeVoice noise diagnostic samples")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=[
            "vibevoice-1.5b-coreml",
            "vibevoice-1.5b+coreml",
            "vibevoice-7b-coreml",
            "vibevoice-7b+coreml",
        ],
        default=["vibevoice-1.5b-coreml"],
        help="Backends to test",
    )
    parser.add_argument(
        "--ref-audio",
        nargs="+",
        default=["voices/bart.wav", "voices/anouk.wav"],
        help="Reference clips for two-speaker dialogue",
    )
    parser.add_argument(
        "--single-speaker-refs",
        nargs="+",
        default=["voices/bart.wav", "voices/anouk.wav"],
        help="Single-speaker references to test in isolation",
    )
    parser.add_argument(
        "--dialogue-text",
        default=DEFAULT_DIALOGUE,
        help="Dialogue test text",
    )
    parser.add_argument(
        "--single-speaker-text",
        default=DEFAULT_SINGLE_SPEAKER_TEXT,
        help="Single-speaker isolation text",
    )
    parser.add_argument(
        "--out-dir",
        default="audio_tests/diagnostics/vibevoice",
        help="Output directory",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=8120,
        help="Base port; increments per backend",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=240.0,
        help="Seconds to wait for backend startup",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1200.0,
        help="Seconds to wait for each synthesis request",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Seed sweep for quality config",
    )
    parser.add_argument(
        "--skip-single-speaker",
        action="store_true",
        help="Skip per-reference single-speaker isolation tests",
    )
    parser.add_argument(
        "--skip-dialogue",
        action="store_true",
        help="Skip the male/female dialogue cases",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate one dialogue sample per backend, plus single-speaker samples",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args.backends, args.seeds, quick=args.quick)
    cases_by_backend: dict[str, list[DiagnosticCase]] = {}
    for case in cases:
        cases_by_backend.setdefault(case.backend, []).append(case)

    failures: list[str] = []

    for backend_index, backend in enumerate(args.backends):
        backend_port = args.base_port + backend_index
        base_url = f"http://127.0.0.1:{backend_port}"
        print(f"[start] backend={backend} port={backend_port}")

        process = _start_backend(repo_root, backend, backend_port)
        try:
            _wait_until_ready(base_url, process, args.startup_timeout)

            for case in ([] if args.skip_dialogue else cases_by_backend.get(backend, [])):
                target = out_dir / backend / f"{_slug(case.name)}.mp3"
                print(f"[run] {backend}: {case.name}")
                try:
                    _synthesize_case(
                        base_url=base_url,
                        backend=backend,
                        ref_audio=args.ref_audio,
                        text=args.dialogue_text,
                        case=case,
                        out_path=target,
                        request_timeout_s=args.request_timeout,
                    )
                    print(f"[ok]  {target}")
                except Exception as exc:
                    message = f"{backend}/{case.name}: {exc}"
                    failures.append(message)
                    print(f"[fail] {message}", file=sys.stderr)

            if not args.skip_single_speaker:
                for ref in args.single_speaker_refs:
                    ref_name = _slug(Path(ref).stem)
                    case = DiagnosticCase(
                        name=f"{backend}_single_{ref_name}_steps20_cfg13_seed42",
                        backend=backend,
                        diffusion_steps=20,
                        cfg_scale=1.3,
                        seed=42,
                        solver="dpm",
                        quantize=None,
                        quantize_diffusion=False,
                    )
                    target = out_dir / backend / "single_speaker" / f"{_slug(case.name)}.mp3"
                    print(f"[run] {backend}: single-speaker ref={ref}")
                    try:
                        _synthesize_case(
                            base_url=base_url,
                            backend=backend,
                            ref_audio=[ref],
                            text=args.single_speaker_text,
                            case=case,
                            out_path=target,
                            request_timeout_s=args.request_timeout,
                        )
                        print(f"[ok]  {target}")
                    except Exception as exc:
                        message = f"{backend}/{case.name}: {exc}"
                        failures.append(message)
                        print(f"[fail] {message}", file=sys.stderr)

        finally:
            _stop_backend(process)

    manifest = {
        "dialogueText": args.dialogue_text,
        "singleSpeakerText": args.single_speaker_text,
        "backends": args.backends,
        "refAudio": args.ref_audio,
        "singleSpeakerRefs": args.single_speaker_refs,
        "seeds": args.seeds,
        "generatedAtEpochSeconds": int(time.time()),
        "failures": failures,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    if failures:
        print("\nCompleted with failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"\nDone. Diagnostic samples generated in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
