#!/usr/bin/env python3
"""Clean voice reference WAV files with ffmpeg for stable cloning quality.

Default processing chain:
- mono downmix
- resample to 24 kHz
- high-pass and low-pass filtering
- mild declick
- loudness normalization
- PCM 16-bit WAV output

Examples:
    uv run python scripts/preprocess_voice_refs.py --all
    uv run python scripts/preprocess_voice_refs.py --input voices/new_voice.wav
    uv run python scripts/preprocess_voice_refs.py --all --in-place
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_FILTER = "highpass=f=70,lowpass=f=11000,adeclick,loudnorm=I=-20:TP=-2:LRA=7"


def _discover_inputs(voices_dir: Path, inputs: list[str], include_all: bool) -> list[Path]:
    paths: list[Path] = []

    for value in inputs:
        p = Path(value)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists() and p.is_file() and p.suffix.lower() == ".wav":
            paths.append(p)

    if include_all:
        paths.extend(sorted((voices_dir.resolve()).glob("*.wav")))

    unique: dict[str, Path] = {}
    for p in paths:
        unique[str(p.resolve())] = p.resolve()
    return sorted(unique.values())


def _run_ffmpeg(ffmpeg: str, src: Path, dst: Path, af: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "24000",
        "-af",
        af,
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}: {proc.stderr.strip()}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess voice reference WAV files")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input WAV file path (repeatable)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every .wav file in voices/",
    )
    parser.add_argument(
        "--voices-dir",
        default="voices",
        help="Directory containing source voice WAV files",
    )
    parser.add_argument(
        "--out-dir",
        default="voices/clean",
        help="Output directory when not using --in-place",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite source files instead of writing to --out-dir",
    )
    parser.add_argument(
        "--suffix",
        default="-clean",
        help="Output suffix before .wav when not using --in-place",
    )
    parser.add_argument(
        "--af",
        default=DEFAULT_FILTER,
        help="ffmpeg audio filter chain",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without running ffmpeg",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg is required. Install with: brew install ffmpeg", file=sys.stderr)
        return 2

    voices_dir = Path(args.voices_dir)
    if not voices_dir.is_absolute():
        voices_dir = Path.cwd() / voices_dir

    inputs = _discover_inputs(voices_dir, args.input, args.all)
    if not inputs:
        print("No input WAV files found. Use --input <file.wav> or --all", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir

    failures: list[str] = []

    for src in inputs:
        if args.in_place:
            dst = src
        else:
            dst = out_dir / f"{src.stem}{args.suffix}.wav"

        print(f"[plan] {src} -> {dst}")
        if args.dry_run:
            continue

        try:
            if args.in_place:
                temp_dst = dst.with_name(dst.stem + ".tmp-clean.wav")
                _run_ffmpeg(ffmpeg, src, temp_dst, args.af)
                temp_dst.replace(dst)
            else:
                _run_ffmpeg(ffmpeg, src, dst, args.af)
            print(f"[ok]   {dst}")
        except Exception as exc:
            failures.append(str(exc))
            print(f"[fail] {exc}", file=sys.stderr)

    if failures:
        print("\nCompleted with failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
