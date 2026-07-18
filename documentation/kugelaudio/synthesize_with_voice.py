#!/usr/bin/env python3
"""
Synthesize speech using a KugelAudio voice embedding (.pt file).

Usage:
    python synthesize_with_voice.py [--voice VOICE_NAME] [--text TEXT] [--output FILE]

Example:
    python synthesize_with_voice.py --voice anouk --text "Hello world" --output hello.mp3

This script demonstrates how to use a pre-encoded voice embedding (.pt file)
to generate speech from text. It assumes the voice embedding exists in the
voices/ directory (created by encode_voices_kugelaudio.py).

Run from within the kugelaudio-open directory with the venv active:
    source .venv/bin/activate
    python synthesize_with_voice.py --voice anouk
"""

import argparse
import subprocess
import sys
import torch
from pathlib import Path

try:
    from kugelaudio_open import (
        KugelAudioForConditionalGenerationInference,
        KugelAudioProcessor,
    )
except ImportError:
    print("Error: KugelAudio not installed.")
    print("Make sure you are running from inside the kugelaudio-open directory with:")
    print("  source .venv/bin/activate")
    print("  python synthesize_with_voice.py ...")
    sys.exit(1)


def synthesize_with_voice(
    text: str,
    voice_name: str = "anouk",
    output_path: str = "output.mp3",
    voices_dir: str = "voices",
) -> None:
    """
    Synthesize speech using a pre-encoded voice embedding (.pt file).

    Args:
        text: Text to synthesize
        voice_name: Name of the voice (without .pt extension)
        output_path: Path to save the output audio file
        voices_dir: Directory containing .pt voice files
    """
    voices_path = Path(voices_dir)
    pt_file = voices_path / f"{voice_name}.pt"

    # Validate voice file exists
    if not pt_file.exists():
        available = [p.stem for p in voices_path.glob("*.pt")] if voices_path.exists() else []
        print(f"Error: Voice file not found: {pt_file}")
        if available:
            print(f"Available voices: {', '.join(sorted(available))}")
        else:
            print(f"No .pt files found in '{voices_dir}'. Run encode_voices_kugelaudio.py first.")
        sys.exit(1)

    model_id = "kugelaudio/kugelaudio-0-open"

    print(f"Loading KugelAudio model from {model_id}...")
    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    model.eval()
    model.model.strip_encoders()

    processor = KugelAudioProcessor.from_pretrained(model_id)

    print(f"Loading voice embedding from {pt_file}...")
    voice_cache = torch.load(str(pt_file), map_location="cpu", weights_only=True)
    print(f"   Voice shape: {voice_cache['acoustic_mean'].shape}")

    print(f'Synthesizing: "{text}"')
    inputs = processor(
        text=text,
        voice_cache=voice_cache,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            cfg_scale=3.0,
            max_new_tokens=2048,
        )

    audio = outputs.speech_outputs[0]

    # Determine output path
    output = Path(output_path)
    wav_path = output.with_suffix(".wav")

    processor.save_audio(audio, str(wav_path))
    print(f"Saved audio to {wav_path}")

    # Convert to MP3 if requested and ffmpeg is available
    if output.suffix.lower() == ".mp3":
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), str(output)],
                check=True,
                capture_output=True,
            )
            wav_path.unlink()
            print(f"Converted to MP3: {output}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Note: ffmpeg not available. Audio saved as WAV: {wav_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize speech using a KugelAudio voice embedding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--voice", default="anouk", help="Voice name / .pt file stem (default: anouk)")
    parser.add_argument("--text", default="Hello! This is a test of KugelAudio voice synthesis.", help="Text to synthesize")
    parser.add_argument("--output", default="output.mp3", help="Output file path (default: output.mp3)")
    parser.add_argument("--voices-dir", default="voices", help="Directory containing .pt voice files (default: voices)")
    args = parser.parse_args()

    synthesize_with_voice(
        text=args.text,
        voice_name=args.voice,
        output_path=args.output,
        voices_dir=args.voices_dir,
    )


if __name__ == "__main__":
    main()
