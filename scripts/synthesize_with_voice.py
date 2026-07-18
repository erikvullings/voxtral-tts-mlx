#!/usr/bin/env python3
"""
Synthesize English text using a KugelAudio voice embedding (.pt file).

Usage:
    uv run python synthesize_with_voice.py [--voice VOICE_NAME] [--output FILE]

Example:
    uv run python synthesize_with_voice.py --voice anouk --output hello.mp3

This script demonstrates how to use a pre-encoded voice embedding (.pt file)
to generate speech from text. It assumes the voice embedding exists in the
voices/ directory (created by encode_voices_kugelaudio.py).

Prerequisites:
    - voices/anouk.pt (or other voice .pt file) must exist
    - Run this script in the kugelaudio-open/ directory
"""

import argparse
import sys
import torch
from pathlib import Path

try:
    from kugelaudio_open import KugelAudio, KugelAudioProcessor
except ImportError:
    print("Error: KugelAudio not installed. Run 'uv sync' in the kugelaudio-open directory.")
    sys.exit(1)


def synthesize_with_voice(
    text: str,
    voice_name: str = "anouk",
    output_path: str = "output.mp3",
    voices_dir: str = "voices",
) -> None:
    """
    Synthesize English text using a pre-encoded voice embedding.

    Args:
        text: English text to synthesize
        voice_name: Name of the voice (without .pt extension)
        output_path: Path to save the output MP3 file
        voices_dir: Directory containing .pt voice files
    """
    voices_path = Path(voices_dir)
    pt_file = voices_path / f"{voice_name}.pt"

    # Validate voice file exists
    if not pt_file.exists():
        print(f"Error: Voice file not found: {pt_file}")
        print(f"Available voices in {voices_dir}/:")
        if voices_path.exists():
            for pt in voices_path.glob("*.pt"):
                print(f"  - {pt.stem}")
        sys.exit(1)

    print(f"📦 Loading KugelAudio model...")
    model = KugelAudio.from_pretrained("kugelaudio/kugelaudio-0-open")
    processor = KugelAudioProcessor.from_pretrained("kugelaudio/kugelaudio-0-open")

    print(f"🎤 Loading voice embedding from {pt_file}...")
    voice_embedding = torch.load(pt_file, weights_only=True, map_location="cpu")
    print(f"   Voice shape: {voice_embedding.shape}")

    print(f"🗣️  Synthesizing: \"{text}\"")
    with torch.no_grad():
        # Prepare text input
        inputs = processor(text=text, return_tensors="pt", language="en")

        # Generate speech with the voice embedding
        # The model expects voice embeddings to be passed as part of the generation context
        output = model.generate(
            input_ids=inputs["input_ids"],
            speaker_embeddings=voice_embedding,
            cfg_scale=3.0,
            num_inference_steps=100,
        )

    # Extract and save audio
    audio_values = output["waveform"].squeeze().cpu().numpy()
    sample_rate = model.config.sample_rate if hasattr(model, "config") else 24000

    # Save as WAV first (easier), then optionally convert to MP3
    import soundfile as sf

    wav_path = Path(output_path).with_suffix(".wav")
    sf.write(wav_path, audio_values, sample_rate)
    print(f"✅ Saved audio to {wav_path}")

    # Optional: Convert to MP3 if ffmpeg is available
    if output_path.endswith(".mp3"):
        try:
            import subprocess

            subprocess.run(
                ["ffmpeg", "-i", str(wav_path), "-q:a", "9", "-y", output_path],
                check=True,
                capture_output=True,
            )
            print(f"✅ Converted to MP3: {output_path}")
            wav_path.unlink()  # Remove WAV file
        except (FileNotFoundError, subprocess.CalledProcessError):
            print(f"⚠️  ffmpeg not available; keeping WAV file: {wav_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize English text using a KugelAudio voice embedding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--voice",
        default="anouk",
        help="Voice name / .pt file stem (default: anouk)",
    )
    parser.add_argument(
        "--text",
        default="Hello! This is a test of the KugelAudio voice synthesis system.",
        help="Text to synthesize (default: sample text)",
    )
    parser.add_argument(
        "--output",
        default="output.mp3",
        help="Output file path (default: output.mp3)",
    )
    parser.add_argument(
        "--voices-dir",
        default="voices",
        help="Directory containing .pt voice files (default: voices)",
    )

    args = parser.parse_args()
    synthesize_with_voice(
        text=args.text,
        voice_name=args.voice,
        output_path=args.output,
        voices_dir=args.voices_dir,
    )


if __name__ == "__main__":
    main()
