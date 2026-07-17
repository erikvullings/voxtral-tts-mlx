#!/usr/bin/env python3
"""
Encode WAV voice files to KugelAudio .pt voice embeddings.

Usage:
    uv run python scripts/encode_voices_kugelaudio.py [--input DIR] [--output DIR]

Example:
    uv run python scripts/encode_voices_kugelaudio.py --input my_voices --output voices

This script reads WAV files from --input (default: "my_voices"), processes them through
KugelAudioProcessor to extract speaker embeddings, and saves them as PyTorch .pt files
in --output (default: "voices/"). Each output file can then be referenced by the
KugelAudio TTS backend via the voice_reference_path parameter.

Prerequisites:
    - KugelAudio model must be cached: huggingface-hub will auto-download on first run
      (~17GB, stored in ~/.cache/huggingface/hub/)
    - Voice input files should be clean, single-speaker WAV clips (0.5–3 seconds recommended)
"""

import json
import torch
from pathlib import Path
from kugelaudio_open import KugelAudioProcessor


def create_voices(wav_folder="my_voices", output_dir="voices"):
    """
    Convert WAV files to KugelAudio voice embeddings.

    Args:
        wav_folder: Directory containing .wav voice reference files
        output_dir: Directory to write .pt embeddings and voices.json manifest
    """
    processor = KugelAudioProcessor.from_pretrained("kugelaudio/kugelaudio-0-open")
    wav_path = Path(wav_folder)
    voices_path = Path(output_dir)
    voices_path.mkdir(exist_ok=True)

    if not wav_path.exists():
        print(f"Error: Input directory '{wav_folder}' does not exist.")
        return

    wav_files = sorted(wav_path.glob("*.wav"))
    if not wav_files:
        print(f"Error: No .wav files found in '{wav_folder}'.")
        return

    voices_data = {}

    for wav_file in wav_files:
        voice_name = wav_file.stem.lower().replace(" ", "_").replace("-", "_")
        pt_path = voices_path / f"{voice_name}.pt"

        print(f"Processing {wav_file.name} → {voice_name}.pt")

        try:
            # Load audio through KugelAudioProcessor
            audio_inputs = processor.audio_processor(str(wav_file), return_tensors="pt")

            # Find the audio tensor (key name varies by processor version)
            audio_tensor = None
            for key in ["audio", "input_values", "input_features", "values"]:
                if key in audio_inputs:
                    audio_tensor = audio_inputs[key]
                    break

            if audio_tensor is None:
                # Fallback: take first tensor value
                for v in audio_inputs.values():
                    if isinstance(v, torch.Tensor):
                        audio_tensor = v
                        break

            if audio_tensor is None:
                raise ValueError("Could not find audio tensor in processor output")

            # Mean pooling to create speaker embedding
            # Remove extra dimensions and average over time
            if audio_tensor.dim() > 2:
                embedding = audio_tensor.mean(dim=1)  # average over time
            else:
                embedding = audio_tensor.mean(dim=0) if audio_tensor.dim() == 2 else audio_tensor

            # Ensure 2D (batch x features)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            torch.save(embedding, pt_path)
            print(f"✓ Saved {pt_path} (shape: {embedding.shape})")

            voices_data[voice_name] = {
                "file": f"{voice_name}.pt",
                "description": f"Custom voice from {wav_file.name}",
                "language": "nl",
            }

        except Exception as e:
            print(f"✗ Failed {wav_file.name}: {e}")

    # Save voices.json manifest
    with open(voices_path / "voices.json", "w", encoding="utf-8") as f:
        json.dump(voices_data, f, indent=2)

    print(f"\nFinished! Created {len(voices_data)} voice embeddings in ./{voices_path}/")
    print(
        f"Use any of the {len(voices_data)} voice names with the KugelAudio API:\n"
        f'  curl -X POST http://localhost:8003/v1/kugelaudio/speech \\\n'
        f'    -d \'{{"text": "...", "voice_reference_path": "{list(voices_data.keys())[0]}"}}\'  \\\n'
        f'    --output output.mp3'
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Encode WAV voice files to KugelAudio .pt embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        default="my_voices",
        help="Input directory with .wav files (default: my_voices)",
    )
    parser.add_argument(
        "--output",
        default="voices",
        help="Output directory for .pt embeddings (default: voices)",
    )
    args = parser.parse_args()

    create_voices(wav_folder=args.input, output_dir=args.output)
