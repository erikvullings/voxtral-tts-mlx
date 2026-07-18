# KugelAudio Voice Synthesis: From WAV to Speech

A step-by-step guide to encode custom voices from WAV files and use them for text-to-speech synthesis with KugelAudio.

## Overview

[KugelAudio](https://github.com/Kugelaudio/kugelaudio-open) is an open-source, on-device text-to-speech model that supports **voice cloning** through custom speaker embeddings. This guide walks you through:

1. **Setup**: Clone and install KugelAudio
2. **Encode**: Convert your WAV voice files to `.pt` embeddings
3. **Synthesize**: Generate speech with your custom voices

**Note:** This package includes three files:

- `VOICE_SYNTHESIS_GUIDE.md` (this guide)
- `encode_voices_kugelaudio.py` (voice encoding script)
- `synthesize_with_voice.py` (synthesis script)

After cloning kugelaudio-open, copy `encode_voices_kugelaudio.py` and `synthesize_with_voice.py` into the kugelaudio-open root directory (alongside the original scripts that came with the repo). Alternatively, you can clone my [fork](https://github.com/erikvullings/kugelaudio-open), that already contains these fixes and more.

## Prerequisites

- **Python 3.11+** (verify with `python --version`)
- **uv** package manager ([install here](https://docs.astral.sh/uv/))
- **~20 GB free disk space** (for HuggingFace model cache)
- **WAV voice files** (0.5–3 seconds each, single speaker, mono or stereo)
- Optional: **ffmpeg** (for MP3 conversion; install via `brew install ffmpeg`)

## Step 1: Clone and Setup KugelAudio

```bash
git clone https://github.com/Kugelaudio/kugelaudio-open.git
cd kugelaudio-open
uv sync
source .venv/bin/activate
```

This creates a Python virtual environment and installs KugelAudio dependencies.

**Verification:**

```bash
python -c "from kugelaudio_open import KugelAudio; print('✓ KugelAudio ready')"
```

## Step 2: Prepare Your Voice Files

Create a directory for your voice samples:

```bash
mkdir my_voices
```

Copy or create your WAV files in this directory. Each file should be:

- **Format**: WAV (PCM, mono or stereo)
- **Duration**: 0.5–3 seconds
- **Sample rate**: 16 kHz or 24 kHz preferred (auto-resampled)
- **Content**: Clean speech, minimal background noise
- **Filename**: `speaker_name.wav` (e.g., `anouk.wav`, `john.wav`)

**Example** (if you have WAV files elsewhere):

```bash
cp ~/dev/tts-mlx/voices/*.wav ./my_voices/
```

## Step 3: Encode Voices to `.pt` Embeddings

The `encode_voices_kugelaudio.py` script extracts speaker embeddings from WAV files:

```bash
python encode_voices_kugelaudio.py --input my_voices --output voices
```

**What it does:**

- Reads each `.wav` file from `my_voices/`
- Processes it through KugelAudio's audio encoder
- Computes a speaker embedding (acoustic characteristics)
- Saves as `voices/<speaker_name>.pt`

**Output:**

```text
Processing anouk.wav → anouk.pt
✓ Saved voices/anouk.pt (shape: torch.Size([1, 64]))
Processing john.wav → john.pt
✓ Saved voices/john.pt (shape: torch.Size([1, 64]))
...
Finished! Created 2 voice embeddings in ./voices/
```

## Step 4: Synthesize Speech with Your Voices

Use the `synthesize_with_voice.py` script to generate speech. **Important:** Run it with `uv run` to use the virtual environment:

```bash
# Basic usage (default: "anouk" voice, sample text)
uv run python synthesize_with_voice.py

# Custom voice and text
uv run python synthesize_with_voice.py \
  --voice anouk \
  --text "Good morning! Welcome to KugelAudio." \
  --output greeting.mp3
```

Alternatively, activate the virtual environment once and run without `uv run`:

```bash
source .venv/bin/activate
python synthesize_with_voice.py --voice anouk --text "Good morning!" --output greeting.mp3
```

**Command options:**

- `--voice VOICE_NAME` — Voice to use (default: `anouk`)
- `--text TEXT` — Text to synthesize (default: sample text)
- `--output FILE` — Output file path (default: `output.mp3`)
- `--voices-dir DIR` — Directory with `.pt` files (default: `voices`)

**Output example:**

```text
📦 Loading KugelAudio model...
🎤 Loading voice embedding from voices/anouk.pt...
   Voice shape: torch.Size([1, 64])
🗣️  Synthesizing: "Good morning! Welcome to KugelAudio."
✅ Saved audio to greeting.wav
✅ Converted to MP3: greeting.mp3
```

## Complete Example Workflow

```bash
# 1. Clone and setup
git clone https://github.com/Kugelaudio/kugelaudio-open.git
cd kugelaudio-open
uv sync
source .venv/bin/activate  # Activate the virtual environment

# 2. Prepare voice files
mkdir my_voices
cp /path/to/your/voice_samples/*.wav ./my_voices/. # <== REPLACE

# 3. Encode voices to embeddings
python encode_voices_kugelaudio.py --input my_voices --output voices

# 4. List available voices
ls voices/*.pt | sed 's/.*\///; s/\.pt//'

# 5. Synthesize speech with a voice (environment already activated, so no uv run needed)
python synthesize_with_voice.py \
  --voice anouk \
  --text "Hello world from KugelAudio!" \
  --output hello.mp3

# 6. Play the result
open hello.mp3  # macOS
# or: mpv hello.mp3 / vlc hello.mp3
```

## Troubleshooting

### "KugelAudio not installed" error

The script needs to run in the virtual environment. Choose one:

**Option 1: Use `uv run`** (no activation needed):
```bash
uv run python synthesize_with_voice.py --voice anouk
```

**Option 2: Activate the venv first**:
```bash
source .venv/bin/activate
python synthesize_with_voice.py --voice anouk
```

Then you can run `python` commands without `uv run` for that terminal session.

### "Voice file not found: voices/anouk.pt"

Ensure you've run the encoding script and voices are in the `voices/` directory:

```bash
ls voices/
# Should show: anouk.pt, john.pt, etc.
```

### "KugelAudio not installed"

Re-run setup:

```bash
uv sync
source .venv/bin/activate
```

### Model downloads very slowly

KugelAudio downloads ~17 GB on first run. This is cached in `~/.cache/huggingface/hub/`. Subsequent runs use the cache.

### ffmpeg conversion to MP3 fails

Install ffmpeg:

```bash
brew install ffmpeg  # macOS
apt install ffmpeg   # Ubuntu/Debian
```

Or skip MP3 and use the WAV output instead.

### Audio quality is poor

- Check your voice WAV files for background noise; re-record if needed
- Ensure WAV files are single-speaker clips (not mixed voices)
- Try longer voice samples (1–2 seconds is often better than 0.5 seconds)

## Next Steps

### Integrate into Your Application

Copy the `synthesize_with_voice.py` logic into your app:

```python
from pathlib import Path
import torch
from kugelaudio_open import KugelAudio, KugelAudioProcessor

model = KugelAudio.from_pretrained("kugelaudio/kugelaudio-0-open")
processor = KugelAudioProcessor.from_pretrained("kugelaudio/kugelaudio-0-open")
voice = torch.load("voices/anouk.pt", weights_only=True)

inputs = processor(text="Hello!", return_tensors="pt", language="en")
output = model.generate(input_ids=inputs["input_ids"], speaker_embeddings=voice)
```

### Batch Processing

Encode and synthesize multiple voices/texts. First activate the environment:

```bash
source .venv/bin/activate

for voice in voices/*.pt; do
  name=$(basename $voice .pt)
  python synthesize_with_voice.py --voice $name --output output_$name.mp3
done
```

Or use `uv run` without activating:

```bash
for voice in voices/*.pt; do
  name=$(basename $voice .pt)
  uv run python synthesize_with_voice.py --voice $name --output output_$name.mp3
done
```

### Server Deployment

See [tts-mlx](https://github.com/erikvullings/tts-mlx) for a FastAPI server that exposes KugelAudio + custom voices via REST API.

## References

- **KugelAudio GitHub**: <https://github.com/Kugelaudio/kugelaudio-open>
- **HuggingFace Model**: <https://huggingface.co/kugelaudio/kugelaudio-0-open>
- **tts-mlx** (this repo's FastAPI wrapper): <https://github.com/erikvullings/tts-mlx>

---

Happy voice cloning! 🎤🔊
