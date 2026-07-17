from __future__ import annotations

import os
from typing import Any, Iterable, Optional

import numpy as np
import soundfile as sf


def resolve_reference_audio_path(voice_path: Optional[str]) -> Optional[str]:
    if not voice_path:
        return None

    candidate = voice_path.strip()
    if not candidate:
        return None

    if os.path.isfile(candidate):
        return candidate

    if "/" not in candidate and "\\" not in candidate:
        for alt in (
            os.path.join("voices", candidate),
            os.path.join("voices", f"{candidate}.wav"),
            f"{candidate}.wav",
        ):
            if os.path.isfile(alt):
                return alt

    return None


def coerce_audio_array(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()

    arr = np.asarray(audio, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr.astype(np.float32, copy=False)


def collect_generation_audio(
    results: Iterable[Any],
    *,
    default_sample_rate: int,
) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    sample_rate = int(default_sample_rate)

    for result in results:
        audio = getattr(result, "audio", None)
        if audio is None:
            continue
        chunks.append(coerce_audio_array(audio))

        result_sample_rate = getattr(result, "sample_rate", None)
        if result_sample_rate is None:
            result_sample_rate = getattr(result, "sr", None)
        if result_sample_rate:
            sample_rate = int(result_sample_rate)

    if not chunks:
        return np.zeros(0, dtype=np.float32), sample_rate

    if len(chunks) == 1:
        return chunks[0], sample_rate

    return np.concatenate(chunks), sample_rate


def write_audio_output(
    output_path: str,
    audio: np.ndarray,
    *,
    sample_rate: int,
) -> str:
    if output_path.endswith(".mp3"):
        if "MP3" in sf.available_formats():
            sf.write(output_path, audio, samplerate=sample_rate, format="MP3")
            return output_path

        wav_path = output_path[:-4] + ".wav"
        sf.write(wav_path, audio, samplerate=sample_rate, format="WAV")
        return wav_path

    sf.write(output_path, audio, samplerate=sample_rate, format="WAV")
    return output_path
