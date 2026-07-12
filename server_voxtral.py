# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "pydantic",
#     "requests",
#     "soundfile",
#     "pyrubberband",
# ]
# ///

import os
import re
import threading
from typing import Any, Iterable, Optional, Protocol, cast

import numpy as np
import soundfile as sf
from api_shared import create_app


class _TTSModel(Protocol):
    def generate(self, *, text: str, voice: str) -> Iterable[Any]: ...


class RealVoxtralEngine:
    MLX_MODEL_ID = "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit"
    PRESET_VOICES = {
        "casual_male",
        "casual_female",
        "cheerful_female",
        "neutral_male",
        "neutral_female",
        "fr_male",
        "fr_female",
        "es_male",
        "es_female",
        "de_male",
        "de_female",
        "it_male",
        "it_female",
        "pt_male",
        "pt_female",
        "nl_male",
        "nl_female",
        "ar_male",
        "hi_male",
        "hi_female",
    }
    VOICES_DIR = "voices"

    def __init__(self):
        self._model: Optional[_TTSModel] = None
        self._lock = threading.Lock()
        self._is_warmed_up = False

    def _warm_up_model(self, model: _TTSModel) -> None:
        if self._is_warmed_up:
            return

        try:
            # One-time throwaway inference to stabilize first-token decoding noise.
            for _ in model.generate(text=".", voice="nl_female"):
                pass
            self._is_warmed_up = True
            print("🔥 Voxtral warm-up completed.")
        except Exception as e:
            # Warm-up should never block synthesis; keep serving with cleanup fallback.
            print(f"⚠️ Voxtral warm-up skipped: {e}")

    def _load_model(self):
        with self._lock:
            if self._model is None:
                print(f"📦 Loading Voxtral MLX model ({self.MLX_MODEL_ID})...")
                from mlx_audio.tts.utils import load

                self._model = cast(_TTSModel, load(self.MLX_MODEL_ID))
                print("✅ Model loaded.")

            model = self._model
            if model is not None and not self._is_warmed_up:
                self._warm_up_model(model)

    def _resolve_voice(self, voice: Optional[str]) -> str:
        """Resolve a Voxtral preset voice name.

        The MLX Voxtral backend only supports its built-in preset IDs and does
        not support local reference-audio voice cloning.
        """
        if not voice:
            return "nl_female"

        normalized = voice.strip().lower()
        if normalized in self.PRESET_VOICES:
            return normalized

        # Detect local reference-audio requests so we can fail clearly.
        candidates = [normalized, voice]
        if "/" not in voice and "\\" not in voice:
            candidates.extend(
                [
                    os.path.join(self.VOICES_DIR, normalized),
                    os.path.join(self.VOICES_DIR, voice),
                ]
            )
        for candidate in candidates:
            if candidate.lower().endswith(".wav") and os.path.isfile(candidate):
                raise ValueError(
                    "Local voice reference files are not supported by the Voxtral backend. "
                    "Use a preset voice or run the Chatterbox backend (server_chatterbox.py) "
                    "for voice cloning."
                )
            wav_candidate = f"{candidate}.wav"
            if os.path.isfile(wav_candidate):
                raise ValueError(
                    "Local voice reference files are not supported by the Voxtral backend. "
                    "Use a preset voice or run the Chatterbox backend (server_chatterbox.py) "
                    "for voice cloning."
                )

        raise ValueError(
            f"Unknown voice '{voice}', expected one of {sorted(self.PRESET_VOICES)}"
        )

    def list_voices(self) -> list[str]:
        return sorted(self.PRESET_VOICES)

    def _cleanup_start_artifact(
        self, audio: np.ndarray, sample_rate: int = 24000
    ) -> np.ndarray:
        if audio.size == 0:
            return audio

        # Detect stable speech onset and skip low-level startup hiss.
        max_scan = min(int(0.35 * sample_rate), audio.size)
        window_samples = max(1, int(0.008 * sample_rate))
        squared = audio[:max_scan] ** 2
        kernel = np.ones(window_samples, dtype=np.float32) / window_samples
        moving_power = np.convolve(squared, kernel, mode="valid")
        moving_rms = np.sqrt(np.maximum(moving_power, 0.0))

        trim_idx = 0
        if moving_rms.size > 0:
            floor_region = moving_rms[: max(1, int(0.05 * sample_rate))]
            noise_floor = float(np.percentile(floor_region, 30))
            peak_rms = float(np.max(moving_rms))

            # Adaptive threshold: above local floor, but still catches soft speech.
            onset_threshold = max(0.0035, noise_floor * 3.5, peak_rms * 0.08)
            active = moving_rms > onset_threshold

            sustain_samples = max(1, int(0.012 * sample_rate))
            sustained = np.convolve(
                active.astype(np.int32),
                np.ones(sustain_samples, dtype=np.int32),
                mode="valid",
            )
            sustained_idx = np.flatnonzero(sustained >= sustain_samples)

            if sustained_idx.size > 0:
                # Keep a tiny lead-in so plosives are not clipped.
                lead_in = int(0.004 * sample_rate)
                trim_idx = max(0, int(sustained_idx[0]) - lead_in)

        if trim_idx > 0:
            audio = audio[trim_idx:]

        if audio.size == 0:
            return audio

        # Suppress quiet startup hiss in the first 40 ms.

        gate_samples = min(int(0.04 * sample_rate), audio.size)
        gate_threshold = 0.0012
        if gate_samples > 0:
            prefix = audio[:gate_samples]
            audio[:gate_samples] = np.where(
                np.abs(prefix) < gate_threshold, 0.0, prefix
            )

        # Force absolute silence for a tiny prefix to eliminate encoder boundary pops.
        hard_mute_samples = min(int(0.006 * sample_rate), audio.size)
        if hard_mute_samples > 0:
            audio[:hard_mute_samples] = 0.0

        # Smooth fade-in to avoid audible transition from gating to speech.
        fade_samples = min(int(0.03 * sample_rate), audio.size)
        if fade_samples > 1:
            audio[:fade_samples] *= np.linspace(
                0.0, 1.0, fade_samples, dtype=audio.dtype
            )

        return audio

    def _trim_warmup_prefix(
        self, audio: np.ndarray, sample_rate: int = 24000
    ) -> np.ndarray:
        """Drop a small MLX warmup prefix before the main speech starts."""
        if audio.size == 0:
            return audio

        # Voxtral MLX can emit a short noisy lead-in before the first stable speech.
        # Trim a tiny fixed prefix so the adaptive cleanup does not have to preserve it.
        warmup_trim_samples = min(int(0.04 * sample_rate), audio.size)
        if warmup_trim_samples <= 0:
            return audio

        return audio[warmup_trim_samples:]

    def _apply_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """Pitch-preserving time-stretch via Rubber Band Library (pyrubberband)."""
        speed = max(0.25, min(4.0, speed))
        if abs(speed - 1.0) < 0.001 or audio.size == 0:
            return audio
        import pyrubberband

        # Prepend silence so the stretcher has frames to stabilize before speech starts.
        # The stretched pre-roll is pad/speed samples long and is trimmed afterwards.
        pad = int(0.1 * 24000)  # 100 ms
        padded = np.concatenate([np.zeros(pad, dtype=audio.dtype), audio])
        stretched = pyrubberband.time_stretch(padded, 24000, speed).astype(audio.dtype)
        return stretched[round(pad / speed) :]

    @staticmethod
    def _split_markup_segments(text: str) -> list[tuple[str, float]]:
        """Split input into speech chunks and pauses.

        Supports explicit break tags and paragraph pauses (blank lines).
        """
        normalized = text.replace("\r\n", "\n")
        normalized = re.sub(r"\n\s*\n+", ' <break time="450ms"/> ', normalized)

        break_pattern = re.compile(
            r'<break\s+time="(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s)"\s*/?>',
            flags=re.IGNORECASE,
        )

        segments: list[tuple[str, float]] = []
        cursor = 0
        for match in break_pattern.finditer(normalized):
            piece = normalized[cursor : match.start()].strip()
            val = float(match.group("value"))
            unit = match.group("unit").lower()
            pause_s = val / 1000.0 if unit == "ms" else val
            pause_s = max(0.0, min(3.0, pause_s))
            segments.append((piece, pause_s))
            cursor = match.end()

        tail = normalized[cursor:].strip()
        if tail or not segments:
            segments.append((tail, 0.0))

        return segments

    def synthesize(
        self, text: str, voice_path: Optional[str], output_path: str, **kwargs
    ) -> str:
        self._load_model()
        model = self._model
        if model is None:
            raise RuntimeError("Voxtral model failed to initialize")
        voice = self._resolve_voice(voice_path)

        speed = float(kwargs.get("speed", 1.0))
        segment_plan = self._split_markup_segments(text)

        stitched: list[np.ndarray] = []
        inter_segment_gap = np.zeros(int(0.08 * 24000), dtype=np.float32)
        for idx, (segment_text, pause_s) in enumerate(segment_plan):
            if segment_text:
                part_chunks: list[np.ndarray] = []
                for result in model.generate(text=segment_text, voice=voice):
                    part_chunks.append(np.array(result.audio))

                part_audio = (
                    np.concatenate(part_chunks)
                    if part_chunks
                    else np.zeros(0, dtype=np.float32)
                )
                part_audio = self._apply_speed(part_audio, speed)
                part_audio = self._trim_warmup_prefix(part_audio, sample_rate=24000)
                part_audio = self._cleanup_start_artifact(part_audio, sample_rate=24000)

                if part_audio.size > 0:
                    if idx > 0 and stitched:
                        stitched.append(inter_segment_gap)
                    stitched.append(part_audio)

            if pause_s > 0:
                stitched.append(np.zeros(int(24000 * pause_s), dtype=np.float32))

        audio = np.concatenate(stitched) if stitched else np.zeros(0, dtype=np.float32)

        if output_path.endswith(".mp3"):
            # Python 3.14 removed audioop; avoid pydub and write MP3 via soundfile if available.
            if "MP3" in sf.available_formats():
                sf.write(output_path, audio, samplerate=24000, format="MP3")
                return output_path

            wav_path = output_path[:-4] + ".wav"
            sf.write(wav_path, audio, samplerate=24000, format="WAV")
            return wav_path

        sf.write(output_path, audio, samplerate=24000, format="WAV")
        return output_path


voxtral_engine = RealVoxtralEngine()
app = create_app(
    title="Voxtral TTS Translation Layer",
    engine=voxtral_engine,
    voice_response_model=list[str],
    supports_ssml_breaks=True,
    backend_capabilities={
        "model": RealVoxtralEngine.MLX_MODEL_ID,
        "voiceCloning": {
            "supported": False,
            "notes": "Preset voice IDs only; local reference audio is not supported.",
        },
        "voiceSelection": {
            "type": "preset",
            "presets": sorted(RealVoxtralEngine.PRESET_VOICES),
        },
        "ssmlProsody": {
            "tagParsing": True,
            "supportedTags": ["<break time=\"Xms\"/>", "<break time=\"Xs\"/>"] ,
            "notes": "Break tags are parsed by the adapter. Emphasis tags are not interpreted.",
        },
        "languageConditioning": {
            "apiDefaultLanguage": None,
            "notes": "Language field is accepted but this adapter does not require it for synthesis.",
        },
    },
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_voxtral:app", host="0.0.0.0", port=8000, reload=True)
