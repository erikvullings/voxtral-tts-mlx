from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from api_shared import OpenAISpeechRequest, VoxtralExtendedRequest, create_app
from request_normalizer import build_stage_directions, normalize_request_text
from server_mlx_audio import write_audio_output


@dataclass(frozen=True)
class VibeVoiceVariant:
    backend_tag: str
    route_prefix: str
    model_id: str
    coreml_semantic: bool


class VibeVoiceSegment(BaseModel):
    text: str
    emotion: Optional[str] = None
    speaker: Optional[int] = None


class VibeVoiceMlxOpenAISpeechRequest(OpenAISpeechRequest):
    model: str = "gafiatulin/vibevoice-7b-mlx"
    voice: str = "default"
    emotion: Optional[str] = None
    segments: list[VibeVoiceSegment] | None = None
    use_stage_directions: bool = False
    speaker_names: list[str] | None = None
    ref_audio: list[str] | None = None
    max_speech_tokens: int | None = None
    diffusion_steps: int | None = None
    cfg_scale: float | None = None
    seed: int | None = None
    solver: str | None = None
    quantize: int | None = None
    quantize_diffusion: bool = False


class VibeVoiceMlxSpeechRequest(VoxtralExtendedRequest):
    voice_reference_path: Optional[str] = "default"
    emotion: Optional[str] = None
    segments: list[VibeVoiceSegment] | None = None
    use_stage_directions: bool = False
    speaker_names: list[str] | None = None
    ref_audio: list[str] | None = None
    max_speech_tokens: int | None = None
    diffusion_steps: int | None = None
    cfg_scale: float | None = None
    seed: int | None = None
    solver: str | None = None
    quantize: int | None = None
    quantize_diffusion: bool = False


class VibeVoiceMlxEngine:
    MIN_REFERENCE_SECONDS = 3.0
    RECOMMENDED_REFERENCE_SECONDS = 8.0

    EMOTION_MAP = {
        "neutral": "normal",
        "normal": "normal",
        "happy": "excited",
        "excited": "excited",
        "angry": "shouting",
        "sad": "calm",
        "serious": "calm",
        "calm": "calm",
    }

    def __init__(self, variant: VibeVoiceVariant):
        self.variant = variant

    @staticmethod
    def _voice_stems() -> list[str]:
        voices_dir = Path("voices")
        if not voices_dir.exists():
            return []
        stems = sorted({p.stem for p in voices_dir.glob("*.wav")})
        return stems

    @staticmethod
    def _resolve_voice_file(name: str) -> Optional[str]:
        candidate = Path(name)
        if candidate.exists() and candidate.is_file():
            return str(candidate)

        voices_dir = Path("voices")
        for option in [
            voices_dir / name,
            voices_dir / f"{name}.wav",
        ]:
            if option.exists() and option.is_file():
                return str(option)
        return None

    def _resolve_speaker_names(
        self,
        *,
        voice_path: Optional[str],
        speaker_names: list[str] | None,
    ) -> list[str]:
        if speaker_names:
            cleaned = [s.strip() for s in speaker_names if s and s.strip()]
            if cleaned:
                return cleaned

        if voice_path and voice_path.strip():
            # Single-speaker default when caller picks one explicit voice.
            return [voice_path.strip()]

        # Default two-speaker map with explicit male/female preference.
        defaults = ["bart", "anouk"]

        available = set(self._voice_stems())
        resolved: list[str] = []
        for name in defaults:
            if name in available:
                resolved.append(name)
            elif available:
                resolved.append(sorted(available)[0])

        if not resolved:
            return defaults

        # Ensure at least two speakers for multi-speaker prompts.
        if len(resolved) == 1:
            resolved.append(resolved[0])
        return resolved

    def _resolve_ref_audio(self, speaker_names: list[str], explicit: Any) -> list[str]:
        refs: list[str] = []
        if isinstance(explicit, list):
            for item in explicit:
                path = self._resolve_voice_file(str(item))
                if path:
                    refs.append(path)

        if refs:
            return refs

        for name in speaker_names:
            path = self._resolve_voice_file(name)
            if path:
                refs.append(path)
        return refs

    @staticmethod
    def _auto_max_speech_tokens(prompt: str, speaker_count: int, requested: Any) -> int:
        if requested is not None:
            try:
                value = int(requested)
                return max(64, min(5000, value))
            except (TypeError, ValueError):
                pass

        words = len(prompt.split())

        # Typical speech averages around 2-3 words/sec.
        estimated_seconds = max(4.0, words / 2.2)

        # VibeVoice generates at about 7.5 speech tokens/sec.
        estimated_tokens = int(estimated_seconds * 7.5)

        # Add a modest turn-transition margin for multi-speaker dialogue.
        margin = 60 + max(0, speaker_count - 1) * 20
        return max(100, min(5000, estimated_tokens + margin))

    @staticmethod
    def _normalize_diffusion_steps(requested: Any) -> int:
        if requested is None:
            return 20
        try:
            value = int(requested)
        except (TypeError, ValueError):
            raise ValueError("VibeVoice diffusion_steps must be an integer") from None
        return max(5, min(50, value))

    @staticmethod
    def _normalize_cfg_scale(requested: Any) -> float:
        if requested is None:
            return 1.3
        try:
            return float(requested)
        except (TypeError, ValueError):
            raise ValueError("VibeVoice cfg_scale must be a float") from None

    @staticmethod
    def _normalize_seed(requested: Any) -> int:
        if requested is None:
            return 42
        try:
            return int(requested)
        except (TypeError, ValueError):
            raise ValueError("VibeVoice seed must be an integer") from None

    @staticmethod
    def _normalize_solver(requested: Any) -> str:
        if requested is None:
            return "dpm"
        value = str(requested).strip().lower()
        allowed = {"dpm", "sde", "ddpm"}
        if value not in allowed:
            raise ValueError("VibeVoice solver must be one of: dpm, sde, ddpm")
        return value

    def _validate_reference_audio(self, ref_audio: list[str]) -> None:
        if not ref_audio:
            return
        try:
            import soundfile as sf
        except Exception:
            return

        too_short: list[str] = []
        for path in ref_audio:
            try:
                info = sf.info(path)
            except Exception:
                continue

            if not info.samplerate:
                continue
            duration = float(info.frames) / float(info.samplerate)
            if duration < self.MIN_REFERENCE_SECONDS:
                too_short.append(f"{Path(path).name} ({duration:.2f}s)")

        if too_short:
            details = ", ".join(too_short)
            raise ValueError(
                "Reference audio too short: "
                f"{details}. Provide at least {self.MIN_REFERENCE_SECONDS:.0f}s of speech "
                f"(recommended {self.RECOMMENDED_REFERENCE_SECONDS:.0f}-{10:.0f}s)."
            )

    @staticmethod
    def _vibevoice_command() -> list[str]:
        local_pkg = Path(__file__).resolve().parent / "vibevoice_mlx" / "e2e_pipeline.py"
        if local_pkg.exists():
            return ["uv", "run", "python", "-m", "vibevoice_mlx.e2e_pipeline"]

        cloned_repo = Path("/tmp/vibevoice-mlx") / "vibevoice_mlx" / "e2e_pipeline.py"
        if cloned_repo.exists():
            return [
                "uv",
                "run",
                "--directory",
                "/tmp/vibevoice-mlx",
                "python",
                "-m",
                "vibevoice_mlx.e2e_pipeline",
            ]

        if shutil_which("vibevoice-mlx"):
            return ["vibevoice-mlx"]

        raise RuntimeError(
            "VibeVoice-MLX runner not found. Expected one of:\n"
            "1) vendored package at ./vibevoice_mlx\n"
            "2) local clone at /tmp/vibevoice-mlx\n"
            "3) installed CLI 'vibevoice-mlx'"
        )

    def _build_prompt(
        self,
        *,
        text: str,
        emotion: Optional[str],
        segments: Any,
        use_stage_directions: bool,
        language: Optional[str],
        speaker_count: int,
    ) -> str:
        if isinstance(segments, list) and segments:
            parsed: list[VibeVoiceSegment] = []
            for item in segments:
                if isinstance(item, dict):
                    seg = VibeVoiceSegment.model_validate(item)
                elif isinstance(item, VibeVoiceSegment):
                    seg = item
                else:
                    continue
                parsed.append(seg)

            zero_based = any(
                seg.speaker is not None and int(seg.speaker) == 0 for seg in parsed
            )

            lines: list[str] = []
            for seg in parsed:
                content = normalize_request_text(seg.text)
                if not content:
                    continue
                if use_stage_directions and seg.emotion:
                    content = normalize_request_text(
                        content,
                        stage_directions=build_stage_directions(seg.emotion),
                    )
                speaker_idx = int(seg.speaker) if seg.speaker is not None else 1
                if zero_based:
                    speaker_idx += 1
                speaker_idx = max(1, min(4, speaker_idx))
                lines.append(f"Speaker {speaker_idx}: {content}")
            if lines:
                return "\n".join(lines)

        base = normalize_request_text(text)
        if not base:
            raise ValueError("Text input cannot be empty")

        # Normalize transcript speaker labels to 1-based format expected by upstream prompts.
        def _normalize_speaker_label(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            if idx <= 0:
                return "Speaker 1:"
            return f"Speaker {idx}:"

        has_speaker_labels = bool(re.search(r"(?im)^\s*speaker\s*\d+\s*:", base))
        base = re.sub(r"(?im)^\s*speaker\s*(\d+)\s*:", _normalize_speaker_label, base)
        if speaker_count <= 1 and not has_speaker_labels:
            lines = [ln.strip() for ln in base.splitlines() if ln.strip()]
            if lines:
                base = "\n".join(f"Speaker 1: {ln}" for ln in lines)
            else:
                base = f"Speaker 1: {base}"

        mapped = self.EMOTION_MAP.get((emotion or "").strip().lower(), "normal")
        if use_stage_directions and mapped:
            base = normalize_request_text(base, stage_directions=build_stage_directions(mapped))
        return base

    def list_voices(self) -> dict[str, Any]:
        stems = self._voice_stems()
        return {
            "fromVoicesFolder": stems,
            "defaultPair": ["bart", "anouk"],
        }

    @staticmethod
    def _normalize_quantize(requested: Any) -> Optional[int]:
        if requested is None:
            return None
        try:
            value = int(requested)
        except (TypeError, ValueError):
            raise ValueError("VibeVoice quantize must be 4 or 8") from None
        if value not in {4, 8}:
            raise ValueError("VibeVoice quantize must be 4 or 8")
        return value

    def synthesize(
        self,
        text: str,
        voice_path: Optional[str],
        output_path: str,
        **kwargs: Any,
    ) -> str:
        speaker_names = self._resolve_speaker_names(
            voice_path=voice_path,
            speaker_names=kwargs.get("speaker_names"),
        )
        ref_audio = self._resolve_ref_audio(speaker_names, kwargs.get("ref_audio"))

        prompt = self._build_prompt(
            text=text,
            emotion=kwargs.get("emotion"),
            segments=kwargs.get("segments"),
            use_stage_directions=bool(kwargs.get("use_stage_directions", False)),
            language=kwargs.get("language"),
            speaker_count=len(speaker_names),
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_output = tmp.name

        cmd = [
            *self._vibevoice_command(),
            "--model",
            self.variant.model_id,
            "--text",
            prompt,
            "--output",
            temp_output,
        ]

        if self.variant.coreml_semantic:
            cmd.append("--coreml-semantic")

        cmd.extend([
            "--diffusion-steps",
            str(self._normalize_diffusion_steps(kwargs.get("diffusion_steps"))),
            "--cfg-scale",
            str(self._normalize_cfg_scale(kwargs.get("cfg_scale"))),
            "--seed",
            str(self._normalize_seed(kwargs.get("seed"))),
            "--solver",
            self._normalize_solver(kwargs.get("solver")),
            "--silence-detection",
            "--trim-trailing-silence",
        ])

        quantize = self._normalize_quantize(kwargs.get("quantize"))
        if quantize is not None:
            cmd.extend(["--quantize", str(quantize)])

        if bool(kwargs.get("quantize_diffusion", False)):
            cmd.append("--quantize-diffusion")

        max_speech_tokens = self._auto_max_speech_tokens(
            prompt,
            speaker_count=len(speaker_names),
            requested=kwargs.get("max_speech_tokens"),
        )
        cmd.extend(["--max-speech-tokens", str(max_speech_tokens)])

        if ref_audio:
            self._validate_reference_audio(ref_audio)
            cmd.append("--ref-audio")
            cmd.extend(ref_audio)

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"vibevoice-mlx failed: {proc.stderr.strip()}")

        try:
            import soundfile as sf

            audio, sample_rate = sf.read(temp_output, dtype="float32")
        finally:
            try:
                os.unlink(temp_output)
            except OSError:
                pass

        return write_audio_output(output_path, audio, sample_rate=int(sample_rate))


def shutil_which(program: str) -> Optional[str]:
    import shutil

    return shutil.which(program)


def create_vibevoice_mlx_app(variant: VibeVoiceVariant):
    engine = VibeVoiceMlxEngine(variant)
    return create_app(
        title=f"VibeVoice MLX ({variant.backend_tag}) Translation Layer",
        engine=engine,
        voice_response_model=dict[str, Any],
        route_prefix=variant.route_prefix,
        openai_request_model=VibeVoiceMlxOpenAISpeechRequest,
        extended_request_model=VibeVoiceMlxSpeechRequest,
        supports_ssml_emphasis=False,
        supports_ssml_breaks=False,
        backend_capabilities={
            "backendTag": variant.backend_tag,
            "model": variant.model_id,
            "coremlSemantic": variant.coreml_semantic,
            "voiceCloning": {
                "supported": True,
                "inputs": ["ref_audio", "voice", "speaker_names"],
                "refAudioCliMapping": "ref_audio -> --ref-audio",
            },
            "speakerRouting": {
                "supported": True,
                "maxSpeakers": 4,
                "routingFormat": "Speaker N: ...",
                "speakerIndexing": "1-based in prompts",
                "notes": "Preferred flow matches upstream CLI: text transcript with Speaker 1/Speaker 2 labels plus ref_audio list.",
                "singleSpeakerDefaults": {
                    "whenVoiceProvidedWithoutSpeakerNames": "use-single-speaker",
                    "autoLabelWhenMissing": "Speaker 1",
                },
            },
            "voiceReferenceResolution": {
                "voiceField": "voice",
                "speakerNamesField": "speaker_names",
                "refAudioField": "ref_audio",
                "resolutionOrder": [
                    "explicit ref_audio",
                    "speaker_names -> voices/<name>.wav",
                    "voice -> single speaker and voices/<voice>.wav",
                    "default pair bart/anouk",
                ],
            },
            "generationBudgeting": {
                "maxSpeechTokens": {
                    "clientConfigRequired": False,
                    "auto": True,
                    "defaultPolicy": "auto-by-estimated-duration-and-speaker-count",
                    "clampRange": [100, 5000],
                    "overrideField": "max_speech_tokens",
                }
            },
            "diffusion": {
                "requestFields": ["diffusion_steps", "cfg_scale", "seed", "solver"],
                "defaults": {
                    "diffusionSteps": 20,
                    "cfgScale": 1.3,
                    "seed": 42,
                    "solver": "dpm",
                },
                "ranges": {
                    "diffusionSteps": [5, 50],
                },
            },
            "silenceHandling": {
                "enabledByDefault": True,
                "cliFlags": ["--silence-detection", "--trim-trailing-silence"],
            },
            "quantization": {
                "supported": True,
                "llmBackbone": [4, 8],
                "diffusionHeadInt8": True,
                "requestFields": ["quantize", "quantize_diffusion"],
            },
            "languageConditioning": {
                "apiDefaultLanguage": None,
                "notes": "No adapter-level language prompt is injected; follow upstream-style transcript prompting.",
            },
            "ssmlProsody": {
                "tagParsing": False,
                "supportedTags": [],
                "notes": "Use plain transcript speaker routing; stage-direction support is adapter-only and optional.",
            },
        },
    )
