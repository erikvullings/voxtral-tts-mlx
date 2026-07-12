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
    quantize: int | None = None
    quantize_diffusion: bool = False


class VibeVoiceMlxEngine:
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

        # Default two-speaker map with explicit male/female preference.
        defaults = ["bart", "anouk"]
        if voice_path and voice_path.strip():
            defaults[1] = voice_path.strip()

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
                return max(160, min(2600, value))
            except (TypeError, ValueError):
                pass

        words = len([w for w in prompt.replace("\n", " ").split(" ") if w.strip()])
        # VibeVoice needs a generous budget for stable multi-line dialogue continuation.
        estimated = int(words * 18.0) + 700 + max(0, speaker_count - 1) * 300
        return max(1200, min(5000, estimated))

    @staticmethod
    def _vibevoice_command() -> list[str]:
        local_pkg = Path("vibevoice_mlx") / "e2e_pipeline.py"
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

        base = re.sub(r"(?im)^\s*speaker\s*(\d+)\s*:", _normalize_speaker_label, base)

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
            use_stage_directions=bool(kwargs.get("use_stage_directions", True)),
            language=kwargs.get("language"),
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
        else:
            cmd.append("--no-semantic")

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
            },
            "generationBudgeting": {
                "maxSpeechTokens": {
                    "clientConfigRequired": False,
                    "auto": True,
                    "defaultPolicy": "auto-by-input-length-and-speaker-count",
                    "clampRange": [1200, 5000],
                    "overrideField": "max_speech_tokens",
                }
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
