from __future__ import annotations

from typing import Iterable


def normalize_base_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    # Keep intentional paragraph breaks but collapse runs of blank lines.
    compact: list[str] = []
    last_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank:
            if not last_blank:
                compact.append("")
            last_blank = True
        else:
            compact.append(line.strip())
            last_blank = False

    normalized = "\n".join(compact).strip()
    return normalized


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []

    out: list[str] = []
    for item in value:
        token = str(item).strip()
        if token:
            out.append(token)
    return out


def build_higgs_tokens(
    *,
    emotion: str | Iterable[str] | None = None,
    style: str | Iterable[str] | None = None,
    sfx: str | Iterable[str] | None = None,
    prosody: str | Iterable[str] | None = None,
) -> list[str]:
    tokens: list[str] = []
    for item in _as_list(emotion):
        tokens.append(f"<|emotion:{item}|>")
    for item in _as_list(style):
        tokens.append(f"<|style:{item}|>")
    for item in _as_list(sfx):
        tokens.append(f"<|sfx:{item}|>")
    for item in _as_list(prosody):
        tokens.append(f"<|prosody:{item}|>")
    return tokens


def build_stage_directions(
    *labels: str | Iterable[str] | None,
) -> list[str]:
    directions: list[str] = []
    for label in labels:
        for item in _as_list(label):
            directions.append(f"[{item.lower()}]")
    return directions


def normalize_request_text(
    text: str,
    *,
    prefix_tokens: Iterable[str] | None = None,
    stage_directions: Iterable[str] | None = None,
    paragraph_pause_seconds: float | None = None,
    paragraph_pause_format: str | None = None,
) -> str:
    normalized = normalize_base_text(text)
    if not normalized:
        return normalized

    parts: list[str] = []
    for token in prefix_tokens or []:
        token_value = str(token).strip()
        if token_value:
            parts.append(token_value)
    for direction in stage_directions or []:
        direction_value = str(direction).strip()
        if direction_value:
            parts.append(direction_value)

    if parts:
        normalized = f"{' '.join(parts)} {normalized}".strip()

    if paragraph_pause_seconds is not None and paragraph_pause_format:
        pause_seconds = max(0.0, min(10.0, float(paragraph_pause_seconds)))
        pause_token = paragraph_pause_format.format(seconds=pause_seconds)
        normalized = normalized.replace("\n\n", f" {pause_token} ")

    return normalized
