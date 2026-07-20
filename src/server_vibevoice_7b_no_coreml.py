from __future__ import annotations

from server_vibevoice_mlx_base import VibeVoiceVariant, create_vibevoice_mlx_app

app = create_vibevoice_mlx_app(
    VibeVoiceVariant(
        backend_tag="vibevoice-7b-coreml",
        route_prefix="vibevoice-7b",
        model_id="vibevoice/VibeVoice-7B",
        coreml_semantic=False,
    )
)
