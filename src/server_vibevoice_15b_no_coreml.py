from __future__ import annotations

from server_vibevoice_mlx_base import VibeVoiceVariant, create_vibevoice_mlx_app

app = create_vibevoice_mlx_app(
    VibeVoiceVariant(
        backend_tag="vibevoice-1.5b-coreml",
        route_prefix="vibevoice-15b",
        model_id="vibevoice/VibeVoice-1.5B",
        coreml_semantic=False,
    )
)
