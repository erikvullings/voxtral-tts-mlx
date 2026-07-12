from __future__ import annotations

from server_vibevoice_mlx_base import VibeVoiceVariant, create_vibevoice_mlx_app

app = create_vibevoice_mlx_app(
    VibeVoiceVariant(
        backend_tag="vibevoice-7b+coreml",
        route_prefix="vibevoice-7b-coreml",
        model_id="gafiatulin/vibevoice-7b-mlx",
        coreml_semantic=True,
    )
)
