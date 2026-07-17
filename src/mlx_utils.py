"""MLX utility functions for warmup and logit penalties.

These utilities can be reused across different MLX-based TTS backends.
"""

from typing import Any, Callable, Optional

import mlx.core as mx


def apply_logit_penalty(
    model: Any,
    token_id: int,
    penalty_strength: float = 5.0,
) -> Any:
    """Patch a model to penalize a specific token during generation.

    This prevents the model from prematurely stopping or over-producing
    a particular token. Commonly used to prevent cutoff at sentence ends
    by penalizing a speech_end_id token.

    Args:
        model: The MLX model instance with a get_lm_logits method
        token_id: Token ID to penalize (e.g., speech_end_id=151653 for KugelAudio)
        penalty_strength: Penalty value to subtract from the logit (default: 5.0)

    Returns:
        The patched model (same object, modified in-place)

    Example:
        >>> from server_kugelaudio import RealKugelAudioEngine
        >>> model = engine._load_model()
        >>> # Prevent speech cutoff by penalizing token 151653
        >>> apply_logit_penalty(model, token_id=151653, penalty_strength=5.0)
    """
    original_get_lm_logits: Callable[[mx.array], mx.array] = model.get_lm_logits

    def patched_get_lm_logits(hidden_states: mx.array) -> mx.array:
        logits = original_get_lm_logits(hidden_states)
        # Apply penalty to target token to discourage its selection
        # logits shape: (batch_size, seq_len, vocab_size)
        # We penalize all positions' predictions for this token
        if logits.ndim >= 1 and logits.shape[-1] > token_id:
            logits[..., token_id] -= penalty_strength
        return logits

    model.get_lm_logits = patched_get_lm_logits
    return model


def warmup_mlx_model(
    model: Any,
    generate_fn: Callable[..., Any],
    **generate_kwargs: Any,
) -> None:
    """Warm up an MLX model by running a minimal generation.

    On Apple Silicon, MLX uses lazy evaluation — computation graphs are
    compiled the first time they are executed. Without a warmup, the first
    real synthesis call incurs extra latency and can produce noise while
    Metal shaders compile. Running a dummy generation here forces all paths
    to compile once at startup.

    Args:
        model: The MLX model instance (primarily used for optional cleanup)
        generate_fn: Callable that invokes model.generate(...) or similar
        **generate_kwargs: Arguments to pass to generate_fn (e.g., text, voice, max_tokens)

    Example:
        >>> from mlx_utils import warmup_mlx_model
        >>> def warmup_kugel(m):
        ...     return warmup_mlx_model(
        ...         m,
        ...         lambda: (t for t in m.generate(text="Hi.", voice="default", max_tokens=10)),
        ...     )
    """
    print("🔥 Warming up MLX model...")
    try:
        # Run the generation and exhaust the iterator
        result = generate_fn(**generate_kwargs)
        if hasattr(result, "__iter__"):
            for _ in result:
                pass
        # Clear the Metal cache to release compilation memory
        try:
            mx.clear_cache()
        except Exception:  # pylint: disable=broad-except
            pass
        print("✅ MLX model warmup complete.")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"⚠️  MLX model warmup failed (non-fatal): {exc}")
