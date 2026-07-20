"""Run the community VibeVoice inference stack with API-supplied references."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--community-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ref-audio", nargs="*", default=[])
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.pop("PYTORCH_MPS_LOW_WATERMARK_RATIO", None)
    os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    community_root = Path(args.community_root).resolve()
    if not community_root.is_dir():
        raise RuntimeError(f"Community VibeVoice checkout not found: {community_root}")
    sys.path.insert(0, str(community_root))

    import torch

    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = VibeVoiceProcessor.from_pretrained(args.model)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        device_map=None,
    )
    model.to(device)
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.diffusion_steps)

    text = Path(args.text_file).read_text(encoding="utf-8").replace("’", "'")
    inputs = processor(
        text=[text],
        voice_samples=[args.ref_audio] if args.ref_audio else None,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=args.cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=True,
        is_prefill=bool(args.ref_audio),
    )
    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        raise RuntimeError("Community VibeVoice returned no audio")
    processor.save_audio(outputs.speech_outputs[0], output_path=args.output)


if __name__ == "__main__":
    main()
