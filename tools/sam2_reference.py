#!/usr/bin/env python
"""Dump per-stage SAM 2 reference tensors (CPU, float32) for the Rust port's reference tests.

Usage: python -I tools/sam2_reference.py MODEL_DIR OUT.safetensors
MODEL_DIR holds the facebook/sam2.1-hiera-tiny files (config.json, model.safetensors, ...).
Versions are pinned in tools/requirements-sam2-reference.txt. The input is synthetic, so no
media is committed; the weights are never copied into the repo.
"""
import sys

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Sam2Model, Sam2Processor


def synthetic_image(size=256):
    """A textured disc over a textured background (deterministic)."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    bg = 0.3 + 0.2 * np.sin(xx / 9.0) * np.cos(yy / 7.0)
    disc = np.hypot(xx - size * 0.45, yy - size * 0.5) < size * 0.22
    img = np.stack([bg, bg * 0.8 + 0.1, bg * 0.6 + 0.3], -1)
    img[disc] = np.stack([0.85 + 0.1 * np.sin(xx / 5), 0.4 + 0.1 * np.cos(yy / 5), 0.15 + 0 * xx], -1)[disc]
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    return (img.clip(0, 1) * 255).astype(np.uint8)


def main(model_dir, out):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = Sam2Model.from_pretrained(model_dir).eval()
    proc = Sam2Processor.from_pretrained(model_dir)
    image = synthetic_image()
    inputs = proc(images=image, input_points=[[[[115.0, 128.0]]]], input_labels=[[[1]]], return_tensors="pt")
    taps = {}

    def tap(name):
        def hook(_m, _i, o):
            outs = o if isinstance(o, (tuple, list)) else [o]
            for k, t in enumerate(outs):
                if isinstance(t, torch.Tensor):
                    taps[f"{name}.{k}"] = t.detach().float().contiguous()
        return hook

    # Stage taps by module path: patch embed, each Hiera block, the neck, prompt encoder, decoder.
    for name, mod in model.named_modules():
        parts = name.split(".")
        if name in ("vision_encoder.backbone.patch_embed", "vision_encoder.neck", "prompt_encoder", "mask_decoder") or (
            len(parts) == 4 and parts[:3] == ["vision_encoder", "backbone", "blocks"]
        ):
            mod.register_forward_hook(tap(name))
    with torch.no_grad():
        res = model(**inputs, multimask_output=True)
    taps["input.pixel_values"] = inputs["pixel_values"].float()
    taps["out.iou_scores"] = res.iou_scores.float()
    taps["out.pred_masks"] = res.pred_masks.float()
    save_file({k: v for k, v in taps.items()}, out)
    print(f"wrote {len(taps)} tensors to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
