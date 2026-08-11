#!/usr/bin/env python3
"""Single-image SAM inference with a box prompt.

Runs inside the sam3 Docker container.
Usage:
    python3 SAM_box.py --image /path/to/image.jpg --box x1,y1,x2,y2 [--output /output/dir]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# 模型加载（单例）
# ---------------------------------------------------------------------------

_processor = None


def get_processor(checkpoint: str = "/workspace/model/sam3.pt", device: str = "cuda"):
    """Load SAM3 model and return a Sam3Processor (cached)."""
    global _processor
    if _processor is not None:
        return _processor

    print(f"Loading SAM model from {checkpoint}...", file=sys.stderr)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(checkpoint_path=checkpoint, device=device, eval_mode=True)
    _processor = Sam3Processor(model, resolution=1008, confidence_threshold=0.3)
    print("SAM model loaded.", file=sys.stderr)
    return _processor


# ---------------------------------------------------------------------------
# 单图框选推理
# ---------------------------------------------------------------------------

def segment_with_box(
    image_path: str,
    box: tuple[int, int, int, int],
    output_dir: str,
    device: str = "cuda",
) -> dict:
    """Run SAM3 inference on an image with a box prompt.

    Args:
        image_path: Path to the input image.
        box: (x1, y1, x2, y2) in pixel coordinates.
        output_dir: Directory to save results.
        device: 'cuda' or 'cpu'.

    Returns:
        dict with keys: success, mask_path, overlay_path, box
    """
    x1, y1, x2, y2 = box
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    pil_image = Image.open(image_path).convert("RGB")
    w, h = pil_image.size

    # Clamp box to image dimensions and convert XYXY → CXYWH normalized to [0, 1]
    x1 = max(0.0, min(float(x1), w))
    y1 = max(0.0, min(float(y1), h))
    x2 = max(x1 + 1, min(float(x2), w))
    y2 = max(y1 + 1, min(float(y2), h))
    # SAM3 expects [center_x, center_y, width, height] normalized
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    box_norm = [cx, cy, bw, bh]

    stem = Path(image_path).stem

    try:
        processor = get_processor(device=device)

        # Set image
        with torch.autocast(device, dtype=torch.bfloat16):
            state = processor.set_image(pil_image)

        # Add box geometric prompt — set_text_prompt is NOT needed here.
        # add_geometric_prompt internally sets "visual" text when no text is set,
        # so the model relies purely on the geometric (box) prompt.
        with torch.autocast(device, dtype=torch.bfloat16):
            state = processor.add_geometric_prompt(box=box_norm, label=True, state=state)

        # Extract masks — could be list of tensors or a single batched tensor
        masks = state.get("masks", None)
        scores = state.get("scores", [])

        if masks is None:
            return {"success": False, "error": "Model returned no masks"}

        # Handle tensor masks: shape [N, 1, H, W] or [N, H, W]
        if isinstance(masks, torch.Tensor):
            n_masks = masks.shape[0] if masks.dim() >= 3 else 1
            if n_masks == 0:
                return {"success": False, "error": "Model returned no masks"}
            best_idx = 0
            if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                best_idx = int(torch.argmax(scores).item())
            elif isinstance(scores, list) and len(scores) > 0:
                scores_np = np.array([s.item() if hasattr(s, "item") else float(s) for s in scores])
                best_idx = int(np.argmax(scores_np))
            mask_tensor = masks[best_idx]
            mask_np = mask_tensor.cpu().numpy().squeeze()
        elif isinstance(masks, list):
            if len(masks) == 0:
                return {"success": False, "error": "Model returned no masks"}
            best_idx = 0
            if len(scores) > 0:
                scores_np = np.array([s.item() if hasattr(s, "item") else float(s) for s in scores])
                best_idx = int(np.argmax(scores_np))
            mask_np = masks[best_idx].cpu().numpy().squeeze() if isinstance(masks[best_idx], torch.Tensor) else np.array(masks[best_idx]).squeeze()
        else:
            return {"success": False, "error": f"Unexpected mask type: {type(masks)}"}

        # Save mask as PNG
        mask_uint8 = (mask_np.astype(np.uint8)) * 255
        mask_path = out_dir / f"{stem}_mask.png"
        Image.fromarray(mask_uint8).save(str(mask_path))

        # Save overlay (original + green mask highlight)
        img_np = np.array(pil_image)  # [H, W, 3] RGB
        overlay = img_np.copy()
        mask_3c = mask_np[:, :, None].astype(np.float32)
        overlay = (overlay * (1 - 0.5 * mask_3c) + np.array([0, 255, 0]) * 0.5 * mask_3c).astype(np.uint8)
        overlay_path = out_dir / f"{stem}_overlay.png"
        Image.fromarray(overlay).save(str(overlay_path))

        return {
            "success": True,
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "score": float(scores[best_idx].item()) if isinstance(scores, torch.Tensor) and scores.numel() > 0 else (float(scores[best_idx]) if isinstance(scores, list) and len(scores) > best_idx else None),
        }

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM3 box-prompt segmentation")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--box", required=True, help="Box coords: x1,y1,x2,y2")
    parser.add_argument("--output", default="/tmp/sam_box_output", help="Output directory")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    parts = [int(float(p)) for p in args.box.split(",")]
    if len(parts) != 4:
        print(json.dumps({"success": False, "error": "box must be x1,y1,x2,y2"}))
        sys.exit(1)

    result = segment_with_box(
        image_path=args.image,
        box=(parts[0], parts[1], parts[2], parts[3]),
        output_dir=args.output,
        device=args.device,
    )

    print(json.dumps(result))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
