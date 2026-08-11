#!/usr/bin/env python3
"""Single-image SAM inference with a point prompt (click-to-segment).

Runs inside the sam3 Docker container.
Usage:
    python3 SAM_point.py --image /path/to/image.jpg --point x,y [--output /output/dir]
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
_model = None


def get_processor(checkpoint: str = "/workspace/model/sam3.pt", device: str = "cuda"):
    """Load SAM3 model with interactivity enabled and return (model, processor) cached."""
    global _processor, _model
    if _processor is not None:
        return _model, _processor

    print(f"Loading SAM model from {checkpoint}...", file=sys.stderr)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    _model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        device=device,
        eval_mode=True,
        enable_inst_interactivity=True,  # Required for predict_inst()
    )
    _processor = Sam3Processor(_model, resolution=1008, confidence_threshold=0.5)
    print("SAM model loaded.", file=sys.stderr)
    return _model, _processor


# ---------------------------------------------------------------------------
# 单图点选推理
# ---------------------------------------------------------------------------

def segment_with_point(
    image_path: str,
    point: tuple[int, int],
    output_dir: str,
    device: str = "cuda",
) -> dict:
    """Run SAM3 inference on an image with a point prompt.

    Uses model.predict_inst() for interactive click-to-segment.

    Args:
        image_path: Path to the input image.
        point: (x, y) in pixel coordinates.
        output_dir: Directory to save results.
        device: 'cuda' or 'cpu'.

    Returns:
        dict with keys: success, mask_path, overlay_path, contour_path, point, score
    """
    x, y = point
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    pil_image = Image.open(image_path).convert("RGB")
    w, h = pil_image.size

    # Clamp point to image dimensions
    px = max(0, min(int(x), w - 1))
    py = max(0, min(int(y), h - 1))

    stem = Path(image_path).stem

    try:
        model, processor = get_processor(device=device)

        # Set image features (once)
        with torch.autocast(device, dtype=torch.bfloat16):
            state = processor.set_image(pil_image)

        # Point-prompt inference
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            masks, scores, _ = model.predict_inst(
                inference_state=state,
                point_coords=np.array([[px, py]]),
                point_labels=np.array([1]),   # 1 = foreground point
                multimask_output=True,         # 3 candidate masks → pick best
            )

        # predict_inst returns numpy arrays: masks (N, H, W), scores (N,)
        if masks is None or len(masks) == 0:
            return {"success": False, "error": "Model returned no masks"}

        best_idx = int(np.argmax(scores))
        soft_np = masks[best_idx].astype(np.float32)  # already in [0, 1], H×W
        best_score = float(scores[best_idx])

        img_np = np.array(pil_image)  # [H, W, 3] RGB

        # --- Binary mask PNG (for precise mask extraction) ---
        mask_bin = (soft_np >= 0.5).astype(np.uint8) * 255
        mask_path = out_dir / f"{stem}_mask.png"
        Image.fromarray(mask_bin).save(str(mask_path))

        # --- Soft mask overlay (smooth edges, green highlight) ---
        soft_3c = soft_np[:, :, None].astype(np.float32)
        green = np.array([0, 220, 40], dtype=np.float32)
        alpha = 0.45 * soft_3c
        overlay = (img_np.astype(np.float32) * (1 - alpha) + green * alpha).clip(0, 255).astype(np.uint8)
        overlay_path = out_dir / f"{stem}_overlay.png"
        Image.fromarray(overlay).save(str(overlay_path))

        # --- Contour overlay (outline for clarity) ---
        from scipy import ndimage
        struct = ndimage.generate_binary_structure(2, 2)
        eroded = ndimage.binary_erosion(soft_np >= 0.5, structure=struct, iterations=2)
        contour = (soft_np >= 0.5) & ~eroded
        contour_overlay = img_np.copy()
        contour_overlay[contour] = [0, 255, 0]
        contour_path = out_dir / f"{stem}_contour.png"
        Image.fromarray(contour_overlay).save(str(contour_path))

        return {
            "success": True,
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "contour_path": str(contour_path),
            "point": [px, py],
            "score": best_score,
        }

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM3 point-prompt segmentation")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--point", required=True, help="Point coords: x,y")
    parser.add_argument("--output", default="/tmp/sam_point_output", help="Output directory")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    parts = [int(float(p)) for p in args.point.split(",")]
    if len(parts) != 2:
        print(json.dumps({"success": False, "error": "point must be x,y"}))
        sys.exit(1)

    result = segment_with_point(
        image_path=args.image,
        point=(parts[0], parts[1]),
        output_dir=args.output,
        device=args.device,
    )

    print(json.dumps(result))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
