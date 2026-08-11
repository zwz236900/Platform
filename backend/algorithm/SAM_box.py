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
    _processor = Sam3Processor(model, resolution=1008, confidence_threshold=0.5)
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

        # Extract best mask — prefer soft logits for smoother edges.
        # masks_logits: float in [0, 1] before binarization; masks: binary at threshold 0.5.
        masks_soft = state.get("masks_logits", None)  # soft mask, preferred
        masks = state.get("masks", None)  # binary mask, fallback
        scores = state.get("scores", [])

        if masks_soft is None and masks is None:
            return {"success": False, "error": "Model returned no masks"}

        def _pick_best_idx(n: int) -> int:
            """Pick the best mask index from scores."""
            if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                return int(torch.argmax(scores).item())
            if isinstance(scores, list) and len(scores) > 0:
                arr = np.array([s.item() if hasattr(s, "item") else float(s) for s in scores])
                return int(np.argmax(arr))
            return 0

        def _tensor_to_np(t: torch.Tensor, idx: int) -> np.ndarray:
            return t[idx].cpu().numpy().squeeze()

        # Extract the best soft mask (float32, [0, 1])
        if masks_soft is not None:
            if isinstance(masks_soft, torch.Tensor):
                n = masks_soft.shape[0] if masks_soft.dim() >= 3 else 1
                if n == 0:
                    return {"success": False, "error": "Model returned no masks"}
                best_idx = _pick_best_idx(n)
                soft_np = _tensor_to_np(masks_soft, best_idx)
            elif isinstance(masks_soft, list):
                if len(masks_soft) == 0:
                    return {"success": False, "error": "Model returned no masks"}
                best_idx = _pick_best_idx(len(masks_soft))
                item = masks_soft[best_idx]
                soft_np = item.cpu().numpy().squeeze() if isinstance(item, torch.Tensor) else np.array(item).squeeze()
            else:
                return {"success": False, "error": f"Unexpected masks_logits type: {type(masks_soft)}"}
        else:
            # Fallback to binary mask
            if isinstance(masks, torch.Tensor):
                n = masks.shape[0] if masks.dim() >= 3 else 1
                if n == 0:
                    return {"success": False, "error": "Model returned no masks"}
                best_idx = _pick_best_idx(n)
                soft_np = _tensor_to_np(masks, best_idx).astype(np.float32)
            elif isinstance(masks, list):
                if len(masks) == 0:
                    return {"success": False, "error": "Model returned no masks"}
                best_idx = _pick_best_idx(len(masks))
                item = masks[best_idx]
                bin_np = item.cpu().numpy().squeeze() if isinstance(item, torch.Tensor) else np.array(item).squeeze()
                soft_np = bin_np.astype(np.float32)
            else:
                return {"success": False, "error": f"Unexpected mask type: {type(masks)}"}

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

        # --- Contour overlay (outline + soft fill for clarity) ---
        from scipy import ndimage
        struct = ndimage.generate_binary_structure(2, 2)
        eroded = ndimage.binary_erosion(soft_np >= 0.5, structure=struct, iterations=2)
        contour = (soft_np >= 0.5) & ~eroded
        # Draw contour in bright green
        contour_overlay = img_np.copy()
        contour_overlay[contour] = [0, 255, 0]
        contour_path = out_dir / f"{stem}_contour.png"
        Image.fromarray(contour_overlay).save(str(contour_path))

        # Compute per-mask scores for debug
        if isinstance(scores, torch.Tensor) and scores.numel() > 0:
            best_score = float(scores[best_idx].item())
        elif isinstance(scores, list) and len(scores) > best_idx:
            s = scores[best_idx]
            best_score = float(s.item()) if hasattr(s, "item") else float(s)
        else:
            best_score = None

        return {
            "success": True,
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "contour_path": str(contour_path),
            "box": [int(x1), int(y1), int(x2), int(y2)],
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
