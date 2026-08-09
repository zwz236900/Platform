"""Scheme B — Apply SAM masks to images, producing RGBA PNGs with transparent background.

Usage (CLI):
    python -m algorithm.image_mask --project Untitled --prompt object

Usage (API):
    from algorithm.image_mask import run
    result = run("Untitled", prompt_slug="object")
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path("/home/zwz/Code/DATA")


def merge_masks(sam_dir: Path, image_name: str, prompt_slug: str = "object") -> np.ndarray | None:
    """Merge all SAM masks for *image_name* (logical OR).

    Mask naming convention: ``{base}_{prompt_slug}_mask{j}.png``

    Returns a (H, W) bool ndarray, or None if no masks found.
    """
    base = Path(image_name).stem
    merged = None
    # Find all masks for this image + prompt
    for mask_path in sorted(sam_dir.glob(f"{base}_{prompt_slug}_mask*.png")):
        mask = np.array(Image.open(mask_path))  # (H, W) 0/255
        if merged is None:
            merged = mask >= 128
        else:
            merged |= mask >= 128
    return merged


def apply_mask(
    image_path: Path,
    mask: np.ndarray,
    output_path: Path,
) -> None:
    """Apply binary mask to image, writing an RGBA PNG.

    - Mask=True  → original RGB + alpha=255
    - Mask=False → RGB=(0,0,0) + alpha=0
    """
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img)  # (H, W, 3) uint8

    # Ensure mask matches image dimensions
    if mask.shape[:2] != (img_arr.shape[0], img_arr.shape[1]):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((img_arr.shape[1], img_arr.shape[0]),
                                   Image.NEAREST)
        mask = np.array(mask_img) >= 128

    out = np.zeros((img_arr.shape[0], img_arr.shape[1], 4), dtype=np.uint8)
    out[mask, :3] = img_arr[mask]
    out[mask, 3] = 255
    # non-mask stays (0,0,0,0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "RGBA").save(output_path, compress_level=6)


def run(
    project_name: str,
    prompt_slug: str = "object",
    *,
    image_dir: str | None = None,
    sam_dir: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run image masking for an entire project.

    Returns dict with keys: ``success``, ``output_dir``, ``num_images``, ``error``.
    """
    project = DATA_DIR / project_name
    if not project.is_dir():
        return {"success": False, "output_dir": "", "num_images": 0,
                "error": f"Project not found: {project}"}

    # Resolve paths
    img_dir = Path(image_dir or (project / "SFM_output" / "undistorted" / "images"))
    sam = Path(sam_dir or (project / "SAM_output"))
    out = Path(output_dir or (project / "algorithm_output" / "input_masked"))

    if not img_dir.is_dir():
        return {"success": False, "output_dir": "", "num_images": 0,
                "error": f"Image directory not found: {img_dir}"}
    if not sam.is_dir():
        return {"success": False, "output_dir": "", "num_images": 0,
                "error": f"SAM output directory not found: {sam}"}

    # Load manifest to know which images have masks
    manifest_path = sam / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        # Get unique image names that have results
        masked_images = {r["image"] for r in manifest.get("results", [])}
    else:
        # Fallback: list all images in image_dir
        masked_images = {p.name for p in img_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

    count = 0
    for img_name in sorted(masked_images):
        # Find the original image (could be in input_image or undistorted)
        img_path = img_dir / img_name
        if not img_path.exists():
            # Try with different extension
            stem = Path(img_name).stem
            candidates = list(img_dir.glob(f"{stem}.*"))
            if candidates:
                img_path = candidates[0]
            else:
                continue

        mask = merge_masks(sam, img_name, prompt_slug)
        if mask is None:
            continue

        # Output filename: same stem, always .png
        out_name = Path(img_name).stem + ".png"
        apply_mask(img_path, mask, out / out_name)
        count += 1

    if count == 0:
        return {"success": False, "output_dir": str(out), "num_images": 0,
                "error": "No masks found for any image"}
    return {"success": True, "output_dir": str(out), "num_images": count, "error": None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply SAM masks to images (RGBA output)")
    parser.add_argument("--project", required=True, help="Project name under DATA/")
    parser.add_argument("--prompt", default="object", help="SAM prompt slug (default: object)")
    parser.add_argument("--image-dir", help="Override input image directory")
    parser.add_argument("--sam-dir", help="Override SAM output directory")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    result = run(
        args.project,
        prompt_slug=args.prompt,
        image_dir=args.image_dir,
        sam_dir=args.sam_dir,
        output_dir=args.output_dir,
    )
    if result["success"]:
        print(f"OK — {result['num_images']} images → {result['output_dir']}")
    else:
        print(f"FAIL — {result['error']}")


if __name__ == "__main__":
    main()
