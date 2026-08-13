"""Scheme A — Multi-view consistent point cloud filtering via SAM masks.

Projects each 3D point into every camera view and compares against the
per-image SAM mask.  Points that land inside mask regions in many views are
kept; points that consistently miss the masks are discarded.

Usage (CLI):
    python -m algorithm.pointcloud_filter --project Untitled --threshold 0.3

Usage (API):
    from algorithm.pointcloud_filter import run
    result = run("Untitled", threshold=0.3)
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .common import (
    load_sparse_model,
    read_ply,
    write_ply,
    project_points,
)

DATA_DIR = Path("/home/zwz/Code/DATA")

# Processing chunk size to keep memory bounded
CHUNK_SIZE = 50_000


def run(
    project_name: str,
    threshold: float = 0.3,
    prompt_slug: str = "object",
    *,
    ply_path: str | None = None,
    sparse_dir: str | None = None,
    sam_dir: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Run multi-view point cloud filtering.

    Returns:
        ``{success, output_path, num_points_before, num_points_after, error}``
    """
    project = DATA_DIR / project_name
    if not project.is_dir():
        return {"success": False, "output_path": "", "num_points_before": 0,
                "num_points_after": 0, "error": f"Project not found: {project}"}

    # Resolve paths (default to undistorted sparse + MVS dense)
    # Check both colmap and openmvs subdirectories, plus direct densify output
    _default_ply = project / "MVS_output" / "colmap" / "fused.ply"
    if not _default_ply.exists():
        _default_ply = project / "MVS_output" / "openmvs" / "scene_dense.ply"
    if not _default_ply.exists():
        _default_ply = project / "MVS_output" / "openmvs" / "scene_dense_mesh.ply"
    if not _default_ply.exists():
        _default_ply = project / "MVS_output" / "scene_dense.ply"  # legacy

    ply_file = Path(ply_path or _default_ply)
    # 去畸变模型现在在 SFM_output/dense/ 下（dense/sparse + dense/images），
    # 而非旧路径 SFM_output/undistorted/。
    sparse = Path(sparse_dir or (project / "SFM_output" / "dense" / "sparse"))
    sam = Path(sam_dir or (project / "SAM_output"))
    out = Path(output_path or (project / "algorithm_output" / "filtered_scene.ply"))

    # Validate
    if not ply_file.exists():
        return {"success": False, "output_path": str(out),
                "num_points_before": 0, "num_points_after": 0,
                "error": f"Point cloud not found: {ply_file}"}
    cameras, images = load_sparse_model(sparse)
    if cameras is None:
        return {"success": False, "output_path": str(out),
                "num_points_before": 0, "num_points_after": 0,
                "error": f"COLMAP 稀疏模型缺失（cameras/images 的 bin 或 txt）: {sparse}"}
    if not sam.is_dir():
        return {"success": False, "output_path": str(out),
                "num_points_before": 0, "num_points_after": 0,
                "error": f"SAM output dir not found: {sam}"}

    # --- Load data ---
    print(f"Loading point cloud: {ply_file}")
    xyz, rgb = read_ply(ply_file)
    n_before = xyz.shape[0]
    print(f"  → {n_before:,} points")

    print(f"Loading sparse model from: {sparse}")
    print(f"  → {len(cameras)} camera(s), {len(images)} image(s)")

    # --- Build per-view data ---
    views: list[dict] = []  # each: {K, R, t, w, h, mask, name}
    n_masks = 0
    for img_id, img in images.items():
        cam = cameras.get(img["cam_id"])
        if cam is None:
            continue

        # Load SAM mask for this image; resize to match camera dimensions
        mask = _load_mask(sam, img["name"], prompt_slug,
                          target_w=cam["width"], target_h=cam["height"])
        if mask is None:
            print(f"  SKIP {img['name']} — no mask found")
            continue
        n_masks += 1

        views.append({
            "K": cam["K"],
            "R": img["R"],
            "t": img["t"],
            "w": cam["width"],
            "h": cam["height"],
            "mask": mask,  # (H, W) bool
            "name": img["name"],
        })

    if not views:
        return {"success": False, "output_path": str(out),
                "num_points_before": n_before, "num_points_after": 0,
                "error": (f"No views with valid masks found "
                          f"(searched {sam / 'point_results'}/ and {sam}/ "
                          f"for {Path(next(iter(images.values()))['name']).stem}_mask*.png "
                          f"& _{prompt_slug}_mask*.png)")}
    print(f"  → {len(views)} views with masks ({n_masks} mask file(s))")

    # --- Batch score ---
    n = n_before
    mask_hits = np.zeros(n, dtype=np.int32)
    visible_count = np.zeros(n, dtype=np.int32)

    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunk_xyz = xyz[start:end]
        chunk_sz = end - start

        for view in views:
            u, v, valid = project_points(
                chunk_xyz, view["K"], view["R"], view["t"],
                view["w"], view["h"],
            )
            visible_count[start:end] += valid.astype(np.int32)

            # For valid projections, check mask
            vi = np.where(valid)[0]
            if len(vi) == 0:
                continue

            # Round to nearest pixel (clamped to mask dimensions)
            ui = np.clip(np.round(u[vi]).astype(np.int32), 0, view["mask"].shape[1] - 1)
            vi_px = np.clip(np.round(v[vi]).astype(np.int32), 0, view["mask"].shape[0] - 1)

            mask_hits[start + vi] += view["mask"][vi_px, ui].astype(np.int32)

        if (start // CHUNK_SIZE) % 5 == 0:
            pct = 100 * end / n
            print(f"  scoring... {pct:.0f}% ({end:,}/{n:,})")

    # --- Score & filter ---
    scores = np.zeros(n, dtype=np.float32)
    nonzero = visible_count > 0
    scores[nonzero] = mask_hits[nonzero].astype(np.float32) / visible_count[nonzero].astype(np.float32)

    keep = scores >= threshold
    n_after = keep.sum()

    print(f"Score stats: min={scores.min():.3f}, median={np.median(scores):.3f}, "
          f"max={scores.max():.3f}, mean={scores.mean():.3f}")
    print(f"Filter: {n_after:,} / {n_before:,} points kept "
          f"({100*n_after/n_before:.1f}%) at threshold={threshold}")

    # --- Write output ---
    print(f"Writing: {out}")
    write_ply(out, xyz[keep], rgb[keep] if rgb.shape[0] == n else None, scores[keep])
    print("Done.")

    return {
        "success": True,
        "input_path": str(ply_file),
        "output_path": str(out),
        "num_points_before": n_before,
        "num_points_after": int(n_after),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mask(sam_dir: Path, image_name: str, prompt_slug: str,
               target_w: int | None = None, target_h: int | None = None,
               results_subdir: str = "point_results") -> np.ndarray | None:
    """Load and merge all SAM masks for one image (logical OR).

    支持两套 SAM_output 布局（都搜，按需并入）：
      - 新版：``SAM_output/{results_subdir}/{stem}_mask.png``（worker 每次分割覆盖写入，
        无 prompt 后缀；box 分割在 box_results/，用 results_subdir 选）。
      - 旧版：``SAM_output/{stem}_{prompt_slug}_mask*.png``（带 prompt 后缀 + 序号）。

    Optionally resizes to (target_h, target_w) to match camera dimensions.
    """
    base = Path(image_name).stem
    found: list[Path] = []
    for d in (sam_dir / results_subdir, sam_dir):
        if not d.is_dir():
            continue
        found += sorted(d.glob(f"{base}_mask*.png"))
        found += sorted(d.glob(f"{base}_{prompt_slug}_mask*.png"))

    merged = None
    for mp in found:
        m = np.array(Image.open(mp)) >= 128
        if merged is None:
            merged = m
        elif m.shape != merged.shape:
            # 异常保护：尺寸不一致时以第一张为准对齐
            m_img = Image.fromarray(m.astype(np.uint8) * 255)
            m_img = m_img.resize((merged.shape[1], merged.shape[0]), Image.NEAREST)
            merged |= np.array(m_img) >= 128
        else:
            merged |= m
    if merged is not None and target_w is not None and target_h is not None:
        if merged.shape[0] != target_h or merged.shape[1] != target_w:
            merged_img = Image.fromarray(merged.astype(np.uint8) * 255)
            merged_img = merged_img.resize((target_w, target_h), Image.NEAREST)
            merged = np.array(merged_img) >= 128
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-view consistent point cloud filtering via SAM masks")
    parser.add_argument("--project", required=True, help="Project name under DATA/")
    parser.add_argument("--threshold", type=float, default=0.3,
                       help="Score threshold for keeping points (default: 0.3)")
    parser.add_argument("--prompt", default="object", help="SAM prompt slug (default: object)")
    parser.add_argument("--ply", help="Override input PLY path")
    parser.add_argument("--sparse-dir", help="Override COLMAP sparse directory")
    parser.add_argument("--sam-dir", help="Override SAM output directory")
    parser.add_argument("--output", help="Override output PLY path")
    args = parser.parse_args()

    result = run(
        args.project,
        threshold=args.threshold,
        prompt_slug=args.prompt,
        ply_path=args.ply,
        sparse_dir=args.sparse_dir,
        sam_dir=args.sam_dir,
        output_path=args.output,
    )
    if result["success"]:
        print(f"OK — {result['num_points_after']:,} / {result['num_points_before']:,} "
              f"points → {result['output_path']}")
    else:
        print(f"FAIL — {result['error']}")


if __name__ == "__main__":
    main()
