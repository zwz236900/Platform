#!/usr/bin/env python3
"""
COLMAP SFM/MVS Pipeline CLI Tool.

A comprehensive command-line interface for running the full COLMAP
Structure-from-Motion and Multi-View Stereo pipeline with structured
JSON output for downstream processing.

Usage:
    # Full pipeline with defaults
    python sfm_cli.py --image_dir ./images --output_dir ./output

    # Sparse reconstruction only
    python sfm_cli.py --image_dir ./images --output_dir ./output --steps extract,match,sfm

    # Dense reconstruction from existing sparse model
    python sfm_cli.py --image_dir ./images --output_dir ./output \\
        --sparse_dir ./output/sparse --steps mvs

    # Custom matching and GPU settings
    python sfm_cli.py --image_dir ./images --output_dir ./output \\
        --matcher sequential --gpu_index 0 --num_threads 8

Output:
    After running, the output directory contains:
        output/
        ├── database.db          # COLMAP database (features + matches)
        ├── sparse/              # Sparse reconstruction results
        │   └── 0/               # First (largest) reconstruction
        ├── dense/               # Dense reconstruction results
        │   ├── images/          # Undistorted images
        │   ├── sparse/          # Undistorted sparse model
        │   ├── stereo/          # PatchMatch Stereo outputs
        │   └── fused.ply        # Final fused point cloud
        └── pipeline_result.json # Structured output data (this file)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pgm", ".ppm"}

STEP_CHOICES = [
    "extract",    # feature extraction
    "match",      # feature matching
    "sfm",        # sparse reconstruction (SfM)
    "undistort",  # image undistortion (prep for MVS)
    "reuse",      # reuse existing undistorted workspace (skip undistort)
    "mvs",        # dense reconstruction (PatchMatch + Fusion)
    "fusion",     # stereo fusion only
    "convert",    # model format conversion (BIN ↔ TXT)
]

MATCHER_CHOICES = ["exhaustive", "sequential", "vocab_tree", "spatial"]

# Full pipeline (default)
FULL_PIPELINE = ["extract", "match", "sfm", "undistort", "mvs"]

# ---------------------------------------------------------------------------
# MVS tunables
# ---------------------------------------------------------------------------
#
# COLMAP exposes these as --PatchMatchStereo.<name> / --StereoFusion.<name>.
# Every one is registered with default=None, which append_opt() turns into
# "flag omitted" → COLMAP's own default applies. This is what keeps callers
# that pass nothing byte-identical to the pre-tunable behaviour.
#
# Maps option name → (type, help text). Defaults are quoted in the help purely
# as a reference; they are NOT re-applied by this script.
PATCH_MATCH_PARAMS: dict[str, tuple[type, str]] = {
    "max_image_size": (int, "处理的最大图像边长（-1 = 不限制）"),
    "window_radius": (int, "匹配窗口半径（默认 5；弱纹理可调大）"),
    "window_step": (int, "窗口滑动步长（默认 1；2 可提速）"),
    "num_samples": (int, "深度采样数（默认 15）"),
    "num_iterations": (int, "PatchMatch 迭代次数（默认 5）"),
    "cache_size": (int, "图像/深度图缓存大小 GB（默认 32）"),
    "geom_consistency_max_cost": (float, "几何一致性最大代价（默认 3）"),
    "filter": (int, "是否启用深度图过滤 0/1（默认 1）"),
    "filter_min_ncc": (float, "过滤 NCC 下限（默认 0.1）"),
    "filter_min_triangulation_angle": (float, "过滤最小三角化角，度（默认 3）"),
    "filter_min_num_consistent": (int, "过滤最少一致视图数（默认 2）"),
    "filter_geom_consistency_max_cost": (float, "过滤几何一致性代价（默认 1）"),
}

FUSION_PARAMS: dict[str, tuple[type, str]] = {
    "max_image_size": (int, "融合处理的最大图像边长（-1 = 不限制）"),
    "min_num_pixels": (int, "一个 3D 点最少像素观测数（默认 5）"),
    "max_num_pixels": (int, "一个 3D 点最多像素观测数（默认 10000）"),
    "max_reproj_error": (float, "最大重投影误差，像素（默认 2）"),
    "max_depth_error": (float, "最大深度相对误差（默认 0.01）"),
    "max_normal_error": (float, "最大法向夹角误差，度（默认 10）"),
    "check_num_images": (int, "融合检查图像数（默认 50）"),
}

FUSION_INPUT_TYPES = ["auto", "geometric", "photometric"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_command(name: str) -> str:
    """Locate a COLMAP subcommand (installed or in a build directory)."""
    # Try direct colmap invocation with subcommand
    if shutil.which("colmap"):
        return name
    # Fallback: look for COLMAP build binaries
    candidates = [
        f"colmap_{name}",
    ]
    for c in candidates:
        if shutil.which(c):
            return c
    # Default: assume colmap CLI with subcommand
    if shutil.which("colmap"):
        return name
    return name  # let it fail with a clear message


def count_images(image_dir: Path) -> int:
    """Count supported image files in a directory (non-recursive)."""
    if not image_dir.is_dir():
        return 0
    return sum(
        1
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )


def append_opt(cmd: list[str], flag: str, value: Any) -> None:
    """Append ``--flag value`` only when value is not None.

    Every tunable in this CLI defaults to None, meaning "don't pass the flag and
    let COLMAP use its own default". That keeps existing callers byte-identical
    in behaviour while still letting callers override any single knob.
    """
    if value is None:
        return
    if isinstance(value, bool):
        value = 1 if value else 0
    cmd += [flag, str(value)]


def resolve_geom_consistency(args: argparse.Namespace) -> bool:
    """Reconcile --no_geom_consistency (historical) with --geom_consistency 0/1.

    Callers must reject the mutually-exclusive combination before relying on
    this; here the tri-state flag wins if it was given.
    """
    if args.geom_consistency is None:
        return not args.no_geom_consistency
    return bool(args.geom_consistency)


def collect_mvs_opts(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pull the --pms_* / --fusion_* tunables into the two option dicts.

    None entries are kept: append_opt() drops them, which is exactly the
    "COLMAP default" behaviour we want for anything the caller didn't set.
    """
    patch_match_opts = {name: getattr(args, f"pms_{name}") for name in PATCH_MATCH_PARAMS}
    fusion_opts = {name: getattr(args, f"fusion_{name}") for name in FUSION_PARAMS}
    return patch_match_opts, fusion_opts


def run_colmap(
    args: list[str],
    *,
    log_file: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    """Run a colmap subcommand with logging and error handling."""
    cmd = ["colmap"] + args
    cmd_str = " ".join(cmd)
    print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] $ {cmd_str}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        print(f"\nERROR: 'colmap' not found in PATH.", file=sys.stderr)
        print(
            "Install COLMAP or ensure it is available in your environment.",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"\nERROR: Command timed out after {timeout}s.", file=sys.stderr)
        raise

    # Print stderr to terminal (COLMAP uses stderr for logging)
    if result.stderr:
        # Only print key lines to keep output manageable
        for line in result.stderr.splitlines():
            if any(kw in line for kw in ("ERROR", "WARNING", "Elapsed", "=>", "===")):
                print(f"  {line}")

    # Write full log to file
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Command: {cmd_str}\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"{'='*60}\n")
            f.write(result.stderr)
            f.write(result.stdout)

    if result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}", file=sys.stderr)
        if result.stderr:
            # Print last few lines for context
            tail = result.stderr.strip().splitlines()[-10:]
            print("\n".join(tail), file=sys.stderr)
        sys.exit(result.returncode)

    return result


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_extract(
    image_dir: Path,
    database_path: Path,
    *,
    single_camera: bool = True,
    camera_model: str = "SIMPLE_RADIAL",
    use_gpu: bool = True,
    gpu_index: int = 0,
    max_num_features: int = 0,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Run COLMAP feature_extractor."""
    num_images = count_images(image_dir)
    print(f"\n{'='*60}")
    print(f"Step 1/5: Feature Extraction  ({num_images} images)")
    print(f"{'='*60}")

    cmd = [
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--ImageReader.camera_model", camera_model,
        "--FeatureExtraction.use_gpu", "1" if use_gpu else "0",
        "--FeatureExtraction.gpu_index", str(gpu_index),
    ]
    # max_num_features=0 is invalid in COLMAP 4.2+; omit to use default (8192)
    if max_num_features > 0:
        cmd += ["--SiftExtraction.max_num_features", str(max_num_features)]

    run_colmap(cmd, log_file=log_file)

    return {"num_images": num_images, "status": "ok"}


def step_match(
    database_path: Path,
    *,
    matcher: str = "exhaustive",
    use_gpu: bool = True,
    gpu_index: int = 0,
    vocab_tree_path: str | None = None,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Run COLMAP feature matching."""
    print(f"\n{'='*60}")
    print(f"Step 2/5: Feature Matching  (matcher={matcher})")
    print(f"{'='*60}")

    if matcher == "exhaustive":
        run_colmap([
            "exhaustive_matcher",
            "--database_path", str(database_path),
            "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
            "--FeatureMatching.gpu_index", str(gpu_index),
        ], log_file=log_file)
    elif matcher == "sequential":
        run_colmap([
            "sequential_matcher",
            "--database_path", str(database_path),
            "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
            "--FeatureMatching.gpu_index", str(gpu_index),
        ], log_file=log_file)
    elif matcher == "vocab_tree":
        vtree = vocab_tree_path or ""
        run_colmap([
            "vocab_tree_matcher",
            "--database_path", str(database_path),
            "--VocabTreeMatching.vocab_tree_path", vtree,
            "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
            "--FeatureMatching.gpu_index", str(gpu_index),
        ], log_file=log_file)
    elif matcher == "spatial":
        run_colmap([
            "spatial_matcher",
            "--database_path", str(database_path),
            "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
            "--FeatureMatching.gpu_index", str(gpu_index),
        ], log_file=log_file)
    else:
        raise ValueError(f"Unknown matcher: {matcher}")

    return {"matcher": matcher, "status": "ok"}


def step_sfm(
    database_path: Path,
    image_dir: Path,
    sparse_dir: Path,
    *,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Run COLMAP mapper (sparse reconstruction)."""
    print(f"\n{'='*60}")
    print("Step 3/5: Sparse Reconstruction (SfM)")
    print(f"{'='*60}")

    sparse_dir.mkdir(parents=True, exist_ok=True)

    run_colmap([
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
    ], log_file=log_file)

    # Count reconstructions
    rec_dirs = sorted([d for d in sparse_dir.iterdir() if d.is_dir()])
    rec_count = len(rec_dirs)
    print(f"  Generated {rec_count} reconstruction(s)")

    result = {"num_reconstructions": rec_count, "reconstructions": []}

    for rd in rec_dirs:
        model_files = list(rd.glob("*.bin")) or list(rd.glob("*.txt"))
        result["reconstructions"].append({
            "name": rd.name,
            "num_files": len(model_files),
        })

    return result


def step_undistort(
    image_dir: Path,
    sparse_dir: Path,
    dense_dir: Path,
    *,
    reconstruction_name: str = "0",
    max_image_size: int | None = None,
    num_patch_match_src_images: int | None = None,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Run COLMAP image_undistorter to prepare for MVS.

    ``num_patch_match_src_images`` decides how many source views each reference
    image gets in stereo/patch-match.cfg. That file is written here and never
    regenerated by the mvs step, so changing it only has an effect when the
    undistort step actually runs (the reuse step just copies an existing
    workspace, frozen cfg included).
    """
    print(f"\n{'='*60}")
    print("Step 4/5: Image Undistortion")
    print(f"{'='*60}")

    input_path = sparse_dir / reconstruction_name
    if not input_path.is_dir():
        print(f"  WARNING: Sparse model not found at {input_path}")
        print(f"  Available: {list(sparse_dir.iterdir())}")
        return {"status": "skipped", "reason": f"sparse model not found: {input_path}"}

    dense_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(input_path),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ]
    append_opt(cmd, "--max_image_size", max_image_size)
    append_opt(cmd, "--num_patch_match_src_images", num_patch_match_src_images)
    run_colmap(cmd, log_file=log_file)

    # image_undistorter 写的是二进制模型（dense/sparse/*.bin），而 OpenMVS 2019 的
    # InterfaceCOLMAP 只读文本格式（cameras.txt/images.txt/points3D.txt）→ 转成 TXT。
    # COLMAP 自身的 dense 重建两种格式都能读，因此统一输出文本对两个引擎都兼容。
    sparse_dir_dense = dense_dir / "sparse"
    if sparse_dir_dense.is_dir():
        txt_tmp = dense_dir / "sparse_txt"
        txt_tmp.mkdir(parents=True, exist_ok=True)
        run_colmap([
            "model_converter",
            "--input_path", str(sparse_dir_dense),
            "--output_path", str(txt_tmp),
            "--output_type", "TXT",
        ], log_file=log_file)
        shutil.rmtree(sparse_dir_dense)
        txt_tmp.rename(sparse_dir_dense)
        print(f"  Converted dense/sparse to TXT for OpenMVS compatibility")

    return {"input_reconstruction": reconstruction_name, "status": "ok"}


def step_mvs(
    dense_dir: Path,
    *,
    use_gpu: bool = True,
    gpu_index: int = 0,
    geom_consistency: bool = True,
    patch_match_opts: dict[str, Any] | None = None,
    fusion_opts: dict[str, Any] | None = None,
    fusion_input_type: str = "auto",
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Run PatchMatch Stereo + Stereo Fusion (dense reconstruction).

    ``patch_match_opts`` / ``fusion_opts`` map a COLMAP option suffix to a value,
    e.g. ``{"window_radius": 7}`` → ``--PatchMatchStereo.window_radius 7``. A
    None value means "leave the flag out and take COLMAP's default", so the
    no-opts call is identical to what this pipeline ran before.
    """
    print(f"\n{'='*60}")
    print("Step 5/5: Dense Reconstruction (MVS)")
    print(f"{'='*60}")

    patch_match_opts = patch_match_opts or {}
    fusion_opts = fusion_opts or {}

    # Resolve/validate input_type before PatchMatch rather than at the fusion
    # call: a mismatch can only end in an error, and failing here avoids burning
    # the whole (expensive) depth-map pass first.
    #
    # input_type must match the depth maps patch_match actually writes, which
    # depends on geom_consistency (see PatchMatchController::ProcessProblem: it
    # writes "<image>.photometric.bin" when consistency is off and adds a
    # "<image>.geometric.bin" pass when it is on). Hardcoding "geometric" made
    # the --no_geom_consistency path fuse nothing at all.
    if fusion_input_type == "auto":
        fusion_input_type = "geometric" if geom_consistency else "photometric"
    elif fusion_input_type == "geometric" and not geom_consistency:
        print(
            "ERROR: --fusion_input_type geometric needs geometric consistency, "
            "but --no_geom_consistency was given; no .geometric.bin depth maps "
            "will exist to fuse.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 5a. PatchMatch Stereo
    print("\n  --- PatchMatch Stereo ---")
    cmd = [
        "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        # Always passed explicitly: pipeline_progress._geom_consistency() reads it
        # back out of the echoed command to tell a two-pass layout from a
        # one-pass one when computing progress.
        "--PatchMatchStereo.geom_consistency", "true" if geom_consistency else "false",
        "--PatchMatchStereo.gpu_index", str(gpu_index),
    ]
    for name, value in patch_match_opts.items():
        append_opt(cmd, f"--PatchMatchStereo.{name}", value)
    run_colmap(cmd, log_file=log_file)

    # 5b. Stereo Fusion
    fused_path = dense_dir / "fused.ply"
    print("\n  --- Stereo Fusion ---")
    cmd = [
        "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", fusion_input_type,
        "--output_path", str(fused_path),
    ]
    for name, value in fusion_opts.items():
        append_opt(cmd, f"--StereoFusion.{name}", value)
    run_colmap(cmd, log_file=log_file)

    result = {
        "status": "ok",
        "fused_ply": str(fused_path),
        "geom_consistency": geom_consistency,
        "fusion_input_type": fusion_input_type,
    }

    if fused_path.exists():
        size_mb = fused_path.stat().st_size / (1024 * 1024)
        result["fused_ply_size_mb"] = round(size_mb, 2)
        print(f"  Fused point cloud: {fused_path} ({size_mb:.1f} MB)")
    else:
        result["status"] = "warning"
        result["warning"] = "fused.ply not created"

    return result


def step_reuse_dense(
    src_dense: Path,
    dst_dense: Path,
    *,
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Reuse an existing undistorted workspace (from SfM undistort step).

    SfM 页面的去畸变步骤已产出 dense/{images,sparse,stereo}（PINHOLE 模型 +
    去畸变图像）。COLMAP MVS 直接复制该工作区到目标 dense 目录，跳过重复的
    image_undistorter，只跑 patch_match + fusion。

    src_dense: 源去畸变工作区（如 SFM_output/dense）
    dst_dense: 目标 dense 目录（如 MVS_output/colmap）
    """
    print(f"\n{'='*60}")
    print("Step: Reuse Undistorted Workspace")
    print(f"{'='*60}")
    print(f"  Source: {src_dense}")
    print(f"  Target: {dst_dense}")
    # The copied stereo/ dir carries a frozen patch-match.cfg, so the
    # image_undistorter tunables never reach this path. Say so rather than
    # letting a caller believe they took effect.
    print("  NOTE: undistortion tunables (max_image_size / src images) do NOT apply")
    print("        on this path -- the existing workspace is copied as-is.")

    if src_dense is None or not src_dense.is_dir():
        print(f"  WARNING: Source workspace not found at {src_dense}")
        return {"status": "skipped", "reason": f"source dense dir not found: {src_dense}"}

    dst_dense.mkdir(parents=True, exist_ok=True)

    # 复制 patch_match 需要的子目录：images（去畸变图）、sparse（PINHOLE 模型）、
    # stereo（patch-match.cfg 等配置文件）
    copied = []
    for sub in ("images", "sparse", "stereo"):
        src = src_dense / sub
        dst = dst_dense / sub
        if not src.is_dir():
            print(f"  WARNING: {sub} missing in source: {src}")
            continue
        if dst.exists():
            print(f"  Removing existing {dst}")
            shutil.rmtree(dst)
        print(f"  Copying {src} -> {dst}")
        shutil.copytree(src, dst)
        copied.append(sub)

    if not copied:
        return {"status": "skipped", "reason": "no usable subdirs found in source"}

    return {"status": "ok", "copied": copied}


def step_convert(
    sparse_dir: Path,
    *,
    model_name: str = "0",
    output_suffix: str = "txt",
    log_file: Path | None = None,
) -> dict[str, Any]:
    """Convert COLMAP model between BIN and TXT formats using model_converter."""
    input_path = sparse_dir / model_name
    output_path = sparse_dir / f"{model_name}_{output_suffix}"

    print(f"\n{'='*60}")
    print(f"Step: Model Conversion  (BIN ↔ TXT)")
    print(f"{'='*60}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    if not input_path.is_dir():
        print(f"  WARNING: Input model not found at {input_path}")
        return {"status": "skipped", "reason": f"input model not found: {input_path}"}

    output_path.mkdir(parents=True, exist_ok=True)

    output_type = output_suffix.upper()
    run_colmap([
        "model_converter",
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--output_type", output_type,
    ], log_file=log_file)

    # Count output files
    output_files = list(output_path.glob(f"*.{output_suffix}"))
    print(f"  Converted {len(output_files)} files")

    return {
        "status": "ok",
        "input": str(input_path),
        "output": str(output_path),
        "output_type": output_type,
        "num_files": len(output_files),
    }


# ---------------------------------------------------------------------------
# JSON output builder
# ---------------------------------------------------------------------------

def build_pipeline_result(
    args: argparse.Namespace,
    steps_run: list[str],
    step_results: dict[str, Any],
    output_dir: Path,
    elapsed_sec: float,
) -> dict[str, Any]:
    """Build the structured pipeline_result.json output."""

    result: dict[str, Any] = {
        "pipeline": {
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "steps_requested": args.steps,
            "steps_completed": steps_run,
            "elapsed_seconds": round(elapsed_sec, 1),
        },
        "input": {
            "image_dir": str(args.image_dir),
            "num_images": count_images(args.image_dir),
            "camera_model": args.camera_model,
            "single_camera": args.single_camera,
        },
        "output": {
            "output_dir": str(output_dir),
            "database_path": str(output_dir / "database.db"),
            "sparse_dir": str(output_dir / "sparse"),
            "dense_dir": str(output_dir / "dense"),
        },
        "steps": step_results,
    }

    # Effective MVS settings for traceability. The full argv is already recorded
    # in pipeline.command; this is the machine-readable form. Only explicitly
    # set tunables are listed -- anything absent means "COLMAP's own default".
    patch_match_opts, fusion_opts = collect_mvs_opts(args)
    result["mvs"] = {
        "geom_consistency": resolve_geom_consistency(args),
        "fusion_input_type": args.fusion_input_type,
        "undistort_max_image_size": args.undistort_max_image_size,
        "num_patch_match_src_images": args.num_patch_match_src_images,
        "patch_match": {k: v for k, v in patch_match_opts.items() if v is not None},
        "fusion": {k: v for k, v in fusion_opts.items() if v is not None},
    }

    # Add sparse reconstruction details if available
    sparse_dir = output_dir / "sparse"
    result["output"]["sparse_models"] = []
    if sparse_dir.is_dir():
        for rd in sorted(sparse_dir.iterdir()):
            if rd.is_dir():
                model_info = {"name": rd.name}
                # Check for model files
                bin_files = sorted([f.name for f in rd.glob("*.bin")])
                txt_files = sorted([f.name for f in rd.glob("*.txt")])
                model_info["files"] = bin_files or txt_files
                model_info["format"] = "bin" if bin_files else "txt" if txt_files else "unknown"
                result["output"]["sparse_models"].append(model_info)

    # Add dense output details
    dense_dir = output_dir / "dense"
    result["output"]["dense_outputs"] = {}
    if dense_dir.is_dir():
        fused = dense_dir / "fused.ply"
        if fused.exists():
            result["output"]["dense_outputs"]["fused_ply"] = str(fused)
            result["output"]["dense_outputs"]["fused_ply_size_mb"] = round(
                fused.stat().st_size / (1024 * 1024), 2
            )
        # Check for stereo results
        stereo_dir = dense_dir / "stereo"
        if stereo_dir.is_dir():
            result["output"]["dense_outputs"]["stereo_patches"] = str(stereo_dir)

    return result


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="COLMAP SFM/MVS Pipeline — automated end-to-end reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python sfm_cli.py --image_dir ./images --output_dir ./output

  # Sparse only, sequential matching, with camera model
  python sfm_cli.py --image_dir ./images --output_dir ./output \\
      --steps extract,match,sfm --matcher sequential --camera_model PINHOLE

  # Resume from sparse (skip extract+match), run dense
  python sfm_cli.py --image_dir ./images --output_dir ./output \\
      --steps sfm,undistort,mvs --sparse_dir ./output/sparse

  # Custom GPU and threads
  python sfm_cli.py --image_dir ./images --output_dir ./output \\
      --gpu_index 1 --num_threads 16

  # With vocabulary tree matching
  python sfm_cli.py --image_dir ./images --output_dir ./output \\
      --matcher vocab_tree --vocab_tree_path /path/to/vocab_tree.bin
        """,
    )

    # ---- Required ----
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for all outputs (database, sparse, dense, results).",
    )

    # ---- Pipeline control ----
    parser.add_argument(
        "--steps",
        type=str,
        default="extract,match,sfm,undistort,mvs",
        help=(
            "Comma-separated pipeline steps to run. "
            f"Choices: {','.join(STEP_CHOICES)}. "
            "Default: extract,match,sfm,undistort,mvs (full pipeline)."
        ),
    )
    parser.add_argument(
        "--sparse_dir",
        type=Path,
        default=None,
        help=(
            "Path to an existing sparse model directory. When provided, "
            "sfm/undistort steps use this instead of the default output/sparse. "
            "Useful for resuming from a previous sparse run."
        ),
    )

    # ---- Dense output ----
    parser.add_argument(
        "--dense_dir",
        type=Path,
        default=None,
        help=(
            "Path for dense MVS output (default: <output_dir>/dense). "
            "Use this to redirect MVS results, e.g. to MVS_output/colmap."
        ),
    )
    parser.add_argument(
        "--reuse_dense_dir",
        type=Path,
        default=None,
        help=(
            "Source of an existing undistorted workspace (e.g. SFM_output/dense) "
            "to copy into --dense_dir when the 'reuse' step runs, skipping "
            "a redundant image_undistorter pass."
        ),
    )

    # ---- Image / camera ----
    parser.add_argument(
        "--single_camera",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use a single (shared) camera intrinsic for all images. Default: 1.",
    )
    parser.add_argument(
        "--camera_model",
        type=str,
        default="SIMPLE_RADIAL",
        help="Camera model for intrinsic estimation. Default: SIMPLE_RADIAL.",
    )

    # ---- Feature extraction ----
    parser.add_argument(
        "--max_num_features",
        type=int,
        default=0,
        help="Maximum number of SIFT features per image (0 = unlimited). Default: 0.",
    )

    # ---- Matching ----
    parser.add_argument(
        "--matcher",
        type=str,
        default="exhaustive",
        choices=MATCHER_CHOICES,
        help="Feature matching strategy. Default: exhaustive.",
    )
    parser.add_argument(
        "--vocab_tree_path",
        type=str,
        default=None,
        help="Path to vocabulary tree file (required when --matcher vocab_tree).",
    )

    # ---- Hardware ----
    parser.add_argument(
        "--use_gpu",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable GPU acceleration. Default: 1.",
    )
    parser.add_argument(
        "--gpu_index",
        type=int,
        default=0,
        help="GPU device index to use. Default: 0.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=-1,
        help="Number of CPU threads (-1 = auto). Default: -1.",
    )

    # ---- MVS ----
    parser.add_argument(
        "--no_geom_consistency",
        action="store_true",
        help="Disable geometric consistency in PatchMatch Stereo.",
    )
    parser.add_argument(
        "--geom_consistency",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "Tri-state form of --no_geom_consistency: 1 forces it on, 0 off. "
            "Omit to keep the default (on). Passing both flags is an error."
        ),
    )
    parser.add_argument(
        "--reconstruction_name",
        type=str,
        default="0",
        help="Name of the sparse reconstruction to use for MVS. Default: 0.",
    )

    # ---- MVS: image_undistorter.
    # Only reached when the undistort step actually runs -- the reuse step just
    # copies an existing workspace, stereo/patch-match.cfg included, so these
    # two have no effect on that path.
    mvs_group = parser.add_argument_group(
        "MVS / undistortion",
        "Forwarded to image_undistorter. No effect when the reuse step is used.",
    )
    mvs_group.add_argument(
        "--undistort_max_image_size",
        type=int,
        default=None,
        help="image_undistorter --max_image_size. Default: -1 (keep resolution).",
    )
    mvs_group.add_argument(
        "--num_patch_match_src_images",
        type=int,
        default=None,
        help=(
            "image_undistorter --num_patch_match_src_images, i.e. how many source "
            "views each reference image gets in stereo/patch-match.cfg. Default: 20."
        ),
    )

    # ---- MVS: PatchMatch Stereo + Stereo Fusion tunables.
    # Registered in a loop so the option name in the CLI, in the COLMAP flag and
    # (via the backend) in the API stay spelled the same way.
    pms_group = parser.add_argument_group(
        "MVS / PatchMatch Stereo",
        "Forwarded to --PatchMatchStereo.<name>. Omit to use COLMAP's default.",
    )
    for name, (typ, help_text) in PATCH_MATCH_PARAMS.items():
        pms_group.add_argument(f"--pms_{name}", type=typ, default=None, help=help_text)

    fusion_group = parser.add_argument_group(
        "MVS / Stereo Fusion",
        "Forwarded to --StereoFusion.<name>. Omit to use COLMAP's default.",
    )
    for name, (typ, help_text) in FUSION_PARAMS.items():
        fusion_group.add_argument(f"--fusion_{name}", type=typ, default=None, help=help_text)
    fusion_group.add_argument(
        "--fusion_input_type",
        type=str,
        choices=FUSION_INPUT_TYPES,
        default="auto",
        help=(
            "Which depth maps stereo_fusion reads. 'auto' (default) follows "
            "geom_consistency: geometric when on, photometric when off. "
            "Naming 'geometric' while geom_consistency is off is an error."
        ),
    )

    # ---- Output control ----
    parser.add_argument(
        "--convert_output_suffix",
        type=str,
        default="txt",
        help="Output format suffix for model_converter step. Default: txt.",
    )
    parser.add_argument(
        "--no_clean",
        action="store_true",
        help="Do NOT clean the output directory before running (default: cleans).",
    )
    parser.add_argument(
        "--json_output",
        type=Path,
        default=None,
        help="Custom path for the JSON result file. Default: <output_dir>/pipeline_result.json.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the COLMAP SFM/MVS pipeline."""
    args = parse_args(argv)
    t_start = time.time()

    # ---- Validate arguments ----
    # Done before any filesystem work so a misuse is reported as a misuse
    # rather than as a missing directory.
    requested_steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for s in requested_steps:
        if s not in STEP_CHOICES:
            print(f"ERROR: Unknown step '{s}'. Choices: {','.join(STEP_CHOICES)}", file=sys.stderr)
            return 1

    # Reconcile the two ways of expressing geometric consistency:
    # --no_geom_consistency is the historical flag (the SfM page drives it);
    # --geom_consistency 0/1 is the tri-state form. Accept either, never both.
    if args.geom_consistency is not None and args.no_geom_consistency:
        print(
            "ERROR: --geom_consistency and --no_geom_consistency are mutually "
            "exclusive; pass only one.",
            file=sys.stderr,
        )
        return 1
    geom_consistency = resolve_geom_consistency(args)

    # Tunable dicts → appended as --PatchMatchStereo.<k> / --StereoFusion.<k>.
    # None values are dropped by append_opt, so COLMAP's own default applies.
    patch_match_opts, fusion_opts = collect_mvs_opts(args)

    # ---- Validate inputs ----
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not image_dir.is_dir():
        print(f"ERROR: image_dir not found: {image_dir}", file=sys.stderr)
        return 1

    num_images = count_images(image_dir)
    if num_images == 0:
        print(f"ERROR: No supported images found in {image_dir}", file=sys.stderr)
        print(f"  Supported formats: {', '.join(SUPPORTED_IMAGE_EXTS)}", file=sys.stderr)
        return 1

    # ---- Setup directories ----
    if not args.no_clean:
        if output_dir.exists():
            print(f"Cleaning existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    database_path = output_dir / "database.db"
    sparse_dir = args.sparse_dir or (output_dir / "sparse")
    dense_dir = args.dense_dir or (output_dir / "dense")
    log_file = output_dir / "pipeline.log"
    json_path = args.json_output or (output_dir / "pipeline_result.json")

    # ---- Print header ----
    print("=" * 60)
    print("  COLMAP SFM/MVS Pipeline")
    print("=" * 60)
    print(f"  Image dir:      {image_dir}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Images found:   {num_images}")
    print(f"  Steps:          {', '.join(requested_steps)}")
    print(f"  Matcher:        {args.matcher}")
    print(f"  Camera model:   {args.camera_model}")
    print(f"  GPU:            {'on' if args.use_gpu else 'off'} (index {args.gpu_index})")
    print(f"  Single camera:  {'yes' if args.single_camera else 'no'}")
    print(f"  Geom consist.:  {'on' if geom_consistency else 'off'}")
    n_overrides = (
        sum(1 for v in patch_match_opts.values() if v is not None)
        + sum(1 for v in fusion_opts.values() if v is not None)
    )
    print(f"  MVS overrides:  {n_overrides} (rest use COLMAP defaults)")
    print("=" * 60)

    # ---- Run pipeline steps ----
    step_results: dict[str, Any] = {}
    steps_run: list[str] = []

    step_map = {
        "extract": lambda: step_extract(
            image_dir, database_path,
            single_camera=bool(args.single_camera),
            camera_model=args.camera_model,
            use_gpu=bool(args.use_gpu),
            gpu_index=args.gpu_index,
            max_num_features=args.max_num_features,
            log_file=log_file,
        ),
        "match": lambda: step_match(
            database_path,
            matcher=args.matcher,
            use_gpu=bool(args.use_gpu),
            gpu_index=args.gpu_index,
            vocab_tree_path=args.vocab_tree_path,
            log_file=log_file,
        ),
        "sfm": lambda: step_sfm(
            database_path, image_dir, sparse_dir,
            log_file=log_file,
        ),
        "undistort": lambda: step_undistort(
            image_dir, sparse_dir, dense_dir,
            reconstruction_name=args.reconstruction_name,
            max_image_size=args.undistort_max_image_size,
            num_patch_match_src_images=args.num_patch_match_src_images,
            log_file=log_file,
        ),
        "reuse": lambda: step_reuse_dense(
            args.reuse_dense_dir, dense_dir,
            log_file=log_file,
        ),
        "mvs": lambda: step_mvs(
            dense_dir,
            use_gpu=bool(args.use_gpu),
            gpu_index=args.gpu_index,
            geom_consistency=geom_consistency,
            patch_match_opts=patch_match_opts,
            fusion_opts=fusion_opts,
            fusion_input_type=args.fusion_input_type,
            log_file=log_file,
        ),
        "fusion": lambda: step_mvs(
            dense_dir,
            use_gpu=bool(args.use_gpu),
            gpu_index=args.gpu_index,
            geom_consistency=geom_consistency,
            patch_match_opts=patch_match_opts,
            fusion_opts=fusion_opts,
            fusion_input_type=args.fusion_input_type,
            log_file=log_file,
        ),
        "convert": lambda: step_convert(
            sparse_dir,
            model_name=args.reconstruction_name,
            output_suffix=args.convert_output_suffix,
            log_file=log_file,
        ),
    }

    try:
        for step in requested_steps:
            result = step_map[step]()
            step_results[step] = result
            steps_run.append(step)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        step_results["_interrupted"] = True
    except Exception as exc:
        print(f"\nPipeline error at step '{steps_run[-1] if steps_run else '?'}': {exc}", file=sys.stderr)
        step_results["_error"] = str(exc)
        # Continue to write partial results

    # ---- Write result JSON ----
    elapsed = time.time() - t_start
    pipeline_result = build_pipeline_result(args, steps_run, step_results, output_dir, elapsed)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_result, f, indent=2, ensure_ascii=False, default=str)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Pipeline Summary")
    print(f"{'='*60}")
    print(f"  Steps completed:  {', '.join(steps_run)}")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Result JSON:      {json_path}")
    print(f"  Sparse dir:       {sparse_dir}")
    print(f"  Dense dir:        {dense_dir}")
    fused_ply = dense_dir / "fused.ply"
    if fused_ply.exists():
        print(f"  Point cloud:      {fused_ply}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
