"""FastAPI backend for the COLMAP SfM/MVS Platform."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import ValidationError

from modules.docker_logger import start_global_monitor
from modules.docker_runner import (
    build_sfm_command,
    get_process,
    is_process_running,
    remove_process,
    run_pipeline_async,
)
from modules.image_preprocessor import strip_metadata, undistort_images
from algorithm.pointcloud_filter import run as run_pointcloud_filter
from algorithm.image_mask import run as run_image_mask
from algorithm.mask_obj import run as run_mask_obj
from modules.logger import get_logger, request_log
from modules.models import (
    CameraModel,
    ColmapMVSParams,
    MatcherType,
    MVSCommand,
    MVSConfig,
    MVSType,
    PipelineConfig,
    PipelineStep,
    TaskStatus,
)
from modules.mvs_runner import (
    build_mvs_command,
    get_mvs_process,
    get_mvs_run_status,
    is_mvs_running,
    remove_mvs_process,
    run_mvs_async,
)
from modules.mask_obj_runner import (
    get_mask_obj_run_status,
    is_mask_obj_running,
    run_mask_obj_async,
)
from modules.gs_runner import (
    GS_COMMANDS,
    build_gs_command,
    get_gs_process,
    get_gs_run_status,
    is_gs_running,
    remove_gs_process,
    run_gs_async,
)
from modules.sam_runner import (
    build_sam_command,
    get_sam_process,
    is_sam_running,
    remove_sam_process,
    run_sam_async,
)
from modules.video_api import video_router
from modules.task_manager import (
    DATA_DIR,
    _algorithm_output_dir,
    _input_image_dir,
    _mvs_output_dir,
    _sfm_output_dir,
    count_images,
    create_project,
    delete_project,
    get_log_tail,
    get_project,
    list_projects,
    sync_project_from_result,
    update_project,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="SfM Platform", version="0.2.0")

# 视频抽帧（video_processor/video_runner/video_api）
app.include_router(video_router)


@app.on_event("startup")
async def _startup():
    """Start global Docker log monitor on boot (auto-follows all containers)."""
    start_global_monitor()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    dt = (time.time() - t0) * 1000
    request_log(request.method, request.url.path, response.status_code, dt)
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_run_steps(project_name: str) -> list[str] | None:
    """Steps of the newest pipeline run, from its ``=== PIPELINE START`` line.

    Returns None when the log is missing/unparseable — callers then fall back
    to trusting the result file (legacy behaviour) rather than guessing.
    """
    lp = _sfm_output_dir(project_name) / "pipeline.log"
    if not lp.is_file():
        return None
    try:
        with open(lp, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        if line.startswith("=== PIPELINE START"):
            idx = line.find(" --steps ")
            if idx < 0:
                return None
            tail = line[idx + len(" --steps "):].split()
            if not tail:
                return None
            return [s.strip() for s in tail[0].split(",") if s.strip()]
    return None


def _pipeline_result_matches_run(project_name: str) -> bool:
    """Is pipeline_result.json the product of the newest run (not a stale one)?

    sfm_cli.py writes pipeline_result.json only at the very end of a run, and
    records the exact steps_requested. docker exec's client exit code is NOT
    trustworthy — when the kernel OOM-kills colmap (-9) mid-run the wrapper
    often still exits 0, leaving an old result file behind. So a result only
    counts as this run's output when its requested steps equal the newest
    ``=== PIPELINE START``'s ``--steps``.
    """
    want = _last_run_steps(project_name)
    if want is None:
        return True  # cannot verify → keep legacy behaviour
    from modules.task_manager import load_pipeline_result
    result = load_pipeline_result(project_name)
    if result is None:
        return False
    requested = (result.get("pipeline") or {}).get("steps_requested", [])
    if isinstance(requested, str):
        requested = [s.strip() for s in requested.split(",") if s.strip()]
    return sorted(requested) == sorted(want)


def _mark_pipeline_failed(project_name: str, message: str) -> None:
    update_project(
        project_name,
        status=TaskStatus.FAILED.value,
        error_message=message,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )


def _reconcile_orphan_colmap(project_name: str) -> None:
    """Resolve a task stuck in ``running`` whose runner registry entry is gone.

    Happens when a pipeline survives a backend restart: docker_runner's
    in-memory ``_running_processes`` dies with the old process, but sfm_cli.py
    keeps running inside colmap-dev (and keeps writing depth maps). When that
    container process is still alive we leave the task running — the /progress
    endpoint counts its artifacts live. When it is gone the run ended without a
    host callback, so we settle status from pipeline_result.json **only if it
    belongs to the newest run** (see _pipeline_result_matches_run); otherwise
    the run was interrupted (OOM kill, crash, …) → failed.
    """
    p = get_project(project_name)
    if p is None or p.get("status") != TaskStatus.RUNNING.value:
        return

    # Imported lazily to keep pipeline_progress (which imports docker_runner +
    # mvs_runner) from touching this module's import graph.
    from modules.pipeline_progress import _colmap_process_alive
    if _colmap_process_alive(project_name):
        return  # genuinely still running (orphan) — live progress continues

    if _pipeline_result_matches_run(project_name):
        sync_project_from_result(project_name)
    else:
        logger.warning(
            "Project %r: orphan run ended without its own result file — "
            "marking failed (interrupted / OOM)", project_name)
        _mark_pipeline_failed(
            project_name,
            "运行中断（进程已结束，未产出本次运行的结果——疑似被系统终止，如内存不足）")


def _check_process_completion(project_name: str) -> None:
    """If the process for project has exited, sync results and clean up."""
    proc = get_process(project_name)
    if proc is None:
        # No registry entry: nothing running, or a container process orphaned by
        # a backend restart. Reconcile a stuck ``running`` task either way.
        _reconcile_orphan_colmap(project_name)
        return
    if proc.poll() is not None:
        if proc.returncode == 0 and _pipeline_result_matches_run(project_name):
            logger.info("Project %r pipeline finished successfully", project_name)
            sync_project_from_result(project_name)
        else:
            if proc.returncode == 0:
                # docker exec returned 0 but no result for THIS run was written
                # (classic OOM signature: colmap killed -9, wrapper survived).
                logger.error(
                    "Project %r pipeline exit 0 but no matching result — "
                    "likely OOM-killed mid-run", project_name)
                _mark_pipeline_failed(
                    project_name,
                    "运行中断（进程退出码为 0 但未产出本次结果——疑似内存不足 OOM 被杀）")
            else:
                logger.error("Project %r pipeline FAILED (exit code %s)",
                             project_name, proc.returncode)
                _mark_pipeline_failed(
                    project_name,
                    f"Pipeline exited with code {proc.returncode}")
        remove_process(project_name)


def _start_pipeline(project_name: str) -> None:
    """Build command, mark running, launch pipeline."""
    p = get_project(project_name)
    if p is None:
        logger.warning("_start_pipeline for unknown project %r", project_name)
        return

    config_dict = p.get("config", {})
    steps = config_dict.get("steps", ["extract", "match", "sfm", "undistort", "mvs"])
    mvs_params, _ = _stored_colmap_mvs(p)

    # Strip IPTC/EXIF metadata before pipeline to prevent OpenImageIO crash
    prep_result = strip_metadata(project_name)
    logger.info("Project %r: pre-run metadata check — success=%s, processed=%s, error=%s",
                project_name, prep_result.get("success"), prep_result.get("processed"),
                prep_result.get("error"))

    cmd = build_sfm_command(
        project_name,
        steps=steps,
        matcher=config_dict.get("matcher", "exhaustive"),
        camera_model=config_dict.get("camera_model", "SIMPLE_RADIAL"),
        single_camera=config_dict.get("single_camera", True),
        use_gpu=config_dict.get("use_gpu", True),
        gpu_index=config_dict.get("gpu_index", 0),
        max_num_features=config_dict.get("max_num_features", 0),
        no_geom_consistency=config_dict.get("no_geom_consistency", False),
        reconstruction_name=config_dict.get("reconstruction_name", "0"),
        vocab_tree_path=config_dict.get("vocab_tree_path"),
        convert_output_suffix=config_dict.get("convert_output_suffix", "txt"),
        mvs_type=config_dict.get("mvs_type", "colmap"),
        mvs_params=mvs_params,
    )

    logger.info("Project %r starting pipeline: %s", project_name, " ".join(cmd))

    # run_pipeline_async 内部会先确保 colmap-dev 容器在跑（调用即启动），
    # 成功返回后才标记 RUNNING：容器起不来时不会留下悬空运行状态。
    run_pipeline_async(project_name, cmd)

    update_project(
        project_name,
        status=TaskStatus.RUNNING.value,
        started_at=datetime.now(timezone.utc).isoformat(),
        error_message=None,
    )


# ---------------------------------------------------------------------------
# API Routes — Projects
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/projects")
async def list_all_projects():
    """List all projects."""
    projects = list_projects()
    summaries = []
    for p in projects:
        pn = p.get("project_name") or p.get("task_id", "")
        if not pn:
            continue
        _check_process_completion(pn)
        p = get_project(pn) or p
        summaries.append({
            "project_name": pn,
            "name": p.get("name", ""),
            "status": p.get("status", TaskStatus.PENDING.value),
            "num_images": p.get("num_images", 0),
            "created_at": p.get("created_at", ""),
            "elapsed_seconds": p.get("elapsed_seconds"),
        })
    return {"projects": summaries}


@app.post("/api/projects")
async def create_new_project(
    name: str = Form(...),
):
    """Create a new project (empty scaffolding, no images yet)."""
    p = create_project(name=name)
    return {
        "success": True,
        "message": f"Project {p['project_name']} created",
        "data": {"project_name": p["project_name"]},
    }


@app.get("/api/projects/{project_name}")
async def get_project_detail(project_name: str):
    """Get full project detail."""
    _check_process_completion(project_name)

    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    config_dict = p.get("config", {})

    return {
        "project_name": p["project_name"],
        "name": p.get("name", ""),
        "status": p.get("status", TaskStatus.PENDING.value),
        "config": config_dict,
        "num_images": p.get("num_images", 0),
        "created_at": p.get("created_at", ""),
        "started_at": p.get("started_at"),
        "finished_at": p.get("finished_at"),
        "error_message": p.get("error_message"),
        "pipeline": p.get("pipeline"),
        "steps": p.get("steps", {}),
        "log_tail": get_log_tail(project_name),
    }


@app.delete("/api/projects/{project_name}")
async def delete_project_endpoint(project_name: str):
    """Delete a project."""
    if is_process_running(project_name):
        raise HTTPException(400, "Project pipeline is still running")
    remove_process(project_name)
    deleted = delete_project(project_name)
    if not deleted:
        raise HTTPException(404, "Project not found")
    logger.info("Project %r deleted via API", project_name)
    return {"success": True, "message": f"Project {project_name} deleted"}


# ---------------------------------------------------------------------------
# API Routes — Images
# ---------------------------------------------------------------------------

@app.post("/api/projects/{project_name}/images")
async def upload_images(
    project_name: str,
    images: list[UploadFile] = File(...),
):
    """Upload images to a project's input_image directory."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    if is_process_running(project_name):
        raise HTTPException(400, "Project pipeline is running — wait for it to finish")

    img_dir = _input_image_dir(project_name)

    for file in images:
        filename = file.filename or "unknown"
        content = await file.read()

        if filename.lower().endswith(".zip"):
            zip_path = img_dir / "_upload.zip"
            zip_path.write_bytes(content)
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    for member in zf.namelist():
                        if member.endswith("/") or "__MACOSX" in member:
                            continue
                        base = Path(member).name
                        if not base:
                            continue
                        ext = Path(member).suffix.lower()
                        if ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff",
                                   ".bmp", ".pgm", ".ppm"}:
                            zf.extract(member, img_dir)
                            extracted = img_dir / member
                            target = img_dir / base
                            if extracted != target:
                                shutil.move(str(extracted), str(target))
            except zipfile.BadZipFile:
                zip_path.unlink()
                raise HTTPException(400, "Invalid zip file")
            zip_path.unlink()
        else:
            ext = Path(filename).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".tif", ".tiff",
                           ".bmp", ".pgm", ".ppm"}:
                continue
            dest = img_dir / filename
            dest.write_bytes(content)

    # Strip IPTC/EXIF metadata to prevent OpenImageIO crash during undistortion
    prep_result = await asyncio.to_thread(strip_metadata, project_name)
    logger.info("Project %r: metadata preprocessing — success=%s, processed=%s",
                project_name, prep_result.get("success"), prep_result.get("processed"))

    num = count_images(project_name)
    update_project(project_name, num_images=num)
    logger.info("Project %r: uploaded images, now %d total", project_name, num)

    return {"success": True, "message": f"{num} images in project", "data": {"num_images": num}}


def _undistorted_image_dir(project_name: str):
    """去畸变图像目录（SfM 去畸变步骤产出，SAM/掩码均在此空间操作）。"""
    return _sfm_output_dir(project_name) / "dense" / "images"


@app.get("/api/projects/{project_name}/images/list")
async def list_project_images(project_name: str):
    """List all undistorted image filenames in a project.

    SAM 在去畸变图上分割（SFM_output/dense/images），这里列出同一份图。
    未跑 SfM 去畸变时返回空列表。
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    img_dir = _undistorted_image_dir(project_name)
    if not img_dir.is_dir():
        return {"project_name": project_name, "images": []}

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pgm", ".ppm"}
    files = sorted([
        f.name for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    ])
    return {"project_name": project_name, "images": files}


@app.get("/api/projects/{project_name}/images/view/{filename:path}")
async def view_project_image(project_name: str, filename: str):
    """Serve an undistorted image file (SFM_output/dense/images)."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    safe = os.path.normpath(filename)
    if safe.startswith("..") or safe.startswith("/"):
        raise HTTPException(400, "Invalid filename")

    img_dir = _undistorted_image_dir(project_name)
    file_path = img_dir / safe
    if not file_path.is_file():
        raise HTTPException(404, f"Image not found: {filename}")

    ext = file_path.suffix.lower()
    media_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".tif": "image/tiff", ".tiff": "image/tiff",
        ".bmp": "image/bmp", ".pgm": "image/x-portable-graymap",
        ".ppm": "image/x-portable-pixmap",
    }
    return FileResponse(file_path, media_type=media_map.get(ext, "image/png"))


# ---------------------------------------------------------------------------
# API Routes — Image Preprocessing
# ---------------------------------------------------------------------------

@app.post("/api/projects/{project_name}/preprocess/undistort")
async def undistort_project_images(project_name: str):
    """Run COLMAP image_undistorter to remove lens distortion.

    Requires that SfM has already completed (sparse/0 must exist).
    Output: SFM_output/undistorted/ containing undistorted images (images/)
    and a new sparse model with PINHOLE cameras (sparse/).
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")
    if is_process_running(project_name):
        raise HTTPException(400, "SfM pipeline is running — wait for it to finish")

    # Check prerequisite: SfM sparse model must exist
    sparse_dir = _sfm_output_dir(project_name) / "sparse" / "0"
    if not sparse_dir.is_dir():
        raise HTTPException(
            400,
            f"SfM sparse model not found at {sparse_dir}. Run the SfM pipeline first.",
        )

    logger.info("Project %r: starting image undistortion", project_name)
    result = await asyncio.to_thread(undistort_images, project_name)

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Undistortion failed"))

    return {
        "success": True,
        "message": f"Undistorted {result['num_images']} images",
        "data": {
            "output_dir": result["output_dir"],
            "num_images": result["num_images"],
        },
    }


@app.post("/api/projects/{project_name}/preprocess/mask-images")
async def mask_project_images(
    project_name: str,
    prompt_slug: str = Form(default="object"),
):
    """Apply SAM masks to images, producing RGBA PNGs with transparent background.

    Uses existing SAM masks in SAM_output/.
    Output: algorithm_output/input_masked/
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    logger.info("Project %r: starting image masking (prompt=%s)", project_name, prompt_slug)
    result = await asyncio.to_thread(run_image_mask, project_name, prompt_slug)

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Image masking failed"))

    return {
        "success": True,
        "message": f"Masked {result['num_images']} images",
        "data": {
            "output_dir": result["output_dir"],
            "num_images": result["num_images"],
        },
    }


@app.post("/api/projects/{project_name}/preprocess/filter-pointcloud")
async def filter_project_pointcloud(
    project_name: str,
    threshold: float = Form(default=0.3),
    prompt_slug: str = Form(default="object"),
    ply_path: str = Form(default=""),
):
    """Filter MVS dense point cloud using multi-view SAM mask consistency.

    Projects each 3D point into all camera views; keeps points that fall
    inside SAM mask regions in most views.

    ply_path: 指定输入点云（COLMAP fused.ply / OpenMVS scene_dense.ply 等）。
        留空 → 自动检测；相对路径 → 相对项目目录解析。
    Requires: MVS dense PLY + COLMAP sparse model + SAM masks.
    Output: algorithm_output/filtered_scene.ply
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(400, "threshold must be between 0.0 and 1.0")

    # 解析输入点云：空 → 自动检测；相对路径 → 相对项目 DATA 目录
    src = ply_path.strip()
    if src and not os.path.isabs(src):
        src = str(DATA_DIR / project_name / src)
    src_or_none = src or None

    logger.info("Project %r: starting pointcloud filter "
                "(threshold=%.2f, prompt=%s, ply=%s)",
                project_name, threshold, prompt_slug, src_or_none)
    result = await asyncio.to_thread(
        run_pointcloud_filter, project_name, threshold, prompt_slug,
        ply_path=src_or_none,
    )

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Point cloud filtering failed"))

    return {
        "success": True,
        "message": f"Kept {result['num_points_after']:,} / {result['num_points_before']:,} points",
        "data": {
            "output_path": result["output_path"],
            "input_path": result.get("input_path", src_or_none or ""),
            "num_points_before": result["num_points_before"],
            "num_points_after": result["num_points_after"],
        },
    }


@app.get("/api/projects/{project_name}/filter/result")
async def filter_result_status(project_name: str):
    """Return status of the filtered point cloud (exists, size, point count).

    Reads only the PLY header to count vertices — cheap even for large files.
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    out = _sfm_output_dir(project_name).parent / "algorithm_output" / "filtered_scene.ply"
    if not out.exists():
        return {"exists": False, "output_path": "", "size_bytes": 0, "num_points": 0}

    try:
        n = _count_ply_points(out)
    except Exception:
        n = 0

    return {
        "exists": True,
        "output_path": str(out),
        "size_bytes": out.stat().st_size,
        "num_points": n,
    }


@app.get("/api/projects/{project_name}/filter/download/{filename:path}")
async def download_filter_file(project_name: str, filename: str):
    """Download a result file from algorithm_output directory."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    safe_path = os.path.normpath(filename)
    if safe_path.startswith("..") or safe_path.startswith("/"):
        raise HTTPException(400, "Invalid filename")

    out_dir = _sfm_output_dir(project_name).parent / "algorithm_output"
    file_path = out_dir / safe_path
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    return FileResponse(file_path, filename=Path(filename).name,
                        media_type="application/octet-stream")


def _count_ply_points(path: Path) -> int:
    """Count vertices in a PLY by reading only its header."""
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[2])
            if line == "end_header":
                break
    return 0


# ---------------------------------------------------------------------------
# COLMAP MVS tunables — request plumbing
# ---------------------------------------------------------------------------

#: Top-level project key holding the last COLMAP MVS settings. Deliberately NOT
#: inside "config": update_project is a shallow dict.update, so writing
#: config=... replaces the whole dict and would drop the SfM settings
#: _start_pipeline reads back. Same precedent as OpenMVS's "mvs_config".
COLMAP_MVS_PARAMS_KEY = "colmap_mvs_params"


async def colmap_mvs_form(
    pms_max_image_size: Optional[int] = Form(default=None),
    pms_window_radius: Optional[int] = Form(default=None),
    pms_window_step: Optional[int] = Form(default=None),
    pms_num_samples: Optional[int] = Form(default=None),
    pms_num_iterations: Optional[int] = Form(default=None),
    pms_cache_size: Optional[int] = Form(default=None),
    pms_geom_consistency_max_cost: Optional[float] = Form(default=None),
    pms_filter: Optional[int] = Form(default=None),
    pms_filter_min_ncc: Optional[float] = Form(default=None),
    pms_filter_min_triangulation_angle: Optional[float] = Form(default=None),
    pms_filter_min_num_consistent: Optional[int] = Form(default=None),
    pms_filter_geom_consistency_max_cost: Optional[float] = Form(default=None),
    fusion_input_type: Optional[str] = Form(default=None),
    fusion_max_image_size: Optional[int] = Form(default=None),
    fusion_min_num_pixels: Optional[int] = Form(default=None),
    fusion_max_num_pixels: Optional[int] = Form(default=None),
    fusion_max_reproj_error: Optional[float] = Form(default=None),
    fusion_max_depth_error: Optional[float] = Form(default=None),
    fusion_max_normal_error: Optional[float] = Form(default=None),
    fusion_check_num_images: Optional[int] = Form(default=None),
) -> ColmapMVSParams:
    """Gather the COLMAP dense-reconstruction tunables from flat form fields.

    Declared field-by-field on purpose. Binding an ``Annotated[ColmapMVSParams,
    Form()]`` model alongside ``/mvs/run``'s existing scalar Form parameters
    makes FastAPI embed it as a single nested field, so the flat fields the
    client actually sends are silently dropped — HTTP 200 with every value
    null. Scalar Form parameters have no such trap.

    Bounds live on the model, not on ``Form(...)``, so an out-of-range value
    comes back as a readable 400 naming the field while a non-numeric one stays
    FastAPI's own 422. Every field defaults to None = "let COLMAP decide", so a
    request that sets nothing produces no flags at all.
    """
    try:
        return ColmapMVSParams(
            pms_max_image_size=pms_max_image_size,
            pms_window_radius=pms_window_radius,
            pms_window_step=pms_window_step,
            pms_num_samples=pms_num_samples,
            pms_num_iterations=pms_num_iterations,
            pms_cache_size=pms_cache_size,
            pms_geom_consistency_max_cost=pms_geom_consistency_max_cost,
            pms_filter=pms_filter,
            pms_filter_min_ncc=pms_filter_min_ncc,
            pms_filter_min_triangulation_angle=pms_filter_min_triangulation_angle,
            pms_filter_min_num_consistent=pms_filter_min_num_consistent,
            pms_filter_geom_consistency_max_cost=pms_filter_geom_consistency_max_cost,
            fusion_input_type=fusion_input_type,
            fusion_max_image_size=fusion_max_image_size,
            fusion_min_num_pixels=fusion_min_num_pixels,
            fusion_max_num_pixels=fusion_max_num_pixels,
            fusion_max_reproj_error=fusion_max_reproj_error,
            fusion_max_depth_error=fusion_max_depth_error,
            fusion_max_normal_error=fusion_max_normal_error,
            fusion_check_num_images=fusion_check_num_images,
        )
    except ValidationError as exc:
        first = exc.errors()[0]
        # loc is empty for whole-model validators (the min/max pixel check),
        # where naming a field would be misleading.
        field = ".".join(str(part) for part in first["loc"])
        msg = str(first["msg"]).removeprefix("Value error, ")
        where = f"{field} — " if field else ""
        raise HTTPException(400, f"COLMAP MVS 参数不合法：{where}{msg}")


def _stored_colmap_mvs(project: dict) -> tuple[ColmapMVSParams, bool]:
    """Rebuild persisted COLMAP MVS settings as ``(params, no_geom_consistency)``.

    The geometric-consistency switch rides along in the same dict because it and
    the fusion ``input_type`` must agree (see the CLI's auto-selection); keeping
    them in one place means resume cannot read back a mismatched pair. Projects
    from before these knobs existed have no such key and get plain defaults.
    """
    stored = dict(project.get(COLMAP_MVS_PARAMS_KEY) or {})
    geom_off = bool(stored.pop("no_geom_consistency", False))
    try:
        return ColmapMVSParams(**stored), geom_off
    except ValidationError:
        # Only reachable via a hand-edited task.json. Fall back to defaults
        # rather than blocking the run, but say so in the log.
        logger.warning("Ignoring unreadable %s: %r", COLMAP_MVS_PARAMS_KEY, stored)
        return ColmapMVSParams(), geom_off


# ---------------------------------------------------------------------------
# API Routes — Pipeline
# ---------------------------------------------------------------------------

@app.post("/api/projects/{project_name}/run")
async def run_pipeline(
    project_name: str,
    steps: str = Form(default="extract,match,sfm,undistort,mvs"),
    matcher: str = Form(default="exhaustive"),
    camera_model: str = Form(default="SIMPLE_RADIAL"),
    single_camera: bool = Form(default=True),
    use_gpu: bool = Form(default=True),
    gpu_index: int = Form(default=0),
    max_num_features: int = Form(default=8192),
    no_geom_consistency: bool = Form(default=False),
    reconstruction_name: str = Form(default="0"),
    convert_output_suffix: str = Form(default="txt"),
    mvs_type: str = Form(default="colmap"),
    mvs_params: ColmapMVSParams = Depends(colmap_mvs_form),
):
    """Configure and run the SfM pipeline for a project."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")
    if is_process_running(project_name):
        raise HTTPException(400, "Pipeline is already running")

    # Validate
    if matcher not in {m.value for m in MatcherType}:
        raise HTTPException(400, f"Invalid matcher: {matcher}")
    if camera_model not in {c.value for c in CameraModel}:
        raise HTTPException(400, f"Invalid camera_model: {camera_model}")

    step_list = [s.strip() for s in steps.split(",") if s.strip()]
    for s in step_list:
        if s not in {ps.value for ps in PipelineStep}:
            raise HTTPException(400, f"Invalid step: {s}")

    # Validate mvs_type
    if mvs_type not in {m.value for m in MVSType}:
        raise HTTPException(400, f"Invalid mvs_type: {mvs_type}. Choices: {[m.value for m in MVSType]}")

    config = PipelineConfig(
        steps=step_list,
        mvs_type=MVSType(mvs_type),
        matcher=MatcherType(matcher),
        camera_model=CameraModel(camera_model),
        single_camera=single_camera,
        use_gpu=use_gpu,
        gpu_index=gpu_index,
        max_num_features=max_num_features,
        no_geom_consistency=no_geom_consistency,
        reconstruction_name=reconstruction_name,
        convert_output_suffix=convert_output_suffix,
    )

    # config is replaced wholesale here (that is this endpoint's contract), but
    # the MVS tunables still go in their own key: resume reads them from there,
    # and _start_pipeline resolves both keys the same way for every entry point.
    update_project(
        project_name,
        config=config.model_dump(),
        **{COLMAP_MVS_PARAMS_KEY: {
            **mvs_params.model_dump(exclude_none=True),
            "no_geom_consistency": no_geom_consistency,
        }},
    )
    logger.info("Project %r: config updated, starting pipeline", project_name)
    await asyncio.to_thread(_start_pipeline, project_name)

    return {"success": True, "message": f"Pipeline started for {project_name}"}


@app.post("/api/projects/{project_name}/stop")
async def stop_pipeline(project_name: str):
    """Stop a running pipeline — kill the subprocess and update status."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    if not is_process_running(project_name):
        raise HTTPException(400, "No pipeline is running for this project")

    proc = get_process(project_name)
    if proc is not None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        except Exception as e:
            logger.warning("Project %r: error killing process: %s", project_name, e)

    remove_process(project_name)
    update_project(
        project_name,
        status=TaskStatus.FAILED.value,
        error_message="Pipeline stopped by user",
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info("Project %r: pipeline stopped by user", project_name)
    return {"success": True, "message": f"Pipeline for {project_name} stopped"}


@app.get("/api/projects/{project_name}/log")
async def get_pipeline_log(project_name: str, lines: int = 100):
    """Get pipeline log."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")
    log_text = get_log_tail(project_name, lines=lines)
    return {"project_name": project_name, "log": log_text or "(no log yet)"}


@app.get("/api/projects/{project_name}/progress")
async def get_pipeline_progress(project_name: str):
    """Host-side pipeline progress snapshot (steps + determinate MVS %).

    The GUI polls this while COLMAP/OpenMVS is running to drive a determinate
    progress bar and a step label. COLMAP's colmap stdout is buffered by
    sfm_cli.py, so the MVS phase is measured by counting the per-image depth
    maps colmap writes onto the bind-mounted DATA dir.
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    from modules.pipeline_progress import compute_pipeline_progress

    return compute_pipeline_progress(project_name)


@app.get("/api/projects/{project_name}/download/{filename:path}")
async def download_file(project_name: str, filename: str):
    """Download a result file from SFM_output directory."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    out_dir = _sfm_output_dir(project_name)
    safe_path = os.path.normpath(filename)
    if safe_path.startswith("..") or safe_path.startswith("/"):
        raise HTTPException(400, "Invalid filename")

    file_path = out_dir / safe_path
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    return FileResponse(file_path, filename=Path(filename).name,
                        media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Helpers — SAM
# ---------------------------------------------------------------------------

SAM_RESULT_MANIFEST = "manifest.json"


def _check_sam_completion(project_name: str) -> None:
    """If the SAM process for project has exited, update status."""
    proc = get_sam_process(project_name)
    if proc is None:
        return
    if proc.poll() is not None:
        if proc.returncode == 0:
            logger.info("Project %r SAM finished successfully", project_name)
        else:
            logger.error("Project %r SAM FAILED (exit code %s)", project_name, proc.returncode)
        remove_sam_process(project_name)


async def _ensure_container(container_name: str, create_hint: str) -> None:
    """Make sure a docker container is running; start it if stopped."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "inspect", "-f", "{{.State.Running}}", container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0 and stdout.decode().strip() == "true":
            return  # already running
    except FileNotFoundError:
        raise HTTPException(500, "Docker not available")

    # Container exists but not running, or doesn't exist — start it
    logger.info("%s container not running, starting...", container_name)
    try:
        # Try docker start first (container exists but stopped)
        proc = await asyncio.create_subprocess_exec(
            "docker", "start", container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker start {container_name} failed")
    except Exception:
        raise HTTPException(
            500,
            f"{container_name} container is not available. Please create it first:\n"
            f"  {create_hint}",
        )

    # Wait for container to be ready
    for _ in range(30):
        await asyncio.sleep(1)
        proc = await asyncio.create_subprocess_exec(
            "docker", "inspect", "-f", "{{.State.Running}}", container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0 and stdout.decode().strip() == "true":
            logger.info("%s container is now running", container_name)
            return
    raise HTTPException(500, f"{container_name} container started but not ready after 30s")


async def _ensure_sam3_container() -> None:
    await _ensure_container(
        "sam3",
        "docker run -d --name sam3 --gpus all -v /path/to/data:/data sam3-image",
    )


# ---------------------------------------------------------------------------
# 持久 SAM Worker 管理（模型常驻 GPU）
# ---------------------------------------------------------------------------

_worker_proc: asyncio.subprocess.Process | None = None
_worker_lock = asyncio.Lock()


async def _ensure_sam3_worker() -> None:
    """Ensure the persistent SAM worker is running (model stays in GPU).

    Spawns `docker exec -i sam3 python3 /workspace/sam_worker.py --serve`
    and waits for the ready signal.  Uses _worker_lock to serialize
    all communication with the single worker process.
    """
    global _worker_proc

    await _ensure_sam3_container()

    # Already running?
    if _worker_proc is not None:
        if _worker_proc.returncode is None:
            # Health check via status command
            try:
                resp = await _send_worker_command({"cmd": "status"}, timeout=5)
                if resp.get("status") == "ready":
                    return
            except Exception:
                pass  # Worker died, restart below
        # Clean up dead process
        try:
            await _worker_proc.wait()
        except Exception:
            pass
        _worker_proc = None

    logger.info("Starting persistent SAM worker...")
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", "--user", f"{os.getuid()}:{os.getgid()}", "-i", "sam3",
        "python3", "/workspace/sam_worker.py",
        "--serve",
        "--checkpoint", "/workspace/model/sam3.pt",
        "--device", "cuda",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Wait for ready signal (timeout 120s for model loading)
    try:
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(500, "SAM worker timed out during model loading (120s)")

    if not line:
        proc.kill()
        stderr_tail = ""
        try:
            err = await asyncio.wait_for(proc.stderr.read(), timeout=3)
            stderr_tail = err.decode()[-300:]
        except Exception:
            pass
        raise HTTPException(500, f"SAM worker exited during startup. Stderr: {stderr_tail}")

    try:
        ready = json.loads(line.decode())
    except json.JSONDecodeError:
        proc.kill()
        raise HTTPException(500, f"SAM worker sent invalid ready signal: {line.decode()[:100]}")

    if ready.get("status") != "ready":
        proc.kill()
        raise HTTPException(500, f"SAM worker unexpected status: {ready}")

    _worker_proc = proc
    logger.info("SAM worker ready — model loaded to GPU")


async def _send_worker_command(cmd: dict, timeout: int = 60) -> dict:
    """Send a single JSON command to the worker and return the JSON response."""
    async with _worker_lock:
        if _worker_proc is None or _worker_proc.returncode is not None:
            raise HTTPException(500, "SAM worker is not running — call _ensure_sam3_worker() first")

        payload = json.dumps(cmd) + "\n"
        try:
            _worker_proc.stdin.write(payload.encode())
            await _worker_proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            raise HTTPException(500, f"SAM worker pipe broken: {e}")

        try:
            line = await asyncio.wait_for(_worker_proc.stdout.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            raise HTTPException(500, f"SAM worker timed out ({timeout}s) on command: {cmd.get('cmd')}")

        if not line:
            raise HTTPException(500, "SAM worker stdout closed unexpectedly")

        try:
            return json.loads(line.decode())
        except json.JSONDecodeError:
            raise HTTPException(500, f"SAM worker invalid response: {line.decode()[:200]}")


async def _release_sam3_worker() -> None:
    """Send shutdown command to the worker and clean up."""
    global _worker_proc
    if _worker_proc is None or _worker_proc.returncode is not None:
        _worker_proc = None
        return

    try:
        await _send_worker_command({"cmd": "shutdown"}, timeout=5)
    except Exception:
        pass

    try:
        await asyncio.wait_for(_worker_proc.wait(), timeout=10)
    except asyncio.TimeoutError:
        _worker_proc.kill()
        await _worker_proc.wait()
    _worker_proc = None
    logger.info("SAM worker released — GPU memory freed")


# ---------------------------------------------------------------------------
# API Routes — SAM
# ---------------------------------------------------------------------------

@app.post("/api/projects/{project_name}/sam/box-segment")
async def sam_box_segment(
    project_name: str,
    filename: str = Form(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
):
    """Run SAM inference on a single image with a box prompt (persistent GPU worker)."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    safe = os.path.normpath(filename)
    if safe.startswith("..") or safe.startswith("/"):
        raise HTTPException(400, "Invalid filename")

    img_dir = _undistorted_image_dir(project_name)
    if not img_dir.is_dir():
        raise HTTPException(400, "未找到去畸变图像（SFM_output/dense/images），请先完成 SfM 去畸变步骤")
    image_path = img_dir / safe
    if not image_path.is_file():
        raise HTTPException(404, f"Image not found: {filename}")

    container_image = f"/data/{project_name}/SFM_output/dense/images/{safe}"
    container_output = f"/data/{project_name}/SAM_output/box_results"
    stem = Path(safe).stem

    await _ensure_sam3_worker()

    logger.info("Project %r: SAM box-segment on %s box=(%d,%d,%d,%d)",
                project_name, safe, x1, y1, x2, y2)

    # Set image on worker (no-op if same as last image)
    await _send_worker_command({"cmd": "set_image", "path": container_image})

    # Run box inference
    result = await _send_worker_command({
        "cmd": "box",
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "output_dir": container_output,
        "stem": stem,
    })

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "SAM inference failed"))

    return {
        "success": True,
        "message": "Box segmentation completed",
        "data": {
            "mask_path": result.get("mask_path"),
            "overlay_path": result.get("overlay_path"),
            "contour_path": result.get("contour_path"),
            "box": result.get("box"),
            "score": result.get("score"),
        },
    }


@app.post("/api/projects/{project_name}/sam/point-segment")
async def sam_point_segment(
    project_name: str,
    filename: str = Form(...),
    points: str = Form(default=""),
    x: int = Form(default=0),   # deprecated, kept for backward compatibility
    y: int = Form(default=0),   # deprecated, kept for backward compatibility
    req_id: str = Form(default=""),  # 批量分割请求标识，原样回传供前端校验响应是否过期
):
    """Run SAM inference on a single image with point prompt(s) (persistent GPU worker).

    points: JSON string like "[[800,600,1],[700,500,0]]"
            label=1 positive (foreground), label=0 negative (background)
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    safe = os.path.normpath(filename)
    if safe.startswith("..") or safe.startswith("/"):
        raise HTTPException(400, "Invalid filename")

    img_dir = _undistorted_image_dir(project_name)
    if not img_dir.is_dir():
        raise HTTPException(400, "未找到去畸变图像（SFM_output/dense/images），请先完成 SfM 去畸变步骤")
    image_path = img_dir / safe
    if not image_path.is_file():
        raise HTTPException(404, f"Image not found: {filename}")

    # Parse points — new multi-point JSON format, or fallback to deprecated x/y
    if points:
        point_list = json.loads(points)  # [[x,y,label], ...]
    else:
        point_list = [[x, y, 1]]

    if not point_list:
        raise HTTPException(400, "At least one point is required")

    container_image = f"/data/{project_name}/SFM_output/dense/images/{safe}"
    container_output = f"/data/{project_name}/SAM_output/point_results"
    stem = Path(safe).stem

    await _ensure_sam3_worker()

    pos_count = sum(1 for pt in point_list if (pt[2] if len(pt) > 2 else 1) == 1)
    neg_count = len(point_list) - pos_count
    logger.info("Project %r: SAM point-segment on %s points=+%d-%d",
                project_name, safe, pos_count, neg_count)

    # Set image on worker (no-op if same as last image)
    await _send_worker_command({"cmd": "set_image", "path": container_image})

    # Run point inference
    result = await _send_worker_command({
        "cmd": "point",
        "points": point_list,
        "output_dir": container_output,
        "stem": stem,
    })

    if not result.get("success"):
        raise HTTPException(500, result.get("error", "SAM inference failed"))

    return {
        "success": True,
        "message": "Point segmentation completed",
        "req_id": req_id,
        "data": {
            "mask_path": result.get("mask_path"),
            "overlay_path": result.get("overlay_path"),
            "contour_path": result.get("contour_path"),
            "points": result.get("points", []),
            "score": result.get("score"),
        },
    }


@app.post("/api/projects/{project_name}/sam/warmup")
async def sam_warmup(project_name: str):
    """Pre-load the SAM model to GPU by running a lightweight inference.

    Uses the first available image in the project to trigger model loading.
    """
    """Pre-load the SAM model to GPU by starting the persistent worker."""
    await _ensure_sam3_worker()
    return {"success": True, "message": "SAM model loaded to GPU"}


@app.get("/api/projects/{project_name}/sam/server-status")
async def sam_server_status(project_name: str):
    """Return the current status of the SAM persistent worker."""
    global _worker_proc
    alive = _worker_proc is not None and _worker_proc.returncode is None
    resp = {"running": alive, "model_loaded": False}
    if alive:
        try:
            status = await _send_worker_command({"cmd": "status"}, timeout=3)
            resp["model_loaded"] = status.get("model_loaded", False)
        except Exception:
            resp["running"] = False
    return resp


@app.post("/api/projects/{project_name}/sam/server-release")
async def sam_server_release(project_name: str):
    """Release the SAM model from GPU by shutting down the persistent worker."""
    await _release_sam3_worker()
    return {"success": True, "message": "SAM model released from GPU"}


@app.get("/api/projects/{project_name}/sam/box-result/{filename}")
async def get_sam_box_result(project_name: str, filename: str):
    """Serve a SAM box-segmentation result image from SAM_output/box_results/."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    safe = os.path.normpath(filename)
    if safe.startswith("..") or safe.startswith("/"):
        raise HTTPException(400, "Invalid filename")

    result_dir = DATA_DIR / project_name / "SAM_output" / "box_results"
    file_path = result_dir / safe
    if not file_path.is_file():
        raise HTTPException(404, f"Result image not found: {filename}")

    ext = file_path.suffix.lower()
    media_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    return FileResponse(file_path, media_type=media_map.get(ext, "image/png"))


@app.get("/api/projects/{project_name}/sam/point-result/{filename}")
async def get_sam_point_result(project_name: str, filename: str):
    """Serve a SAM point-segmentation result image from SAM_output/point_results/."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    safe = os.path.normpath(filename)
    if safe.startswith("..") or safe.startswith("/"):
        raise HTTPException(400, "Invalid filename")

    result_dir = DATA_DIR / project_name / "SAM_output" / "point_results"
    file_path = result_dir / safe
    if not file_path.is_file():
        raise HTTPException(404, f"Result image not found: {filename}")

    ext = file_path.suffix.lower()
    media_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    return FileResponse(file_path, media_type=media_map.get(ext, "image/png"))


@app.post("/api/projects/{project_name}/sam/points/save")
async def save_sam_points(project_name: str, stem: str = Form(...), points: str = Form(...)):
    """Persist point-click markers so they survive app restarts.

    points: JSON string like "[[800,600,1],[700,500,0]]"

    Stores in SAM_output/point_marks/ (NOT point_results/) to avoid
    permission conflicts with Docker-created output directories.
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    point_dir = DATA_DIR / project_name / "SAM_output" / "point_marks"
    point_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = os.path.basename(stem)
    json_path = point_dir / f"{safe_stem}_points.json"

    # Validate JSON
    try:
        pt_list = json.loads(points)
        if not isinstance(pt_list, list):
            raise ValueError("points must be a list")
    except Exception as e:
        raise HTTPException(400, f"Invalid points JSON: {e}")

    json_path.write_text(json.dumps(pt_list, ensure_ascii=False), encoding="utf-8")
    return {"success": True, "stem": safe_stem, "count": len(pt_list)}


@app.get("/api/projects/{project_name}/sam/points/load/{stem}")
async def load_sam_points(project_name: str, stem: str):
    """Load persisted point-click markers for an image."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    safe_stem = os.path.basename(stem)
    if safe_stem.startswith("..") or safe_stem.startswith("/"):
        raise HTTPException(400, "Invalid stem")

    json_path = DATA_DIR / project_name / "SAM_output" / "point_marks" / f"{safe_stem}_points.json"
    if not json_path.is_file():
        return {"points": [], "exists": False}

    try:
        pts = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {"points": [], "exists": True}

    return {"points": pts, "exists": True}


@app.post("/api/projects/{project_name}/sam/run")
async def run_sam(
    project_name: str,
    prompts: str = Form(default="object"),
    confidence: float = Form(default=0.5),
    save_masks: bool = Form(default=True),
    save_masks_rle: bool = Form(default=False),
    vis: bool = Form(default=False),
    batch_size: int = Form(default=1),
    device: str = Form(default="cuda"),
):
    """Run SAM3 inference on a project's input images.

    Args:
        prompts: Comma-separated text prompts (e.g. "car,person,dog").
        confidence: Detection confidence threshold (0.0–1.0).
        save_masks: Export each mask as a standalone PNG.
        save_masks_rle: Include RLE-encoded masks in the JSON output.
        vis: Generate matplotlib visualizations during inference (memory-heavy, off by default).
        batch_size: Number of images to batch through the backbone (reduces GPU idle time).
        device: "cuda" or "cpu".
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")
    if is_sam_running(project_name):
        raise HTTPException(400, "SAM inference is already running")

    prompt_list = [s.strip() for s in prompts.split(",") if s.strip()]
    if not prompt_list:
        raise HTTPException(400, "At least one prompt is required")

    cmd = build_sam_command(
        project_name,
        prompts=prompt_list,
        confidence=confidence,
        device=device,
        save_masks=save_masks,
        save_masks_rle=save_masks_rle,
        vis=vis,
        batch_size=batch_size,
    )

    logger.info("Project %r starting SAM: prompts=%s", project_name, prompt_list)
    await asyncio.to_thread(run_sam_async, project_name, cmd)

    return {
        "success": True,
        "message": f"SAM inference started for {project_name}",
        "data": {
            "prompts": prompt_list,
            "confidence": confidence,
            "save_masks": save_masks,
        },
    }


@app.get("/api/projects/{project_name}/sam/result")
async def get_sam_result(project_name: str):
    """Get SAM inference results (manifest.json content)."""
    _check_sam_completion(project_name)

    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    manifest_path = _sfm_output_dir(project_name).parent / "SAM_output" / SAM_RESULT_MANIFEST

    if not manifest_path.exists():
        return {
            "project_name": project_name,
            "ready": False,
            "message": "No SAM results yet. Run SAM inference first.",
            "results": [],
        }

    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {
            "project_name": project_name,
            "ready": False,
            "message": "Failed to parse SAM results.",
            "results": [],
        }

    return {
        "project_name": project_name,
        "ready": True,
        "total_images": data.get("total_images", 0),
        "total_prompts": data.get("total_prompts", 0),
        "results": data.get("results", []),
    }


@app.get("/api/projects/{project_name}/sam/log")
async def get_sam_log(project_name: str, lines: int = 100):
    """Get SAM inference log."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    log_path = _sfm_output_dir(project_name).parent / "SAM_output" / "sam.log"

    if not log_path.exists():
        return {"project_name": project_name, "log": "(no log yet)"}

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return {"project_name": project_name, "log": "".join(all_lines[-lines:])}
    except OSError:
        return {"project_name": project_name, "log": "(error reading log)"}


# ---------------------------------------------------------------------------
# Helpers — MVS
# ---------------------------------------------------------------------------


def _check_mvs_completion(project_name: str) -> None:
    """If the MVS process for project has exited, update status."""
    proc = get_mvs_process(project_name)
    if proc is None:
        return
    if proc.poll() is not None:
        if proc.returncode == 0:
            logger.info("Project %r MVS finished successfully", project_name)
        else:
            logger.error("Project %r MVS FAILED (exit code %s)", project_name, proc.returncode)
        remove_mvs_process(project_name)


# ---------------------------------------------------------------------------
# API Routes — MVS
# ---------------------------------------------------------------------------


@app.post("/api/projects/{project_name}/mvs/run")
async def run_mvs_pipeline(
    project_name: str,
    mvs_type: str = Form(default="openmvs"),
    command: str = Form(default="pipeline"),
    model: str = Form(default="0"),
    resolution_level: int = Form(default=1),
    max_resolution: int = Form(default=3200),
    min_resolution: int = Form(default=640),
    number_views: int = Form(default=5),
    no_cuda: bool = Form(default=False),
    refine_resolution: int = Form(default=1),
    scales: int = Form(default=2),
    gradient_step: float = Form(default=45.0),
    texture_resolution: int = Form(default=1),
    export_type: str = Form(default="ply"),
    no_geom_consistency: bool = Form(default=False),
    mvs_params: ColmapMVSParams = Depends(colmap_mvs_form),
):
    """Run an MVS pipeline step (COLMAP or OpenMVS) for a project.

    The dense-reconstruction tunables and ``no_geom_consistency`` are COLMAP
    only; the OpenMVS branch below ignores them entirely.
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    # Validate mvs_type
    if mvs_type not in {m.value for m in MVSType}:
        raise HTTPException(400, f"Invalid mvs_type: {mvs_type}. Choices: {[m.value for m in MVSType]}")

    if mvs_type == "colmap":
        # --- COLMAP MVS ---
        # 注册表 + 容器内孤儿进程（后端重启后仍活着）都算“正在运行”
        from modules.pipeline_progress import _colmap_process_alive
        if is_process_running(project_name) or _colmap_process_alive(project_name):
            raise HTTPException(400, "COLMAP pipeline is already running")

        from modules.docker_runner import build_sfm_command, run_pipeline_async as run_colmap_async

        # SfM 的去畸变步骤已产出 SFM_output/dense/{images,sparse,stereo}（PINHOLE 模型
        # + 去畸变图像）。存在时 COLMAP MVS 复用该工作区（reuse 步骤只复制，跳过重复
        # image_undistorter），再跑 patch_match + fusion；不存在时回退 undistort + mvs。
        sfm_dense = _sfm_output_dir(project_name) / "dense"
        can_reuse = (sfm_dense / "images").is_dir() and (sfm_dense / "sparse").is_dir()
        steps = ["reuse", "mvs"] if can_reuse else ["undistort", "mvs"]
        reuse_dense_dir = f"/data/{project_name}/SFM_output/dense" if can_reuse else None

        # 几何一致性关掉时只有 .photometric.bin，fusion 读 geometric 会产出空点云。
        # CLI 也会挡（同样报错退出），这里提前拦下给出可读的 400。
        if mvs_params.fusion_input_type == "geometric" and no_geom_consistency:
            raise HTTPException(
                400,
                "fusion_input_type=geometric 需要几何一致性（当前 no_geom_consistency=true）"
                "——请改用 photometric 或 auto，或打开几何一致性")

        cmd = build_sfm_command(
            project_name,
            steps=steps,
            use_gpu=not no_cuda,
            gpu_index=0,
            no_geom_consistency=no_geom_consistency,
            mvs_type="colmap",
            reuse_dense_dir=reuse_dense_dir,
            mvs_params=mvs_params,
        )
        logger.info("Project %r: starting COLMAP MVS", project_name)
        # run_colmap_async 内部先确保 colmap-dev 容器在跑，成功后标记 RUNNING
        await asyncio.to_thread(run_colmap_async, project_name, cmd)
        # 存下来供 resume 沿用：续算跑在同一 workspace 上，参数必须与原次一致。
        # 写在启动成功之后——启动失败时 workspace 没动，旧参数才是对的那一份。
        update_project(
            project_name,
            status=TaskStatus.RUNNING.value,
            **{COLMAP_MVS_PARAMS_KEY: {
                **mvs_params.model_dump(exclude_none=True),
                "no_geom_consistency": no_geom_consistency,
            }},
        )
        return {
            "success": True,
            "message": f"COLMAP MVS started for {project_name}",
            "data": {"mvs_type": "colmap"},
        }

    # --- OpenMVS ---
    if is_mvs_running(project_name):
        raise HTTPException(400, "MVS pipeline is already running")

    # Validate command
    if command not in {c.value for c in MVSCommand}:
        raise HTTPException(400, f"Invalid MVS command: {command}. Choices: {[c.value for c in MVSCommand]}")

    if export_type not in ("ply", "obj"):
        raise HTTPException(400, "export_type must be 'ply' or 'obj'")

    config = MVSConfig(
        mvs_type=MVSType(mvs_type),
        command=MVSCommand(command),
        model=model,
        resolution_level=resolution_level,
        max_resolution=max_resolution,
        min_resolution=min_resolution,
        number_views=number_views,
        no_cuda=no_cuda,
        refine_resolution=refine_resolution,
        scales=scales,
        gradient_step=gradient_step,
        texture_resolution=texture_resolution,
        export_type=export_type,
    )
    update_project(project_name, mvs_config=config.model_dump())

    cmd = build_mvs_command(
        project_name,
        command=command,
        model=model,
        resolution_level=resolution_level,
        max_resolution=max_resolution,
        min_resolution=min_resolution,
        number_views=number_views,
        no_cuda=no_cuda,
        refine_resolution=refine_resolution,
        scales=scales,
        gradient_step=gradient_step,
        texture_resolution=texture_resolution,
        export_type=export_type,
    )

    logger.info("Project %r: starting OpenMVS %s", project_name, command)
    await asyncio.to_thread(run_mvs_async, project_name, cmd)

    return {
        "success": True,
        "message": f"OpenMVS {command} started for {project_name}",
        "data": {"mvs_type": "openmvs", "command": command},
    }


@app.post("/api/projects/{project_name}/mvs/resume")
async def resume_colmap_mvs(project_name: str):
    """断点续算 COLMAP MVS（不清历史深度图）。

    项目的 COLMAP MVS（或整条 SfM+MVS）被中断/被杀后，MVS_output/colmap
    （去畸变 images + PINHOLE sparse + stereo/patch-match.cfg + 部分 depth_maps）
    通常还在。此端点对**同一 workspace** 直接重跑 patch_match_stereo +
    stereo_fusion —— colmap 会跳过已有深度图/法向图的影像只补缺（见
    PatchMatchController::ProcessProblem 的 ExistsFile 短路），因此**不经过
    reuse 步骤**，也就不会像 GUI 普通重跑那样先把 stereo 整目录删掉再拷空的。

    前置条件：
      - MVS_output/colmap 已由之前的 undistort/reuse 就绪（有 images/sparse/
        stereo/patch-match.cfg）；
      - 当前没有其他 COLMAP 管线在写该 workspace（注册表进程或容器内孤儿都算）；
      - 尚未生成 fused.ply（若深度图已齐且 fused.ply 已存在即算完成，拒绝）。
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    # colmap 同一 workspace 是否已有别的 sfm_cli.py 在跑（注册表或容器内孤儿都算）。
    # 惰性 import 与 run_mvs_pipeline / _reconcile_orphan_colmap 保持一致。
    from modules.pipeline_progress import (
        _colmap_process_alive,
        _depth_stats,
        _patch_match_refs,
    )
    if is_process_running(project_name) or _colmap_process_alive(project_name):
        raise HTTPException(
            400,
            "COLMAP pipeline 正在运行（可能仍在写同一 workspace）——请等它结束后再续算")

    workspace = _mvs_output_dir(project_name, "colmap")
    cfg = workspace / "stereo" / "patch-match.cfg"
    if not ((workspace / "images").is_dir()
            and (workspace / "sparse").is_dir() and cfg.is_file()):
        raise HTTPException(
            400,
            f"{workspace} 尚无可用续算的工作区（缺 images/sparse/"
            "stereo/patch-match.cfg）。请先跑一次完整的 COLMAP MVS 生成初始 workspace。")

    refs = _patch_match_refs(workspace)
    if not refs:
        raise HTTPException(400, "stereo/patch-match.cfg 为空或不可读，无法续算")

    # 续算不接收 UI 参数，一律沿用上次存的（同一 workspace 里新旧参数混算会让
    # fusion 读到不一致的深度图）。历史项目无该键 → 全 None，行为同从前。
    mvs_params, geom_off = _stored_colmap_mvs(p)
    if geom_off:
        # 关掉几何一致性时只写 .photometric.bin，单趟；下面的完成判据与
        # _depth_stats 分桶都按两趟设计，续算会永远判不出「已齐」。
        raise HTTPException(
            400,
            "该项目上次运行时关闭了几何一致性（单趟深度图，只有 .photometric.bin），"
            "续算需要两趟布局。请重跑一次完整的 COLMAP MVS。")

    stats = _depth_stats(workspace)
    photo, geo = stats["photometric"], stats["geometric"]

    fused_ply = workspace / "fused.ply"
    if photo >= refs and geo >= refs and fused_ply.is_file():
        raise HTTPException(
            400,
            f"深度图已齐（photometric {photo}/{refs}、geometric {geo}/{refs}）"
            f"且 {fused_ply.name} 已存在——已完成，无需续算")

    # 续算沿用上次的两趟布局（geom_consistency=true，上面已挡掉 false），
    # 否则与存量 .photometric.bin / .geometric.bin 不一致会让 fusion 读不到图。
    from modules.docker_runner import (
        build_sfm_command,
        run_pipeline_async as run_colmap_async,
    )
    config = p.get("config", {})
    cmd = build_sfm_command(
        project_name,
        steps=["mvs"],
        use_gpu=config.get("use_gpu", True),
        gpu_index=config.get("gpu_index", 0),
        mvs_type="colmap",
        mvs_params=mvs_params,
    )
    logger.info("Project %r: resuming COLMAP MVS (no reuse) — %s",
                project_name, " ".join(cmd))
    await asyncio.to_thread(run_colmap_async, project_name, cmd)
    update_project(
        project_name,
        status=TaskStatus.RUNNING.value,
        started_at=datetime.now(timezone.utc).isoformat(),
        error_message=None,
    )

    miss_photo = max(refs - photo, 0)
    miss_geo = max(refs - geo, 0)
    msg = (f"COLMAP MVS 续算已启动：已有 photometric {photo}/{refs}、"
           f"geometric {geo}/{refs}，将补算 {miss_photo} 张 photometric、"
           f"{miss_geo} 张 geometric 后执行融合")
    return {
        "success": True,
        "message": msg,
        "data": {
            "mvs_type": "colmap",
            "steps": ["mvs"],
            "reused_workspace": True,
            "photo_done": photo,
            "geo_done": geo,
            "refs": refs,
        },
    }


@app.get("/api/projects/{project_name}/mvs/log")
async def get_mvs_log(project_name: str, lines: int = 100,
                      mvs_type: str = "openmvs"):
    """Get MVS pipeline log.

    Query params:
        mvs_type: MVS type (\"colmap\" or \"openmvs\", default \"openmvs\").
            - openmvs: reads MVS_output/openmvs/mvs_pipeline.log
            - colmap: reads SFM_output/pipeline.log (COLMAP MVS uses SfM log)
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    if mvs_type == "colmap":
        log_path = _sfm_output_dir(project_name) / "pipeline.log"
    else:
        log_path = _mvs_output_dir(project_name, mvs_type) / "mvs_pipeline.log"

    if not log_path.exists():
        return {"project_name": project_name, "log": "(no log yet)"}

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return {"project_name": project_name, "log": "".join(all_lines[-lines:])}
    except OSError:
        return {"project_name": project_name, "log": "(error reading log)"}


def _clean_surrogates(obj: object) -> object:
    """把 surrogateescape 产生的代理字符还原为正常 UTF-8。

    历史数据防御：openmvs 容器（Python 3.6 + ascii locale）曾把非 ASCII 项目名
    解码成 \\udc 开头的代理字符并写入结果 JSON，响应渲染时会导致
    UnicodeEncodeError: surrogates not allowed。这里还原成可编码的 UTF-8。
    """
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
    if isinstance(obj, list):
        return [_clean_surrogates(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _clean_surrogates(v) for k, v in obj.items()}
    return obj


@app.get("/api/projects/{project_name}/mvs/result")
async def get_mvs_result(project_name: str, mvs_type: str = "openmvs"):
    """Get MVS pipeline result (mvs_pipeline_result.json).

    Query params:
        mvs_type: MVS 引擎类型（openmvs / colmap），各自读自己的结果文件。
            - openmvs: MVS_output/openmvs/mvs_pipeline_result.json（兼容旧布局 MVS_output/ 根目录）
            - colmap:  SFM_output/pipeline_result.json
    """
    _check_mvs_completion(project_name)

    # 后端权威的 MVS 运行状态 — 前端据此判定完成（替代解析日志关键字）。
    if mvs_type == "colmap":
        # colmap 走 SfM 的 task.json 状态机
        _check_process_completion(project_name)
        task_status = (get_project(project_name) or {}).get("status", "pending")
        run_status = {
            "running": "running",
            "completed": "ok",
            "failed": "failed",
        }.get(task_status, "idle")
        exit_code = None
    else:
        st = get_mvs_run_status(project_name)
        run_status = st.get("status", "idle") if st else "idle"
        exit_code = (st or {}).get("exit_code")

    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    if mvs_type == "colmap":
        result_path = _sfm_output_dir(project_name) / "pipeline_result.json"
    else:
        # 区分引擎：openmvs 结果写在 MVS_output/openmvs/ 子目录
        from modules.task_manager import _mvs_result_json
        result_path = _mvs_result_json(project_name, "openmvs")
        if not result_path.exists():
            # 兼容旧布局：结果直接写在 MVS_output/ 根目录
            result_path = _mvs_output_dir(project_name) / "mvs_pipeline_result.json"

    if not result_path.exists():
        return {
            "project_name": project_name,
            "ready": False,
            "message": "暂无 MVS 结果：尚未运行任何 MVS 命令，或该命令未生成结果文件。",
            "run_status": run_status,
            "exit_code": exit_code,
        }

    try:
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {
            "project_name": project_name,
            "ready": False,
            "message": "Failed to parse MVS results.",
            "run_status": run_status,
            "exit_code": exit_code,
        }
    # 历史数据防御：容器 ascii locale 曾写入代理字符，还原后渲染不再 500
    data = _clean_surrogates(data)

    if mvs_type == "colmap":
        # colmap 的 pipeline_result.json 结构与 openmvs 不同，做一次映射
        pipeline = data.get("pipeline") or {}
        steps = {}
        for name in pipeline.get("steps_completed", []):
            steps[name] = {"status": "ok", "elapsed": None, "output": ""}
        return {
            "project_name": project_name,
            "ready": True,
            "status": "ok" if steps else "pending",
            "steps": steps,
            "total_elapsed": pipeline.get("elapsed_seconds"),
            "run_status": run_status,
            "exit_code": exit_code,
        }

    return {
        "project_name": project_name,
        "ready": True,
        "status": data.get("status"),
        "project": data.get("project"),
        "steps": data.get("steps", {}),
        "total_elapsed": data.get("total_elapsed"),
        "run_status": run_status,
        "exit_code": exit_code,
    }


@app.get("/api/projects/{project_name}/mvs/files")
async def list_mvs_files(project_name: str, mvs_type: str = "openmvs"):
    """List MVS 产物文件，按所属步骤标注，排除日志/结果描述等非产物。

    每个产物带 step（来自哪一步）与 path（相对 MVS_output/<type>/ 的路径），
    前端下拉框据此展示「步骤 · 文件」，而不是整目录倾倒。
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    mvs_dir = _mvs_output_dir(project_name, mvs_type)
    products: list[dict] = []
    if mvs_dir.is_dir():
        # 文件名 → 步骤（顺序敏感：更具体的后缀先匹配）
        for f in sorted(mvs_dir.iterdir()):
            if not f.is_file():
                continue
            if f.name in ("mvs_pipeline.log", "mvs_pipeline_result.json"):
                continue  # 日志 / 结果描述，不是可下载产物
            step = "接口"
            for key, label in (("texture", "纹理贴图"),
                               ("refine", "网格精化"),
                               ("mesh", "网格重建"),
                               ("dense", "稠密化")):
                if key in f.name:
                    step = label
                    break
            products.append({"step": step, "label": f.name, "path": f.name})

    return {"project_name": project_name, "products": products}


@app.get("/api/projects/{project_name}/mvs/download/{filename:path}")
async def download_mvs_file(project_name: str, filename: str,
                            mvs_type: str = ""):
    """Download a result file from MVS_output directory.

    Query params:
        mvs_type: Optional MVS type (\"colmap\" or \"openmvs\"). If not
            given, searches all MVS subdirectories.
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    safe_path = os.path.normpath(filename)
    if safe_path.startswith("..") or safe_path.startswith("/"):
        raise HTTPException(400, "Invalid filename")

    mvs_dir = _mvs_output_dir(project_name, mvs_type)
    file_path = mvs_dir / safe_path
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    return FileResponse(file_path, filename=Path(filename).name,
                        media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# API Routes — 掩码目标重建（mask_obj）
# ---------------------------------------------------------------------------


def _mask_obj_dir(project_name: str) -> Path:
    return _algorithm_output_dir(project_name) / "mask_obj"


@app.post("/api/projects/{project_name}/mask-obj/run")
async def run_mask_obj_pipeline(
    project_name: str,
    mask_source: str = Form(default="point"),
    padding: int = Form(default=5),
):
    """按 SAM 掩码裁剪去畸变图 → 调整稀疏模型 → 启动 MVS 稠密重建。

    裁剪 + 建模同步执行（纯 Python，快）；成功后才启动 sfm_cli.py --steps mvs
    的 docker 任务。mask_source 选 SAM 掩码来源（point=point_results，box=box_results）。
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")
    if is_mask_obj_running(project_name):
        raise HTTPException(400, "掩码目标重建任务已在运行")

    if mask_source not in ("point", "box"):
        raise HTTPException(400, f"Invalid mask_source: {mask_source}. Choices: point, box")

    result = await asyncio.to_thread(
        run_mask_obj, project_name, mask_source=mask_source, padding=padding)
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "mask_obj 裁剪建模失败"))

    cmd = result["command"]
    logger.info("Project %r: starting mask_obj MVS, %d image(s) cropped",
                project_name, result.get("images_cropped", 0))

    # 裁剪摘要先写进日志，前端轮询立刻能看到
    log_path = _mask_obj_dir(project_name) / "mask_obj.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"=== MASK_OBJ CROP {project_name} ===\n")
        f.write(f"  裁剪图像: {result['images_cropped']}  "
                f"跳过(无掩码): {result['images_skipped_no_mask']}  "
                f"跳过(无文件): {result['images_skipped_no_file']}\n")
        f.write(f"  相机: {result['num_cameras']}  3D 点: {result['num_points3D']}\n")
        f.write(f"  输出: {result['output_dir']}\n")
        f.write(f"  命令: {' '.join(cmd)}\n")

    await asyncio.to_thread(run_mask_obj_async, project_name, cmd)
    return {
        "success": True,
        "message": "掩码目标重建已启动（裁剪完成，MVS 运行中）",
        "data": {
            "images_cropped": result["images_cropped"],
            "images_skipped": result["images_skipped_no_mask"],
        },
    }


@app.get("/api/projects/{project_name}/mask-obj/log")
async def get_mask_obj_log(project_name: str, lines: int = 100):
    """Get mask_obj pipeline log (algorithm_output/mask_obj/mask_obj.log)."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    log_path = _mask_obj_dir(project_name) / "mask_obj.log"
    if not log_path.exists():
        return {"project_name": project_name, "log": "(no log yet)"}

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return {"project_name": project_name, "log": "".join(all_lines[-lines:])}
    except OSError:
        return {"project_name": project_name, "log": "(error reading log)"}


@app.get("/api/projects/{project_name}/mask-obj/result")
async def get_mask_obj_result(project_name: str):
    """返回掩码目标重建状态 + 裁剪统计 + fused.ply 信息。"""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    st = get_mask_obj_run_status(project_name)
    run_status = st.get("status", "idle") if st else "idle"
    exit_code = (st or {}).get("exit_code")

    out = _mask_obj_dir(project_name)
    crop: dict = {}
    crop_json = out / "crop_result.json"
    if crop_json.is_file():
        try:
            with open(crop_json, encoding="utf-8") as f:
                crop = json.load(f)
        except (json.JSONDecodeError, OSError):
            crop = {}

    fused = out / "fused.ply"
    fused_mb = round(fused.stat().st_size / (1024 * 1024), 2) if fused.is_file() else None
    # 只要 fused.ply 在且当前不在运行/失败，就视为完成。
    # （后端重启后 run_status 会丢失成 idle，但仍应显示既有结果。）
    ready = fused.is_file() and run_status not in ("running", "failed")

    if run_status == "failed":
        status, message = "failed", "掩码目标重建失败（MVS 进程异常退出）"
    elif run_status == "running":
        status, message = "running", "掩码目标重建运行中..."
    elif ready:
        status, message = "ok", "掩码目标重建成功"
    else:
        status, message = "pending", "暂无结果：尚未运行，或 MVS 未产出 fused.ply"

    return {
        "project_name": project_name,
        "run_status": run_status,
        "exit_code": exit_code,
        "ready": ready,
        "status": status,
        "images_cropped": crop.get("images_cropped"),
        "images_skipped_no_mask": crop.get("images_skipped_no_mask"),
        "num_cameras": crop.get("num_cameras"),
        "num_points3D": crop.get("num_points3D"),
        "fused_size_mb": fused_mb,
        "message": message,
    }


@app.get("/api/projects/{project_name}/mask-obj/files")
async def list_mask_obj_files(project_name: str):
    """List 可下载产物（fused.ply / cropped_images.zip / pipeline.log）。"""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    out = _mask_obj_dir(project_name)
    products: list[dict] = []
    if out.is_dir():
        for name, step, label in (
            ("fused.ply", "稠密重建", "fused.ply"),
            ("cropped_images.zip", "裁剪图", "cropped_images.zip"),
            ("pipeline.log", "日志", "pipeline.log"),
        ):
            if (out / name).is_file():
                products.append({"step": step, "label": label, "path": name})
    return {"project_name": project_name, "products": products}


@app.get("/api/projects/{project_name}/mask-obj/download/{filename:path}")
async def download_mask_obj_file(project_name: str, filename: str):
    """Download a result file from algorithm_output/mask_obj directory."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    safe_path = os.path.normpath(filename)
    if safe_path.startswith("..") or safe_path.startswith("/"):
        raise HTTPException(400, "Invalid filename")

    file_path = _mask_obj_dir(project_name) / safe_path
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    return FileResponse(file_path, filename=Path(filename).name,
                        media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Helpers — 3DGS
# ---------------------------------------------------------------------------

def _gs_output_dir(project_name: str) -> Path:
    return _sfm_output_dir(project_name).parent / "3DGS_output"


def _check_gs_completion(project_name: str) -> None:
    """If the 3DGS process for project has exited, update status."""
    proc = get_gs_process(project_name)
    if proc is None:
        return
    if proc.poll() is not None:
        if proc.returncode == 0:
            logger.info("Project %r 3DGS finished successfully", project_name)
        else:
            logger.error("Project %r 3DGS FAILED (exit code %s)", project_name, proc.returncode)
        remove_gs_process(project_name)


# ---------------------------------------------------------------------------
# API Routes — 3DGS
# ---------------------------------------------------------------------------


@app.post("/api/projects/{project_name}/3dgs/run")
async def run_3dgs_pipeline(
    project_name: str,
    command: str = Form(default="full"),
    iterations: int = Form(default=30_000),
    resolution: int = Form(default=1),
    white_background: bool = Form(default=False),
    quiet: bool = Form(default=False),
    checkpoint_interval: int = Form(default=0),
    start_checkpoint: str = Form(default=""),
):
    """Run a 3D Gaussian Splatting pipeline step for a project.

    Args:
        command: "prepare", "train", "render", or "full" (prepare → train → render).
        iterations: Training iterations (default 30000).
        resolution: Image downscale factor (default 1 = full resolution).
        white_background: Use white background instead of black.
        quiet: Suppress non-essential output.
        checkpoint_interval: Save checkpoint every N iterations (0 = default 7000/30000 only).
        start_checkpoint: Path to .pth checkpoint to resume training from.
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")
    if is_gs_running(project_name):
        raise HTTPException(400, "3DGS pipeline is already running")

    if command not in GS_COMMANDS:
        raise HTTPException(400, f"Invalid 3DGS command: {command}. Choices: {GS_COMMANDS}")

    # Verify prerequisites for train/render.
    # 3DGS 需要去畸变图像 + PINHOLE 模型：来自 SfM undistort 的输出
    # (SFM_output/dense/{images,sparse})。未跑 undistort 时 dense 目录不存在。
    if command in ("train", "full"):
        dense_sparse = _sfm_output_dir(project_name) / "dense" / "sparse"
        dense_images = _sfm_output_dir(project_name) / "dense" / "images"
        if not dense_sparse.exists() or not dense_images.exists():
            raise HTTPException(
                400,
                "未找到去畸变输出（SFM_output/dense）。3DGS 需要 SfM 的 undistort "
                "步骤输出（去畸变图像 + PINHOLE 模型），请先运行 SfM 管线（含去畸变）。",
            )

    cmd = build_gs_command(
        project_name,
        command=command,
        iterations=iterations,
        resolution=resolution,
        white_background=white_background,
        quiet=quiet,
        checkpoint_interval=checkpoint_interval,
        start_checkpoint=start_checkpoint or None,
    )

    logger.info("Project %r: starting 3DGS %s (iterations=%d, ckpt_interval=%d)",
                project_name, command, iterations, checkpoint_interval)
    await asyncio.to_thread(run_gs_async, project_name, cmd)

    return {
        "success": True,
        "message": f"3DGS {command} started for {project_name}",
        "data": {
            "command": command,
            "iterations": iterations,
            "checkpoint_interval": checkpoint_interval,
            "start_checkpoint": start_checkpoint or None,
        },
    }


@app.get("/api/projects/{project_name}/3dgs/log")
async def get_gs_log(project_name: str, lines: int = 100):
    """Get 3DGS pipeline log."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    log_path = _gs_output_dir(project_name) / "gs_pipeline.log"
    if not log_path.exists():
        return {"project_name": project_name, "log": "(no log yet)"}

    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return {"project_name": project_name, "log": "".join(all_lines[-lines:])}
    except OSError:
        return {"project_name": project_name, "log": "(error reading log)"}


@app.get("/api/projects/{project_name}/3dgs/result")
async def get_gs_result(project_name: str):
    """Get 3DGS pipeline result — list of rendered images.

    返回 run_status（running/ok/failed），前端据此判定完成并停止轮询，
    不再只依赖 ready（prepare 等命令不产渲染图，ready 恒 False）。
    """
    _check_gs_completion(project_name)

    st = get_gs_run_status(project_name)
    run_status = st.get("status", "idle") if st else "idle"
    exit_code = (st or {}).get("exit_code")

    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    output_dir = _gs_output_dir(project_name)

    if not output_dir.exists():
        return {
            "project_name": project_name,
            "ready": False,
            "run_status": run_status,
            "exit_code": exit_code,
            "message": "暂无 3DGS 结果：尚未运行任何 3DGS 命令。",
        }

    # Find the latest iteration directory
    # renders 返回相对 3DGS_output 的路径（如 train/ours_30000/renders/xxx.png），
    # 前端可直接拼 /3dgs/download/<path> 预览 / 下载。
    train_renders_dir = output_dir / "train"
    renders = []
    iteration_name = ""
    if train_renders_dir.exists():
        for ours_dir in sorted(train_renders_dir.iterdir(), reverse=True):
            if ours_dir.is_dir() and ours_dir.name.startswith("ours_"):
                render_subdir = ours_dir / "renders"
                if render_subdir.exists():
                    iteration_name = ours_dir.name
                    rel_prefix = f"train/{ours_dir.name}/renders"
                    renders = sorted([
                        f"{rel_prefix}/{p.name}"
                        for p in render_subdir.iterdir()
                        if p.suffix.lower() == ".png"
                    ])
                break

    has_point_cloud = (output_dir / "point_cloud").exists()
    # 点云 PLY 相对路径（若有），前端可下载
    point_cloud_ply = ""
    pc_root = output_dir / "point_cloud"
    if pc_root.exists():
        for d in sorted(pc_root.iterdir(), reverse=True):
            ply = d / "point_cloud.ply"
            if ply.exists():
                point_cloud_ply = str(ply.relative_to(output_dir))
                break

    return {
        "project_name": project_name,
        "ready": True,
        "run_status": run_status,
        "exit_code": exit_code,
        "renders_count": len(renders),
        "renders": renders,
        "iteration_name": iteration_name,
        "has_point_cloud": has_point_cloud,
        "point_cloud_ply": point_cloud_ply,
    }


@app.get("/api/projects/{project_name}/3dgs/files")
async def list_gs_files(project_name: str):
    """列出 3DGS 产物：只有高斯点云。

    3DGS 运行对用户有意义的最终产物就是高斯点云（point_cloud.ply）。
    检查点、渲染图、日志、中间配置等都不放进下载框，避免干扰。
    """
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    gs_dir = _gs_output_dir(project_name)
    products: list[dict] = []
    if gs_dir.is_dir():
        # 点云（各迭代，最新在前）
        pc_root = gs_dir / "point_cloud"
        if pc_root.is_dir():
            pc_dirs = [d for d in pc_root.iterdir()
                       if d.is_dir() and d.name.startswith("iteration_")]
            pc_dirs.sort(
                key=lambda d: int(d.name.split("_")[1]), reverse=True)
            for d in pc_dirs:
                ply = d / "point_cloud.ply"
                if ply.is_file():
                    products.append({
                        "step": "训练",
                        "label": f"点云（{d.name}）",
                        "path": str(ply.relative_to(gs_dir)).replace(os.sep, "/"),
                    })

    return {"project_name": project_name, "products": products}


@app.get("/api/projects/{project_name}/3dgs/download/{filename:path}")
async def download_gs_file(project_name: str, filename: str):
    """Download a result file from 3DGS_output directory."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")

    gs_dir = _gs_output_dir(project_name)
    safe_path = os.path.normpath(filename)
    if safe_path.startswith("..") or safe_path.startswith("/"):
        raise HTTPException(400, "Invalid filename")

    file_path = gs_dir / safe_path
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    return FileResponse(file_path, filename=Path(filename).name,
                        media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend page."""
    html_path = FRONTEND_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Frontend not found. Create frontend/index.html</h2>")
