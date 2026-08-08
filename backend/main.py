"""FastAPI backend for the COLMAP SfM/MVS Platform."""

from __future__ import annotations

import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from modules.docker_logger import start_global_monitor
from modules.docker_runner import (
    build_sfm_command,
    get_process,
    is_process_running,
    preprocess_images,
    remove_process,
    run_pipeline_async,
)
from modules.logger import get_logger, request_log
from modules.models import (
    CameraModel,
    MatcherType,
    PipelineConfig,
    PipelineStep,
    TaskStatus,
)
from modules.task_manager import (
    _input_image_dir,
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

def _check_process_completion(project_name: str) -> None:
    """If the process for project has exited, sync results and clean up."""
    proc = get_process(project_name)
    if proc is None:
        return
    if proc.poll() is not None:
        if proc.returncode == 0:
            logger.info("Project %r pipeline finished successfully", project_name)
            sync_project_from_result(project_name)
        else:
            logger.error("Project %r pipeline FAILED (exit code %s)", project_name, proc.returncode)
            update_project(
                project_name,
                status=TaskStatus.FAILED.value,
                error_message=f"Pipeline exited with code {proc.returncode}",
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
        remove_process(project_name)


def _start_pipeline(project_name: str) -> None:
    """Build command, mark running, launch pipeline."""
    p = get_project(project_name)
    if p is None:
        logger.warning("_start_pipeline for unknown project %r", project_name)
        return

    config_dict = p.get("config", {})
    steps = config_dict.get("steps", ["extract", "match", "sfm", "undistort", "mvs"])

    # Strip IPTC/EXIF metadata before pipeline to prevent OpenImageIO crash
    prep_result = preprocess_images(project_name)
    logger.info("Project %r: pre-run metadata check — success=%s, processed=%s",
                project_name, prep_result.get("success"), prep_result.get("processed"))

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
    )

    logger.info("Project %r starting pipeline: %s", project_name, " ".join(cmd))

    update_project(
        project_name,
        status=TaskStatus.RUNNING.value,
        started_at=datetime.now(timezone.utc).isoformat(),
        error_message=None,
    )

    run_pipeline_async(project_name, cmd)


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
    prep_result = preprocess_images(project_name)
    logger.info("Project %r: metadata preprocessing — success=%s, processed=%s",
                project_name, prep_result.get("success"), prep_result.get("processed"))

    num = count_images(project_name)
    update_project(project_name, num_images=num)
    logger.info("Project %r: uploaded images, now %d total", project_name, num)

    return {"success": True, "message": f"{num} images in project", "data": {"num_images": num}}


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

    config = PipelineConfig(
        steps=step_list,
        matcher=MatcherType(matcher),
        camera_model=CameraModel(camera_model),
        single_camera=single_camera,
        use_gpu=use_gpu,
        gpu_index=gpu_index,
        max_num_features=max_num_features,
        no_geom_consistency=no_geom_consistency,
        reconstruction_name=reconstruction_name,
    )

    update_project(project_name, config=config.model_dump())
    logger.info("Project %r: config updated, starting pipeline", project_name)
    _start_pipeline(project_name)

    return {"success": True, "message": f"Pipeline started for {project_name}"}


@app.get("/api/projects/{project_name}/log")
async def get_pipeline_log(project_name: str, lines: int = 100):
    """Get pipeline log."""
    p = get_project(project_name)
    if p is None:
        raise HTTPException(404, "Project not found")
    log_text = get_log_tail(project_name, lines=lines)
    return {"project_name": project_name, "log": log_text or "(no log yet)"}


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
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend page."""
    html_path = FRONTEND_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>Frontend not found. Create frontend/index.html</h2>")
