"""Project lifecycle management — JSON files under DATA_DIR/<project_name>/."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modules.logger import get_logger
from modules.models import PipelineConfig, TaskStatus

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("/home/zwz/Code/DATA")

# Pre-created output directories (mirrors existing convention)
OUTPUT_DIRS = ["SFM_output", "MVS_output", "3DGS_output", "PoinTR_output", "SAM_output"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _project_dir(project_name: str) -> Path:
    return DATA_DIR / project_name


def _project_json(project_name: str) -> Path:
    return _project_dir(project_name) / "task.json"


def _input_image_dir(project_name: str) -> Path:
    return _project_dir(project_name) / "Input" / "input_image"


def _input_video_dir(project_name: str) -> Path:
    return _project_dir(project_name) / "Input" / "input_video"


def _sfm_output_dir(project_name: str) -> Path:
    return _project_dir(project_name) / "SFM_output"


def _result_json(project_name: str) -> Path:
    """Path to pipeline_result.json (inside SFM_output)."""
    return _sfm_output_dir(project_name) / "pipeline_result.json"


def _log_path(project_name: str) -> Path:
    """Path to pipeline.log (inside SFM_output)."""
    return _sfm_output_dir(project_name) / "pipeline.log"


# ---------------------------------------------------------------------------
# Scaffold
# ---------------------------------------------------------------------------

def _scaffold_project(project_name: str) -> None:
    """Create the standard directory layout for a project."""
    base = _project_dir(project_name)
    (base / "Input" / "input_image").mkdir(parents=True, exist_ok=True)
    (base / "Input" / "input_video").mkdir(parents=True, exist_ok=True)
    for d in OUTPUT_DIRS:
        (base / d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create_project(name: str, config: PipelineConfig | None = None) -> dict[str, Any]:
    """Create a new project with standard directory layout."""
    # Sanitize name
    safe_name = name.strip().replace("/", "_").replace("\\", "_")
    if not safe_name:
        safe_name = f"project_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    if _project_dir(safe_name).exists():
        # Return existing project
        existing = get_project(safe_name)
        if existing:
            return existing

    now = datetime.now(timezone.utc).isoformat()

    project = {
        "project_name": safe_name,
        "name": name.strip(),
        "status": TaskStatus.PENDING.value,
        "config": config.model_dump() if config else PipelineConfig().model_dump(),
        "num_images": 0,
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "error_message": None,
        "pipeline": None,
        "steps": {},
    }

    _scaffold_project(safe_name)
    _save_project(safe_name, project)
    logger.info("Project %r created", safe_name)
    return project


def get_project(project_name: str) -> dict[str, Any] | None:
    """Load a project from disk."""
    path = _project_json(project_name)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_projects() -> list[dict[str, Any]]:
    """List all projects ordered by creation time (newest first)."""
    if not DATA_DIR.exists():
        return []

    projects = []
    for d in sorted(DATA_DIR.iterdir(), reverse=True):
        if not d.is_dir() or d.name.startswith("."):
            continue
        tj = d / "task.json"
        if tj.exists():
            try:
                with open(tj, encoding="utf-8") as f:
                    projects.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    return projects


def update_project(project_name: str, **fields) -> dict[str, Any] | None:
    """Update fields on a project."""
    p = get_project(project_name)
    if p is None:
        return None
    p.update(fields)
    _save_project(project_name, p)
    return p


def delete_project(project_name: str) -> bool:
    """Delete a project and all its data."""
    pd = _project_dir(project_name)
    if not pd.exists():
        return False
    shutil.rmtree(pd)
    logger.info("Project %r deleted (dir=%s)", project_name, pd)
    return True


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def count_images(project_name: str) -> int:
    """Count images in the project's input_image directory."""
    img_dir = _input_image_dir(project_name)
    if not img_dir.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pgm", ".ppm"}
    return sum(1 for p in img_dir.iterdir()
               if p.is_file() and p.suffix.lower() in exts)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

def load_pipeline_result(project_name: str) -> dict[str, Any] | None:
    """If the pipeline has finished, load pipeline_result.json."""
    rj = _result_json(project_name)
    if not rj.exists():
        return None
    try:
        with open(rj, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_log_tail(project_name: str, lines: int = 80) -> str | None:
    """Return the last N lines of the pipeline log."""
    lp = _log_path(project_name)
    if not lp.exists():
        return None
    try:
        with open(lp, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except OSError:
        return None


def sync_project_from_result(project_name: str) -> None:
    """After pipeline completes, sync pipeline_result.json → task.json."""
    result = load_pipeline_result(project_name)
    if result is None:
        return

    updates = {
        "pipeline": result.get("pipeline"),
        "steps": result.get("steps", {}),
        "num_images": result.get("input", {}).get("num_images", 0),
        "status": TaskStatus.COMPLETED.value,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if result.get("pipeline", {}).get("elapsed_seconds"):
        updates["elapsed_seconds"] = result["pipeline"]["elapsed_seconds"]

    steps_completed = result.get("pipeline", {}).get("steps_completed", [])
    steps_requested = result.get("pipeline", {}).get("steps_requested", [])
    if steps_completed != steps_requested:
        updates["status"] = TaskStatus.FAILED.value

    update_project(project_name, **updates)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _save_project(project_name: str, project: dict) -> None:
    with open(_project_json(project_name), "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False, default=str)
