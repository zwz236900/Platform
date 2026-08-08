"""Docker exec wrapper for running sfm_cli.py inside the colmap-dev container."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from modules.docker_logger import LOG_DIR as DOCKER_LOG_DIR, start_follow
from modules.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTAINER_NAME = "colmap-dev"
SFM_CLI_PATH = "/colmap/sfm_cli.py"         # path inside container
DATA_DIR_HOST = Path("/home/zwz/Code/DATA")  # mounted at /data in container
DATA_DIR_CONTAINER = Path("/data")
CONTAINER_LOG_PATH = DOCKER_LOG_DIR / f"{CONTAINER_NAME}.log"

# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_sfm_command(
    project_name: str,
    *,
    steps: list[str] | None = None,
    matcher: str = "exhaustive",
    camera_model: str = "SIMPLE_RADIAL",
    single_camera: bool = True,
    use_gpu: bool = True,
    gpu_index: int = 0,
    max_num_features: int = 0,
    no_geom_consistency: bool = False,
    reconstruction_name: str = "0",
    vocab_tree_path: str | None = None,
) -> list[str]:
    """Build the docker exec command to run sfm_cli.py inside the container.

    Project layout inside container:
        /data/<project>/Input/input_image/   →  --image_dir
        /data/<project>/SFM_output/          →  --output_dir
    """
    image_dir = DATA_DIR_CONTAINER / project_name / "Input" / "input_image"
    output_dir = DATA_DIR_CONTAINER / project_name / "SFM_output"

    steps_str = ",".join(steps) if steps else "extract,match,sfm,undistort,mvs"

    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "python3", SFM_CLI_PATH,
        "--image_dir", str(image_dir),
        "--output_dir", str(output_dir),
        "--steps", steps_str,
        "--matcher", matcher,
        "--camera_model", camera_model,
        "--single_camera", "1" if single_camera else "0",
        "--use_gpu", "1" if use_gpu else "0",
        "--gpu_index", str(gpu_index),
        "--max_num_features", str(max_num_features),
        "--reconstruction_name", reconstruction_name,
    ]

    if no_geom_consistency:
        cmd.append("--no_geom_consistency")
    if vocab_tree_path:
        cmd += ["--vocab_tree_path", vocab_tree_path]

    return cmd


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

_SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pgm", ".ppm")


def preprocess_images(project_name: str) -> dict:
    """Strip all metadata (IPTC/EXIF) from uploaded images inside the colmap-dev container.

    This prevents OpenImageIO from crashing on invalid IPTC metadata
    (``encode_iptc_iim_one_tag`` assertion failure) during image undistortion.

    Returns a dict with keys:
        success (bool)
        processed (int) — number of files actually touched
        error (str | None) — error message if the command failed
    """
    image_dir = DATA_DIR_CONTAINER / project_name / "Input" / "input_image"
    log = get_logger(__name__)

    # Only pass supported image extensions to exiftool
    ext_args: list[str] = []
    for ext in _SUPPORTED_EXTS:
        ext_args += ["-ext", ext.lstrip(".")]

    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "exiftool", "-all=", "-overwrite_original",
        *ext_args,
        str(image_dir) + "/",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # generous timeout for slow disks
        )
    except FileNotFoundError:
        log.warning("Project %r: docker not available for image preprocessing", project_name)
        return {"success": False, "processed": 0, "error": "docker not found"}
    except subprocess.TimeoutExpired:
        log.warning("Project %r: image preprocessing timed out", project_name)
        return {"success": False, "processed": 0, "error": "timeout"}

    # exiftool outputs "N image files updated" on success; parse that line
    processed = 0
    if result.returncode == 0 and result.stdout:
        for line in result.stdout.splitlines():
            if "image files updated" in line:
                try:
                    processed = int(line.strip().split()[0])
                except ValueError:
                    pass

    # exit code 1 usually means "nothing to do" — not an error
    if result.returncode not in (0, 1):
        log.warning(
            "Project %r: exiftool exited with %d — stderr: %s",
            project_name, result.returncode, result.stderr[:200] if result.stderr else "",
        )
        return {"success": False, "processed": processed,
                "error": f"exiftool exit code {result.returncode}"}

    log.info("Project %r: metadata stripped from %d image(s)", project_name, processed)
    return {"success": True, "processed": processed, "error": None}


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

_running_processes: dict[str, subprocess.Popen] = {}


def run_pipeline_async(
    project_name: str,
    cmd: list[str],
    on_complete: callable | None = None,
) -> subprocess.Popen | None:
    """Run the pipeline command asynchronously in a background thread."""

    def _target():
        logger.info("Project %r: launching docker exec...", project_name)

        # Ensure container log follower is running
        start_follow(CONTAINER_NAME)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr → stdout for real-time line order
            text=True,
            bufsize=1,  # line-buffered
        )
        _running_processes[project_name] = proc

        # Prepare log paths
        pipeline_log = DATA_DIR_HOST / project_name / "SFM_output" / "pipeline.log"
        pipeline_log.parent.mkdir(parents=True, exist_ok=True)

        # Stream output line-by-line to both log files in real time
        DOCKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(pipeline_log, "a", encoding="utf-8") as proj_f, \
             open(CONTAINER_LOG_PATH, "a", encoding="utf-8") as dock_f:

            header = f"\n{'='*60}\n=== PIPELINE START {project_name} | {' '.join(cmd)}\n{'='*60}\n"
            proj_f.write(header)
            proj_f.flush()
            dock_f.write(header)
            dock_f.flush()

            for line in proc.stdout:
                proj_f.write(line)
                proj_f.flush()
                dock_f.write(line)
                dock_f.flush()

            proc.wait()
            logger.info("Project %r: docker exited with code %d", project_name, proc.returncode)

            footer = f"\n{'='*60}\n=== PIPELINE END {project_name} | exit_code: {proc.returncode}\n{'='*60}\n"
            proj_f.write(footer)
            proj_f.flush()
            dock_f.write(footer)
            dock_f.flush()

        if on_complete:
            on_complete(project_name, proc.returncode)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(0.1)

    return _running_processes.get(project_name)


# ---------------------------------------------------------------------------
# Process registry
# ---------------------------------------------------------------------------

def get_process(project_name: str) -> subprocess.Popen | None:
    return _running_processes.get(project_name)


def remove_process(project_name: str) -> None:
    _running_processes.pop(project_name, None)


def is_process_running(project_name: str) -> bool:
    proc = _running_processes.get(project_name)
    if proc is None:
        return False
    return proc.poll() is None
