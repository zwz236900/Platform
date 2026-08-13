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
    convert_output_suffix: str = "txt",
    mvs_type: str = "colmap",
) -> list[str]:
    """Build the docker exec command to run sfm_cli.py inside the container.

    Project layout inside container:
        /data/<project>/Input/input_image/   →  --image_dir
        /data/<project>/SFM_output/          →  --output_dir
        /data/<project>/MVS_output/<type>/   →  --dense_dir (when mvs in steps)
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
        "--no_clean",
    ]

    # If mvs step is included, redirect dense output to MVS_output/<type>/
    if steps and any(s in ("mvs", "fusion") for s in steps):
        dense_dir = DATA_DIR_CONTAINER / project_name / "MVS_output" / mvs_type
        cmd += ["--dense_dir", str(dense_dir)]

    if convert_output_suffix:
        cmd += ["--convert_output_suffix", convert_output_suffix]
    if no_geom_consistency:
        cmd.append("--no_geom_consistency")
    if vocab_tree_path:
        cmd += ["--vocab_tree_path", vocab_tree_path]

    return cmd


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

_running_processes: dict[str, subprocess.Popen] = {}


def _ensure_colmap_running() -> None:
    """调用前确保 colmap-dev 容器在跑；未跑则 docker start（不常驻，随调用启动）。

    SfM 与 colmap MVS 都经此 runner 执行 docker exec，调用时统一拉起容器。
    同步执行：容器起不来直接抛错 → 上层 500。调用方应在标记 RUNNING 前调用，
    以免留下悬空运行状态。
    """
    try:
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True, text=True, timeout=15,
        )
        if inspect.returncode == 0 and inspect.stdout.strip() == "true":
            return  # already running
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError("Docker not available; cannot ensure colmap-dev container")

    logger.info("%s container not running, starting before exec...", CONTAINER_NAME)
    start = subprocess.run(
        ["docker", "start", CONTAINER_NAME],
        capture_output=True, text=True, timeout=30,
    )
    if start.returncode != 0:
        raise RuntimeError(
            f"Failed to start {CONTAINER_NAME} container: {start.stderr.strip()}\n"
            f"Create it first: docker run -d --name {CONTAINER_NAME} --gpus all "
            f"-v /home/zwz/Code/SFM/colmap:/colmap "
            f"-v /home/zwz/Code/DATA:/data "
            f"-v /home/zwz/Code/platform/shared_data:/shared_data "
            f"platform/colmap:latest tail -f /dev/null"
        )
    logger.info("%s container started", CONTAINER_NAME)


def run_pipeline_async(
    project_name: str,
    cmd: list[str],
    on_complete: callable | None = None,
) -> subprocess.Popen | None:
    """Run the pipeline command asynchronously in a background thread."""

    # 调用即启动：确保 colmap-dev 容器在跑（不常驻），失败则同步抛错
    _ensure_colmap_running()

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
