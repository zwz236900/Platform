"""Docker exec wrapper for running gs_cli.py inside the gs3 container."""

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

CONTAINER_NAME = "gs3"
GS_CLI_PATH = "/workspace/3DGS/gs_cli.py"       # path inside container
DATA_DIR_HOST = Path("/home/zwz/Code/DATA")       # mounted at /data in container
DATA_DIR_CONTAINER = Path("/data")
CONTAINER_LOG_PATH = DOCKER_LOG_DIR / f"{CONTAINER_NAME}.log"

# Valid subcommands
GS_COMMANDS = ["prepare", "train", "render", "full"]

# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def build_gs_command(
    project_name: str,
    *,
    command: str = "full",
    iterations: int = 30_000,
    resolution: int = 1,
    white_background: bool = False,
    quiet: bool = False,
    checkpoint_interval: int = 0,
    start_checkpoint: str | None = None,
) -> list[str]:
    """Build the docker exec command to run gs_cli.py inside the gs3 container.

    Project layout inside container:
        /data/<project>/Input/input_image/   →  --image-dir
        /data/<project>/SFM_output/sparse/0/ →  --sfm-dir
        /data/<project>/3DGS_input/          →  --source
        /data/<project>/3DGS_output/         →  -m / --model
    """
    source_dir = DATA_DIR_CONTAINER / project_name / "3DGS_input"
    model_dir = DATA_DIR_CONTAINER / project_name / "3DGS_output"
    image_dir = DATA_DIR_CONTAINER / project_name / "Input" / "input_image"
    sfm_dir = DATA_DIR_CONTAINER / project_name / "SFM_output" / "sparse" / "0"

    cmd = [
        "docker", "exec", "-e", "PYTHONUNBUFFERED=1", CONTAINER_NAME,
        "python3", GS_CLI_PATH,
        command,
        "--source", str(source_dir),
        "--model", str(model_dir),
    ]

    if command in ("prepare", "full"):
        cmd += [
            "--image-dir", str(image_dir),
            "--sfm-dir", str(sfm_dir),
        ]

    if command in ("train", "full"):
        cmd += [
            "--iterations", str(iterations),
        ]
        # Always pass explicit resolution to avoid 3DGS auto-downscaling
        cmd += ["--resolution", str(resolution)]
        if white_background:
            cmd.append("--white-background")
        if quiet:
            cmd.append("--quiet")
        if checkpoint_interval > 0:
            cmd += ["--checkpoint-interval", str(checkpoint_interval)]
        if start_checkpoint:
            cmd += ["--start-checkpoint", start_checkpoint]

    if command == "render":
        cmd += [
            "--iteration", str(iterations),
        ]
        if quiet:
            cmd.append("--quiet")

    return cmd


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

_running_gs_processes: dict[str, subprocess.Popen] = {}


def run_gs_async(
    project_name: str,
    cmd: list[str],
    on_complete: callable | None = None,
) -> subprocess.Popen | None:
    """Run the 3DGS command asynchronously in a background thread."""

    def _target():
        logger.info("Project %r: launching 3DGS docker exec...", project_name)

        # Ensure container log follower is running
        start_follow(CONTAINER_NAME)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _running_gs_processes[project_name] = proc

        # Prepare log paths
        gs_log = DATA_DIR_HOST / project_name / "3DGS_output" / "gs_pipeline.log"
        gs_log.parent.mkdir(parents=True, exist_ok=True)

        # Stream output to both log files in real time
        DOCKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(gs_log, "a", encoding="utf-8") as proj_f, \
             open(CONTAINER_LOG_PATH, "a", encoding="utf-8") as dock_f:

            header = f"\n{'='*60}\n=== 3DGS START {project_name} | {' '.join(cmd)}\n{'='*60}\n"
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
            logger.info("Project %r: 3DGS docker exited with code %d", project_name, proc.returncode)

            footer = f"\n{'='*60}\n=== 3DGS END {project_name} | exit_code: {proc.returncode}\n{'='*60}\n"
            proj_f.write(footer)
            proj_f.flush()
            dock_f.write(footer)
            dock_f.flush()

        if on_complete:
            on_complete(project_name, proc.returncode)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(0.1)

    return _running_gs_processes.get(project_name)


# ---------------------------------------------------------------------------
# Process registry
# ---------------------------------------------------------------------------

def get_gs_process(project_name: str) -> subprocess.Popen | None:
    return _running_gs_processes.get(project_name)


def remove_gs_process(project_name: str) -> None:
    _running_gs_processes.pop(project_name, None)


def is_gs_running(project_name: str) -> bool:
    proc = _running_gs_processes.get(project_name)
    if proc is None:
        return False
    return proc.poll() is None
