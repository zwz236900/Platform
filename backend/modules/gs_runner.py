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
        /data/<project>/SFM_output/dense/images/  →  --image-dir (去畸变后的图像)
        /data/<project>/SFM_output/dense/sparse/  →  --sfm-dir (去畸变后的 COLMAP 模型)
        /data/<project>/3DGS_input/               →  --source
        /data/<project>/3DGS_output/              →  -m / --model

    3DGS 需要 PINHOLE 相机 + 去畸变图像，因此使用 SfM undistort 阶段的输出
    （dense/images + dense/sparse），而非原始输入图（Input/input_image）与
    原始稀疏模型（SFM_output/sparse/0）。
    """
    source_dir = DATA_DIR_CONTAINER / project_name / "3DGS_input"
    model_dir = DATA_DIR_CONTAINER / project_name / "3DGS_output"
    image_dir = DATA_DIR_CONTAINER / project_name / "SFM_output" / "dense" / "images"
    sfm_dir = DATA_DIR_CONTAINER / project_name / "SFM_output" / "dense" / "sparse"

    cmd = [
        "docker", "exec", "-e", "PYTHONUNBUFFERED=1", CONTAINER_NAME,
        "python3", GS_CLI_PATH,
        command,
        "--source", str(source_dir),
    ]

    # --model 只在 train/render/full 子命令中存在（prepare 无此参数）
    if command in ("train", "render", "full"):
        cmd += ["--model", str(model_dir)]

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

# 最近一次 3DGS 运行的结果状态，keyed by project_name。
# 前端据此判定完成（替代解析日志关键字 / 只看 ready），进程退出即落定，幂等。
_gs_run_state: dict[str, dict] = {}


def _finalize_gs_state(project_name: str, exit_code: int) -> None:
    """记录一次 3DGS 运行的最终状态（进程退出时调用，幂等）。"""
    _gs_run_state[project_name] = {
        "status": "ok" if exit_code == 0 else "failed",
        "exit_code": exit_code,
    }


def get_gs_run_status(project_name: str) -> dict | None:
    """返回当前 3DGS 运行状态；本会话从未运行过返回 None。

    - {"status": "running"}                       进程存活
    - {"status": "ok"|"failed", "exit_code": int} 已结束
    """
    proc = _running_gs_processes.get(project_name)
    if proc is not None and proc.poll() is None:
        return {"status": "running"}
    if proc is not None:
        # 进程已退出而 _target 尚未落定状态（罕见竞态）→ 就地补齐
        _finalize_gs_state(project_name, proc.returncode)
    return _gs_run_state.get(project_name)


def _ensure_gs_running() -> None:
    """调用前确保 gs3 容器在跑；未跑则 docker start（不常驻，随调用启动）。

    与 openmvs / colmap-dev 的 runner 保持一致：容器按需拉起，失败同步抛错
    → 上层 500，不留悬空运行状态。
    """
    try:
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True, text=True, timeout=15,
        )
        if inspect.returncode == 0 and inspect.stdout.strip() == "true":
            return  # already running
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError("Docker not available; cannot ensure gs3 container")

    logger.info("%s container not running, starting before exec...", CONTAINER_NAME)
    start = subprocess.run(
        ["docker", "start", CONTAINER_NAME],
        capture_output=True, text=True, timeout=30,
    )
    if start.returncode != 0:
        raise RuntimeError(
            f"Failed to start {CONTAINER_NAME} container: {start.stderr.strip()}\n"
            f"Create it first: docker run -d --name {CONTAINER_NAME} --gpus all "
            f"-v /home/zwz/Code/DATA:/data "
            f"-v /home/zwz/Code/3DGS:/workspace/3DGS "
            f"platform/gs:latest tail -f /dev/null"
        )
    logger.info("%s container started", CONTAINER_NAME)


def run_gs_async(
    project_name: str,
    cmd: list[str],
    on_complete: callable | None = None,
) -> subprocess.Popen | None:
    """Run the 3DGS command asynchronously in a background thread."""

    # 调用即启动：确保 gs3 容器在跑（不常驻），失败则同步抛错
    _ensure_gs_running()

    # 同步标记运行中：保证 POST 返回时，前端立即查询到 run_status=running，
    # 不会误判为 idle 而提前停表（docker exec 启动可能慢于 join(0.1)）。
    _gs_run_state[project_name] = {"status": "running", "exit_code": None}

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
            # 落定运行状态（幂等；get_gs_run_status 也会在观察到期时补齐）
            _finalize_gs_state(project_name, proc.returncode)

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
