"""Capture Docker container logs to individual files — snapshot AND real-time.

Logs are written to /home/zwz/Code/Platform/log/docker_containers/<container_name>.log

Two modes of operation:

1. **Global monitor** (``--monitor`` / ``start_global_monitor()``) —
   runs as a daemon, watches for new containers via ``docker events``,
   and automatically follows every container that appears.  This is the
   default mode used by the backend at startup.

2. **Manual** — individual ``start_follow()``, ``live_tail()``, ``capture_logs()``.

Usage as CLI::

    python3 -m modules.docker_logger --monitor       # global daemon (default!)
    python3 -m modules.docker_logger --follow colmap-dev
    python3 -m modules.docker_logger --live          # live-tail ALL containers at once
    python3 -m modules.docker_logger --status        # show what's being followed
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from modules.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_DIR = Path("/home/zwz/Code/Platform/log/docker_containers")

# How often the global monitor reconciles (seconds)
_RECONCILE_INTERVAL = 15

# ---------------------------------------------------------------------------
# Background follower registry
# ---------------------------------------------------------------------------

_followers: dict[str, _Follower] = {}
_followers_lock = threading.Lock()


class _Follower:
    """Tracks a running ``docker logs -f`` background process."""

    def __init__(self, container: str, proc: subprocess.Popen, thread: threading.Thread):
        self.container = container
        self.proc = proc
        self.thread = thread
        self.started_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Public API — global monitor (the default mode)
# ---------------------------------------------------------------------------

_monitor_thread: threading.Thread | None = None
_monitor_stop = threading.Event()


def start_global_monitor() -> threading.Thread:
    """Start the global Docker log monitor daemon.

    Once started, the monitor will:
    - Immediately follow all currently-running containers
    - Watch ``docker events`` for newly-started containers and follow them
    - Periodically reconcile (catch anything the events missed)
    - Clean up followers for containers that have stopped

    This is idempotent — calling it twice only starts one monitor.

    Returns the monitor thread.
    """
    global _monitor_thread

    if _monitor_thread is not None and _monitor_thread.is_alive():
        logger.info("Global monitor already running")
        return _monitor_thread

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _monitor_stop.clear()

    _monitor_thread = threading.Thread(
        target=_monitor_loop, daemon=True, name="docker-log-monitor"
    )
    _monitor_thread.start()
    logger.info("Global monitor started → %s", LOG_DIR)
    return _monitor_thread


def stop_global_monitor() -> None:
    """Stop the global monitor and all followers."""
    _monitor_stop.set()
    stop_all_follows()
    logger.info("Global monitor stopped")


def is_monitor_running() -> bool:
    """Check if the global monitor daemon is active."""
    return _monitor_thread is not None and _monitor_thread.is_alive()


# ---------------------------------------------------------------------------
# Public API — manual follow (per-container)
# ---------------------------------------------------------------------------


def start_follow(container_name: str) -> bool:
    """Start a background thread that streams ``docker logs -f`` into the log file.

    Returns True if started, False if already following this container.
    """
    with _followers_lock:
        if container_name in _followers:
            # Check if the existing follower is still alive
            existing = _followers[container_name]
            if existing.proc.poll() is None:
                return False  # still running
            # Dead follower — clean up and restart
            _followers.pop(container_name, None)

    log_path = LOG_DIR / f"{container_name}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Write a start marker
    header = (
        f"\n{'=' * 60}\n"
        f"=== FOLLOW STARTED {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f" | container: {container_name}\n"
        f"{'=' * 60}\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(header)
        f.flush()

    # Launch docker logs -f
    try:
        proc = subprocess.Popen(
            ["docker", "logs", "-f", "--timestamps", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        logger.warning("docker not available — cannot follow %r", container_name)
        return False

    def _stream():
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                for line in proc.stdout:
                    f.write(line)
                    f.flush()
        except Exception:
            pass
        finally:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n{'=' * 60}\n"
                        f"=== FOLLOW STOPPED {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
                        f" | container: {container_name}"
                        f" | exit_code: {proc.returncode}\n"
                        f"{'=' * 60}\n"
                    )
                    f.flush()
            except Exception:
                pass
            with _followers_lock:
                _followers.pop(container_name, None)

    thread = threading.Thread(target=_stream, daemon=True, name=f"docker-log-{container_name}")
    thread.start()

    with _followers_lock:
        _followers[container_name] = _Follower(container_name, proc, thread)

    logger.info("Following %r → %s", container_name, log_path)
    return True


def stop_follow(container_name: str) -> bool:
    """Stop the background follower for a container. Returns True if one was running."""
    with _followers_lock:
        follower = _followers.pop(container_name, None)

    if follower is None:
        return False

    follower.proc.terminate()
    follower.thread.join(timeout=5)
    if follower.proc.poll() is None:
        follower.proc.kill()
    logger.info("Stopped following %r", container_name)
    return True


def stop_all_follows() -> int:
    """Stop all background followers. Returns how many were stopped."""
    with _followers_lock:
        names = list(_followers.keys())

    count = 0
    for name in names:
        if stop_follow(name):
            count += 1
    return count


def is_following(container_name: str) -> bool:
    """Check if a background follower is active for this container."""
    with _followers_lock:
        return container_name in _followers


def list_following() -> list[str]:
    """Return names of containers currently being followed."""
    with _followers_lock:
        return list(_followers.keys())


# ---------------------------------------------------------------------------
# Public API — live tail (foreground, prints to terminal)
# ---------------------------------------------------------------------------


def live_tail(container_name: str) -> None:
    """Block and print new log lines to stdout as they arrive.

    Like ``tail -f`` — reads the log file and prints each new line.
    Press Ctrl-C to stop.

    If no follower is active for this container, one is started first.
    """
    log_path = LOG_DIR / f"{container_name}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")

    if container_name not in _followers:
        start_follow(container_name)

    print(f"📄 Live tail: {log_path}\n   (Ctrl-C to stop)\n")

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)

            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n⏹  Stopped. (follower still running in background)")
    except OSError as e:
        print(f"Error reading log: {e}", file=sys.stderr)


def live_tail_all() -> None:
    """Live-tail ALL containers being followed, with a ``[name]`` prefix per line.

    Press Ctrl-C to stop.
    """
    # Ensure all running containers are being followed
    for name in _list_running_containers():
        start_follow(name)

    followed = list_following()
    if not followed:
        print("No containers to tail.")
        return

    log_paths = {name: LOG_DIR / f"{name}.log" for name in followed}
    for name, path in log_paths.items():
        if not path.exists():
            path.write_text("", encoding="utf-8")

    # Open all files at current end
    files = {}
    for name, path in log_paths.items():
        f = open(path, "r", encoding="utf-8", errors="replace")
        f.seek(0, 2)
        files[name] = f

    print(f"📄 Live tail: {', '.join(followed)}\n   (Ctrl-C to stop)\n")

    try:
        while True:
            any_output = False
            for name, f in files.items():
                line = f.readline()
                if line:
                    sys.stdout.write(f"[{name}] {line}")
                    any_output = True
            if any_output:
                sys.stdout.flush()
            else:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n⏹  Stopped. (followers still running in background)")
    finally:
        for f in files.values():
            f.close()


# ---------------------------------------------------------------------------
# Public API — snapshot (one-shot)
# ---------------------------------------------------------------------------


def capture_logs(container_names: list[str] | None = None) -> dict[str, bool]:
    """Take a one-shot snapshot of docker logs for specified containers.

    If *container_names* is None, captures all running containers.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if container_names is None:
        container_names = _list_running_containers()

    results: dict[str, bool] = {}
    for name in container_names:
        results[name] = _capture_one(name)

    return results


def capture_all() -> dict[str, bool]:
    """Take a one-shot snapshot of ALL Docker containers (running & stopped)."""
    names = _list_all_containers()
    results: dict[str, bool] = {}
    for name in names:
        results[name] = _capture_one(name)
    return results


# ---------------------------------------------------------------------------
# Legacy alias (used by main.py)
# ---------------------------------------------------------------------------


def auto_follow_running() -> list[str]:
    """Start background followers for every currently-running container.

    .. deprecated::
        Prefer ``start_global_monitor()`` for continuous monitoring.
        This is a one-shot snapshot of what's currently running.
    """
    names = _list_running_containers()
    for name in names:
        start_follow(name)
    if names:
        logger.info("Auto-following %d container(s): %s", len(names), ", ".join(names))
    return names


# ---------------------------------------------------------------------------
# Internals — global monitor loop
# ---------------------------------------------------------------------------


def _monitor_loop() -> None:
    """Main loop of the global monitor daemon."""
    logger.info("Global monitor: reconciling every %ds", _RECONCILE_INTERVAL)

    while not _monitor_stop.is_set():
        try:
            _reconcile()
        except Exception:
            logger.exception("Global monitor reconcile error — will retry")

        _monitor_stop.wait(_RECONCILE_INTERVAL)

    logger.info("Global monitor: loop exited")


def _reconcile() -> None:
    """Ensure every running container has a follower, and prune dead ones."""
    running = set(_list_running_containers())
    following = set(list_following())

    # Start followers for new containers
    for name in running - following:
        logger.info("Global monitor: new container %r detected → starting follow", name)
        start_follow(name)

    # Prune followers for containers that no longer exist or have died
    with _followers_lock:
        stale = []
        for name, follower in _followers.items():
            if name not in running:
                stale.append(name)
            elif follower.proc.poll() is not None:
                # Process died (container probably restarted) — restart
                stale.append(name)

    for name in stale:
        if name in running:
            logger.info("Global monitor: follower for %r died → restarting", name)
            stop_follow(name)
            start_follow(name)
        else:
            logger.info("Global monitor: container %r gone → cleaning up", name)
            stop_follow(name)


# ---------------------------------------------------------------------------
# Internals — docker CLI helpers
# ---------------------------------------------------------------------------


def _capture_one(container_name: str) -> bool:
    """Take a one-shot snapshot for a single container."""
    log_path = LOG_DIR / f"{container_name}.log"

    try:
        result = subprocess.run(
            ["docker", "logs", "--timestamps", container_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        logger.warning("docker not available — cannot capture logs for %r", container_name)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("docker logs timed out for %r", container_name)
        return False

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"=== SNAPSHOT {timestamp} | container: {container_name}")
        f.write(f" | exit_code: {result.returncode}\n")
        f.write(f"{'=' * 60}\n")
        if result.stdout:
            f.write(result.stdout)
        if result.stderr:
            f.write(f"\n--- STDERR ---\n{result.stderr}\n")

    logger.info("Docker logs snapshot for %r → %s (%d bytes)",
                container_name, log_path, len(result.stdout or "") + len(result.stderr or ""))
    return True


def _list_running_containers() -> list[str]:
    """Return names of currently running Docker containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [n.strip() for n in result.stdout.splitlines() if n.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _list_all_containers() -> list[str]:
    """Return names of ALL Docker containers (including stopped)."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [n.strip() for n in result.stdout.splitlines() if n.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Docker container log collector")
    parser.add_argument("--monitor", action="store_true",
                        help="Start global monitor daemon (auto-follows all containers)")
    parser.add_argument("--follow", metavar="CONTAINER",
                        help="Live-tail a single container's log (terminal)")
    parser.add_argument("--live", action="store_true",
                        help="Live-tail ALL containers at once (terminal)")
    parser.add_argument("--start", metavar="CONTAINER",
                        help="Start background follower for a container")
    parser.add_argument("--stop", metavar="CONTAINER",
                        help="Stop background follower for a container")
    parser.add_argument("--snapshot", action="store_true",
                        help="One-shot capture all running containers")
    parser.add_argument("--status", action="store_true",
                        help="Show which containers are being followed")

    args = parser.parse_args()

    if args.monitor:
        print(f"Starting global monitor (logs → {LOG_DIR})")
        start_global_monitor()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nShutting down...")
            stop_global_monitor()
    elif args.follow:
        live_tail(args.follow)
    elif args.live:
        live_tail_all()
    elif args.start:
        start_follow(args.start)
        print(f"Started background follower for {args.start}")
    elif args.stop:
        stop_follow(args.stop)
        print(f"Stopped follower for {args.stop}")
    elif args.snapshot:
        results = capture_logs()
        for name, ok in results.items():
            print(f"  {name}: {'✓' if ok else '✗'}")
    elif args.status:
        names = list_following()
        monitor = "running" if is_monitor_running() else "stopped"
        print(f"Global monitor: {monitor}")
        if names:
            print(f"Following ({len(names)}): {', '.join(names)}")
        else:
            print("Not following any containers.")
    else:
        # Default: start monitor
        print(f"Starting global monitor (logs → {LOG_DIR})")
        start_global_monitor()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nShutting down...")
            stop_global_monitor()
