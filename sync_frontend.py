#!/usr/bin/env python3
"""
Watch WSL directory and one-way sync to Windows directory.
Usage: python3 sync_frontend.py
"""

import os
import subprocess
import time
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# === CONFIG ===
SRC = "/home/zwz/Code/Platform/frontend"
DST = "/mnt/e/Photogrammetry/Frontend"

# Patterns to exclude (rsync --exclude)
EXCLUDES = [
    ".venv",
    "__pycache__",
    ".git",
    "node_modules",
    "*.pyc",
    ".DS_Store",
]

# Debounce time in seconds (batch changes within this window)
DEBOUNCE = 1.5


class SyncHandler(FileSystemEventHandler):
    def __init__(self):
        self._pending = False
        self._last_event = 0

    def _trigger_sync(self):
        now = time.time()
        self._last_event = now
        if not self._pending:
            self._pending = True
            while True:
                elapsed = time.time() - self._last_event
                if elapsed >= DEBOUNCE:
                    break
                time.sleep(0.2)
            self._pending = False
            self._run_rsync()

    def _run_rsync(self):
        cmd = ["rsync", "-av", "--delete"]
        for ex in EXCLUDES:
            cmd += ["--exclude", ex]
        cmd += [SRC + "/", DST + "/"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            count = len([l for l in result.stdout.splitlines()
                        if l and not l.startswith("sending ") and not l.startswith("sent ")])
            if count > 0 or result.stderr:
                ts = time.strftime("%H:%M:%S")
                if count > 0:
                    print(f"[{ts}] Synced {count} file(s)")
                if result.stderr:
                    print(f"[{ts}] {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"[{time.strftime('%H:%M:%S')}] WARNING: rsync timed out")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {e}")

    def on_created(self, event):
        print(f"[{time.strftime('%H:%M:%S')}] + {event.src_path}")
        self._trigger_sync()

    def on_modified(self, event):
        print(f"[{time.strftime('%H:%M:%S')}] ~ {event.src_path}")
        self._trigger_sync()

    def on_deleted(self, event):
        print(f"[{time.strftime('%H:%M:%S')}] - {event.src_path}")
        self._trigger_sync()

    def on_moved(self, event):
        print(f"[{time.strftime('%H:%M:%S')}] > {event.src_path} -> {event.dest_path}")
        self._trigger_sync()


def main():
    if not os.path.isdir(SRC):
        print(f"ERROR: Source directory not found: {SRC}")
        sys.exit(1)
    if not os.path.isdir(DST):
        print(f"ERROR: Target directory not found: {DST}")
        sys.exit(1)

    print(f"Watching: {SRC}")
    print(f"Syncing to: {DST}")
    print(f"Excludes: {', '.join(EXCLUDES)}")
    print("Press Ctrl+C to stop.\n")

    handler = SyncHandler()
    observer = Observer()
    observer.schedule(handler, SRC, recursive=True)
    observer.start()

    try:
        while observer.is_alive():
            observer.join(1)
    except KeyboardInterrupt:
        print("\nStopping sync...")
        observer.stop()
    observer.join()
    print("Done.")


if __name__ == "__main__":
    main()
