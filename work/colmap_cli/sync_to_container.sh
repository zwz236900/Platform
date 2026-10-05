#!/usr/bin/env bash
# Push sfm_cli.py into the running colmap-dev container.
#
# Why this exists: sfm_cli.py is baked into the platform/colmap:latest image
# (COPY . /colmap/), NOT bind-mounted. Editing the copy here on the host has no
# effect until it is copied in, and a container recreated from the image gets
# the OLD script back. So: after any `docker run` of a new colmap-dev container
# or after restoring the image, re-run this script.
#
# Permanent fix (not done here, would require recreating the container): add a
# single-file bind mount, which overrides just the CLI and survives rebuilds:
#     -v /home/zwz/Code/Platform/work/colmap_cli/sfm_cli.py:/colmap/sfm_cli.py
# Do NOT bind-mount a whole directory onto /colmap -- that hides the ~6.7 GB
# COLMAP source and build tree baked into the image at that path.
#
# Usage: ./sync_to_container.sh [container_name]

set -euo pipefail

CONTAINER="${1:-colmap-dev}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sfm_cli.py"
DST="/colmap/sfm_cli.py"

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: source not found: $SRC" >&2
    exit 1
fi

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "ERROR: container '$CONTAINER' does not exist." >&2
    echo "Create it first, e.g.:" >&2
    echo "  docker run -d --name $CONTAINER --gpus all --shm-size=8g \\" >&2
    echo "    -v /home/zwz/Code/DATA:/data $CONTAINER_IMAGE tail -f /dev/null" >&2
    exit 1
fi

docker cp "$SRC" "$CONTAINER:$DST"

# Verify the copy landed and is syntactically valid inside the container.
docker exec "$CONTAINER" python3 -m py_compile "$DST"
echo "OK: $SRC -> $CONTAINER:$DST (py_compile passed)"
