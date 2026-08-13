"""Shared utilities for algorithm modules.

- COLMAP binary model reader (cameras.bin, images.bin)
- PLY point-cloud reader / writer (binary little-endian, float xyz + uchar rgb)
- Camera projection helpers
"""

import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# COLMAP camera model ID → params count (undistorted will always be PINHOLE)
# ---------------------------------------------------------------------------
CAMERA_MODEL_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE  f, cx, cy
    1: 4,   # PINHOLE         fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL   f, cx, cy, k
    3: 5,   # RADIAL          f, cx, cy, k1, k2
    4: 8,   # OPENCV          fx, fy, cx, cy, k1, k2, p1, p2
    5: 8,   # OPENCV_FISHEYE  fx, fy, cx, cy, k1, k2, k3, k4
}


# ---------------------------------------------------------------------------
# COLMAP binary readers
# ---------------------------------------------------------------------------

def read_cameras_bin(path: Path) -> dict[int, dict[str, Any]]:
    """Read cameras.bin → dict[cam_id, {model_id, width, height, params, K}].

    ``K`` is a (3,3) numpy intrinsic matrix built from the model params.
    """
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            n_params = CAMERA_MODEL_PARAMS.get(model_id, 0)
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))

            # Build intrinsic matrix K
            if model_id in (0, 2):  # SIMPLE_PINHOLE or SIMPLE_RADIAL
                fx, cx, cy = params[0], params[1], params[2]
                fy = fx
            elif model_id == 1:  # PINHOLE
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]

            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0,  1]], dtype=np.float64)

            cameras[cam_id] = {
                "model_id": model_id,
                "width": int(w),
                "height": int(h),
                "params": params,
                "K": K,
            }
    return cameras


def read_images_bin(path: Path) -> dict[int, dict[str, Any]]:
    """Read images.bin → dict[img_id, {qvec, tvec, cam_id, name, R, t}].

    ``R`` is the (3,3) world-to-camera rotation matrix,
    ``t`` is the (3,) translation vector.
    """
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]

            # Read null-terminated name
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8", errors="replace")

            n_pts = struct.unpack("<Q", f.read(8))[0]
            # Skip 2D point observations: (x double, y double, point3D_id uint64) each
            f.seek(n_pts * 24, os.SEEK_CUR)

            qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
            tvec = np.array([tx, ty, tz], dtype=np.float64)
            R = qvec2rotmat(qvec)

            images[img_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "cam_id": cam_id,
                "name": name,
                "R": R,
                "t": tvec,
            }
    return images


def read_cameras_text(path: Path) -> dict[int, dict[str, Any]]:
    """Read cameras.txt → dict[cam_id, {model_id, width, height, params, K}].

    每行: CAMERA_ID MODEL WIDTH HEIGHT PARAMS…（空格分隔，与 BIN 版本返回同构）。
    """
    model_ids = {
        "SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2, "RADIAL": 3,
        "OPENCV": 4, "OPENCV_FISHEYE": 5, "FULL_OPENCV": 6,
    }
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model_name = parts[1]
            w, h = int(parts[2]), int(parts[3])
            params = np.array([float(x) for x in parts[4:]], dtype=np.float64)

            # 与 BIN 版本一致：SIMPLE_* 只有单个焦距 f
            if model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                fx, fy, cx, cy = params[0], params[0], params[1], params[2]
            else:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]

            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0,  1]], dtype=np.float64)

            cameras[cam_id] = {
                "model_id": model_ids.get(model_name, 1),
                "width": int(w),
                "height": int(h),
                "params": params,
                "K": K,
            }
    return cameras


def read_images_text(path: Path) -> dict[int, dict[str, Any]]:
    """Read images.txt → dict[img_id, {qvec, tvec, cam_id, name, R, t}].

    每张图两行：首行 IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME，
    次行为 2D 观测（过滤算法不关心，跳过）。
    """
    images = {}
    with open(path, "r", encoding="utf-8") as f:
        raw = f.readlines()
    i = 0
    while i < len(raw):
        line = raw[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 9:
            i += 1
            continue
        img_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
        cam_id = int(parts[8])
        name = parts[9] if len(parts) > 9 else ""
        R = qvec2rotmat(qvec)

        images[img_id] = {
            "qvec": qvec,
            "tvec": tvec,
            "cam_id": cam_id,
            "name": name,
            "R": R,
            "t": tvec,
        }
        i += 2  # 跳过紧随其后的 2D 观测行
    return images


def load_sparse_model(sparse_dir: Path):
    """按需读取 COLMAP 稀疏模型（bin 或 txt 自动识别）。

    Returns:
        (cameras, images) dict；目录里两者都缺时返回 (None, None)。
    """
    cameras = images = None
    for name in ("cameras.bin", "cameras.txt"):
        if (sparse_dir / name).exists():
            cameras = (read_cameras_bin if name.endswith(".bin")
                       else read_cameras_text)(sparse_dir / name)
            break
    for name in ("images.bin", "images.txt"):
        if (sparse_dir / name).exists():
            images = (read_images_bin if name.endswith(".bin")
                      else read_images_text)(sparse_dir / name)
            break
    if cameras is None or images is None:
        return None, None
    return cameras, images


# ---------------------------------------------------------------------------
# Quaternion / rotation
# ---------------------------------------------------------------------------

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (qw, qx, qy, qz) to 3×3 rotation matrix R_w2c."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_points(
    xyz: np.ndarray,      # (N, 3)
    K: np.ndarray,         # (3, 3)
    R: np.ndarray,         # (3, 3) world-to-camera
    t: np.ndarray,         # (3,)
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points into image plane.

    Returns:
        u:     (N,) float — column (x) pixel coordinate
        v:     (N,) float — row (y) pixel coordinate
        valid: (N,) bool  — True if in front of camera AND within image bounds
    """
    # Camera coordinates: X_cam = R @ X_world + t
    xyz_cam = xyz @ R.T + t[np.newaxis, :]  # (N, 3)
    depth = xyz_cam[:, 2]

    # Perspective division
    u = (K[0, 0] * xyz_cam[:, 0]) / depth + K[0, 2]
    v = (K[1, 1] * xyz_cam[:, 1]) / depth + K[1, 2]

    valid = (depth > 0) & (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)

    return u, v, valid


# ---------------------------------------------------------------------------
# PLY I/O (binary little-endian only, float xyz + uchar rgb)
# ---------------------------------------------------------------------------

def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary PLY point cloud.

    Supports common schemas:
      - ``float x, y, z`` + ``uchar red, green, blue``  (MVS, COLMAP)
      - ``float x, y, z`` + ``float nx, ny, nz`` + ``uchar red, green, blue``

    Returns:
        xyz:  (N, 3) float64
        rgb:  (N, 3) uint8
    """
    with open(path, "rb") as f:
        # --- Parse header ---
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

    # Find vertex count
    n_verts = 0
    properties: list[tuple[str, str]] = []  # (name, type)
    in_vertex = False
    for line in header_lines:
        if line.startswith("element vertex "):
            n_verts = int(line.split()[2])
            in_vertex = True
        elif line.startswith("element ") and in_vertex:
            in_vertex = False  # next element (e.g. face)
        elif in_vertex and line.startswith("property "):
            parts = line.split()
            properties.append((parts[2], parts[1]))

    if n_verts == 0:
        raise ValueError(f"No vertices found in {path}")

    # Build dtype from properties
    dtype_map = {
        "float": "f4", "float32": "f4",
        "double": "f8", "float64": "f8",
        "uchar": "u1", "uint8": "u1",
        "int": "i4", "int32": "i4",
        "uint": "u4", "uint32": "u4",
    }
    prop_names = []
    prop_formats = []
    for name, ptype in properties:
        prop_names.append(name)
        prop_formats.append(dtype_map.get(ptype, "f4"))

    dtype = np.dtype({"names": prop_names, "formats": prop_formats})
    data = np.fromfile(path, dtype=dtype, count=n_verts,
                       offset=_ply_data_offset(path))

    # Extract xyz
    x_names = [n for n in prop_names if n in ("x", "y", "z") or n == "x"]
    if "x" in prop_names and "y" in prop_names and "z" in prop_names:
        xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    else:
        raise ValueError(f"Cannot determine xyz columns from: {prop_names}")

    # Extract rgb if present
    if "red" in prop_names and "green" in prop_names and "blue" in prop_names:
        rgb = np.column_stack([data["red"], data["green"], data["blue"]]).astype(np.uint8)
    elif "r" in prop_names and "g" in prop_names and "b" in prop_names:
        rgb = np.column_stack([data["r"], data["g"], data["b"]]).astype(np.uint8)
    else:
        rgb = np.zeros((n_verts, 3), dtype=np.uint8)

    return xyz, rgb


def _ply_data_offset(path: Path) -> int:
    """Find byte offset after 'end_header\\n'."""
    with open(path, "rb") as f:
        content = f.read()
    marker = b"end_header\n"
    idx = content.find(marker)
    if idx == -1:
        marker = b"end_header\r\n"
        idx = content.find(marker)
    if idx == -1:
        raise ValueError("PLY header not found")
    return idx + len(marker)


def write_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray | None = None,
    scores: np.ndarray | None = None,
) -> None:
    """Write a binary-little-endian PLY point cloud.

    Args:
        xyz: (N, 3) float64 or float32
        rgb: (N, 3) uint8, optional
        scores: (N,) float64, optional — stored as extra ``float score`` property
    """
    n = xyz.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build header
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    props = ["<3f", 3 * 4]  # (format, stride)

    if rgb is not None:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        props[0] += ",3B"
        props[1] += 3

    if scores is not None:
        header.append("property float score")
        props[0] += ",f"
        props[1] += 4

    header.append("end_header\n")

    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))

        # Build structured array with explicit field names
        dtype_descr: list[tuple[str, str]] = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ]
        arr_data = []
        if rgb is not None:
            dtype_descr += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        if scores is not None:
            dtype_descr.append(("score", "f4"))

        dtype = np.dtype(dtype_descr)
        arr = np.zeros(n, dtype=dtype)
        arr["x"] = xyz[:, 0].astype(np.float32)
        arr["y"] = xyz[:, 1].astype(np.float32)
        arr["z"] = xyz[:, 2].astype(np.float32)
        if rgb is not None:
            arr["red"] = rgb[:, 0]
            arr["green"] = rgb[:, 1]
            arr["blue"] = rgb[:, 2]
        if scores is not None:
            arr["score"] = scores.astype(np.float32)

        arr.tofile(f)
