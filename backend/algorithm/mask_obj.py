"""掩码目标重建：按 SAM 掩码裁剪去畸变图 → 调整稀疏模型 → 复用现有 MVS 稠密重建。

流程（与用户指定一致）：
  1. 读 ``SAM_output/{point|box}_results`` 的掩码，按掩码 bbox（外扩 padding%）裁剪
     去畸变图 ``SFM_output/dense/images``，存 ``algorithm_output/mask_obj/images/``。
  2. 用裁剪几何调整 PINHOLE 稀疏模型（主点平移 cx-x0, cy-y0），
     写 ``algorithm_output/mask_obj/sparse/``（扁平 TXT，与 dense/sparse 同布局）。
  3. 打包裁剪图 ``cropped_images.zip``。
  4. 返回 ``sfm_cli.py --steps mvs`` 的 docker 命令（与现有 COLMAP MVS 同款）。

掩码与去畸变图同空间同尺寸（SAM 修正后均为 2519×1642）；旧掩码（原图空间）由
``_load_mask`` 按目标尺寸 NEAREST resize 兜底。

Usage (CLI):
    python -m algorithm.mask_obj --project 跑通流程 --mask-source point --padding 5

Usage (API):
    from algorithm.mask_obj import run
    result = run("跑通流程", mask_source="point", padding=5)
"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .common import read_cameras_text, read_images_text
from .pointcloud_filter import _load_mask

DATA_DIR = Path("/home/zwz/Code/DATA")

CONTAINER_NAME = "colmap-dev"
SFM_CLI_PATH = "/colmap/sfm_cli.py"


def _read_observations(images_txt: Path) -> dict[int, list[tuple[float, float, int]]]:
    """读 images.txt 的 2D 观测行 → dict[img_id, [(x, y, point3d_id), ...]]。

    每张图两行：位姿行 + 观测行；观测行是 ``x y point3d_id`` 三元组（-1 表示无 3D 对应）。
    """
    obs: dict[int, list[tuple[float, float, int]]] = {}
    with open(images_txt, encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        img_id = int(parts[0])
        obs_line = lines[i].strip() if i < len(lines) else ""
        i += 1
        vals = obs_line.split()
        triples = []
        for k in range(0, len(vals) - 2, 3):
            triples.append((float(vals[k]), float(vals[k + 1]), int(float(vals[k + 2]))))
        obs[img_id] = triples
    return obs


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(
    project_name: str,
    mask_source: str = "point",
    padding: int = 5,
    prompt_slug: str = "object",
) -> dict:
    """裁剪去畸变图 + 调整稀疏模型 + 打包，返回待执行的 sfm docker 命令。

    Returns:
        ``{success, images_cropped, images_skipped_no_mask, images_skipped_no_file,
        num_cameras, num_points3D, output_dir, command, error}``
    """
    project = DATA_DIR / project_name
    if not project.is_dir():
        return {"success": False, "error": f"Project not found: {project}"}

    dense = project / "SFM_output" / "dense"
    dense_images = dense / "images"
    dense_sparse = dense / "sparse"
    sam = project / "SAM_output"
    out = project / "algorithm_output" / "mask_obj"
    out_images = out / "images"
    out_sparse = out / "sparse"

    if not dense_images.is_dir():
        return {"success": False,
                "error": "未找到去畸变图像（SFM_output/dense/images），请先完成 SfM 去畸变步骤"}
    if not sam.is_dir():
        return {"success": False, "error": f"SAM output dir not found: {sam}"}

    cameras = read_cameras_text(dense_sparse / "cameras.txt") \
        if (dense_sparse / "cameras.txt").is_file() else None
    images = read_images_text(dense_sparse / "images.txt") \
        if (dense_sparse / "images.txt").is_file() else None
    if cameras is None or images is None:
        return {"success": False,
                "error": f"COLMAP 稀疏模型缺失（dense/sparse 的 cameras.txt/images.txt）: {dense_sparse}"}

    # 2D 观测（PatchMatch 的 __auto__ 源图选择依赖共视 track）
    observations = _read_observations(dense_sparse / "images.txt") \
        if (dense_sparse / "images.txt").is_file() else {}

    results_subdir = "box_results" if mask_source == "box" else "point_results"

    # 目标尺寸 = 去畸变相机尺寸（SfM 默认单相机，PINHOLE 全图一致）
    first_cam = next(iter(cameras.values()))
    t_w, t_h = first_cam["width"], first_cam["height"]

    # 清掉本功能专属的旧产物，避免残留数据混入本次运行
    for p in (out_images, out_sparse, out / "stereo"):
        if p.is_dir():
            shutil.rmtree(p)
    for name in ("fused.ply", "fused.ply.vis", "pipeline.log", "pipeline_result.json",
                 "cropped_images.zip", "crop_result.json"):
        f = out / name
        if f.is_file():
            f.unlink()

    out_images.mkdir(parents=True, exist_ok=True)

    cam_map: dict[tuple, int] = {}      # (w,h,cx,cy) → 新 cam_id
    new_cams: dict[int, dict] = {}
    image_rows: list[str] = []          # images.txt 内容（含空观察行）
    cropped_names: list[str] = []       # 裁剪图文件名（保持遍历顺序）
    point_tracks: dict[int, list[tuple[int, int]]] = {}  # pid → [(img_id, point2D_idx)]
    n_cropped = 0
    n_no_mask = 0
    n_no_file = 0
    total_obs = 0

    # 逐张稀疏图中的图（保证位姿齐全；dense/images 与之同名一一对应）
    for img_id, img in images.items():
        name = img["name"]
        cam = cameras.get(img["cam_id"])
        if cam is None:
            n_no_file += 1
            continue
        src = dense_images / name
        if not src.is_file():
            n_no_file += 1
            continue

        mask = _load_mask(sam, name, prompt_slug,
                          target_w=t_w, target_h=t_h, results_subdir=results_subdir)
        if mask is None:
            n_no_mask += 1
            continue
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            n_no_mask += 1
            continue

        # bbox + 外扩 padding% + clamp 到图像范围
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        pw = int(round((x1 - x0) * padding / 100)) + 1
        ph = int(round((y1 - y0) * padding / 100)) + 1

        im = Image.open(src).convert("RGB")
        iw, ih = im.size
        c0 = (max(0, x0 - pw), max(0, y0 - ph))
        c1 = (min(iw, x1 + pw + 1), min(ih, y1 + ph + 1))
        w, h = c1[0] - c0[0], c1[1] - c0[1]

        crop = im.crop((c0[0], c0[1], c1[0], c1[1]))
        crop.save(out_images / name, "JPEG", quality=95)
        cropped_names.append(name)

        # 新相机：主点平移 cx-x0, cy-y0，宽高=裁剪框
        K = cam["K"]
        fx, fy = K[0, 0], K[1, 1]
        cxc = K[0, 2] - c0[0]
        cyc = K[1, 2] - c0[1]
        key = (w, h, round(cxc, 6), round(cyc, 6))
        if key not in cam_map:
            new_cam_id = len(cam_map) + 1
            cam_map[key] = new_cam_id
            new_cams[new_cam_id] = {"w": w, "h": h, "fx": fx, "fy": fy,
                                    "cx": cxc, "cy": cyc}
        new_cam_id = cam_map[key]

        # 观测点裁剪：只保留落在裁剪窗内的，平移 -c0 到裁剪空间（pid 不变）
        kept = [(x, y, pid) for (x, y, pid) in observations.get(img_id, [])
                if c0[0] <= x < c1[0] and c0[1] <= y < c1[1]]
        obs_str = " ".join(f"{x - c0[0]:.3f} {y - c0[1]:.3f} {pid}"
                           for x, y, pid in kept)
        total_obs += len(kept)
        # 记录每个 3D 点的 track（image_id, 观测在裁剪后观察行里的序号）。
        # PatchMatch 的 __auto__ 源图选择与深度范围估计都依赖 track，
        # points3D.txt 的 track 列必须由 images.txt 的观测构建。
        for k, (_, _, pid) in enumerate(kept):
            point_tracks.setdefault(pid, []).append((img_id, k))

        qw, qx, qy, qz = img["qvec"]
        tx, ty, tz = img["tvec"]
        image_rows.append(
            f"{img_id} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} "
            f"{tx:.12g} {ty:.12g} {tz:.12g} {new_cam_id} {name}")
        image_rows.append(obs_str)
        n_cropped += 1

    if n_cropped == 0:
        return {"success": False,
                "error": (f"没有找到带掩码的去畸变图 "
                          f"（searched {sam / results_subdir}/ 与 {sam}/ 的 {Path(next(iter(images.values()))['name']).stem}_mask*.png）")}

    # ---- 写稀疏模型（扁平 TXT，与 dense/sparse 同布局，供 patch_match_stereo 读取）----
    out_sparse.mkdir(parents=True, exist_ok=True)

    with open(out_sparse / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(new_cams)}\n")
        for cid in sorted(new_cams):
            c = new_cams[cid]
            f.write(f"{cid} PINHOLE {c['w']} {c['h']} "
                    f"{c['fx']:.12g} {c['fy']:.12g} {c['cx']:.12g} {c['cy']:.12g}\n")

    with open(out_sparse / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        mean_obs = total_obs / n_cropped if n_cropped else 0
        f.write(f"# Number of images: {n_cropped}, mean observations per image: {mean_obs:.1f}\n")
        f.write("\n".join(image_rows) + "\n")

    # points3D：保留 id x y z r g b error，并重建 track 列（由上面 images.txt 观测得出）。
    # track 为无长度前缀的 (IMAGE_ID POINT2D_IDX)* 对（ReadPoints3DText 读到行尾），
    # PatchMatch 的 ComputeSharedPoints / ComputeDepthRanges 全靠它。
    num_pts = 0
    p3d_src = dense_sparse / "points3D.txt"
    with open(out_sparse / "points3D.txt", "w", encoding="utf-8") as fo:
        if p3d_src.is_file():
            with open(p3d_src, encoding="utf-8") as fi:
                for line in fi:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.split()
                    pid = int(parts[0])
                    track = sorted(point_tracks.get(pid, []))  # 按 img_id 排序，输出稳定
                    base = " ".join(parts[:8])
                    if track:
                        pairs = " ".join(f"{im} {idx}" for im, idx in track)
                        fo.write(f"{base} {pairs}\n")
                    else:
                        fo.write(f"{base}\n")
                    num_pts += 1
        else:
            fo.write("# Number of points: 0\n")

    # ---- patch-match.cfg（image_undistorter 才会生成；patch_match_stereo 必需）----
    # 格式：每张图两行——图名 + "__auto__, N"（自动选 N 张源图），与现有 dense/stereo 一致。
    cfg_dir = out / "stereo"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg_dir / "patch-match.cfg", "w", encoding="utf-8") as f:
        for cname in cropped_names:
            f.write(f"{cname}\n__auto__, 20\n")

    # ---- fusion.cfg（同样是 image_undistorter 生成；stereo_fusion 必需）----
    # 格式：每个参与融合的图名一行（与 patch-match.cfg 的图集一致）。
    with open(cfg_dir / "fusion.cfg", "w", encoding="utf-8") as f:
        for cname in cropped_names:
            f.write(f"{cname}\n")

    # ---- PatchMatch/StereoFusion 输出目录（image_undistorter 会预建；这里手动补齐）----
    # 缺了这些目录 patch_match 写深度图会 "Could not open .../depth_maps/xxx.bin"，
    # 每个视图都会报错但进程不退出，最终没有深度图可融合。
    for sub in ("depth_maps", "normal_maps", "consistency_graphs"):
        (cfg_dir / sub).mkdir(parents=True, exist_ok=True)

    # ---- 打包裁剪图 ----
    zip_path = out / "cropped_images.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_images.iterdir()):
            if p.is_file():
                zf.write(p, arcname=p.name)

    result = {
        "success": True,
        "images_cropped": n_cropped,
        "images_skipped_no_mask": n_no_mask,
        "images_skipped_no_file": n_no_file,
        "num_cameras": len(new_cams),
        "num_points3D": num_pts,
        "mean_observations": round(total_obs / n_cropped, 1),
        "output_dir": str(out),
        "cropped_images_zip": str(zip_path),
        "command": build_command(project_name),
    }
    with open(out / "crop_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ---------------------------------------------------------------------------
# Docker 命令
# ---------------------------------------------------------------------------

def build_command(project_name: str) -> list[str]:
    """构建 MVS 的 docker exec 命令（复用 sfm_cli.py 的 mvs 步骤）。

    工作区即 ``algorithm_output/mask_obj``（images/ + sparse/ 已就位），
    只跑 patch_match_stereo + stereo_fusion，产物 fused.ply 落在工作区。
    """
    base = f"/data/{project_name}/algorithm_output/mask_obj"
    return [
        "docker", "exec", "-e", "PYTHONUNBUFFERED=1", CONTAINER_NAME,
        "python3", SFM_CLI_PATH,
        "--image_dir", f"{base}/images",
        "--output_dir", base,
        "--dense_dir", base,
        "--steps", "mvs",
        "--no_clean",
        "--use_gpu", "1",
        "--gpu_index", "0",
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="掩码目标重建 — 裁剪去畸变图并调整稀疏模型")
    parser.add_argument("--project", required=True, help="Project name under DATA/")
    parser.add_argument("--mask-source", default="point", choices=["point", "box"],
                        help="SAM 掩码来源目录（point_results / box_results）")
    parser.add_argument("--padding", type=int, default=5,
                        help="bbox 外扩百分比（默认 5）")
    args = parser.parse_args()

    result = run(args.project, mask_source=args.mask_source, padding=args.padding)
    if result["success"]:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"FAIL — {result['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
