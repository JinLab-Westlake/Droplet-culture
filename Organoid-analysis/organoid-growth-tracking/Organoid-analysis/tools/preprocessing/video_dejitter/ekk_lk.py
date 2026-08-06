
import os
import re
import cv2
import numpy as np
import pandas as pd

def _to_gray_float(img):
    """
    img: HxW 或 HxWxC，uint8/uint16/float 都可
    return: float32灰度图，范围大致[0,1]
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    mx = float(img.max()) if img.max() > 0 else 1.0
    return img / mx


def estimate_motion_ecc(prev_img, curr_img,
                        motion="translation",
                        n_iters=200,
                        eps=1e-6,
                        gauss_ksize=5):
    prev = _to_gray_float(prev_img)
    curr = _to_gray_float(curr_img)

    if gauss_ksize and gauss_ksize > 1:
        prev = cv2.GaussianBlur(prev, (gauss_ksize, gauss_ksize), 0)
        curr = cv2.GaussianBlur(curr, (gauss_ksize, gauss_ksize), 0)

    if motion == "translation":
        warp_mode = cv2.MOTION_TRANSLATION
    elif motion == "euclidean":
        warp_mode = cv2.MOTION_EUCLIDEAN
    elif motion == "affine":
        warp_mode = cv2.MOTION_AFFINE
    else:
        raise ValueError("motion must be translation/euclidean/affine")

    W = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iters, eps)

    try:
        _, W = cv2.findTransformECC(prev, curr, W, warp_mode, criteria, None, gauss_ksize if gauss_ksize else 1)
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32)

    # --- 关键：把 W(更像 template->input) 取逆，得到 curr->prev ---
    W33 = np.vstack([W, [0, 0, 1]]).astype(np.float32)
    try:
        M33 = np.linalg.inv(W33).astype(np.float32)
        M23 = M33[:2, :]
    except np.linalg.LinAlgError:
        M23 = np.eye(2, 3, dtype=np.float32)

    return M23  # curr -> prev（现在跟 LK 一致）

def estimate_motion_lk_ransac(prev_img, curr_img,
                              max_corners=500,
                              quality=0.01,
                              min_dist=8,
                              block_size=7,
                              win_size=21,
                              max_level=3,
                              ransac_thresh=3.0,
                              gauss_ksize=5,
                              mask=None):
    prev = _to_gray_float(prev_img)
    curr = _to_gray_float(curr_img)

    if gauss_ksize and gauss_ksize > 1:
        prev = cv2.GaussianBlur(prev, (gauss_ksize, gauss_ksize), 0)
        curr = cv2.GaussianBlur(curr, (gauss_ksize, gauss_ksize), 0)

    prev_u8 = np.clip(prev * 255.0, 0, 255).astype(np.uint8)
    curr_u8 = np.clip(curr * 255.0, 0, 255).astype(np.uint8)

    p0 = cv2.goodFeaturesToTrack(
        prev_u8,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_dist,
        blockSize=block_size,
        mask=mask
    )
    if p0 is None or len(p0) < 8:
        return np.eye(2, 3, dtype=np.float32)

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_u8, curr_u8, p0, None,
        winSize=(win_size, win_size),
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    st = st.reshape(-1)
    p0_good = p0.reshape(-1, 2)[st == 1]
    p1_good = p1.reshape(-1, 2)[st == 1]
    if len(p0_good) < 8:
        return np.eye(2, 3, dtype=np.float32)

    A, inliers = cv2.estimateAffinePartial2D(
        p0_good, p1_good,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000,
        confidence=0.99,
        refineIters=10
    )
    if A is None:
        return np.eye(2, 3, dtype=np.float32)

    A33 = np.vstack([A, [0, 0, 1]]).astype(np.float32)
    try:
        M33 = np.linalg.inv(A33).astype(np.float32)
        M = M33[:2, :]
    except np.linalg.LinAlgError:
        M = np.eye(2, 3, dtype=np.float32)

    return M  # curr -> prev

# =================== 批处理辅助函数 ===================

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

def aff23_to_33(M23):
    M33 = np.eye(3, dtype=np.float32)
    M33[:2, :] = M23.astype(np.float32)
    return M33

def list_frames_sorted(input_dir, sort_mode="name"):
    """
    sort_mode:
      - "name": 按文件名排序（最通用）
      - "time": 如果文件名里有你那种 2025y11m19d_08h20m，就按时间排序
    """
    files = [f for f in os.listdir(input_dir)
             if (not f.startswith(".")) and f.lower().endswith(IMG_EXTS)]
    if sort_mode == "name":
        files.sort()
        return [os.path.join(input_dir, f) for f in files], files

    # time sort: 找到最后的时间字段
    def parse_time(fname):
        base = os.path.splitext(fname)[0]
        m = re.search(r"(\d{4})y(\d{2})m(\d{2})d_(\d{2})h(\d{2})m$", base)
        if not m:
            return None
        y, mo, d, h, mi = map(int, m.groups())
        return (y, mo, d, h, mi)

    items = []
    for f in files:
        t = parse_time(f)
        if t is None:
            continue
        items.append((t, f))
    items.sort(key=lambda x: x[0])
    files2 = [f for _, f in items]
    return [os.path.join(input_dir, f) for f in files2], files2

def batch_estimate_motion(
    input_dir,
    out_csv,
    out_npz=None,
    method="ecc",          # "ecc" or "lk"
    ecc_motion="translation",
    sort_mode="name",
    save_warped_preview_dir=None,   # 设成目录就会保存对齐预览图（可选）
):
    paths, names = list_frames_sorted(input_dir, sort_mode=sort_mode)
    if len(paths) < 2:
        raise RuntimeError("帧数不足（至少2张）")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if save_warped_preview_dir:
        os.makedirs(save_warped_preview_dir, exist_ok=True)

    rows = []
    M_pair_list = []  # curr->prev 2x3
    T_ref_list = []   # curr->ref  3x3

    prev_img = None
    T_prev_to_ref = np.eye(3, dtype=np.float32)

    for i, (p, name) in enumerate(zip(paths, names)):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] 读图失败: {p}")
            continue

        if prev_img is None:
            # 第一帧：相对自身
            M23 = np.eye(2, 3, dtype=np.float32)
            T_curr_to_ref = np.eye(3, dtype=np.float32)
        else:
            if method == "ecc":
                M23 = estimate_motion_ecc(prev_img, img, motion=ecc_motion)
            elif method == "lk":
                M23 = estimate_motion_lk_ransac(prev_img, img)
            else:
                raise ValueError("method must be ecc or lk")

            M33 = aff23_to_33(M23)                 # curr -> prev
            T_curr_to_ref = T_prev_to_ref @ M33    # curr -> ref

        # 记录
        rows.append({
            "frame_idx": i,
            "filename": name,
            "M00": float(M23[0,0]), "M01": float(M23[0,1]), "M02": float(M23[0,2]),
            "M10": float(M23[1,0]), "M11": float(M23[1,1]), "M12": float(M23[1,2]),
            "T00": float(T_curr_to_ref[0,0]), "T01": float(T_curr_to_ref[0,1]), "T02": float(T_curr_to_ref[0,2]),
            "T10": float(T_curr_to_ref[1,0]), "T11": float(T_curr_to_ref[1,1]), "T12": float(T_curr_to_ref[1,2]),
            "T20": float(T_curr_to_ref[2,0]), "T21": float(T_curr_to_ref[2,1]), "T22": float(T_curr_to_ref[2,2]),
        })
        M_pair_list.append(M23)
        T_ref_list.append(T_curr_to_ref)

        # 可选：保存对齐预览（把当前帧 warp 到上一帧坐标系）
        if save_warped_preview_dir and prev_img is not None:
            H, W = prev_img.shape[:2]
            warped = cv2.warpAffine(img, M23, (W, H), flags=cv2.INTER_LINEAR)
            out_p = os.path.join(save_warped_preview_dir, f"{i:04d}_{name}")
            cv2.imwrite(out_p, warped)

        # 更新 prev
        prev_img = img
        T_prev_to_ref = T_curr_to_ref

        if (i % 20) == 0:
            print(f"[INFO] processed {i+1}/{len(paths)}")

    # 导出 CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] CSV saved: {out_csv}")

    # 可选：导出 npz（方便后续直接加载矩阵）
    if out_npz:
        np.savez_compressed(
            out_npz,
            filenames=np.array([r["filename"] for r in rows], dtype=object),
            M_pair=np.stack(M_pair_list, axis=0),   # N x 2 x 3
            T_ref=np.stack(T_ref_list, axis=0),     # N x 3 x 3
        )
        print(f"[OK] NPZ saved: {out_npz}")

# batch_align_frames.py
import os
import re
import cv2
import numpy as np
import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# ====== 你已有的两个函数：直接 import 或粘贴 ======
# from tracking_sup.motion_comp import estimate_motion_ecc, estimate_motion_lk_ransac

# 若你在本文件里粘贴，就确保函数名一致：
# estimate_motion_ecc(prev_img, curr_img, ...)
# estimate_motion_lk_ransac(prev_img, curr_img, ...)

def aff23_to_33(M23: np.ndarray) -> np.ndarray:
    M33 = np.eye(3, dtype=np.float32)
    M33[:2, :] = M23.astype(np.float32)
    return M33

def list_frames_sorted(input_dir: str, sort_mode="name"):
    files = [f for f in os.listdir(input_dir)
             if (not f.startswith(".")) and f.lower().endswith(IMG_EXTS)]

    if sort_mode == "name":
        files.sort()
        return [os.path.join(input_dir, f) for f in files], files

    # sort_mode == "time": 识别末尾形如 2025y11m19d_08h20m
    def parse_time(fname):
        base = os.path.splitext(fname)[0]
        m = re.search(r"(\d{4})y(\d{2})m(\d{2})d_(\d{2})h(\d{2})m$", base)
        if not m:
            return None
        return tuple(map(int, m.groups()))

    items = []
    for f in files:
        t = parse_time(f)
        if t is not None:
            items.append((t, f))
    items.sort(key=lambda x: x[0])
    files2 = [f for _, f in items]
    return [os.path.join(input_dir, f) for f in files2], files2

def _warp_to_ref(img, T_curr_to_ref_33, ref_shape, interp=cv2.INTER_LINEAR):
    """
    把当前帧 img warp 到 ref 坐标系。
    T_curr_to_ref_33: 3x3
    ref_shape: (H,W) of reference frame
    """
    H, W = ref_shape[:2]
    M23 = T_curr_to_ref_33[:2, :]  # 2x3
    # borderMode 可以选 CONSTANT/REPLICATE，CONSTANT 会出现黑边
    warped = cv2.warpAffine(
        img, M23, (W, H),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped

def batch_align(
    input_dir: str,
    output_dir: str,
    method="ecc",              # "ecc" or "lk"
    ecc_motion="translation",  # translation/euclidean/affine
    sort_mode="name",          # name/time
    save_csv=True,
    csv_path=None,
    save_pair_warp=False,      # True: 同时保存 curr->prev 的 warp（更直观看抖动是否被消掉）
):
    os.makedirs(output_dir, exist_ok=True)

    paths, names = list_frames_sorted(input_dir, sort_mode=sort_mode)
    if len(paths) == 0:
        raise RuntimeError("输入目录没有图片")
    print(f"[INFO] found {len(paths)} frames")

    # 读第一帧作为 reference
    ref = cv2.imread(paths[0], cv2.IMREAD_UNCHANGED)
    if ref is None:
        raise RuntimeError(f"读第一帧失败: {paths[0]}")
    ref_shape = ref.shape[:2]

    # 输出第一帧（对齐结果就是自身）
    out0 = os.path.join(output_dir, names[0])
    cv2.imwrite(out0, ref)

    # 记录矩阵
    rows = []
    rows.append({
        "frame_idx": 0,
        "filename": names[0],
        "M_curr_to_prev": "I",
        "T_curr_to_ref": "I"
    })

    prev = ref
    T_prev_to_ref = np.eye(3, dtype=np.float32)

    for i in range(1, len(paths)):
        img = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] read failed: {paths[i]}")
            continue

        # 1) 估计 curr->prev
        if method == "ecc":
            M23 = estimate_motion_ecc(prev, img, motion=ecc_motion)
        elif method == "lk":
            M23 = estimate_motion_lk_ransac(prev, img)
        else:
            raise ValueError("method must be ecc or lk")

        # 2) 累计 curr->ref
        M33 = aff23_to_33(M23)            # curr -> prev
        T_curr_to_ref = T_prev_to_ref @ M33

        # 3) warp 当前帧到 ref 并保存
        aligned = _warp_to_ref(img, T_curr_to_ref, ref_shape)
        out_path = os.path.join(output_dir, names[i])
        cv2.imwrite(out_path, aligned)

        # 可选：也保存 curr->prev 的对齐（方便看“抖动是否消除”）
        if save_pair_warp:
            H, W = prev.shape[:2]
            warp_prev = cv2.warpAffine(img, M23, (W, H), flags=cv2.INTER_LINEAR)
            out_path2 = os.path.join(output_dir, f"_pair_{i:04d}_" + names[i])
            cv2.imwrite(out_path2, warp_prev)

        # 4) 记录
        rows.append({
            "frame_idx": i,
            "filename": names[i],
            "M00": float(M23[0,0]), "M01": float(M23[0,1]), "M02": float(M23[0,2]),
            "M10": float(M23[1,0]), "M11": float(M23[1,1]), "M12": float(M23[1,2]),
            "T00": float(T_curr_to_ref[0,0]), "T01": float(T_curr_to_ref[0,1]), "T02": float(T_curr_to_ref[0,2]),
            "T10": float(T_curr_to_ref[1,0]), "T11": float(T_curr_to_ref[1,1]), "T12": float(T_curr_to_ref[1,2]),
            "T20": float(T_curr_to_ref[2,0]), "T21": float(T_curr_to_ref[2,1]), "T22": float(T_curr_to_ref[2,2]),
        })

        prev = img
        T_prev_to_ref = T_curr_to_ref

        if i % 20 == 0:
            print(f"[INFO] processed {i}/{len(paths)-1}")

    if save_csv:
        if csv_path is None:
            csv_path = os.path.join(output_dir, "motion.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[OK] csv saved: {csv_path}")

    print(f"[OK] aligned frames saved in: {output_dir}")

# ====================== 你按需改下面路径直接运行 ======================
if __name__ == "__main__":
    input_dir = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/VID1495_G3_1/darker"
    output_dir = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/VID1495_G3_1/darker/frames_aligned"

    batch_align(
        input_dir=input_dir,
        output_dir=output_dir,
        method="ecc",              # "ecc" / "lk"
        ecc_motion="translation",  # 你是几像素晃动：translation 更稳
        sort_mode="name",          # 或 "time"
        save_csv=True,
        save_pair_warp=False
    )
