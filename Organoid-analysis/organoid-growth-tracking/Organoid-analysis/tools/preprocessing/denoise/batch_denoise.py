import os
import cv2
import numpy as np

def denoise_image(img_bgr, strength=10):
    """
    对原始 BGR 图像整体去噪（支持灰度 / 彩色）
    img_bgr: np.ndarray, HxWxC 或 HxW
    strength: 去噪强度 h，越大去噪越狠，但会更糊一点
    """
    if img_bgr is None:
        return None

    # 灰度图
    if img_bgr.ndim == 2 or img_bgr.shape[2] == 1:
        return cv2.fastNlMeansDenoising(
            img_bgr,
            None,
            h=strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )
    else:
        # 彩色图（BGR）
        return cv2.fastNlMeansDenoisingColored(
            img_bgr,
            None,
            h=strength,       # 亮度通道去噪强度
            hColor=strength,  # 颜色通道去噪强度
            templateWindowSize=7,
            searchWindowSize=21,
        )

def batch_denoise(
    input_dir,
    output_dir,
    strength=10,
    exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
):
    """
    对 input_dir 下所有图像批量去噪并保存到 output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    # 遍历目录
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(exts):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] 读取失败，跳过: {in_path}")
            continue

        denoised = denoise_image(img, strength=strength)
        if denoised is None:
            print(f"[WARN] 去噪失败，跳过: {in_path}")
            continue

        ok = cv2.imwrite(out_path, denoised)
        if not ok:
            print(f"[WARN] 保存失败: {out_path}")
        else:
            print(f"[OK] {in_path} -> {out_path}")

if __name__ == "__main__":
    input_dir = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Partail_datasets/0133_29validation_round1"
    output_dir = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Partail_datasets/0133_29validation_round1/denoised_test"

    batch_denoise(input_dir, output_dir, strength=15)
