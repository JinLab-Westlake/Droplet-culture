import cv2
import numpy as np

def _to_gray_float(img):
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

    M = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iters, eps)

    try:
        _, M = cv2.findTransformECC(prev, curr, M, warp_mode, criteria, None, gauss_ksize if gauss_ksize else 1)
    except cv2.error:
        M = np.eye(2, 3, dtype=np.float32)

    return M  # curr -> prev

