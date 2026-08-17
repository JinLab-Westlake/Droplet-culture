# -*- coding: utf-8 -*-
import os
import json
import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
import re


# path
EXCEL_PATH = r"/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Partail_datasets/tracking_result_VID1506_B10_3.xlsx"
IMAGE_DIR = r"/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Full_datasets/0135_29validation_round3"      #在另一个文件夹找同名图像
OUTPUT_DIR = r"/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Partail_datasets/VID1506_B10_3/output/lighter"



# style
COLOR_L = (0, 0, 0)
COLOR = (0, 255, 0)                    # green
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX   # font style
FONT_SCALE = 1.6                   # font size
FONT_THICKNESS = 2                   # font thickness
BOX_THICKNESS = 2                      # box thickness

# mask fusion style
MASK_BLEND_BASE = 0.3   # overlay[m] * 0.6
MASK_BLEND_COLOR = 0.4  # color * 0.4

# the whole graph is fused
FINAL_ALPHA = 0.8       # cv2.addWeighted(overlay, 0.8, image, 0.2, 0)



# Time annotation param
TIME_INTERVAL_HOURS = 2      # The interval between each image was 2 hours
TIME_FONT_SCALE = 1.4
TIME_THICKNESS = 3
TIME_MARGIN_X = 25
TIME_MARGIN_Y = 25
TIME_COLOR = (0, 0, 0)       # Black text
TIME_OUTLINE_COLOR = (255, 255, 255)  # White stroke to prevent confusion



# vedio param
VIDEO_PATH = os.path.join(OUTPUT_DIR, "tracking_labeled_video.mp4")
VIDEO_FPS = 2          # Several frames per second. 2 means that each image stays for 0.5 seconds
VIDEO_CODEC = "mp4v"   # mp4v



def natural_key(s):
    """
    Naturally sorted by the number in the filename:
    img_2.tif will be listed before img_10.tif
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def make_video_from_images(image_paths, video_path, fps=2):
    """
    A sequence of images is used to synthesize a video
    """
    if len(image_paths) == 0:
        print("[WARN] no images to make video")
        return

    first = cv2.imread(image_paths[0])
    if first is None:
        print(f"[WARN] failed to read first image: {image_paths[0]}")
        return

    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {video_path}")

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARN] failed to read frame: {img_path}")
            continue

        # If the sizes don't match, resize uniformly to the size of the first image
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)

    writer.release()
    print(f"[OK] video saved: {video_path}")


def format_time_label(frame_idx, interval_hours=2):
    """
    frame_idx: The number of images, starting at 0
    0 -> Time: 0h
    1 -> Time: 2h
    12 -> Time: 1day0h
    13 -> Time: 1day2h
    """
    total_hours = frame_idx * interval_hours

    if total_hours < 24:
        return f"Time: {total_hours}h"

    days = total_hours // 24
    hours = total_hours % 24
    return f"Time: {days}day{hours}h"



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def decode_coco_rle(rle_json_str):
    """
    mask_rle in Excel decoded to bool mask
    """
    if pd.isna(rle_json_str) or rle_json_str in ["", None, False]:
        return None

    if isinstance(rle_json_str, dict):
        rle = rle_json_str
    else:
        rle = json.loads(rle_json_str)

    rle["counts"] = rle["counts"].encode("utf-8")
    mask = mask_utils.decode(rle)   # [H, W], uint8
    return mask.astype(bool)


def draw_labeled_image_from_excel(image, rows, save_path, font_scale=0.6, time_text = None):
    """
    - Fixed green
    - Fixed FONT_HERSHEY_SIMPLEX
    - Overlay the mask first
    - Draw the bbox again
    - Write object_id again
    """
    overlay = image.copy()

    for _, row in rows.iterrows():
        obj_id = int(row["object_id"])
        obj_area = int(row["mask_area"])

        x1 = int(round(float(row["x1"])))
        y1 = int(round(float(row["y1"])))
        x2 = int(round(float(row["x2"])))
        y2 = int(round(float(row["y2"])))

        # read mask
        mask = None
        if "mask_rle" in row.index:
            mask = decode_coco_rle(row["mask_rle"])

        # overlay mask
        if mask is not None:
            overlay[mask] = (
                overlay[mask] * MASK_BLEND_BASE
                + np.array(COLOR, dtype=np.uint8) * MASK_BLEND_COLOR
            ).astype(np.uint8)

        # drawing bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR, BOX_THICKNESS)

        # object_id
        text = str(obj_id)
        cv2.putText(
            overlay,
            text,
            (x1, max(0, y1 - 5)),
            FONT_FACE,
            font_scale,
            COLOR_L,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
        
        area_scale = 0.8
        area_text = str(obj_area)
        cv2.putText(
            overlay,
            area_text,
            (x2 + 5, y1+25),
            FONT_FACE,
            area_scale,
            (0,130,0),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
        


    # fuse with original image
    labeled = cv2.addWeighted(overlay, FINAL_ALPHA, image, 1 - FINAL_ALPHA, 0)

    # Time
    if time_text is not None:
        h, w = labeled.shape[:2]
        org = (TIME_MARGIN_X, h - TIME_MARGIN_Y)

        cv2.putText(
            labeled,
            time_text,
            org,
            FONT_FACE,
            TIME_FONT_SCALE,
            TIME_OUTLINE_COLOR,
            TIME_THICKNESS + 3,
            cv2.LINE_AA,
        )

        # black font
        cv2.putText(
            labeled,
            time_text,
            org,
            FONT_FACE,
            TIME_FONT_SCALE,
            TIME_COLOR,
            TIME_THICKNESS,
            cv2.LINE_AA,
        )

    cv2.imwrite(save_path, labeled)

def main():
    ensure_dir(OUTPUT_DIR)

    df = pd.read_excel(EXCEL_PATH)
    
    

    required_cols = ["filename", "object_id", "x1", "y1", "x2", "y2","mask_area"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Excel lacks essential columns: {c}")

    # Prevent object_id from being read as float
    df["object_id"] = df["object_id"].astype(int)
    df["mask_area"] = df["mask_area"].astype(int)

    grouped = df.groupby("filename", sort=False)

    total = 0
    saved = 0
    
    saved_image_paths = []

    # for fname, rows in grouped:
    for frame_idx, (fname, rows) in enumerate(grouped):
        total += 1
        img_path = os.path.join(IMAGE_DIR, fname)

        if not os.path.exists(img_path):
            print(f"[WARN] image not found: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue

        out_path = os.path.join(OUTPUT_DIR, fname)
        
        time_text = format_time_label(frame_idx, TIME_INTERVAL_HOURS)

        draw_labeled_image_from_excel(
            image=image,
            rows=rows,
            save_path=out_path,
            font_scale=FONT_SCALE,
            time_text=time_text,
        )

        saved += 1
        saved_image_paths.append(out_path)
        print(f"[OK] saved: {out_path}")

    print(f"Done. total = {total}, saved = {saved}")

    # Synthetic video after natural sorting by filename
    saved_image_paths = sorted(saved_image_paths, key=natural_key)
    make_video_from_images(
        image_paths=saved_image_paths,
        video_path=VIDEO_PATH,
        fps=VIDEO_FPS,
    )


if __name__ == "__main__":
    main()