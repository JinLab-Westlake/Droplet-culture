# # -*- coding: utf-8 -*-
# import os
# import json
# import cv2
# import numpy as np
# import pandas as pd
# from pycocotools import mask as mask_utils


# # =========================
# # 可调参数
# # =========================
# EXCEL_PATH = r"/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Partail_datasets/VID1506_G3_4/output/tracking_result_VID1506_G3_4.xlsx"
# IMAGE_DIR = r"/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Full_datasets/0134_29validation_round2"      # 另一个文件夹，里面是同名图像
# OUTPUT_DIR = r"/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Partail_datasets/VID1506_G3_4/output/lighter"

# # 是否画 mask / bbox / id
# DRAW_MASK = True
# DRAW_BBOX = True
# DRAW_ID = True

# # 数字大小相关，你主要改这几个
# FONT_SCALE = 1.2          # 数字大小，越大越大，例如 0.6 / 1.0 / 1.5 / 2.0
# FONT_THICKNESS = 2        # 数字粗细
# TEXT_PADDING = 4          # 数字背景框内边距
# TEXT_OFFSET_Y = 8         # 数字离框顶端的偏移

# # mask 显示透明度
# MASK_ALPHA = 0.35

# # bbox 线宽
# BOX_THICKNESS = 2

# # 如果想让数字更醒目，可以给文字加底色
# USE_TEXT_BG = True


# # =========================
# # 工具函数
# # =========================
# def ensure_dir(path):
#     os.makedirs(path, exist_ok=True)


# def color_from_id(obj_id):
#     """
#     给每个 object_id 一个稳定颜色
#     """
#     obj_id = int(obj_id)
#     np.random.seed(obj_id * 17 + 23)
#     color = np.random.randint(50, 255, size=3).tolist()
#     return tuple(int(c) for c in color)   # BGR 风格下也能直接用


# def decode_coco_rle(rle_json_str):
#     """
#     Excel 里的 mask_rle 是 JSON 字符串
#     这里解码回 [H, W] 的 mask
#     """
#     if pd.isna(rle_json_str) or rle_json_str in ["", None, False]:
#         return None

#     if isinstance(rle_json_str, dict):
#         rle = rle_json_str
#     else:
#         rle = json.loads(rle_json_str)

#     rle["counts"] = rle["counts"].encode("utf-8")
#     mask = mask_utils.decode(rle)   # [H, W], uint8
#     return mask.astype(bool)


# def draw_text_with_bg(img, text, org, color,
#                       font_scale=1.0, thickness=2,
#                       padding=4, bg_color=(0, 0, 0)):
#     """
#     在图上写字，带背景框
#     org: 左下角坐标
#     """
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
#     x, y = org

#     # 背景框
#     x1 = x - padding
#     y1 = y - th - padding
#     x2 = x + tw + padding
#     y2 = y + baseline + padding

#     # 防止越界
#     x1 = max(0, x1)
#     y1 = max(0, y1)
#     x2 = min(img.shape[1] - 1, x2)
#     y2 = min(img.shape[0] - 1, y2)

#     cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
#     cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


# def draw_mask_on_image(image, mask, color, alpha=0.35):
#     """
#     把 mask 半透明叠加到图像
#     color: BGR tuple
#     """
#     if mask is None:
#         return image

#     out = image.copy()
#     color_arr = np.array(color, dtype=np.uint8)

#     out[mask] = (out[mask] * (1 - alpha) + color_arr * alpha).astype(np.uint8)
#     return out


# def render_one_image(image, rows):
#     """
#     rows: 同一个 filename 对应的所有 Excel 行
#     """
#     vis = image.copy()

#     # 先画 mask，再画框和数字
#     if DRAW_MASK:
#         for _, row in rows.iterrows():
#             mask = None
#             if "mask_rle" in row.index:
#                 mask = decode_coco_rle(row["mask_rle"])

#             if mask is None:
#                 continue

#             obj_id = int(row["object_id"])
#             color = color_from_id(obj_id)
#             vis = draw_mask_on_image(vis, mask, color=color, alpha=MASK_ALPHA)

#     for _, row in rows.iterrows():
#         obj_id = int(row["object_id"])
#         color = color_from_id(obj_id)

#         # 读 bbox
#         has_box = all(k in row.index for k in ["x1", "y1", "x2", "y2"])
#         if has_box:
#             x1 = int(round(float(row["x1"])))
#             y1 = int(round(float(row["y1"])))
#             x2 = int(round(float(row["x2"])))
#             y2 = int(round(float(row["y2"])))
#         else:
#             x1 = y1 = x2 = y2 = None

#         # 画 bbox
#         if DRAW_BBOX and has_box:
#             cv2.rectangle(vis, (x1, y1), (x2, y2), color, BOX_THICKNESS)

#         # 画 object_id
#         if DRAW_ID:
#             text = str(obj_id)

#             if has_box:
#                 tx = x1
#                 ty = max(20, y1 - TEXT_OFFSET_Y)
#             else:
#                 # 没 bbox 时，尝试用 mask_cx / mask_cy
#                 if "mask_cx" in row.index and "mask_cy" in row.index:
#                     tx = int(round(float(row["mask_cx"])))
#                     ty = int(round(float(row["mask_cy"])))
#                 else:
#                     tx, ty = 20, 30

#             if USE_TEXT_BG:
#                 draw_text_with_bg(
#                     vis,
#                     text,
#                     (tx, ty),
#                     color=(255, 255, 255),
#                     font_scale=FONT_SCALE,
#                     thickness=FONT_THICKNESS,
#                     padding=TEXT_PADDING,
#                     bg_color=color
#                 )
#             else:
#                 cv2.putText(
#                     vis,
#                     text,
#                     (tx, ty),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     FONT_SCALE,
#                     color,
#                     FONT_THICKNESS,
#                     cv2.LINE_AA
#                 )

#     return vis


# # =========================
# # 主流程
# # =========================
# def main():
#     ensure_dir(OUTPUT_DIR)

#     df = pd.read_excel(EXCEL_PATH)

#     if "filename" not in df.columns:
#         raise ValueError("Excel 里必须有 filename 列")

#     if "object_id" not in df.columns:
#         raise ValueError("Excel 里必须有 object_id 列")

#     # 防止 object_id 被读成 float2
#     df["object_id"] = df["object_id"].astype(int)

#     # 按文件名分组
#     grouped = df.groupby("filename", sort=False)

#     total = 0
#     saved = 0

#     for fname, rows in grouped:
#         total += 1
#         img_path = os.path.join(IMAGE_DIR, fname)

#         if not os.path.exists(img_path):
#             print(f"[WARN] image not found: {img_path}")
#             continue

#         image = cv2.imread(img_path)
#         if image is None:
#             print(f"[WARN] failed to read image: {img_path}")
#             continue

#         vis = render_one_image(image, rows)

#         out_path = os.path.join(OUTPUT_DIR, fname)
#         cv2.imwrite(out_path, vis)
#         saved += 1
#         print(f"[OK] saved: {out_path}")

#     print(f"Done. total groups = {total}, saved = {saved}")


# if __name__ == "__main__":
#     main()



# -*- coding: utf-8 -*-
import os
import json
import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
import re


# =========================
# 路径参数
# =========================
EXCEL_PATH = r"/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Partail_datasets/tracking_result_VID1506_B10_3.xlsx"
IMAGE_DIR = r"/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Full_datasets/0135_29validation_round3"      #在另一个文件夹找同名图像
OUTPUT_DIR = r"/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Partail_datasets/VID1506_B10_3/output/lighter"


# =========================
# 风格参数：默认跟你原来的 code 一致
# =========================
COLOR_L = (0, 0, 0)
COLOR = (0, 255, 0)                    # 固定绿色
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX   # 字体
FONT_SCALE = 1.6                   # 字号，数字大小
FONT_THICKNESS = 2                   # 文字粗细
BOX_THICKNESS = 2                      # 框粗细

# mask 融合风格
MASK_BLEND_BASE = 0.3   # overlay[m] * 0.6
MASK_BLEND_COLOR = 0.4  # color * 0.4

# 最终整图融合
FINAL_ALPHA = 0.8       # cv2.addWeighted(overlay, 0.8, image, 0.2, 0)


# =========================
# Time 标注参数
# =========================
TIME_INTERVAL_HOURS = 2      # 每张图间隔 2 小时
TIME_FONT_SCALE = 1.4
TIME_THICKNESS = 3
TIME_MARGIN_X = 25
TIME_MARGIN_Y = 25
TIME_COLOR = (0, 0, 0)       # 黑色文字
TIME_OUTLINE_COLOR = (255, 255, 255)  # 白色描边，防止看不清


# =========================
# 视频参数
# =========================
VIDEO_PATH = os.path.join(OUTPUT_DIR, "tracking_labeled_video.mp4")
VIDEO_FPS = 2          # 每秒几帧。2 表示每张图停留 0.5 秒
VIDEO_CODEC = "mp4v"   # mp4 推荐 mp4v



def natural_key(s):
    """
    按文件名中的数字自然排序：
    img_2.tif 会排在 img_10.tif 前面
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def make_video_from_images(image_paths, video_path, fps=2):
    """
    把一组图片按顺序合成视频
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

        # 如果尺寸不一致，统一 resize 到第一张图大小
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)

    writer.release()
    print(f"[OK] video saved: {video_path}")


def format_time_label(frame_idx, interval_hours=2):
    """
    frame_idx: 第几张图，从 0 开始
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
    Excel 中的 mask_rle 解码成 bool mask
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
    按你原来的 draw_labeled_image 风格来画：
    - 固定绿色
    - 固定 FONT_HERSHEY_SIMPLEX
    - 先叠加 mask
    - 再画 bbox
    - 再写 object_id
    """
    overlay = image.copy()

    for _, row in rows.iterrows():
        obj_id = int(row["object_id"])
        obj_area = int(row["mask_area"])

        x1 = int(round(float(row["x1"])))
        y1 = int(round(float(row["y1"])))
        x2 = int(round(float(row["x2"])))
        y2 = int(round(float(row["y2"])))

        # 读取 mask
        mask = None
        if "mask_rle" in row.index:
            mask = decode_coco_rle(row["mask_rle"])

        # 叠加 mask
        if mask is not None:
            overlay[mask] = (
                overlay[mask] * MASK_BLEND_BASE
                + np.array(COLOR, dtype=np.uint8) * MASK_BLEND_COLOR
            ).astype(np.uint8)

        # 画 bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR, BOX_THICKNESS)

        # 写 object_id
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
        


    # # 和原图再融合一次
    # labeled = cv2.addWeighted(overlay, FINAL_ALPHA, image, 1 - FINAL_ALPHA, 0)
    # cv2.imwrite(save_path, labeled)

    # 和原图再融合一次
    labeled = cv2.addWeighted(overlay, FINAL_ALPHA, image, 1 - FINAL_ALPHA, 0)

    # 在左下角写 Time
    if time_text is not None:
        h, w = labeled.shape[:2]
        org = (TIME_MARGIN_X, h - TIME_MARGIN_Y)

        # 白色描边
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

        # 黑色主体
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
            raise ValueError(f"Excel 缺少必要列: {c}")

    # 防止 object_id 读成 float
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
        # draw_labeled_image_from_excel(
        #     image=image,
        #     rows=rows,
        #     save_path=out_path,
        #     font_scale=FONT_SCALE,   # 改这里控制数字大小
        # )
        
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

    # 按文件名自然排序后合成视频
    saved_image_paths = sorted(saved_image_paths, key=natural_key)
    make_video_from_images(
        image_paths=saved_image_paths,
        video_path=VIDEO_PATH,
        fps=VIDEO_FPS,
    )


if __name__ == "__main__":
    main()