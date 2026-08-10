# -*- coding: utf-8 -*-

import cv2
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import ColorMode
import torch
import atexit
import bisect
import multiprocessing as mp
from collections import deque


from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import os

from detectron2.config import get_cfg

import sys
sys.path.insert(0, "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main")

import argparse
from detectron2.config import get_cfg

from swinb.config import add_swinb_config
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np

import re
import pandas as pd
from datetime import datetime

import json
from pycocotools import mask as mask_utils





class VisualizationDemo:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, confidence_threshold=0.5, min_size=0):
        """
        Args:
            image (np.ndarray): (H, W, C), BGR format.
            confidence_threshold (float): confidence threshold.
            min_size (int): minimal width and height or area of boxes to keep.
        Returns:
            predictions, vis_output, instances
        """
        vis_output = None
        predictions = self.predictor(image)
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)

                # Step 1: confidence threshold
                high_confidence_mask = instances.scores > confidence_threshold
                instances = instances[high_confidence_mask]

                # Step 2: IoU-based suppression
                def compute_iou(box1, box2):
                    x1, y1, x2, y2 = box1
                    x3, y3, x4, y4 = box2
                    xi1, yi1 = max(x1, x3), max(y1, y3)
                    xi2, yi2 = min(x2, x4), min(y2, y4)
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (x4 - x3) * (y4 - y3)
                    union_area = area1 + area2 - inter_area
                    return inter_area / union_area if union_area > 0 else 0

                boxes = instances.pred_boxes.tensor.cpu().numpy()
                scores = instances.scores.cpu().numpy()
                keep = []

                for i in range(len(boxes)):
                    if i in keep:
                        continue
                    discard = False
                    for j in keep:
                        iou = compute_iou(boxes[i], boxes[j])
                        if iou > 0.0:
                            if scores[i] <= scores[j]:
                                discard = True
                                break
                    if not discard:
                        keep.append(i)

                # Apply IoU-based keep
                instances = instances[keep]

                # # Step 3: min_size filtering (based on width/height or area)
                # boxes = instances.pred_boxes.tensor.cpu().numpy()
                # sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                # size_mask = sizes >= min_size
                # instances = instances[size_mask]
                # Step 3: min_size filtering (prefer mask area)
                if min_size > 0:
                    if instances.has("pred_masks"):
                        # pred_masks: (N, H, W) bool/uint8 tensor
                        masks = instances.pred_masks  # torch.Tensor
                        # mask面积=mask为True的像素数
                        mask_areas = masks.flatten(1).sum(dim=1)  # (N,)
                        size_mask = mask_areas >= min_size
                        instances = instances[size_mask]
                    else:
                        # fallback: box area
                        boxes = instances.pred_boxes.tensor
                        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                        size_mask = box_areas >= min_size
                        instances = instances[size_mask]

                
                

                # ==== 用自己的 id 替换类别，从 0 开始 ====
                num_inst = len(instances)
                if num_inst > 0:
                    custom_ids = torch.arange(num_inst, dtype=torch.int64)
                    # 用自定义 id 覆盖原来的类别 id
                    instances.pred_classes = custom_ids
                # ======================================

                # Visualize
                vis_output = visualizer.draw_instance_predictions(predictions=instances)


        return predictions, vis_output, instances

    
    

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break



def get_patch_window(start_box, patch_size = 224, image_size = 1980):
    x1, y1 = start_box[:2]
    # make sure patch not out of image
    x1 = int(min(x1, image_size - patch_size))
    y1 = int(min(y1, image_size - patch_size))
    return np.array([x1, y1, x1 + patch_size, y1 + patch_size])

def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1+x2) // 2, (y1+y2) // 2])

def is_inside(patch,box):
    cx,cy = box_center(box)
    return patch[0] <= cx <= patch[2] and patch[1] <= cy <= patch[3]    
    
def group_boxes_from_left_top(boxes, patch_size = 224, image_size = 1980):
    boxes = boxes.copy()
    indices = np.arange(len(boxes))
    visited = np.zeros(len(boxes), dtype = bool)
    
    # from the left top side of the object in image
    sort_idx = np.lexsort((boxes[:,0], boxes[:,1]))
    boxes = boxes[sort_idx]
    indices = indices[sort_idx]
    
    patches = []
    
    for i, box in enumerate(boxes):
        if visited[i]:
            continue
        patch = get_patch_window(box, patch_size, image_size)
        included = []
        
        for j, other_box in enumerate(boxes):
            if visited[j]:
                continue
            if is_inside(patch, other_box):
                included.append(indices[j])
                visited[j] = True
            
        # patches.append({
        #     "patch_box": patch,
        #     "box_indices": included
        # })
        patches.append(patch)

    return patches



cfg = get_cfg()
add_swinb_config(cfg) 

# cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/configs/transfiner/mask_rcnn_swinb_FPN_3x.yaml")

cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO/config.yaml")
cfg.MODEL.WEIGHTS = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO/model_0003849.pth"


# cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/configs/transfiner/mask_rcnn_swinb_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x/model_0010199.pth"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg.freeze()

# Initialize the tool
demo = VisualizationDemo(cfg, instance_mode=ColorMode.IMAGE)




# ==== tracking ====
def collect_frames_by_prefix(input_root, prefix):
    """
    在 input_root 下，收集所有以 prefix 开头的图像，
    并根据文件名里的时间字段排序。

    例如:
      prefix = "VID1513_A1_1"
      文件: VID1513_A1_1_2025y11m19d_08h20m.jpg
    """
    frames = []  # list of (t, fpath, fname)

    for fname in os.listdir(input_root):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        if fname.startswith("."):
            continue

        # 必须以 prefix_ 开头
        if not fname.startswith(prefix + "_"):
            continue

        # 解析时间
        _, t = parse_key_and_time(fname)
        if t is None:
            continue

        fpath = os.path.join(input_root, fname)
        frames.append((t, fpath, fname))

    # 按时间排序
    frames.sort(key=lambda x: x[0])
    return frames



# ===================== 文件名解析 =====================

def parse_key_and_time(filename):
    """
    输入: VID1513_A1_1_2025y11m19d_08h20m.jpg
    输出:
        key = "VID1513_A1_1"
        t   = datetime(2025, 11, 19, 8, 20)
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    if len(parts) < 5:
        return None, None

    key = "_".join(parts[0:3])  # VID1513_A1_1

    # 剩下部分合起来: 2025y11m19d_08h20m
    time_str = "_".join(parts[3:])  # 2025y11m19d_08h20m

    # 用正则解析
    m = re.match(r"(\d{4})y(\d{2})m(\d{2})d_(\d{2})h(\d{2})m", time_str)
    if not m:
        return key, None
    year, month, day, hour, minute = map(int, m.groups())
    t = datetime(year, month, day, hour, minute)
    return key, t

# ===================== 余弦距离 =====================

def cosine_distance(f1, f2):
    """
    f1, f2: 1D 向量 (numpy array)
    返回: 1 - cos_sim，范围 [0, 2]
    """
    denom = (np.linalg.norm(f1) * np.linalg.norm(f2))
    if denom == 0:
        return 1.0
    sim = float(np.dot(f1, f2) / denom)
    return 1.0 - sim

def get_instance_features(instances, image_shape):
    """
    给当前帧的所有 instances 生成特征向量
    只用中心点位置，不要面积:
        feature = [cx_norm, cy_norm]
    """
    h, w = image_shape[:2]
    boxes = instances.pred_boxes.tensor.cpu().numpy()  # [N, 4]
    feats = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cx_norm = cx / w
        cy_norm = cy / h
        feats.append(np.array([cx_norm, cy_norm], dtype=np.float32))

    return np.stack(feats, axis=0) if feats else np.zeros((0, 2), dtype=np.float32)

def euclidean_distance(p1, p2):
    return float(np.linalg.norm(p1 - p2))



def update_tracks(prev_tracks, det_feats, dist_thresh=0.05, max_missed=0):
    """
    prev_tracks: 轨迹列表，每个元素:
        {
            "id": int,
            "feat": np.array([cx_norm, cy_norm]),
            "missed": int,
            "active": bool
        }
    det_feats: 当前帧检测特征 [N, 2]，为归一化中心坐标
    dist_thresh: 欧式距离阈值（0~sqrt(2），这里从 0.15 先试起）
    返回:
        assignments: 长度 N 的列表，每个元素:
            {
              "track_id": 轨迹ID或None,
              "is_new": bool,
              "children": [parent_ids...]  # 多父时使用
            }
        prev_tracks: 更新 missed/active 后的轨迹列表
    """
    num_det = det_feats.shape[0]
    assignments = [
        {"track_id": None, "is_new": False, "children": []}
        for _ in range(num_det)
    ]

    # ===== 情况 0：当前帧没有检测 =====
    if num_det == 0:
        for tr in prev_tracks:
            if tr.get("active", True):
                tr["missed"] = tr.get("missed", 0) + 1
                if tr["missed"] >= max_missed:
                    tr["active"] = False
        return assignments, prev_tracks

    # ===== 情况 1：还没有任何历史轨迹 =====
    if len(prev_tracks) == 0:
        for i in range(num_det):
            assignments[i]["is_new"] = True   # 由外层创建新轨迹
        return assignments, prev_tracks

    # ===== 情况 2：正常匹配 =====
    parents_by_det = {i: [] for i in range(num_det)}

    # 只用 active 轨迹参与匹配
    for tr in prev_tracks:
        if not tr.get("active", True):
            continue
        tr_id = tr["id"]
        tr_feat = tr["feat"]  # [2]

        # 计算 tr_feat 到所有 det_feat 的距离，向量化避免写 for i
        dists = np.linalg.norm(det_feats - tr_feat[None, :], axis=1)  # [N]
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        # 在阈值之内，就把这个 track 作为该 detection 的一个 parent
        if min_dist < dist_thresh:
            parents_by_det[min_idx].append(tr_id)

    matched_track_ids = set()

    for det_idx in range(num_det):
        parents = parents_by_det[det_idx]

        if len(parents) == 0:
            # 没有任何 parent -> 交给外层当新 root
            assignments[det_idx]["is_new"] = True

        elif len(parents) == 1:
            # 一个 parent：延续该轨迹
            tr_id = parents[0]
            assignments[det_idx]["track_id"] = tr_id
            assignments[det_idx]["is_new"] = False
            assignments[det_idx]["children"] = []  # 单父不放到 children
            matched_track_ids.add(tr_id)

        else:
            # 多个 parent：在外层创建一个新 child，children 记录所有父 ID
            assignments[det_idx]["is_new"] = True
            assignments[det_idx]["children"] = parents[:]  # 多父
            matched_track_ids.update(parents)

    # ===== 根据 matched_track_ids 更新 prev_tracks 的 missed/active =====
    for tr in prev_tracks:
        if not tr.get("active", True):
            continue
        tr_id = tr["id"]
        if tr_id in matched_track_ids:
            tr["missed"] = 0
            tr["active"] = True
        else:
            tr["missed"] = tr.get("missed", 0) + 1
            if tr["missed"] >= max_missed:
                tr["active"] = False

    return assignments, prev_tracks






def draw_labeled_image(image, instances, det_obj_ids, save_path):
    """
    在 image 上叠加 mask 和 bbox + index (object_id)，然后保存到 save_path
    det_obj_ids: 长度 = len(instances)，每个 detection 对应的 object_id
    """
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    masks = None
    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()  # [N, H, W]

    overlay = image.copy()
    H, W = image.shape[:2]

    for i, box in enumerate(boxes):
        obj_id = det_obj_ids[i]
        color = (0, 255, 0)  # 红色

        # 叠加 mask
        if masks is not None:
            m = masks[i].astype(bool)
            # 只在 mask 区域做颜色融合
            overlay[m] = (overlay[m] * 0.6 + np.array(color, dtype=np.uint8) * 0.4).astype(np.uint8)

        # 画 bbox
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # 写 index（object_id）
        text = str(obj_id)
        cv2.putText(
            overlay,
            text,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    # 和原图再做一次融合，让视觉更柔和一点
    alpha = 0.8
    labeled = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    cv2.imwrite(save_path, labeled)

def collect_frames_by_prefix(input_root, prefix):
    """
    在 input_root 下，收集所有以 prefix 开头的图像，
    并根据文件名里的时间字段排序。

    例如:
      prefix = "VID1513_A1_1"
      文件: VID1513_A1_1_2025y11m19d_08h20m.jpg
    返回: [(t, fpath, fname), ...]，按 t 升序
    """
    frames = []

    for fname in os.listdir(input_root):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        if fname.startswith("."):
            continue

        # 必须以 prefix_ 开头
        if not fname.startswith(prefix + "_"):
            continue

        _, t = parse_key_and_time(fname)
        if t is None:
            continue

        fpath = os.path.join(input_root, fname)
        frames.append((t, fpath, fname))

    frames.sort(key=lambda x: x[0])  # 按时间排序
    return frames


def encode_mask_to_coco_rle(mask_bool):
    """
    mask_bool: [H, W] 的 bool numpy array
    返回可直接写入 Excel 的 JSON 字符串
    """
    mask_u8 = np.asfortranarray(mask_bool.astype(np.uint8))  # COCO RLE 要求 Fortran order
    rle = mask_utils.encode(mask_u8)  # dict: {'size': [H, W], 'counts': bytes}

    # counts 默认是 bytes，Excel/JSON 不方便存，转成 utf-8 字符串
    rle["counts"] = rle["counts"].decode("utf-8")

    # 存成 JSON 字符串，后面读回来也方便
    return json.dumps(rle, ensure_ascii=False)


# ===================== 主流程：读图 + 检测 + Tracking + 导出 Excel =====================

def run_tracking_for_prefix(input_root, demo, excel_out_path, labeled_out_root, prefix):
    """
    只对某个指定前缀（例如 VID1513_A1_1）下的图像做 tracking。
    """
    os.makedirs(labeled_out_root, exist_ok=True)

    # 1. 收集该前缀下的所有图像并按时间排序
    frames = collect_frames_by_prefix(input_root, prefix)
    if not frames:
        print(f"[WARN] no frames found for prefix = {prefix}")
        return

    print(f"Processing prefix = {prefix}, num_frames = {len(frames)}")

    next_track_id = 1
    tracks = []   # 所有轨迹，元素结构: {"id", "feat", "missed", "active"}

    all_rows = []

    # 每个 prefix 单独一个子目录放标注图像
    key_labeled_dir = os.path.join(labeled_out_root, prefix)
    os.makedirs(key_labeled_dir, exist_ok=True)

    for frame_idx, (t, fpath, fname) in enumerate(frames, start=1):
        image = cv2.imread(fpath)
        if image is None:
            print(f"[WARN] failed to read image: {fpath}")
            continue

        # predictions, vis_output, instances = demo.run_on_image(image)

        # if len(instances) == 0:
        #     print(f"[INFO] no instances in {fname}")
        #     continue
        
        # # ==== 特征（建议是 [cx_norm, cy_norm]）====
        # det_feats = get_instance_features(instances, image.shape)

        # # ==== 更新 track ====
        # assignments, tracks = update_tracks(tracks, det_feats, dist_thresh=0.08, max_missed=3)
        
        predictions, vis_output, instances = demo.run_on_image(image)

        if len(instances) == 0:
            print(f"[INFO] no instances in {fname}")
            continue

        # ========= 用 mask 面积过滤小目标 =========
        if instances.has("pred_masks"):
            masks = instances.pred_masks.cpu().numpy().astype(bool)  # [N, H, W]
            mask_areas = masks.reshape(masks.shape[0], -1).sum(axis=1)  # [N]

            keep = mask_areas >= 500
            if keep.sum() == 0:
                print(f"[INFO] all instances < 300 pixels in {fname}, skip frame")
                continue

            # 只保留面积 >= 300 的实例
            instances = instances[keep]
        # 如果没有 pred_masks，就暂时不按面积筛（也可以改成用 bbox 面积）
        # ======================================

        det_feats = get_instance_features(instances, image.shape)
        assignments, tracks = update_tracks(tracks, det_feats, dist_thresh=0.08, max_missed=3)

        
        

        num_det = len(assignments)
        det_obj_ids = [-1] * num_det
        det_parents = [None] * num_det

        for det_idx, assign in enumerate(assignments):
            feat = det_feats[det_idx]
            is_new = assign.get("is_new", False)
            track_id = assign.get("track_id", None)
            parents = assign.get("children", []) or []  # 多父情况才会填

            # === 旧轨迹延续（单父）===
            if (not is_new) and (track_id is not None):
                # 这个 detection 继续用父轨迹的 id
                pid = track_id
                det_obj_ids[det_idx] = pid
                det_parents[det_idx] = [pid]

                # 更新该轨迹的特征
                for tr in tracks:
                    if tr["id"] == pid:
                        tr["feat"] = feat
                        tr["missed"] = 0
                        tr["active"] = True
                        break

            # === 多父合并 -> 新 child，一个子轨迹，多父 parent_ids ===
            elif is_new and len(parents) > 0:
                child_id = next_track_id
                next_track_id += 1

                det_obj_ids[det_idx] = child_id
                det_parents[det_idx] = parents[:]   # detection 可以有多个 parent_id

                # 新 child 进入轨迹列表
                tracks.append({
                    "id": child_id,
                    "feat": feat,
                    "missed": 0,
                    "active": True,
                })

                # 这些 parent 可以选择标记为 inactive（看你需求）
                for tr in tracks:
                    if tr["id"] in parents:
                        tr["active"] = False

            # === 纯新 root（没有任何 parent）===
            else:
                child_id = next_track_id
                next_track_id += 1

                det_obj_ids[det_idx] = child_id
                det_parents[det_idx] = []  # 没有父节点

                tracks.append({
                    "id": child_id,
                    "feat": feat,
                    "missed": 0,
                    "active": True,
                })

        # ====== 写 Excel , 基于 mask 统计大小和位置 ======
        boxes = instances.pred_boxes.tensor.cpu().numpy()  # [N, 4]

        masks = None
        if instances.has("pred_masks"):
            masks = instances.pred_masks.cpu().numpy().astype(bool)  # [N, H, W]

        H, W = image.shape[:2]

        for det_idx, obj_id in enumerate(det_obj_ids):
            parent_ids = det_parents[det_idx]

            x1, y1, x2, y2 = boxes[det_idx]
            w = x2 - x1
            h = y2 - y1

            mask_rle = None

            if masks is not None:
                m = masks[det_idx]            # [H, W], bool
                mask_area = int(m.sum())

                if mask_area > 0:
                    ys, xs = np.where(m)
                    cx = float(xs.mean())
                    cy = float(ys.mean())

                    # 新增：保存 COCO RLE
                    mask_rle = encode_mask_to_coco_rle(m)
                else:
                    mask_area = 0
                    cx = float((x1 + x2) / 2.0)
                    cy = float((y1 + y2) / 2.0)

                    # 空 mask 也可以存一个空的 RLE，或者直接 None
                    mask_rle = None
            else:
                mask_area = int(w * h)
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                mask_rle = None

            row = {
                "key": prefix,
                "filename": fname,
                "timestamp": t,
                "frame_idx": frame_idx,
                "object_id": obj_id,
                "parent_ids": ",".join(map(str, parent_ids)) if parent_ids else "",

                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "bbox_w": float(w),
                "bbox_h": float(h),

                "mask_area": mask_area,
                "mask_cx": cx,
                "mask_cy": cy,
                "mask_cx_norm": cx / W,
                "mask_cy_norm": cy / H,

                # 新增
                "mask_rle": mask_rle,
            }
            all_rows.append(row)
            
        # 画图
        labeled_save_path = os.path.join(key_labeled_dir, fname)
        draw_labeled_image(image, instances, det_obj_ids, labeled_save_path)

    # 3. 导出 Excel
    if not all_rows:
        print("[WARN] no rows to save.")
        return

    df = pd.DataFrame(all_rows)
    df.sort_values(by=["key", "frame_idx", "object_id"], inplace=True)
    df.to_excel(excel_out_path, index=False)
    print(f"Excel saved to: {excel_out_path}")





if __name__ == "__main__":
    input_folder = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Full_datasets/0135_29validation_round3/darker"
    excel_out = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Partail_datasets/tracking_result_VID1513_A4_2.xlsx"
    labeled_out_root = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/CRISPR_validition/Partail_datasets/VID1513_A4_2/output"

    prefix = "VID1495_G5_1"

    run_tracking_for_prefix(input_folder, demo, excel_out, labeled_out_root, prefix)


