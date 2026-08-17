# -*- coding: utf-8 -*-

import cv2
import torch
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import os
import sys
from pathlib import Path
# Morphi-models/Morphi/Instance-segment
INSTANCE_SEGMENT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "Morphi"
    / "Instance-segment"
)
sys.path.insert(0, str(INSTANCE_SEGMENT_ROOT))
from detectron2.config import get_cfg
from swinb.config import add_swinb_config
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

                # confidence threshold
                high_confidence_mask = instances.scores > confidence_threshold
                instances = instances[high_confidence_mask]

                # IoU-based suppression
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

                # min_size filtering (prefer mask area)
                if min_size > 0:
                    if instances.has("pred_masks"):
                        # pred_masks: (N, H, W) bool/uint8 tensor
                        masks = instances.pred_masks  # torch.Tensor
                        # mask area = Number of pixels where mask is True
                        mask_areas = masks.flatten(1).sum(dim=1)  # (N,)
                        size_mask = mask_areas >= min_size
                        instances = instances[size_mask]
                    else:
                        # fallback: box area
                        boxes = instances.pred_boxes.tensor
                        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                        size_mask = box_areas >= min_size
                        instances = instances[size_mask]

                
                

                # Replace the category with its own id, starting at 0
                num_inst = len(instances)
                if num_inst > 0:
                    custom_ids = torch.arange(num_inst, dtype=torch.int64)
                    # Override the original category id with the custom id
                    instances.pred_classes = custom_ids

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

cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO/config.yaml")
cfg.MODEL.WEIGHTS = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO/model_0003849.pth"


cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg.freeze()

# Initialize the tool
demo = VisualizationDemo(cfg, instance_mode=ColorMode.IMAGE)




# ==== tracking ====
def collect_frames_by_prefix(input_root, prefix):
    """
    Under input_root, collect all images that start with prefix,
    And sort by the time field in the filename.

    For example:
      prefix = "VID1513_A1_1"
      file: VID1513_A1_1_2025y11m19d_08h20m.jpg
    """
    frames = []  # list of (t, fpath, fname)

    for fname in os.listdir(input_root):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        if fname.startswith("."):
            continue

        # must start from prefix_
        if not fname.startswith(prefix + "_"):
            continue

        # parsing time
        _, t = parse_key_and_time(fname)
        if t is None:
            continue

        fpath = os.path.join(input_root, fname)
        frames.append((t, fpath, fname))

    # Sort by time
    frames.sort(key=lambda x: x[0])
    return frames



# Filename resolution
def parse_key_and_time(filename):
    """
    Input: VID1513_A1_1_2025y11m19d_08h20m.jpg
    Output:
        key = "VID1513_A1_1"
        t   = datetime(2025, 11, 19, 8, 20)
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    if len(parts) < 5:
        return None, None

    key = "_".join(parts[0:3])  # VID1513_A1_1

    #  2025y11m19d_08h20m
    time_str = "_".join(parts[3:])

    # Using regular parsing
    m = re.match(r"(\d{4})y(\d{2})m(\d{2})d_(\d{2})h(\d{2})m", time_str)
    if not m:
        return key, None
    year, month, day, hour, minute = map(int, m.groups())
    t = datetime(year, month, day, hour, minute)
    return key, t

# cousin distance

def cosine_distance(f1, f2):
    """
    f1, f2: 1D vectors (numpy array)
    Return: 1-cos_sim, range [0, 2]
    """
    denom = (np.linalg.norm(f1) * np.linalg.norm(f2))
    if denom == 0:
        return 1.0
    sim = float(np.dot(f1, f2) / denom)
    return 1.0 - sim

def get_instance_features(instances, image_shape):
    """
    Generate feature vectors for all instances of the current frame
    Only the center position, not the area:
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
    prev_tracks: List of trajectories, each element:
        {
            "id": int,
            "feat": np.array([cx_norm, cy_norm]),
            "missed": int,
            "active": bool
        }
    det_feats: The detection feature [N, 2] of the current frame is the normalized center coordinate
    dist_thresh: Euclidean distance threshold ,here is 0.05)
    :return:
        assignments: A list of length N, with each element:
            {
              "track_id": Trajectory ID or None,
              "is_new": bool,
              "children": [parent_ids...]  # Used with multiple parents
            }
        prev_tracks: Update the list of trajectories after missed/active
    """
    num_det = det_feats.shape[0]
    assignments = [
        {"track_id": None, "is_new": False, "children": []}
        for _ in range(num_det)
    ]

    # 1. When there is no detection in the current frame
    if num_det == 0:
        for tr in prev_tracks:
            if tr.get("active", True):
                tr["missed"] = tr.get("missed", 0) + 1
                if tr["missed"] >= max_missed:
                    tr["active"] = False
        return assignments, prev_tracks

    # 2. When there is no historical track yet
    if len(prev_tracks) == 0:
        for i in range(num_det):
            assignments[i]["is_new"] = True   # New trajectories are created by the outer layer
        return assignments, prev_tracks

    # 3. Normal match
    parents_by_det = {i: [] for i in range(num_det)}

    # Only active trajectories are involved in matching
    for tr in prev_tracks:
        if not tr.get("active", True):
            continue
        tr_id = tr["id"]
        tr_feat = tr["feat"]  # [2]

        # The distance of tr_feat to all det_feat is calculated
        dists = np.linalg.norm(det_feats - tr_feat[None, :], axis=1)  # [N]
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        # Within the threshold, the track is used as a parent for the detection
        if min_dist < dist_thresh:
            parents_by_det[min_idx].append(tr_id)

    matched_track_ids = set()

    for det_idx in range(num_det):
        parents = parents_by_det[det_idx]

        if len(parents) == 0:
            # There is no parent -> given to the outer layer as a new root
            assignments[det_idx]["is_new"] = True

        elif len(parents) == 1:
            # A parent: continues the trajectory
            tr_id = parents[0]
            assignments[det_idx]["track_id"] = tr_id
            assignments[det_idx]["is_new"] = False
            assignments[det_idx]["children"] = []  # 单父不放到 children
            matched_track_ids.add(tr_id)

        else:
            # Multiple parents-Create a new child in the outer layer,
            # and children keeps track of all the parent ids
            assignments[det_idx]["is_new"] = True
            assignments[det_idx]["children"] = parents[:]  # 多父
            matched_track_ids.update(parents)

    # Update the missed/active of prev_tracks according to matched_track_ids
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
    Overlay mask and bbox + index (object_id) on the image and save to save_path
    det_obj_ids: length = len(instances), object_id for each detection
    """
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    masks = None
    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()  # [N, H, W]

    overlay = image.copy()
    H, W = image.shape[:2]

    for i, box in enumerate(boxes):
        obj_id = det_obj_ids[i]
        color = (0, 255, 0)  # red

        # 叠加 mask
        if masks is not None:
            m = masks[i].astype(bool)
            # Color blending is done only in the mask region
            overlay[m] = (overlay[m] * 0.6 + np.array(color, dtype=np.uint8) * 0.4).astype(np.uint8)

        # drawing bbox
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        #  index（object_id）
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

    # Blend it with the original image again
    alpha = 0.8
    labeled = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    cv2.imwrite(save_path, labeled)

def collect_frames_by_prefix(input_root, prefix):
    """
    Under input_root, collect all images that start with prefix,
    And sort by the time field in the filename.

    For example:
    prefix = "VID1513_A1_1"
    File: VID1513_A1_1_2025y11m19d_08h20m.jpg
    Return: [(t, fpath, fname),...] In ascending order of t
    """
    frames = []

    for fname in os.listdir(input_root):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        if fname.startswith("."):
            continue

        # must start from prefix_
        if not fname.startswith(prefix + "_"):
            continue

        _, t = parse_key_and_time(fname)
        if t is None:
            continue

        fpath = os.path.join(input_root, fname)
        frames.append((t, fpath, fname))

    frames.sort(key=lambda x: x[0])  # sorted by time
    return frames


def encode_mask_to_coco_rle(mask_bool):
    """
    mask_bool: [H, W] 的 bool numpy array
    Returns a JSON string that can be written directly into Excel
    """
    mask_u8 = np.asfortranarray(mask_bool.astype(np.uint8))  # COCO RLE require Fortran order
    rle = mask_utils.encode(mask_u8)  # dict: {'size': [H, W], 'counts': bytes}

    # counts defaults to bytes, Excel/JSON is not convenient to store,
    # convert to utf-8 string
    rle["counts"] = rle["counts"].decode("utf-8")

    # save as json
    return json.dumps(rle, ensure_ascii=False)


# Main: Read graph + detection + Tracking + Export Excel

def run_tracking_for_prefix(input_root, demo, excel_out_path, labeled_out_root, prefix):
    """
    Do tracking only for images under a given prefix (for example VID1513_A1_1).
    """
    os.makedirs(labeled_out_root, exist_ok=True)

    # All images under that prefix are collected and sorted by time
    frames = collect_frames_by_prefix(input_root, prefix)
    if not frames:
        print(f"[WARN] no frames found for prefix = {prefix}")
        return

    print(f"Processing prefix = {prefix}, num_frames = {len(frames)}")

    next_track_id = 1
    tracks = []   # element structure: {"id", "feat", "missed", "active"}

    all_rows = []

    # Each prefix has a separate subdirectory to record and
    # play the annotated image
    key_labeled_dir = os.path.join(labeled_out_root, prefix)
    os.makedirs(key_labeled_dir, exist_ok=True)

    for frame_idx, (t, fpath, fname) in enumerate(frames, start=1):
        image = cv2.imread(fpath)
        if image is None:
            print(f"[WARN] failed to read image: {fpath}")
            continue

        predictions, vis_output, instances = demo.run_on_image(image)

        if len(instances) == 0:
            print(f"[INFO] no instances in {fname}")
            continue

        # filter by size
        if instances.has("pred_masks"):
            masks = instances.pred_masks.cpu().numpy().astype(bool)  # [N, H, W]
            mask_areas = masks.reshape(masks.shape[0], -1).sum(axis=1)  # [N]

            keep = mask_areas >= 500
            if keep.sum() == 0:
                print(f"[INFO] all instances < 300 pixels in {fname}, skip frame")
                continue

            instances = instances[keep]

        det_feats = get_instance_features(instances, image.shape)
        assignments, tracks = update_tracks(tracks, det_feats, dist_thresh=0.08, max_missed=3)

        
        

        num_det = len(assignments)
        det_obj_ids = [-1] * num_det
        det_parents = [None] * num_det

        for det_idx, assign in enumerate(assignments):
            feat = det_feats[det_idx]
            is_new = assign.get("is_new", False)
            track_id = assign.get("track_id", None)
            parents = assign.get("children", []) or []  # in multi-parents situation

            # Continuation of old trajectory (single parent)
            if (not is_new) and (track_id is not None):
                pid = track_id
                det_obj_ids[det_idx] = pid
                det_parents[det_idx] = [pid]

                # update feature of trajectory
                for tr in tracks:
                    if tr["id"] == pid:
                        tr["feat"] = feat
                        tr["missed"] = 0
                        tr["active"] = True
                        break

            # Multi-parent merge -> new child, one child trace, multi-parent parent_ids
            elif is_new and len(parents) > 0:
                child_id = next_track_id
                next_track_id += 1

                det_obj_ids[det_idx] = child_id
                det_parents[det_idx] = parents[:]   # detection have multi-parent_id

                # The new child enters the list of trajectories
                tracks.append({
                    "id": child_id,
                    "feat": feat,
                    "missed": 0,
                    "active": True,
                })

                for tr in tracks:
                    if tr["id"] in parents:
                        tr["active"] = False

            # new root（without any parent）===
            else:
                child_id = next_track_id
                next_track_id += 1

                det_obj_ids[det_idx] = child_id
                det_parents[det_idx] = []

                tracks.append({
                    "id": child_id,
                    "feat": feat,
                    "missed": 0,
                    "active": True,
                })


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

                    # saving COCO RLE
                    mask_rle = encode_mask_to_coco_rle(m)
                else:
                    mask_area = 0
                    cx = float((x1 + x2) / 2.0)
                    cy = float((y1 + y2) / 2.0)

                    # An empty mask can also store an empty RLE, or simply None
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

                # new
                "mask_rle": mask_rle,
            }
            all_rows.append(row)
            
        # drawing
        labeled_save_path = os.path.join(key_labeled_dir, fname)
        draw_labeled_image(image, instances, det_obj_ids, labeled_save_path)

    # export Excel
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


