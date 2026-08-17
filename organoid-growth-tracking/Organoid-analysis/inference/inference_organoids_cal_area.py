# -*- coding: utf-8 -*-

from detectron2.structures import Instances
from copy import deepcopy
import cv2
import torch
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
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
import argparse
from detectron2.config import get_cfg
from scipy.spatial.distance import cdist
from swinb.config import add_swinb_config
import numpy as np



def compute_circularity(mask):
    """
    mask: torch.Tensor of shape [H, W] or np.ndarray
    Returns: float (circularity)
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)



def compute_diameter(mask):
    """
    Calculate the maximum, minimum, and average diameter (in pixels)
    Input: mask is either torch.Tensor or np.ndarray
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy().astype(np.uint8)
    else:
        mask_np = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, 0.0, 0.0

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 2:
        return 0.0, 0.0, 0.0

    points = cnt.squeeze()  # shape: (N, 2)


    dists = cdist(points, points)
    max_d = np.max(dists)
    min_d = np.min(dists[np.nonzero(dists)])
    mean_d = np.mean(dists)

    return max_d, min_d, mean_d




def is_touching_border(mask, border_width=1):
    """
    Determining if the mask is touching the edge of the image (default width is 1 pixel)
    mask: np.ndarray of shape (H, W), 0/1
    """
    h, w = mask.shape
    top = mask[:border_width, :].any()
    bottom = mask[-border_width:, :].any()
    left = mask[:, :border_width].any()
    right = mask[:, -border_width:].any()
    return top or bottom or left or right



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

    def run_on_image(self, image,confidence_threshold = 0.7, min_size =0, min_circularity = 0.0):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
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
                high_confidence_mask = instances.scores < confidence_threshold
                instances = instances[high_confidence_mask]

                # Mask IoU-based suppression
                def compute_mask_iou(mask1, mask2):
                    intersection = (mask1 & mask2).sum().item()
                    union = (mask1 | mask2).sum().item()
                    return intersection / union if union > 0 else 0

                masks = instances.pred_masks.cpu()  # shape: [N, H, W]
                scores = instances.scores.cpu().numpy()
                keep = []

                for i in range(len(masks)):
                    if i in keep:
                        continue
                    discard = False
                    for j in keep:
                        iou = compute_mask_iou(masks[i], masks[j])
                        if iou > 0.01:
                            if scores[i] <= scores[j]:
                                discard = True
                                break
                    if not discard:
                        keep.append(i)

                # Apply Mask-IoU-based keep
                instances = instances[keep]

                # min_size filtering based on mask area
                mask_areas = instances.pred_masks.sum(dim=(1, 2)).cpu().numpy()
                size_mask = mask_areas >= min_size
                instances = instances[size_mask]
                
                circularity_scores = []
                for mask in instances.pred_masks:
                    circ = compute_circularity(mask)
                    circularity_scores.append(circ)

                circularity_scores = np.array(circularity_scores)
                circularity_mask = circularity_scores >= min_circularity
                instances = instances[circularity_mask]
                
                # keep_non_border = []
                # masks_np = instances.pred_masks.cpu().numpy()
                # for i, m in enumerate(masks_np):
                #     if not is_touching_border(m):
                #         keep_non_border.append(i)
                # instances = instances[keep_non_border]
                
                diameters = []
                circularity_mask = instances.pred_masks.cpu().numpy()
                for i, m in enumerate(circularity_mask):
                    # if not is_touching_border(m):
                    d_max, d_min, d_mean = compute_diameter(m)
                    diameters.append((d_max))
                    # keep_non_border.append(i)
                        
                print(diameters)

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




import csv




cfg = get_cfg()
add_swinb_config(cfg)  # load swinb


cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO/config.yaml")
cfg.MODEL.WEIGHTS = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO/model_0003049.pth"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg.freeze()

# Initialize the tool
demo = VisualizationDemo(cfg, instance_mode=ColorMode.IMAGE)

# Input and output directories
input_folder = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/tracing/test"
output_folder = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/tracing/test/output"


# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)


# csv path
count_csv_path = os.path.join(output_folder, "instance_counts_C.csv")


# write into csv
with open(count_csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "ID", "Area", "Circularity", "d_max", "d_min", "d_mean"])

    
    
    # main
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".png", ".tif", ".jpeg", ".tiff")) and not filename.startswith('.'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = cv2.imread(input_path)
            predictions, vis_output, instances = demo.run_on_image(image)
        

            # instances = predictions["instances"].to("cpu")
            # high_confidence_mask = instances.scores > 0.0
            # instances = instances[high_confidence_mask]
            
            for i, m in enumerate(instances.pred_masks):
                if not is_touching_border(m):
                    d_max, d_min, d_mean = compute_diameter(m)
                    circularity = compute_circularity(m)
                    area = m.sum().item()  # torch.Tensor -> int
                    writer.writerow([filename, i, area, circularity, d_max, d_min, d_mean])
            

            #  the number of mask
            instance_count = len(instances)
            # writer.writerow(['C_'+ filename, instance_count])

            # only drawing mask
            only_mask_instances = Instances(image.shape[:2])
            only_mask_instances.pred_masks = deepcopy(instances.pred_masks)

            visualizer = Visualizer(image, metadata=demo.metadata, instance_mode=ColorMode.IMAGE)
            vis_output = visualizer.draw_instance_predictions(only_mask_instances)

            if vis_output:
                vis_image = vis_output.get_image()[:, :, ::-1]
                cv2.imwrite(output_path, vis_image)
                print(f"{filename} - {instance_count} instances - Saved to {output_path}")
            else:
                print(f"{filename} - no detections")
    
