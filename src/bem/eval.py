from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics.utils.metrics import ap_per_class, box_iou

from .background import BackgroundExtractorTorch
from .dataset import build_coco_gt, resolve_yolo_dataset
from .model_wrappers import build_detector
from .scoring import rescore, robust_cosine_similarity


@dataclass
class EvalConfig:
    data: Path
    weights: str = "yolo11m.pt"
    device: str = "cuda:0"
    imgsz: int = 640
    mode: str = "baseline"
    conf: float = 0.001
    embedding_window: int = 20
    alpha: float = 0.6
    gamma: float = 1.0
    use_boxes: bool = True
    sliding: bool = False
    masking_mode: bool = True
    random_order: bool = False
    seed: int = 0
    augment: bool = False
    output_dir: Path = Path("runs")


def _xywh2xyxy(xywh):
    x, y, w, h = xywh
    return [x, y, x + w, y + h]


def _build_ultra_inputs_for_ap(preds: List[Dict], gt_dict: Dict, iou_thr: float = 0.50):
    from collections import defaultdict

    pred_by_img, gt_by_img = defaultdict(list), defaultdict(list)
    for pred in preds:
        pred_by_img[pred["image_id"]].append(pred)
    for ann in gt_dict["annotations"]:
        gt_by_img[ann["image_id"]].append(ann)

    tp_list, conf_list, pred_cls_list, target_cls = [], [], [], []
    for image_id, gts in gt_by_img.items():
        gt_boxes = [_xywh2xyxy(ann["bbox"]) for ann in gts]
        gt_cls = [ann["category_id"] for ann in gts]
        target_cls.extend(gt_cls)
        preds_img = sorted(pred_by_img.get(image_id, []), key=lambda item: -item["score"])
        if not preds_img:
            continue
        pred_xyxy = torch.tensor([_xywh2xyxy(pred["bbox"]) for pred in preds_img], dtype=torch.float32)
        gt_xyxy = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4), dtype=torch.float32)
        ious = box_iou(pred_xyxy, gt_xyxy) if gt_boxes else torch.zeros((len(preds_img), 0))
        used_gt = set()
        for i, pred in enumerate(preds_img):
            conf_list.append(float(pred["score"]))
            pred_cls_list.append(int(pred["category_id"]))
            matched = 0
            if gt_xyxy.shape[0] > 0:
                best = int(torch.argmax(ious[i]).item())
                if ious[i, best].item() >= iou_thr and best not in used_gt and int(pred["category_id"]) == int(gt_cls[best]):
                    matched = 1
                    used_gt.add(best)
            tp_list.append(matched)
    tp = np.asarray(tp_list, dtype=np.float32)
    if tp.ndim == 1:
        tp = tp[:, None]
    return tp, np.asarray(conf_list, dtype=np.float32), np.asarray(pred_cls_list, dtype=np.int32), np.asarray(target_cls, dtype=np.int32)


def evaluate_dataset(cfg: EvalConfig) -> Dict[str, float]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path, images_dir, labels_dir, names = resolve_yolo_dataset(cfg.data)
    filter_classes = []
    if yaml_path.name == "dataset.yaml":
        lw = cfg.weights.lower()
        filter_classes = [] if "world" in lw else ([0, 2] if "voc" in lw else [2])

    gt_dict, image_paths, id_by_path = build_coco_gt(images_dir, labels_dir, names, filter_classes)
    (cfg.output_dir / "gt.json").write_text(json.dumps(gt_dict), encoding="utf-8")
    if cfg.random_order:
        rng = random.Random(cfg.seed)
        rng.shuffle(image_paths)

    model = build_detector(cfg.weights, cfg.device, cfg.imgsz, names)
    bem = None
    if cfg.mode == "bem":
        bem = BackgroundExtractorTorch(window_size=cfg.embedding_window, keep_sliding=cfg.sliding, use_boxes=cfg.use_boxes, masking_mode=cfg.masking_mode)

    preds: List[Dict] = []
    bg_embedding = None
    cos_sim_score = 0.0
    bg_ext_index = 0
    start = time.time()
    processed = 0

    for img_path in tqdm(image_paths, desc=f"Evaluating {cfg.data.name} [{cfg.mode}]"):
        if torch.cuda.is_available() and processed % 5 == 0:
            torch.cuda.empty_cache()
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        processed += 1

        with torch.no_grad():
            result, feat = model.inference(frame, augment=cfg.augment)

        if bg_embedding is not None and bem is not None:
            cos_sim_score = robust_cosine_similarity(feat, bg_embedding)

        raw_dets = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            if names == ["person"]:
                cls_id = 0
            else:
                raw_cls = int(box.cls[0])
                cls_name = result.names[raw_cls]
                if cls_name not in names:
                    continue
                cls_id = names.index(cls_name)
            raw_dets.append([x1, y1, x2, y2, score, cls_id])

        dets, bg_boxes = [], []
        if raw_dets:
            scores = np.asarray([det[4] for det in raw_dets], dtype=float)
            scores = rescore(scores, cos_sim_score, alpha=cfg.alpha, gamma=cfg.gamma)
            for (x1, y1, x2, y2, _orig, cls_id), new_score in zip(raw_dets, scores):
                if len(names) < 2 and int(cls_id) != 0:
                    continue
                if float(new_score) < cfg.conf:
                    continue
                dets.append((x1, y1, x2, y2, float(new_score), int(cls_id)))
                bg_boxes.append([x1, y1, x2, y2])

        if bem is not None:
            bg_frame = bem.update(frame, boxes=bg_boxes, box_format="xyxy", normalized=False)
            # Preserve legacy timing: update the prototype after the window has elapsed.
            if bg_ext_index > cfg.embedding_window:
                if bg_frame is not None:
                    _, bg_embedding = model.inference(bg_frame)
                bg_ext_index = 0
            else:
                bg_ext_index += 1

        image_id = id_by_path[img_path]
        for x1, y1, x2, y2, score, cls_id in dets:
            preds.append({"image_id": int(image_id), "category_id": int(cls_id), "bbox": [float(max(0.0, x1)), float(max(0.0, y1)), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))], "score": float(score)})

    elapsed_ms = (time.time() - start) * 1000.0
    avg_latency_ms = elapsed_ms / max(processed, 1)
    pred_path = cfg.output_dir / f"pred_{cfg.mode}.json"
    pred_path.write_text(json.dumps(preds), encoding="utf-8")

    tp, confs, pred_cls, target_cls = _build_ultra_inputs_for_ap(preds, gt_dict, iou_thr=0.50)
    if tp.size > 0:
        result = ap_per_class(tp, confs, pred_cls, target_cls, plot=False, save_dir=cfg.output_dir)
        _, _, _, _, _, ap, _, p_curve, _, _, x, _ = result
        p_auc = float(np.trapz(p_curve, x, axis=1).mean()) if p_curve.size else 0.0
        mAP50 = float(ap[:, 0].mean()) if ap.ndim == 2 and ap.shape[1] else float(ap.mean()) if ap.size else 0.0
    else:
        mAP50, p_auc = 0.0, 0.0

    # Recall is still computed above to preserve the original evaluation path,
    # but it is intentionally excluded from console/file outputs.
    out = {"mAP@0.50": mAP50, "P-AUC": p_auc, "latency_ms_per_image": avg_latency_ms, "num_images": float(processed)}
    pd.DataFrame([{**asdict(cfg), **out}]).to_csv(cfg.output_dir / f"results_{cfg.mode}.csv", index=False)
    (cfg.output_dir / f"results_{cfg.mode}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out
