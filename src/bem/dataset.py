from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2


def load_yaml_minimal(yaml_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def parse_names(yaml_path: Path) -> List[str]:
    text = yaml_path.read_text(encoding="utf-8")
    if "names:" in text and "person" in text and "truck" not in text:
        return ["person"]
    if yaml_path.name == "dataset.yaml":
        return ["truck", "car", "van", "bus"]
    return ["person"]


def yolo_label_to_boxes(label_path: Path, img_wh: Tuple[int, int]) -> List[Tuple[int, float, float, float, float]]:
    width, height = img_wh
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, cx, cy, bw, bh = line.split()[:5]
        cx, cy, bw, bh = float(cx) * width, float(cy) * height, float(bw) * width, float(bh) * height
        boxes.append((int(cls), cx - bw / 2, cy - bh / 2, bw, bh))
    return boxes


def resolve_yolo_dataset(root: Path) -> Tuple[Path, Path, Path, List[str]]:
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No data.yaml or dataset.yaml found under {root}")
    meta = load_yaml_minimal(yaml_path)
    base = Path(meta.get("path", root.as_posix()))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    if "data" not in str(base):
        base = root
    images = base / meta.get("test", meta.get("val", "images/val"))
    labels = base / str(meta.get("test", meta.get("val", "labels/val"))).replace("images", "labels")
    if not images.exists():
        images = base / "images"
    if not labels.exists():
        labels = base / "labels"
    return yaml_path, images, labels, parse_names(yaml_path)


def build_coco_gt(images_dir: Path, labels_dir: Path, category_names: List[str], filter_classes: List[int]):
    images, annotations = [], []
    categories = [{"id": i, "name": name} for i, name in enumerate(category_names)]
    image_id, ann_id = 1, 1
    valid_paths, id_by_path = [], {}
    img_paths = sorted([p for p in images_dir.glob("*.jpg")])
    for path in img_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        images.append({"id": image_id, "file_name": path.name, "width": w, "height": h})
        valid_paths.append(path)
        id_by_path[path] = image_id
        for cls, x, y, bw, bh in yolo_label_to_boxes(labels_dir / f"{path.stem}.txt", (w, h)):
            if cls in filter_classes:
                continue
            x, y = max(0.0, x), max(0.0, y)
            if x + bw > w: bw = max(0.0, w - x)
            if y + bh > h: bh = max(0.0, h - y)
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": int(cls), "bbox": [float(x), float(y), float(bw), float(bh)], "area": float(bw * bh), "iscrowd": 0})
            ann_id += 1
        image_id += 1
    gt = {"info": {"description": "LLVIP-val converted to COCO", "version": "1.0"}, "licenses": [], "images": images, "annotations": annotations, "categories": categories}
    return gt, valid_paths, id_by_path
