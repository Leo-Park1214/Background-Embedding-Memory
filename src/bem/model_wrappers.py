from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from ultralytics import RTDETR, YOLO, YOLOWorld


def pool_embedding(fm: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(fm, output_size=1)


class RTDETRWithHook:
    def __init__(self, weights: str = "rtdetr-l.pt", device: str = "cuda:0", imgsz: int = 640, hook_idx: int = 9, select_class: Optional[List[str]] = None):
        select_class = select_class or ["person"]
        self.model = RTDETR(weights)
        self.device = device
        self.imgsz = imgsz
        self._feat: Optional[torch.Tensor] = None
        self._handle = self.model.model.model[hook_idx].register_forward_hook(self._save)
        self.idx = [idx for idx, name in self.model.names.items() if name in select_class]

    def _save(self, _module: Any, _inputs: Any, output: Any) -> None:
        self._feat = pool_embedding(output) if isinstance(output, torch.Tensor) and output.dim() == 4 else output

    @torch.no_grad()
    def inference(self, img: Any, augment: bool = False) -> Tuple[Any, Optional[torch.Tensor]]:
        result = self.model(img, device=self.device, imgsz=self.imgsz, conf=0.0, verbose=False, classes=self.idx)[0]
        return result, self._feat


class YOLOWithHook:
    def __init__(self, weights: str = "yolo11m.pt", device: str = "cuda:0", imgsz: int = 640, select_class: Optional[List[str]] = None, hook_idx: Optional[int] = None):
        select_class = select_class or ["person"]
        self.model = YOLOWorld(weights) if "world" in weights.lower() else YOLO(weights)
        if "world" in weights.lower():
            self.model.set_classes(select_class)
        self.device = device
        self.imgsz = imgsz
        self._feat: Optional[torch.Tensor] = None
        if hook_idx is None:
            if "8" in weights:
                hook_idx = 9
            elif "11" in weights:
                hook_idx = 10
            else:
                hook_idx = -2
        self._handle = self.model.model.model[hook_idx].register_forward_hook(self._save)
        self.idx = [idx for idx, name in self.model.names.items() if name in select_class]
        print(f"[model] weights={weights}, device={device}, classes={select_class}->{self.idx}, hook_idx={hook_idx}")

    def _save(self, _module: Any, _inputs: Any, output: Any) -> None:
        self._feat = pool_embedding(output) if isinstance(output, torch.Tensor) and output.dim() == 4 else output

    @torch.no_grad()
    def inference(self, img: Any, augment: bool = False) -> Tuple[Any, Optional[torch.Tensor]]:
        result = self.model(img, device=self.device, imgsz=self.imgsz, conf=0.0, verbose=False, classes=self.idx, augment=augment)[0]
        return result, self._feat


def build_detector(weights: str, device: str, imgsz: int, classes: List[str]):
    lw = weights.lower()
    if "detr" in lw:
        return RTDETRWithHook(weights=weights, device=device, imgsz=imgsz, select_class=classes)
    return YOLOWithHook(weights=weights, device=device, imgsz=imgsz, select_class=classes)
