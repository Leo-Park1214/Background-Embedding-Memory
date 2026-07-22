from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
from torch.cuda.amp import autocast
ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class BackgroundExtractorTorch:
    """
    Efficient 12-frame background extractor implemented in **PyTorch** (GPU-friendly).

    - Dataset-agnostic: works with any video frames you provide.
    - Accepts **object boxes** per frame and excludes those regions from the background model.
    - Call `update(img, boxes=None, box_format='xyxy', normalized=False)` with numpy/torch
      images (H×W or H×W×3). On the 12th push, returns the median background (numpy uint8, H×W×3).
      Until then returns `None`.

    Pipeline
    ------------------------------------------
    1) Temporal **median** (robust seed) over the 12-frame stack
    2) Foreground votes: |I_gray - median_gray| > `diff_thresh`
    3) Optional **detection boxes** -> foreground prior mask (OR with motion mask)
    4) Morphological **open/close** on the mask (pooling-based, fast)
    5) Masked pixels remain as-is in the median image (no inpainting)

    Notes
    -----
    • Accepts uint8 [0,255] or float images ([0,1] or [0,255]).
    • Accepts grayscale or color; output is 3 channels (RGB) uint8.
    • Set `device="cuda"` (default if available) for acceleration.
    """

    window_size: int = 12
    diff_thresh: float = 30.0 / 255.0
    votes_min: int = 2
    morph_open: int = 3
    morph_close: int = 5
    keep_sliding: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_boxes: bool = True
    box_pad: int = 4
    masking_mode:bool = False
    _buf: List[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _box_masks: List[Optional[torch.Tensor]] = field(default_factory=list, init=False, repr=False)
    
    def reset(self) -> None:
        self._buf.clear()
        self._box_masks.clear()

    @torch.no_grad()
    def update(
        self,
        img: ArrayLike,
        boxes: Optional[ArrayLike] = None,
        box_format: str = "xyxy",
        normalized: bool = False,
    ) -> Optional[np.ndarray]:
        frame = self._to_float_chw(img).to(self.device, non_blocking=True)
        self._buf.append(frame)
        if boxes is not None and self.use_boxes:
            H, W = frame.shape[-2], frame.shape[-1]
            box_mask = self._boxes_to_mask(boxes, H, W, box_format=box_format, normalized=normalized)
            self._box_masks.append(box_mask)
        else:
            self._box_masks.append(None)

        if len(self._buf) < self.window_size:
            return None
        if self.masking_mode:
            bg_tensor = self._compute_background_with_mask(self._buf, self._box_masks)
        else:
            bg_tensor = self._compute_background(self._buf, self._box_masks)

        if self.keep_sliding:
            self._buf = self._buf[1:]
            self._box_masks = self._box_masks[1:]
        else:
            self._buf.clear()
            self._box_masks.clear()
        return self._to_uint8_hwc(bg_tensor)

    @torch.no_grad()
    def _compute_background(self, frames: List[torch.Tensor], box_masks: List[Optional[torch.Tensor]]) -> torch.Tensor:
        stack = torch.stack(frames, dim=0)
        T, C, H, W = stack.shape

        median = stack.median(dim=0).values

        gray_w = torch.tensor([0.299, 0.587, 0.114], device=self.device, dtype=stack.dtype)
        if C == 1:
            gray_stack = stack[:, 0]
            gray_median = median[0]
        else:
            gray_stack = (stack * gray_w.view(1, 3, 1, 1)).sum(dim=1)
            gray_median = (median * gray_w.view(3, 1, 1)).sum(dim=0)
        diff = (gray_stack - gray_median).abs()
        mask_stack = diff > self.diff_thresh
        votes = mask_stack.sum(dim=0)
        fg_mask = votes >= self.votes_min

        if any(m is not None for m in box_masks):
            box_union = torch.zeros_like(fg_mask, dtype=torch.bool)
            for m in box_masks:
                if m is not None:
                    box_union |= m
            fg_mask |= box_union

        fg = fg_mask.float().unsqueeze(0).unsqueeze(0)
        if self.morph_open > 0:
            fg = self._morph_open(fg, self.morph_open)
        if self.morph_close > 0:
            fg = self._morph_close(fg, self.morph_close)
        fg_mask = fg[0, 0] > 0.5

        # No inpainting: just keep median image
        return median
    @torch.no_grad()
    def _compute_background_with_mask(self, frames: List[torch.Tensor], box_masks: List[Optional[torch.Tensor]]) -> torch.Tensor:
        """
        B = (Σ_t I_t ⊙ M_t) / (Σ_t M_t)
        여기서 M_t는 각 프레임의 '배경 포함(mask=1)' 영역.
        기존 구현은 (T,C,H,W) 전체 스택을 만들었지만,
        메모리 피크를 줄이기 위해 프레임별 누적합으로 계산한다. 
        """
        if len(frames) > 0:
            # 기본 크기/타입
            C, H, W = frames[0].shape
            device = frames[0].device
            dtype  = frames[0].dtype
            # 누적 버퍼
            numer = torch.zeros((C, H, W), device=device, dtype=dtype)  # Σ(I_t * M_t)
            denom = torch.zeros((1, H, W), device=device, dtype=dtype)  # Σ(M_t)
            plain_sum = torch.zeros((C, H, W), device=device, dtype=dtype)  # Σ I_t (fallback용)
            T = 0

            for f, m in zip(frames, box_masks):
                # 안전장치: 모두 동일 디바이스/형식 보장
                if f.device != device:
                    f = f.to(device)
                if f.dtype != dtype:
                    f = f.to(dtype)

                # 전체 평균용 누적(구멍 채우기 fallback)
                plain_sum += f
                T += 1

                if m is None:
                    # 전 픽셀 배경으로 사용
                    ones = torch.ones((1, H, W), device=device, dtype=dtype)
                    numer += f
                    denom += ones
                else:
                    # 객체(True) 제외 → 배경은 ~m
                    if m.device != device:
                        m = m.to(device)
                    bgm = (~m).to(dtype)           # (H, W)
                    bgm_bc = bgm.unsqueeze(0)      # (1, H, W)
                    numer += f * bgm_bc            # (C, H, W)
                    denom += bgm_bc                # (1, H, W)

            # 평균 계산
            zero_mask = denom == 0                # (1, H, W)
            denom_safe = denom.masked_fill(zero_mask, 1.0)
            bg = numer / denom_safe              # (C, H, W)

            # 분모=0인 위치(모든 프레임이 박스로 가려진 픽셀)는 전체 평균으로 보정
            if zero_mask.any():
                plain_mean = plain_sum / max(T, 1)
                zm = zero_mask.expand(C, H, W)   # (C,H,W)
                bg = torch.where(zm, plain_mean, bg)

        return bg
    def _boxes_to_mask(
        self,
        boxes: ArrayLike,
        H: int,
        W: int,
        box_format: str = "xyxy",
        normalized: bool = False,
    ) -> torch.Tensor:
        if isinstance(boxes, torch.Tensor):
            b = boxes.detach().cpu().numpy()
        else:
            b = np.asarray(boxes)
        if b.size == 0:
            return torch.zeros((H, W), dtype=torch.bool, device=self.device)
        if b.ndim == 1:
            b = b[None, :]

        mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        for x1, y1, x2, y2 in self._iter_boxes_xyxy(b, W, H, box_format, normalized):
            x1 = max(0, int(np.floor(x1)) - self.box_pad)
            y1 = max(0, int(np.floor(y1)) - self.box_pad)
            x2 = min(W - 1, int(np.ceil(x2)) + self.box_pad)
            y2 = min(H - 1, int(np.ceil(y2)) + self.box_pad)
            if x2 > x1 and y2 > y1:
                mask[y1:y2+1, x1:x2+1] = True
        return mask

    def _iter_boxes_xyxy(
        self,
        boxes_np: np.ndarray,
        W: int,
        H: int,
        box_format: str,
        normalized: bool,
    ):
        if box_format not in {"xyxy", "xywh"}:
            raise ValueError("box_format must be 'xyxy' or 'xywh'")
        for b in boxes_np:
            if box_format == "xyxy":
                x1, y1, x2, y2 = b[:4]
            else:
                x, y, w, h = b[:4]
                x1, y1, x2, y2 = x, y, x + w, y + h
            if normalized:
                x1, x2 = x1 * W, x2 * W
                y1, y2 = y1 * H, y2 * H
            yield x1, y1, x2, y2

    def _morph_dilate(self, x: torch.Tensor, k: int) -> torch.Tensor:
        pad = k // 2
        return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)

    def _morph_erode(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return 1.0 - self._morph_dilate(1.0 - x, k)

    def _morph_open(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return self._morph_dilate(self._morph_erode(x, k), k)

    def _morph_close(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return self._morph_erode(self._morph_dilate(x, k), k)

    def _to_float_chw(self, x: ArrayLike) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 2:
                arr = arr[..., None]
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = arr.astype(np.float32)
                if arr.max() > 1.5:
                    arr = arr / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            return torch.from_numpy(arr)
        elif isinstance(x, torch.Tensor):
            t = x
            if t.ndim == 2:
                t = t.unsqueeze(-1)
            if t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                t = t.to(torch.float32) / 255.0
            else:
                t = t.to(torch.float32)
                if t.max() > 1.5:
                    t = t / 255.0
            if t.shape[-1] in (1, 3):
                t = t.permute(2, 0, 1)
            return t
        else:
            raise TypeError("img must be np.ndarray or torch.Tensor")

    def _to_uint8_hwc(self, chw: torch.Tensor) -> np.ndarray:
        chw = chw.clamp(0, 1)
        if chw.shape[0] == 1:
            chw = chw.repeat(3, 1, 1)
        hwc = chw.permute(1, 2, 0).contiguous().cpu().numpy()
        return (hwc * 255.0 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from PIL import Image
    import random

    def load_rgb(path: Path, i) -> np.ndarray:
        os.makedirs("ext_save", exist_ok = True)
        save_path = os.path.join("ext_save", str(i) + ".jpg")
        img = Image.open(path).convert("RGB")
        img.save(save_path)
        return np.array(img)

    def load_yolo_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
        """Read YOLO txt (class cx cy w h in [0,1]) -> (N,4) xyxy absolute pixels."""
        if not label_path.exists():
            return np.zeros((0, 4), dtype=np.float32)
        boxes = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = map(float, parts[:5])
                x1 = (cx - w / 2.0) * img_w
                y1 = (cy - h / 2.0) * img_h
                x2 = (cx + w / 2.0) * img_w
                y2 = (cy + h / 2.0) * img_h
                boxes.append([x1, y1, x2, y2])
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray(boxes, dtype=np.float32)

    parser = argparse.ArgumentParser(description="Sample 12 images from a YOLO dataset and extract background.")
    parser.add_argument("--root", type=str, default="./dataset", help="Root of YOLO dataset (contains images/ and labels/)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Which split to read")
    parser.add_argument("--exts", type=str, nargs="*", default=[".jpg", ".jpeg", ".png", ".bmp"], help="Image extensions to include")
    parser.add_argument("--count", type=int, default=20, help="Number of frames to sample (uses window size)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling order")
    parser.add_argument("--save", type=str, default="background.png", help="Output path for the background image")
    parser.add_argument("--use-boxes", action="store_true", help="Use YOLO labels as object boxes (recommended)")
    parser.add_argument("--sliding", action="store_true", help="Use sliding window (continues after first result)")
    parser.add_argument("--masking", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    # Gather image files
    img_files = [p for p in img_dir.rglob("*") if p.suffix.lower() in set(e.lower() for e in args.exts)]
    if len(img_files) == 0:
        raise RuntimeError(f"No images with extensions {args.exts} found under {img_dir}")

    random.seed(args.seed)
    img_files.sort()  # deterministic base order
    chosen = img_files[:args.count] if args.count > 0 else img_files[:12]
    if len(chosen) < 12:
        raise RuntimeError("Need at least 12 images to build the background model.")

    extractor = BackgroundExtractorTorch(keep_sliding=args.sliding, use_boxes=args.use_boxes)
    extractor.masking_mode = args.masking
    background = None
    for i, img_path in enumerate(chosen):
        img_np = load_rgb(img_path, i)
        H, W = img_np.shape[0], img_np.shape[1]
        if args.use_boxes:
            label_path = (lbl_dir / img_path.relative_to(img_dir)).with_suffix(".txt")
            boxes_xyxy = load_yolo_boxes(label_path, W, H)
        else:
            boxes_xyxy = None

        background = extractor.update(img_np, boxes=boxes_xyxy, box_format="xyxy", normalized=False)

    if background is None:
        raise RuntimeError("Background was not produced; check that exactly/at least 12 frames were fed.")

    # Save result
    if not os.path.exists("ext_save"):
        os.makedirs("ext_save", exist_ok= True)
    idx_folder = int(len(os.listdir("ext_save")))
    idx_ext = os.path.join("ext_save", f"extract_run_{idx_folder}")
    os.makedirs(idx_ext, exist_ok= True)
    save_path = os.path.join(idx_ext, args.save)
    Image.fromarray(background).save(save_path)
    print(f"Saved background to {args.save} | shape={background.shape} dtype={background.dtype}")