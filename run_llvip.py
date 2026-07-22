from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from src.bem.eval import EvalConfig, evaluate_dataset


REPORTED_METRICS = ("mAP@0.50", "P-AUC", "latency_ms_per_image")


def _parse_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _as_yolo_root(path: Path) -> Path:
    """Return a dataset root containing data.yaml/dataset.yaml when possible."""
    if (path / "data.yaml").exists() or (path / "dataset.yaml").exists():
        return path
    if (path / "yolo" / "data.yaml").exists() or (path / "yolo" / "dataset.yaml").exists():
        return path / "yolo"
    return path


def _resolve_scene_root(data_root: Path, modality: str, scene: str) -> Path:
    candidates = [
        data_root / modality / scene,
        data_root / modality / f"Scene{scene}",
        data_root / scene / modality,
        data_root / f"Scene{scene}" / modality,
    ]
    for candidate in candidates:
        root = _as_yolo_root(candidate)
        if (root / "data.yaml").exists() or (root / "dataset.yaml").exists():
            return root
    tried = "\n  - ".join(str(_as_yolo_root(p)) for p in candidates)
    raise FileNotFoundError(
        f"Could not find the {modality} dataset for scene {scene}. Tried:\n  - {tried}"
    )


def _resolve_modality_roots(data_root: Path, modality: str, scenes: Iterable[str]) -> List[Path]:
    scene_list = list(scenes)
    if scene_list:
        return [_resolve_scene_root(data_root, modality, scene) for scene in scene_list]

    root = _as_yolo_root(data_root / modality)
    if not ((root / "data.yaml").exists() or (root / "dataset.yaml").exists()):
        raise FileNotFoundError(
            f"Could not find {modality} YOLO data under {data_root / modality}. "
            f"Expected {data_root / modality / 'yolo' / 'data.yaml'} or a dataset.yaml file."
        )
    return [root]


def _numeric_mean(rows: List[Dict[str, float]]) -> Dict[str, float]:
    frame = pd.DataFrame(rows)
    return {
        metric: float(frame[metric].mean())
        for metric in REPORTED_METRICS
        if metric in frame.columns
    }


def _evaluate_modality(
    modality: str,
    roots: List[Path],
    args: argparse.Namespace,
) -> Dict:
    per_scene: List[Dict] = []
    for index, root in enumerate(roots):
        scene_name = root.parent.name if root.name == "yolo" else root.name
        if len(roots) == 1 and not args.scenes:
            scene_name = modality

        cfg = EvalConfig(
            data=root,
            weights=args.weights,
            device=args.device,
            imgsz=args.imgsz,
            mode=args.mode,
            conf=args.conf,
            embedding_window=args.embedding_window,
            alpha=args.alpha,
            gamma=args.gamma,
            use_boxes=True,
            sliding=True,
            masking_mode=True,
            random_order=args.random_order,
            seed=args.seed,
            augment=args.augment,
            output_dir=args.output_dir / modality / scene_name / args.mode,
        )
        result = evaluate_dataset(cfg)
        per_scene.append({"scene": scene_name, **result})

    return {
        "per_scene": per_scene,
        "mean": _numeric_mean(per_scene),
        "num_images": int(sum(row.get("num_images", 0) for row in per_scene)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run visible and infrared LLVIP evaluations, average scene-level metrics "
            "within each modality, and report the integrated two-modality result."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing visible/ and infrared/.")
    parser.add_argument("--weights", type=str, default="yolo11m.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mode", choices=["baseline", "bem"], default="bem")
    parser.add_argument("--conf", type=float, default=0.0)
    parser.add_argument("--embedding-window", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--random-order", action="store_true")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--scenes",
        type=str,
        default="19,20,21,22,23,24",
        help=(
            "Optional comma-separated scene/background ids, e.g. 19,20,21,22,23,24. "
            "Each scene is evaluated independently before averaging."
        ),
    )
    parser.add_argument("--modalities", type=str, default="visible,infrared")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/llvip_integrated"))
    args = parser.parse_args()
    print(args)
    scenes = _parse_csv(args.scenes)
    modalities = _parse_csv(args.modalities)
    if not modalities:
        raise ValueError("At least one modality must be supplied through --modalities.")

    modality_results: Dict[str, Dict] = {}
    for modality in modalities:
        roots = _resolve_modality_roots(args.data_root, modality, scenes)
        modality_results[modality] = _evaluate_modality(modality, roots, args)

    modality_means = [result["mean"] for result in modality_results.values()]
    total = {
        metric: float(sum(mean[metric] for mean in modality_means if metric in mean) / len(modality_means))
        for metric in REPORTED_METRICS
        if all(metric in mean for mean in modality_means)
    }

    integrated = {
        "mode": args.mode,
        "aggregation": (
            "Each scene/background is evaluated independently. Scene metrics are averaged "
            "within each modality, then modality means are averaged with equal weight."
        ),
        "modalities": modality_results,
        "total": total,
        "num_images": int(sum(result["num_images"] for result in modality_results.values())),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"integrated_{args.mode}.json"
    json_path.write_text(json.dumps(integrated, indent=2), encoding="utf-8")

    summary_rows = []
    for modality, result in modality_results.items():
        summary_rows.append({"scope": modality, **result["mean"], "num_images": result["num_images"]})
    summary_rows.append({"scope": "TOTAL", **total, "num_images": integrated["num_images"]})
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / f"integrated_{args.mode}.csv", index=False
    )

    print("\n=== Integrated summary ===")
    for row in summary_rows:
        print(json.dumps(row, indent=2))
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
