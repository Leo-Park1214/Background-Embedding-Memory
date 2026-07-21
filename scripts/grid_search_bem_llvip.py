from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional hyperparameter sweep used for ablation/tuning.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=str, default="yolo11m.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--alphas", type=str, default="0.1,0.2,0.4,0.6,0.8,0.9,1.0")
    parser.add_argument("--gammas", type=str, default="0.001,0.005,0.01,0.05,0.1,0.5,1.0")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/grid_search"))
    args = parser.parse_args()

    alphas = [float(v) for v in args.alphas.split(",") if v]
    gammas = [float(v) for v in args.gammas.split(",") if v]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for alpha, gamma in itertools.product(alphas, gammas):
        out = args.output_dir / f"a{alpha}_g{gamma}"
        cmd = [sys.executable, "run_llvip.py", "--data-root", str(args.data_root), "--weights", args.weights, "--device", args.device, "--mode", "bem", "--alpha", str(alpha), "--gamma", str(gamma), "--output-dir", str(out)]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        summary_path = out / "summary_bem.json"
        if summary_path.exists():
            records.append({"alpha": alpha, "gamma": gamma, **json.loads(summary_path.read_text(encoding="utf-8"))["mean"]})
    (args.output_dir / "grid_search_summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
