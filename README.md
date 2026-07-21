# Background Embedding Memory

Background Embedding Memory (BEM) is a training-free module for reducing false-positive detections in fixed-camera scenes.

## Installation

Clone the repository and run all commands from the repository root directory.

### Exact reproduction environment

The following environment was used for the experiments reported in the paper:

```bash
git clone https://github.com/Leo-Park1214/Background-Embedding-Memory.git
cd Background-Embedding-Memory

conda create -n bem python=3.11 -y
conda activate bem

pip install -r requirements.txt
```

Use `requirements.txt` to reproduce the experimental environment and reported results as closely as possible.

### Faster installation environment

The following alternative environment provides a faster and more lightweight installation:

```bash
git clone https://github.com/Leo-Park1214/Background-Embedding-Memory.git
cd Background-Embedding-Memory

conda create -n bem python=3.10 -y
conda activate bem

pip install -r requirements2.txt
```

`requirements2.txt` is provided for convenience. It was not the primary environment used to produce the reported results and may contain less strictly pinned dependency versions.

## Repository structure

After downloading the required data and model weights, the expected directory structure is:

```text
Background-Embedding-Memory/
├── data/
│   └── LLVIP/
│       ├── visible/
│       └── infrared/
├── weights/
│   ├── yolo11m.pt
│   ├── yolov8s.pt
│   ├── rtdetr-l.pt
│   └── ...
├── scripts/
│   ├── download_llvip.py
│   └── grid_search_bem_llvip.py
├── src/
│   └── bem/
├── run_llvip.py
├── requirements.txt
├── requirements2.txt
├── LICENSE
└── README.md
```

The exact contents under `data/LLVIP/` are prepared automatically by `scripts/download_llvip.py`.

Experiment outputs are stored under the directory specified by `--output-dir`.

## Model weights

Official COCO-pretrained model weights can be downloaded from the [Ultralytics model documentation](https://docs.ultralytics.com/models/).

The COCO-to-VOC models were fine-tuned on the Pascal VOC dataset from COCO-pretrained checkpoints using the default Ultralytics training configuration:

```python
model.train(data="VOC.yaml")
```

The model weights required to reproduce the reported experiments are publicly available at:

* COCO-pretrained weights: [Ultralytics model documentation](https://docs.ultralytics.com/models/)
* COCO-to-VOC fine-tuned weights: [PUBLIC_MODEL_WEIGHTS_URL]

The weight links must be accessible without requesting permission or signing in.

Place downloaded weights in the `weights/` directory or provide an absolute or relative path through `--weights`.

For example:

```bash
mkdir -p weights
```

```bash
python run_llvip.py \
  --data-root data/LLVIP \
  --weights weights/yolo11m.pt \
  --device cuda:0 \
  --mode baseline \
  --output-dir runs/baseline
```

## Download LLVIP

The LLVIP data used in the experiments can be downloaded and prepared using:

```bash
python scripts/download_llvip.py \
  --source hf \
  --data-root data/LLVIP
```

The download source must be publicly accessible without requesting permission.

To rebuild previously prepared data, run:

```bash
python scripts/download_llvip.py \
  --source hf \
  --data-root data/LLVIP \
  --overwrite
```

After preparation, verify that both modalities are available:

```text
data/LLVIP/
├── visible/
└── infrared/
```

The LLVIP dataset remains subject to its original license and terms of use.

## Baseline

Run the baseline evaluation from the repository root:

```bash
python run_llvip.py \
  --data-root data/LLVIP \
  --weights weights/yolo11m.pt \
  --device cuda:0 \
  --mode baseline \
  --output-dir runs/baseline
```

The baseline results are saved under:

```text
runs/baseline/
└── integrated_baseline.json
```

## BEM

Run the BEM evaluation from the repository root:

```bash
python run_llvip.py \
  --data-root data/LLVIP \
  --weights weights/yolo11m.pt \
  --device cuda:0 \
  --mode bem \
  --embedding-window 25 \
  --alpha 0.6 \
  --gamma 1.0 \
  --output-dir runs/bem
```

The experiment evaluates visible and infrared data from scenes 19–24.

The BEM results are saved under:

```text
runs/bem/
└── integrated_bem.json
```

The appropriate `alpha` and `gamma` values depend on the selected model. The values reported in Table 1 are listed in the hyperparameter section below.

## Hyperparameter grid search

The grid-search script is included at:

```text
scripts/grid_search_bem_llvip.py
```

Run the grid search from the repository root:

```bash
python scripts/grid_search_bem_llvip.py \
  --data data/LLVIP \
  --weights weights/yolo11m.pt \
  --device cuda:0 \
  --alphas 0.1,0.2,0.4,0.6,0.8,0.9,1.0 \
  --gammas 0.001,0.005,0.01,0.05,0.1,0.5,1.0 \
  --output-dir runs/grid_search
```

Every combination of the supplied `alpha` and `gamma` candidates is evaluated.

The expected output structure is:

```text
runs/grid_search/
├── a0.1_g0.001/
├── a0.1_g0.005/
├── ...
└── tuning_summary.json
```

### Best hyperparameter combinations

| Model                   | Alpha | Gamma | Embedding window |
| ----------------------- | ----: | ----: | ---------------: |
| YOLOv11m (COCO)         |   0.2 | 0.005 |               25 |
| YOLOv8s (COCO)          |   0.2 | 0.005 |               25 |
| RT-DETR-L (COCO)        |   0.5 |   0.1 |               25 |
| YOLOv8l-World-v2        |   0.8 |  0.05 |               25 |
| YOLOv8s-World-v2        |   0.7 |  0.05 |               25 |
| YOLOv11m (COCO to VOC)  |   0.7 | 0.005 |               25 |
| YOLOv8s (COCO to VOC)   |   0.7 | 0.005 |               25 |
| RT-DETR-L (COCO to VOC) |   0.6 |  0.05 |               25 |

These hyperparameter combinations correspond to the settings used for the Table 1 results.

To evaluate a specific model using its selected hyperparameters, replace the `--weights`, `--alpha`, and `--gamma` arguments accordingly.

Each reported experiment was repeated five times with different random seeds.

## Main outputs

Experiments save scene-level results and modality-integrated results.

The main output files are:

```text
integrated_baseline.json
integrated_bem.json
```

The output directory is controlled by the `--output-dir` argument.

For example:

```text
runs/
├── baseline/
│   └── integrated_baseline.json
├── bem/
│   └── integrated_bem.json
└── grid_search/
    └── tuning_summary.json
```

## Archived reproducible version

The repository version evaluated for the ICPR 2026 Reproducible Research Label is permanently archived at:

* GitHub release: [GITHUB_RELEASE_URL]
* DOI or SWHID: [DOI_OR_SWHID]

The archived release contains the exact source code, documentation, and configuration files associated with the reproducibility submission.

## License

The source code developed for this repository is released under the [MIT License](LICENSE).

Third-party datasets, pretrained model weights, libraries, and other external artefacts remain subject to their respective original licenses and terms of use. In particular, the MIT License for this repository does not override the licenses of LLVIP, Ultralytics, or any pretrained model weights.
