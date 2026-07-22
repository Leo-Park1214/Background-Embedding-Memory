# Background Embedding Memory

Background Embedding Memory (BEM) is a training-free module for reducing false-positive detections in fixed-camera scenes.

## Installation

Clone the repository and run all commands below from the repository root directory.

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

The following alternative environment provides a faster installation:

```bash
git clone https://github.com/Leo-Park1214/Background-Embedding-Memory.git
cd Background-Embedding-Memory

conda create -n bem python=3.10 -y
conda activate bem

pip install -r requirements2.txt
```

`requirements2.txt` is provided for convenience and faster setup. It was not the primary environment used to produce the reported results and may contain less strictly pinned dependency versions.

## Repository structure

The repository has the following structure:

```text
Background-Embedding-Memory/
├── data/
│   └── LLVIP/
│       ├── visible/
│       │   ├── 19/
│       │   │   └── yolo/
│       │   │       ├── data.yaml
│       │   │       ├── images/
│       │   │       │   ├── train/
│       │   │       │   └── val/
│       │   │       └── labels/
│       │   │           ├── train/
│       │   │           └── val/
│       │   ├── 20/
│       │   ├── 21/
│       │   ├── 22/
│       │   ├── 23/
│       │   └── 24/
│       └── infrared/
│           ├── 19/
│           ├── 20/
│           ├── 21/
│           ├── 22/
│           ├── 23/
│           └── 24/
├── weights/
│   ├── yolo11m.pt
│   ├── yolov8s.pt
│   ├── rtdetr-l.pt
│   └── ...
├── scripts/
│   └── download_llvip.py
├── src/
│   └── bem/
│       ├── __init__.py
│       ├── background.py
│       ├── dataset.py
│       ├── eval.py
│       ├── model_wrappers.py
│       └── scoring.py
├── grid_search_bem_llvip.py
├── run_llvip.py
├── pyproject.toml
├── requirements.txt
├── requirements2.txt
├── LICENSE
└── README.md
```

The LLVIP scene folders and their YOLO-format files are generated automatically by `scripts/download_llvip.py`.

Experiment outputs are stored under the directory supplied through `--output-dir`.

## Model weights

Official COCO-pretrained model weights can be downloaded from the [Ultralytics model documentation](https://docs.ultralytics.com/models/).

The COCO-to-VOC models were fine-tuned on the Pascal VOC dataset from COCO-pretrained checkpoints using the default Ultralytics training configuration:

```python
model.train(data="VOC.yaml")
```

The model weights used for the reported experiments are available from:

- COCO-pretrained weights: [Ultralytics model documentation](https://docs.ultralytics.com/models/)
- COCO-to-VOC fine-tuned weights: [Google Drive](https://drive.google.com/drive/folders/1HIBy5mrBvQDF7GKeTFqo4WOpSacVQcMK?usp=sharing)

The fine-tuned-weight link must be accessible without requesting permission or signing in.

Place downloaded weights in the `weights/` directory or provide an absolute or relative path through `--weights`.

For example:

```bash
mkdir -p weights
```

## Download and prepare LLVIP

Download LLVIP and prepare visible and infrared YOLO-format datasets for scenes 19–24:

```bash
python scripts/download_llvip.py \
  --source hf \
  --data-root data/LLVIP
```

To rebuild already prepared data:

```bash
python scripts/download_llvip.py \
  --source hf \
  --data-root data/LLVIP \
  --overwrite
```

The prepared data should contain the following scene-level roots:

```text
data/LLVIP/
├── visible/
│   ├── 19/yolo/data.yaml
│   ├── 20/yolo/data.yaml
│   ├── 21/yolo/data.yaml
│   ├── 22/yolo/data.yaml
│   ├── 23/yolo/data.yaml
│   └── 24/yolo/data.yaml
└── infrared/
    ├── 19/yolo/data.yaml
    ├── 20/yolo/data.yaml
    ├── 21/yolo/data.yaml
    ├── 22/yolo/data.yaml
    ├── 23/yolo/data.yaml
    └── 24/yolo/data.yaml
```

The LLVIP dataset remains subject to its original license and terms of use.

## Baseline evaluation

Run the baseline evaluation from the repository root:

```bash
python run_llvip.py \
  --data-root data/LLVIP \
  --weights weights/yolo11m.pt \
  --device cuda:0 \
  --mode baseline \
  --embedding-window 25 \
  --output-dir runs/baseline
```

By default, the command evaluates the visible and infrared modalities for scenes 19–24.

The main baseline outputs are:

```text
runs/baseline/
├── integrated_baseline.json
├── integrated_baseline.csv
├── visible/
└── infrared/
```

## BEM evaluation

Run BEM from the repository root:

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

The main BEM outputs are:

```text
runs/bem/
├── integrated_bem.json
├── integrated_bem.csv
├── visible/
└── infrared/
```

Each scene is evaluated independently. Scene-level metrics are first averaged within each modality, and the visible and infrared modality means are then averaged with equal weight. The integrated result is stored under the `total` field of `integrated_baseline.json` or `integrated_bem.json`.

### Optional evaluation arguments

The integrated runner supports the following optional arguments:

```text
--scenes 19,20,21,22,23,24
--modalities visible,infrared
--random-order
--seed 20
--augment
--imgsz 640
--conf 0.0
```

For example:

```bash
python run_llvip.py \
  --data-root data/LLVIP \
  --weights weights/yolo11m.pt \
  --device cuda:0 \
  --mode bem \
  --embedding-window 25 \
  --alpha 0.2 \
  --gamma 0.005 \
  --random-order \
  --seed 20 \
  --output-dir runs/bem_seed20
```

## Hyperparameter grid search

The grid-search script is located in the repository root:

```text
grid_search_bem_llvip.py
```

Run it from the repository root using `--data-root`:

```bash
python grid_search_bem_llvip.py \
  --data-root data/LLVIP \
  --weights weights/yolo11m.pt \
  --device cuda:0 \
  --alphas 0.1,0.2,0.4,0.6,0.8,0.9,1.0 \
  --gammas 0.001,0.005,0.01,0.05,0.1,0.5,1.0 \
  --output-dir runs/grid_search
```

Every Cartesian-product combination of the supplied `alpha` and `gamma` values is evaluated by invoking `run_llvip.py` in BEM mode.

Each combination stores its authoritative integrated result in its own directory:

```text
runs/grid_search/
├── a0.1_g0.001/
│   ├── integrated_bem.json
│   └── integrated_bem.csv
├── a0.1_g0.005/
│   ├── integrated_bem.json
│   └── integrated_bem.csv
├── ...
└── grid_search_summary.json
```

The metrics for an individual combination are available under the `total` field of that combination's `integrated_bem.json`.

> **Implementation note:** The current grid-search runner executes every combination and writes the per-combination `integrated_bem.json` files. Before relying on `grid_search_summary.json`, ensure that the script collects `integrated_bem.json["total"]`; the per-combination files are the authoritative outputs.

### Best hyperparameter combinations

| Model | Alpha | Gamma | Embedding window |
|---|---:|---:|---:|
| YOLOv11m (COCO) | 0.2 | 0.005 | 25 |
| YOLOv8s (COCO) | 0.2 | 0.005 | 25 |
| RT-DETR-L (COCO) | 0.5 | 0.1 | 25 |
| YOLOv8l-World-v2 | 0.8 | 0.05 | 25 |
| YOLOv8s-World-v2 | 0.7 | 0.05 | 25 |
| YOLOv11m (COCO to VOC) | 0.7 | 0.005 | 25 |
| YOLOv8s (COCO to VOC) | 0.7 | 0.005 | 25 |
| RT-DETR-L (COCO to VOC) | 0.6 | 0.05 | 25 |

These combinations correspond to the settings used for the Table 1 results.

To evaluate a selected combination, replace `--weights`, `--alpha`, and `--gamma` in the BEM evaluation command.

Each reported experiment was repeated five times with different random seeds. The `--seed` argument is available in `run_llvip.py` for seed-controlled runs.

## Main outputs

The integrated runner writes JSON and CSV summaries:

```text
integrated_baseline.json
integrated_baseline.csv
integrated_bem.json
integrated_bem.csv
```

The JSON output contains:

```text
mode
aggregation
modalities
├── visible
│   ├── per_scene
│   ├── mean
│   └── num_images
└── infrared
    ├── per_scene
    ├── mean
    └── num_images
total
num_images
```

The `total` field contains the final equal-weight average of the visible and infrared modality means.

## Associated paper

**BEM: Training-Free Background Embedding Memory for False-Positive Suppression in Real-Time Fixed-Background Camera**

- arXiv: [2604.11714](https://arxiv.org/abs/2604.11714)
- Paper DOI: [10.48550/arXiv.2604.11714](https://doi.org/10.48550/arXiv.2604.11714)

## Archived reproducible version

Add the final permanent software identifier after archiving the RRPR release:

- GitHub release: `[ADD_FINAL_GITHUB_RELEASE_URL]`
- Software DOI or SWHID: `[ADD_ZENODO_DOI_OR_SWHID]`

Do not replace these placeholders with an example DOI. The software DOI or SWHID must identify the archived source-code release used for the reproducibility submission.

## License

The source code developed for this repository is released under the [MIT License](LICENSE).

Third-party datasets, pretrained model weights, libraries, and other external artefacts remain subject to their respective original licenses and terms of use. The MIT License for this repository does not override the licenses of LLVIP, Ultralytics, or any pretrained model weights.
