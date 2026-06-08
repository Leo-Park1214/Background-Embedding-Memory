# Background Embedding Memory

Background Embedding Memory (BEM) is a training-free module for reducing false-positive detections in fixed-camera scenes.

## Installation
```bash
git clone https://github.com/Leo-Park1214/Background-Embedding-Memory.git
cd Background-Embedding-Memory

conda create -n bem python=3.11 -y
conda activate bem

pip install -r requirements.txt
```
or
```bash
git clone https://github.com/Leo-Park1214/Background-Embedding-Memory.git
cd Background-Embedding-Memory

conda create -n bem python=3.10 -y
conda activate bem

pip install -r requirements2.txt
```

We use requirements.txt for our experiments; however, environment setup with requirements2.txt is faster.

## Model weights

Model weights can be downloaded from the [Ultralytics model documentation](https://docs.ultralytics.com/models/).

Place the downloaded weight file in the repository or pass its path through `--weights`.

## Download LLVIP

```bash
python scripts/download_llvip.py \
  --source hf \
  --data-root data/LLVIP
```

To rebuild existing prepared data:

```bash
python scripts/download_llvip.py \
  --source hf \
  --data-root data/LLVIP \
  --overwrite
```

## Baseline

```bash
python run_llvip.py \
  --data-root data/LLVIP \
  --weights path/to/model.pt \
  --device cuda:0 \
  --mode baseline \
  --output-dir runs/baseline
```

## BEM

```bash
python run_llvip.py \
  --data-root data/LLVIP \
  --weights path/to/model.pt \
  --device cuda:0 \
  --mode bem \
  --embedding-window 20 \
  --alpha 0.6 \
  --gamma 1.0 \
  --output-dir runs/bem
```

The experiment evaluates visible and infrared data from scenes 19–24 and saves the integrated result in the output directory.

## Hyperparameter grid search

Grid search is executed with `grid_search_bem_llvip.py`.

```bash
python scripts/grid_search_bem_llvip.py \
  --data data/LLVIP \
  --weights path/to/model.pt \
  --device cuda:0 \
  --alphas 0.1,0.2,0.4,0.6,0.8,0.9,1.0 \
  --gammas 0.001,0.005,0.01,0.05,0.1,0.5,1.0 \
  --output-dir runs/grid_search
```

Every combination of the supplied `alpha` and `gamma` candidates is evaluated.

```text
runs/grid_search/
├── a0.2_g0.5/
├── a0.2_g1.0/
├── ...
└── tuning_summary.json
```
### Best hyperparameter combinations

model | alpha | gamma | embedding-window
|---|---|---|---|
yolo11m(coco) | 0.2 | 0.005 | 25 
yolov8s(coco) | 0.2 | 0.005 | 25
rtdetr-l(coco) | 0.5 | 0.1 | 25
yolov8-l-world-v2 | 0.8 | 0.05 | 25
yolov8-s-world-v2 | 0.7 | 0.05 | 25
yolo11m(coco -> voc) | 0.7 | 0.005 | 25
yolov8s(coco -> voc) | 0.7 | 0.005 | 25
rtdetr-l(coco -> voc) | 0.6 | 0.05 | 25

This setting is our table1 results setting

coco->voc model is fine-tuned from coco pretrained model with ultralytics basic setting (model.train(data=VOC.yaml)) 
To ensure stability, we run each experiment five times.
## Main outputs

Eexperiments save scene-level results and an integrated result:

```text
integrated_baseline.json
integrated_bem.json
```
## License

MIT License
