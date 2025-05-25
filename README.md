# Fine‑Tuning **ResNet‑18** on **Caltech‑101**

> **Transfer‑learning beats training from scratch by **≈ +22 pp** on a 30‑shot Caltech‑101 split.**

This repository contains the full PyTorch implementation, training scripts and utilities for the experiments described in the report:

> *Huang Jichuan & Zhao Junming – “Fine‑tuning Pre‑trained Convolutional Neural Networks for Caltech‑101 Image Classification: A Comparative Study”*

---

## Directory Layout

```
caltech101_finetuning/
├── data_utils.py          # dataset download, split & transforms
├── model_utils.py         # model building & differential LR helper
├── train_caltech101.py    # training / evaluation loop
├── download.py            # one‑click Caltech‑101 fetch
├── requirements.txt       # exact Python package versions
└── README.md              # you are here
```
---

## Quick Start

```bash
# 1. Create & activate a Python ≥3.9 environment
conda create -n caltech101 python=3.9 -y
conda activate caltech101

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download Caltech‑101 (~130 MB) and split 30/val‑rest
python download.py  # creates data/Caltech101/

# 4. Random Initialization (Run 2 by default)
python train_caltech101.py \
  --data-root /mnt/data/jichuan/ICL4Minimax/ICL4minimax/Caltech-101_Classification/data \
  --model resnet18 \
  --scratch \
  --epochs 300 \
  --batch-size 64 \
  --img-size 224 \
  --lr-backbone 2.5e-3 \
  --lr-fc 2.5e-3 \
  --weight-decay 2e-3 \
  --mixup-alpha 0.4 \
  --cutmix-alpha 1.0 \
  --workers 8 \
  --log-dir runs/r18_scratch_reg

# 5. Fine‑tune the pre‑trained ResNet‑18 (Run 5 by default)
python train_caltech101.py \
  --data-root /mnt/data/jichuan/ICL4Minimax/ICL4minimax/Caltech-101_Classification/data \
  --model resnet18 \
  --epochs 240 \
  --batch-size 96 \
  --img-size 224 \
  --lr-backbone 2e-4 \
  --lr-fc 4e-3 \
  --weight-decay 5e-5 \
  --mixup-alpha 0.2 \
  --cutmix-alpha 1.0 \
  --workers 8 \
  --log-dir runs/r18_pretrain_lowwd_mix02
```
TensorBoard logs are written under `checkpoints/<run_name>/tb/`. Start TensorBoard via `tensorboard --logdir checkpoints/`.

---

## Reproducing the Published Runs

| Run   | Scratch | Epochs | Batch |  LR BB |  LR FC |  WD  | MixUp α | CutMix α |  Val‑Acc  |
| ----- | ------- | :----: | :---: | :----: | :----: | :--: | :-----: | :------: | :-------: |
| **1** | ✔       |   300  |   64  |  3e‑3  |  3e‑3  | 1e‑3 |   0.4   |    1.0   |   0.712   |
| **2** | ✔       |   300  |   64  | 2.5e‑3 | 2.5e‑3 | 2e‑3 |   0.4   |    1.0   | **0.723** |
| **3** | ✖       |   200  |  128  |  2e‑4  |  5e‑3  | 1e‑4 |   0.4   |    1.0   |   0.932   |
| **4** | ✖       |   200  |  128  |  3e‑4  |  5e‑3  | 1e‑4 |   0.4   |    1.0   |   0.935   |
| **5** | ✖       |   240  |   96  |  2e‑4  |  4e‑3  | 5e‑5 |   0.2   |    1.0   | **0.940** |

---

## Download Trained Weights

```bash
# Best fine‑tuned model (Run 5)
wget -O resnet18_caltech101_run5.pth \
  "https://drive.google.com/uc?export=download&id=1cu5nu9MQMsNCxg1yRiNeK9cqiHB2wR8c"
```
---

## Key Training Flags

| Flag                               | Purpose                                    | 
| ---------------------------------- | ------------------------------------------ | 
| `--scratch`                        | Train from random init instead of ImageNet |
| `--epochs`                         | Training epochs                            | 
| `--batch_size`                     | Global batch size                          | 
| `--lr_backbone` / `--lr_fc`        | Differential learning rates                | 
| `--weight_decay`                   | L2 WD                                      | 
| `--mixup_alpha` / `--cutmix_alpha` | Augmentation strengths                     | 
| `--img_size`                       | Input resolution                           | 
| `--workers`                        | Data‑loader processes                      |

Run `python train_caltech101.py --help` for all options.

---

Happy fine‑tuning! 
