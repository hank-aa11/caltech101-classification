# Fine‑Tuning **ResNet‑18** on **Caltech‑101**

> **Transfer‑learning beats training from scratch by **≈ +22 pp** on a 30‑shot Caltech‑101 split.**

This repository contains the full PyTorch implementation, training scripts and utilities for the experiments described in the report:

> *Huang Jichuan & Zhao Junming – “Fine‑tuning Pre‑trained Convolutional Neural Networks for Caltech‑101 Image Classification: A Comparative Study”*
> *(see `report.tex`)*

---

## 📂 Directory Layout

```
caltech101_finetuning/
├── data_utils.py          # dataset download, split & transforms
├── model_utils.py         # model building & differential LR helper
├── train_caltech101.py    # training / evaluation loop
├── download.py            # one‑click Caltech‑101 fetch
├── checkpoints/           # *.pth weights & TensorBoard logs
│   ├── r18_pretrain_lowwd_mix02/  # Run 5 (best) – 0.940 Val‑Acc
│   └── ...
├── requirements.txt       # exact Python package versions
└── README.md              # you are here
```

Feel free to rename the root folder; paths are handled via arguments.

---

## ⚙️  Quick Start

```bash
# 1. Create & activate a Python ≥3.9 environment
conda create -n caltech101 python=3.9 -y
conda activate caltech101

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download Caltech‑101 (~130 MB) and split 30/val‑rest
python download.py  # creates data/Caltech101/

# 4. Fine‑tune the pre‑trained ResNet‑18 (Run 3 by default)
python train_caltech101.py \
    --exp r18_pretrain_main \
    --data_root data/Caltech101/ \
    --output_dir checkpoints/
```

TensorBoard logs are written under `checkpoints/<run_name>/tb/`. Start TensorBoard via `tensorboard --logdir checkpoints/`.

---

## 🏋️‍♀️  Reproducing the Published Runs

| Run   | Scratch | Epochs | Batch |  LR BB |  LR FC |  WD  | MixUp α | CutMix α |  Val‑Acc  |
| ----- | ------- | :----: | :---: | :----: | :----: | :--: | :-----: | :------: | :-------: |
| **1** | ✔       |   300  |   64  |  3e‑3  |  3e‑3  | 1e‑3 |   0.4   |    1.0   |   0.712   |
| **2** | ✔       |   300  |   64  | 2.5e‑3 | 2.5e‑3 | 2e‑3 |   0.4   |    1.0   |   0.723   |
| **3** | ✖       |   200  |  128  |  2e‑4  |  5e‑3  | 1e‑4 |   0.4   |    1.0   |   0.932   |
| **4** | ✖       |   200  |  128  |  3e‑4  |  5e‑3  | 1e‑4 |   0.4   |    1.0   |   0.935   |
| **5** | ✖       |   240  |   96  |  2e‑4  |  4e‑3  | 5e‑5 |   0.2   |    1.0   | **0.940** |

Use the exact preset names found in `train_caltech101.py`:

```bash
python train_caltech101.py --exp r18_pretrain_lowwd_mix02  # Run 5
```

Pre‑trained ImageNet weights are automatically downloaded by `torchvision` on first run.

---

## 💾  Download Trained Weights

```bash
# Best fine‑tuned model (Run 5)
wget -O resnet18_caltech101_run5.pth \
  "https://drive.google.com/uc?export=download&id=1cu5nu9MQMsNCxg1yRiNeK9cqiHB2wR8c"
```

Load in Python:

```python
import torch, model_utils
model = model_utils.get_model(num_classes=101, pretrained=False)
state_dict = torch.load('resnet18_caltech101_run5.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
```

---

## 🔧  Key Training Flags

| Flag                               | Purpose                                    | Default            |
| ---------------------------------- | ------------------------------------------ | ------------------ |
| `--scratch`                        | Train from random init instead of ImageNet | `False`            |
| `--epochs`                         | Training epochs                            | *preset‑dependent* |
| `--batch_size`                     | Global batch size                          | *preset‑dependent* |
| `--lr_backbone` / `--lr_fc`        | Differential learning rates                | —                  |
| `--weight_decay`                   | L2 WD                                      | —                  |
| `--mixup_alpha` / `--cutmix_alpha` | Augmentation strengths                     | —                  |
| `--img_size`                       | Input resolution                           | 224                |
| `--workers`                        | Data‑loader processes                      | 8                  |

Run `python train_caltech101.py --help` for all options.

---

## 📈  Expected GPU Utilisation

| GPU            | Batch 128 (FP32) | Peak VRAM |
| -------------- | ---------------- | --------- |
| RTX 3060 12 GB | ✓                | \~7 GB    |
| RTX 2080 8 GB  | ✓ (batch 96)     | \~7 GB    |

Training time for Run 5 on a single RTX 3060 ≈ **2 h**.

---

## 📝  Citation

If you use this code or trained weights in your research, please cite the accompanying report:

```
@report{huang2025caltech101,
  title   = {Fine‑tuning Pre‑trained Convolutional Neural Networks for Caltech‑101 Image Classification: A Comparative Study},
  author  = {Huang, Jichuan and Zhao, Junming},
  year    = {2025},
  note    = {Technical report},
}
```

---

## 🗒️  License

Distributed under the **MIT License** – see `LICENSE` for details.

---

## 🙏  Acknowledgements

* Caltech for releasing the Caltech‑101 dataset.
* The PyTorch & Torchvision teams.
* Original ImageNet pre‑training courtesy of \[PyTorch Hub].

---

Happy fine‑tuning! 🚀
