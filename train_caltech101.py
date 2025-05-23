import argparse
import math
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from data_utils import get_dataloaders  
from model_utils import get_model      

# ------------------------- utility classes ------------------------- #

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: p.clone().detach() for k, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert k in self.shadow
            self.shadow[k].mul_(self.decay).add_(p, alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        self.backup = {}
        for k, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[k] = p.data.clone()
            p.data.copy_(self.shadow[k])

    def restore(self, model: nn.Module):
        for k, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.backup[k])
        self.backup = {}


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)):
    """Compute top‑k accuracy (returns list)."""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True) 
    pred = pred.t()                              
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:k].reshape(-1).float().sum().item() / target.size(0))
    return res


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes: int, smoothing: float = 0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.log_softmax(dim=-1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


# ------------------------- MixUp / CutMix helpers ------------------------- #

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W, H = size[2], size[3]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    # uniform center
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mix_data(x, y, alpha: float, use_cutmix: bool):
    """Return mixed inputs, pairs of targets, and lambda."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    if use_cutmix:
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1.0 - (bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2))
    else:
        x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ------------------------- training / evaluation ------------------------- #

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scheduler,
    ema: EMA,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    args,
):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    total_steps = len(loader)

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # apply MixUp / CutMix with 0.5 probability
        if np.random.rand() < 0.5:
            use_cutmix = np.random.rand() < 0.5  # half of the mixes are CutMix
            imgs, labels_a, labels_b, lam = mix_data(
                imgs, labels,
                alpha=args.cutmix_alpha if use_cutmix else args.mixup_alpha,
                use_cutmix=use_cutmix,
            )
        else:
            labels_a = labels_b = None  # indicates no mix
            lam = 1.0

        # forward
        logits = model(imgs)
        if labels_a is not None:
            loss = mix_criterion(criterion, logits, labels_a, labels_b, lam)
        else:
            loss = criterion(logits, labels)

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        # EMA
        ema.update(model)

        # metric
        acc1 = accuracy(logits, labels, topk=(1,))[0]
        running_loss += loss.item()
        running_acc += acc1

        # log every 20 steps
        if step % 20 == 0:
            global_step = epoch * total_steps + step
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar("Train/Acc", acc1, global_step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)

    return running_loss / total_steps, running_acc / total_steps


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device):
    model.eval()
    loss_sum, acc_sum = 0.0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Test‑time augmentation: original + horizontal flip
        logits = model(imgs) + model(torch.flip(imgs, [3]))
        logits /= 2.0

        loss = criterion(logits, labels)
        acc1 = accuracy(logits, labels, topk=(1,))[0]

        loss_sum += loss.item()
        acc_sum += acc1

    return loss_sum / len(loader), acc_sum / len(loader)


# ------------------------- main ------------------------- #

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data --- #
    train_loader, val_loader, num_classes = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # overwrite train transform (safer – DatasetFolder wraps Subset)
    new_train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        ]
    )
    try:
        train_loader.dataset.dataset.transform = new_train_tf  # Subset → Dataset
    except AttributeError:
        train_loader.dataset.transform = new_train_tf  # fallback

    # --- model & optimizer --- #
    model, param_groups = get_model(
        name=args.model,
        num_classes=num_classes,
        pretrained=not args.scratch,
        lr_backbone=args.lr_backbone,
        lr_fc=args.lr_fc,
    )
    model.to(device)

    if args.scratch:
        criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    max_lr = [g["lr"] for g in optimizer.param_groups]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=1e3,
        final_div_factor=1e3,
    )

    ema = EMA(model, decay=0.999)

    # --- logging & checkpoint --- #
    log_dir = Path("runs") / f"Caltech101_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(str(log_dir))
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            ema,
            device,
            epoch,
            writer,
            args,
        )

        # eval with EMA weights
        ema.apply_to(model)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        ema.restore(model)

        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Acc", val_acc, epoch)

        print(
            f"[Epoch {epoch+1}/{args.epochs}] "
            f"Train L {train_loss:.4f} A {train_acc:.3f} | "
            f"Val L {val_loss:.4f} A {val_acc:.3f} | "
            f"LR {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ema.apply_to(model)
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "ema_state": ema.shadow,
                },
                ckpt_dir / f"{args.model}_best.pth",
            )
            ema.restore(model)

    # save final EMA model
    ema.apply_to(model)
    torch.save(
        {
            "epoch": args.epochs,
            "state_dict": model.state_dict(),
            "ema_state": ema.shadow,
        },
        ckpt_dir / f"{args.model}_final.pth",
    )

    writer.close()
    print(f"Training finished. Best Val Acc = {best_acc:.3f}")


# ------------------------- cli ------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fine‑tune CNNs on Caltech‑101 (strong baseline)")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "alexnet"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr-backbone", type=float, default=2e-4)
    parser.add_argument("--lr-fc", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scratch", action="store_true", help="train from scratch")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--log-dir", default="runs", help="root dir for TensorBoard logs")
    args = parser.parse_args()

    main(args)
