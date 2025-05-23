from pathlib import Path
from collections import defaultdict
import random
from typing import Tuple, List
import torch
from torchvision import datasets, transforms

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _split(labels: List[int], n_train: int = 30, seed: int = 42):
    random.seed(seed)
    cls2idx = defaultdict(list)
    for i, y in enumerate(labels):
        cls2idx[y].append(i)
    train, val = [], []
    for idxs in cls2idx.values():
        random.shuffle(idxs)
        train += idxs[:n_train]
        val   += idxs[n_train:]
    return train, val

def get_dataloaders(data_root: str,
                    img_size=224, batch_size=32, num_workers=4
                    ) -> Tuple[torch.utils.data.DataLoader,
                               torch.utils.data.DataLoader,
                               int]:
    root = Path(data_root).expanduser().resolve()
    cats_dir = root / "101_ObjectCategories"
    if not cats_dir.is_dir():
        raise FileNotFoundError(f"{cats_dir} 不存在，请检查 --data-root")

    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
    ])

    full = datasets.ImageFolder(cats_dir, transform=tf_train)

    if "BACKGROUND_Google" in full.class_to_idx:
        bg_old_lbl = full.class_to_idx["BACKGROUND_Google"]

        keep_samples = [(p, y) for p, y in full.samples if y != bg_old_lbl]
        kept_classes = [c for c in full.classes if c != "BACKGROUND_Google"]
        old2new = {full.class_to_idx[c]: i for i, c in enumerate(kept_classes)}

        full.samples = [(p, old2new[y]) for p, y in keep_samples]
        full.targets = [old2new[y] for _, y in keep_samples]
        full.classes = kept_classes
        full.class_to_idx = {c: i for i, c in enumerate(kept_classes)}
    else:
        if not hasattr(full, "targets"):  
            full.targets = [y for _, y in full.samples]
        old2new = None  

    train_idx, val_idx = _split(full.targets)
    train_ds = torch.utils.data.Subset(full, train_idx)

    val_base = datasets.ImageFolder(cats_dir, transform=tf_val)
    if old2new: 
        val_samples = [(p, y) for p, y in val_base.samples
                       if y in old2new]           
        val_samples = [(p, old2new[y]) for p, y in val_samples]
    else:
        val_samples = val_base.samples

    val_base.samples = val_samples
    val_base.targets = [y for _, y in val_samples]
    val_ds = torch.utils.data.Subset(val_base, val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, len(full.classes)
