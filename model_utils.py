from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models

def get_model(name: str = "resnet18",
              num_classes: int = 101,
              pretrained: bool = True,
              lr_backbone: float = 1e-4,
              lr_fc: float = 1e-3) -> Tuple[nn.Module, list]:
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1
                                if pretrained else None)
        in_dim = model.fc.in_features
        model.fc = nn.Linear(in_dim, num_classes)
        params = [
            {"params": [p for n, p in model.named_parameters()
                        if not n.startswith("fc")], "lr": lr_backbone},
            {"params": model.fc.parameters(), "lr": lr_fc}
        ]
    elif name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1
                               if pretrained else None)
        in_dim = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_dim, num_classes)
        params = [
            {"params": model.features.parameters(), "lr": lr_backbone},
            {"params": model.classifier[:-1].parameters(), "lr": lr_backbone},
            {"params": model.classifier[-1].parameters(), "lr": lr_fc},
        ]
    else:
        raise ValueError(f"Unsupported model {name}")
    return model, params
