
"""ResNet50 binary classifier for fresh/rotten fruit."""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


# Repository root
ROOT = Path(__file__).resolve().parents[2]

# Locally saved ImageNet V1 weights
LOCAL_WEIGHTS = (
    ROOT / "models" / "pretrained"
    / "resnet50_imagenet1k_v1.pth"
)


def build_model(
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """Build ResNet50 with a two-class output head."""

    if pretrained:
        if not LOCAL_WEIGHTS.is_file():
            raise FileNotFoundError(
                f"Local pretrained weights not found: "
                f"{LOCAL_WEIGHTS}\n"
                "Run download_resnet50.py first."
            )

        # Build original ImageNet architecture
        model = resnet50(weights=None)

        # Load pretrained ImageNet parameters
        state_dict = torch.load(
            LOCAL_WEIGHTS,
            map_location="cpu",
            weights_only=True
        )
        model.load_state_dict(state_dict, strict=True)

    else:
        # Initialize randomly
        model = resnet50(weights=None)

    # Optionally freeze pretrained backbone
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # Replace ImageNet classifier with binary head
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.fc.in_features, 2)
    )

    return model
