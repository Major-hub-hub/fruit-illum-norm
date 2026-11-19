"""ResNet-50 binary classifier."""

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_model(pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

