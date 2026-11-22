
from pathlib import Path
import torch
from torchvision.models import resnet50

weight_path = Path("models/pretrained/resnet50_imagenet1k_v1.pth")

# Build the original 1000-class architecture
model = resnet50(weights=None)

# Load ImageNet pretrained parameters locally
state_dict = torch.load(
    weight_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict, strict=True)

# Adapt to the project's two classes
model.fc = torch.nn.Linear(model.fc.in_features, 2)

print("Local ResNet50 loaded successfully!")
print("Classifier:", model.fc)
print("Local weights:", weight_path.resolve())
