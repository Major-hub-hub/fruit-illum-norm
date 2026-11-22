
from pathlib import Path

import torch
from torchvision import models

# 1. Select the exact pretrained weights
weights = models.ResNet50_Weights.IMAGENET1K_V1

# 2. Download weights and construct ResNet50
model = models.resnet50(weights=weights)

# 3. Locate PyTorch's checkpoint cache
cache_path = (
    Path(torch.hub.get_dir())
    / "checkpoints"
    / Path(weights.url).name
)

# 4. Verify the download
assert cache_path.is_file(), "Weights file not found!"

print("ResNet50 successfully loaded!")
print("Weights:", weights.name)
print("Checkpoint:", cache_path)
print("File size (MiB):",
      round(cache_path.stat().st_size / 1024**2, 2))
print("Output features:", model.fc.out_features)

# 5. Save a separate copy in the project
output_dir = Path("models/pretrained")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "resnet50_imagenet1k_v1.pth"

torch.save(model.state_dict(), output_path)

print("Project copy saved to:", output_path.resolve())
