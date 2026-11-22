"""Download and cache the pretrained ResNet-50 weights used by this project.

Run with: conda run -n fruit-split python download_resnet50.py
"""

from pathlib import Path
from urllib.parse import urlparse


def main() -> None:
    try:
        from torch.hub import get_dir
        from torchvision.models import ResNet50_Weights, resnet50
    except (ImportError, OSError, RuntimeError) as exc:
        raise SystemExit(
            "Could not import PyTorch and torchvision. Run this script in the "
            "project environment:\n"
            "  conda run -n fruit-split python download_resnet50.py\n"
            f"Original error: {exc}"
        ) from exc

    # Match src/models/resnet50_binary.py so it can reuse the cached weights.
    weights = ResNet50_Weights.IMAGENET1K_V1
    filename = Path(urlparse(weights.url).path).name
    checkpoint = Path(get_dir()) / "checkpoints" / filename

    print(f"Weights: {weights.name}", flush=True)
    print(f"Cache file: {checkpoint}", flush=True)
    print("Downloading missing weights or loading the cached file...", flush=True)

    # Loading the model also checks that the downloaded state dict is usable.
    resnet50(weights=weights, progress=True)
    print("ResNet-50 weights are cached and loaded successfully.")


if __name__ == "__main__":
    main()
