"""Dataset loader for split CSV files and preprocessing conditions."""

from torch.utils.data import Dataset


class FruitDataset(Dataset):
    """Load image paths and labels from a split CSV."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implement CSV loading and image preprocessing here.")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

