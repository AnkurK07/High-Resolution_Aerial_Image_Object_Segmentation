"""
dataset.py
----------
PyTorch Dataset/DataLoader wrapper around the numpy arrays produced by
preprocessor.py, plus the train/test split.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import config


class SegmentationDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        # images: (N, H, W, 3) float32 in [0,1] -> torch wants (N, 3, H, W)
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_dataloaders(images: np.ndarray, labels: np.ndarray):
    """Split into train/test and wrap both halves in DataLoaders."""
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED,
    )

    train_loader = DataLoader(
        SegmentationDataset(x_train, y_train),
        batch_size=config.BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        SegmentationDataset(x_test, y_test),
        batch_size=config.BATCH_SIZE, shuffle=False,
    )
    return train_loader, test_loader
