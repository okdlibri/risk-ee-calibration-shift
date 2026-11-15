"""
data_utils.py
Data loading utilities for CIFAR-10 (clean) and CIFAR-10-C (corrupted).

CIFAR-10-C
----------
Download from: https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
Extract to:    ./data/CIFAR-10-C/
Expected layout after extraction:
    ./data/CIFAR-10-C/gaussian_noise.npy   (50000, 32, 32, 3) uint8
    ./data/CIFAR-10-C/shot_noise.npy
    ...
    ./data/CIFAR-10-C/labels.npy           (50000,) int64

Each .npy file contains images for 5 severity levels stacked:
    severity s ∈ {1,2,3,4,5} occupies rows [(s-1)*10000 : s*10000]
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import config

# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 standard transforms
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet-style normalisation used throughout to keep consistent with
# pretrained backbones if we later switch to a larger model
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def get_cifar10_loaders(batch_size: int = config.BASELINE_BATCH,
                        val_split: float = 0.1,
                        num_workers: int = config.NUM_WORKERS):
    """
    Returns (train_loader, val_loader, test_loader) for CIFAR-10.

    A 90/10 split of the official 50 000 training images is used for
    train / validation.  The 10 000 test images remain untouched.
    """
    full_train = datasets.CIFAR10(root=config.DATA_DIR, train=True,
                                  download=True, transform=transform_train)
    test_set   = datasets.CIFAR10(root=config.DATA_DIR, train=False,
                                  download=True, transform=transform_test)

    n_val   = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.SEED)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, test_loader


def get_clean_train_loader(batch_size: int = config.EE_BATCH,
                           num_workers: int = config.NUM_WORKERS):
    """
    Returns a DataLoader over the FULL clean training set (no val split).
    Used for computing feature statistics after training is complete.
    """
    full_train = datasets.CIFAR10(root=config.DATA_DIR, train=True,
                                  download=True, transform=transform_test)
    return DataLoader(full_train, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers, pin_memory=True)


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10-C Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CIFAR10C(Dataset):
    """
    CIFAR-10-C: a single corruption type at a single severity level.

    Parameters
    ----------
    corruption : str   one of config.CORRUPTION_TYPES
    severity   : int   1 – 5
    """

    def __init__(self, corruption: str, severity: int,
                 root: str = config.CIFAR10C_DIR,
                 transform=None):
        assert 1 <= severity <= 5, "severity must be in {1,2,3,4,5}"
        assert corruption in config.CORRUPTION_TYPES, \
            f"Unknown corruption: {corruption}"

        data_path  = os.path.join(root, f'{corruption}.npy')
        label_path = os.path.join(root, 'labels.npy')

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"CIFAR-10-C not found at {data_path}.\n"
                "Download from https://zenodo.org/record/2535967 and extract "
                f"to {root}"
            )

        all_data   = np.load(data_path)                        # (50000,32,32,3)
        all_labels = np.load(label_path)                       # (50000,)

        s = severity - 1
        self.data   = all_data  [s * 10000:(s + 1) * 10000]   # (10000,32,32,3)
        self.labels = all_labels[s * 10000:(s + 1) * 10000]   # (10000,)

        self.transform = transform or transform_test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img   = self.data[idx]                     # (32,32,3) uint8
        label = int(self.labels[idx])
        # PIL-free path: convert numpy → float tensor
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(img_t)
        return img_t, label


def get_cifar10c_loader(corruption: str, severity: int,
                        batch_size: int = config.BASELINE_BATCH,
                        num_workers: int = config.NUM_WORKERS) -> DataLoader:
    """Single convenience wrapper."""
    ds = CIFAR10C(corruption, severity)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def cifar10c_available() -> bool:
    """Return True if the CIFAR-10-C files are present on disk."""
    label_path = os.path.join(config.CIFAR10C_DIR, 'labels.npy')
    first_path = os.path.join(config.CIFAR10C_DIR,
                              f'{config.CORRUPTION_TYPES[0]}.npy')
    return os.path.exists(label_path) and os.path.exists(first_path)
