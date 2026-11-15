"""
data_utils_vit.py
Data loaders for CIFAR-100 (clean) and CIFAR-100-C (corrupted).
Images are resized to 224×224 for DeiT-Small.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as T

import config_vit as cfg

# ImageNet-style normalisation used by DeiT pretrained weights
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# ─── Transforms ───────────────────────────────────────────────────────────────

def train_transform():
    return T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

def val_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

def corrupt_transform():
    """Transform for CIFAR-100-C (already 32×32 PNG-like numpy arrays)."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


# ─── Clean CIFAR-100 ──────────────────────────────────────────────────────────

def get_cifar100_loaders(batch_size: int = cfg.BASELINE_BATCH,
                         val_split: float = 0.1,
                         num_workers: int = cfg.NUM_WORKERS):
    """
    Returns (train_loader, val_loader, test_loader).
    90 % of the official training set is used for training, 10 % for validation.
    """
    full_train = torchvision.datasets.CIFAR100(
        root=cfg.DATA_DIR, train=True, download=False,
        transform=train_transform())

    n_val   = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    torch.manual_seed(cfg.SEED)
    train_ds, val_ds = random_split(full_train, [n_train, n_val])

    # Val set needs the val transform, not the aug transform
    val_ds.dataset = torchvision.datasets.CIFAR100(
        root=cfg.DATA_DIR, train=True, download=False,
        transform=val_transform())

    test_ds = torchvision.datasets.CIFAR100(
        root=cfg.DATA_DIR, train=False, download=False,
        transform=val_transform())

    kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ─── CIFAR-100-C ─────────────────────────────────────────────────────────────

class CIFAR100C(Dataset):
    """
    CIFAR-100-C corruption dataset.
    Each .npy file has shape (50000, 32, 32, 3) — 5 severities × 10000 images.
    labels.npy has shape (50000,) but the labels repeat across severities,
    so we use only the first 10000.
    """
    def __init__(self, corruption: str, severity: int,
                 transform=None):
        assert 1 <= severity <= 5, "Severity must be 1-5"
        npy_path = os.path.join(cfg.CIFAR100C_DIR, f'{corruption}.npy')
        lbl_path = os.path.join(cfg.CIFAR100C_DIR, 'labels.npy')

        if not os.path.exists(npy_path):
            raise FileNotFoundError(f'Missing: {npy_path}')

        all_data   = np.load(npy_path)                      # (50000,32,32,3)
        all_labels = np.load(lbl_path)                      # (50000,) or (10000,)

        s = severity - 1
        self.data   = all_data[s * 10000: (s + 1) * 10000] # (10000,32,32,3)
        # Labels may be 10000 or 50000; handle both
        if len(all_labels) == 50000:
            self.labels = all_labels[s * 10000: (s + 1) * 10000]
        else:
            self.labels = all_labels                        # already 10000

        self.transform = transform if transform is not None else corrupt_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img   = self.data[idx]   # uint8 (32,32,3)
        label = int(self.labels[idx])
        img   = self.transform(img)
        return img, label


def get_cifar100c_loader(corruption: str, severity: int,
                         batch_size: int = cfg.EE_BATCH,
                         num_workers: int = cfg.NUM_WORKERS) -> DataLoader:
    ds = CIFAR100C(corruption, severity)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == '__main__':
    tr, va, te = get_cifar100_loaders(batch_size=4)
    imgs, labels = next(iter(te))
    print('Test batch:', imgs.shape, labels.shape)

    ldr = get_cifar100c_loader('gaussian_noise', severity=3, batch_size=4)
    imgs2, labels2 = next(iter(ldr))
    print('Corrupt batch:', imgs2.shape, labels2.shape)
