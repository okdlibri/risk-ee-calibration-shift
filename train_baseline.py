"""
train_baseline.py
Train a clean ResNet18 on CIFAR-10 from scratch.

This produces the backbone weights used by the early-exit model.
The final checkpoint is saved to config.BASELINE_CKPT.

Usage
-----
    python train_baseline.py

Typical result: ~93–94% top-1 accuracy on the CIFAR-10 test set
after 100 epochs with cosine annealing.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import config
from data_utils import get_cifar10_loaders

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(config.SEED)
random.seed(config.SEED)
np.random.seed(config.SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[baseline] Device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def build_baseline():
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, config.NUM_CLASSES)
    return net.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += out.argmax(1).eq(y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(1).eq(y).sum().item()
        total   += x.size(0)
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=config.BASELINE_BATCH
    )

    model     = build_baseline()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=config.BASELINE_LR,
                          momentum=config.BASELINE_MOMENTUM,
                          weight_decay=config.BASELINE_WD,
                          nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.BASELINE_EPOCHS
    )

    best_val_acc = 0.0
    print(f"[baseline] Training for {config.BASELINE_EPOCHS} epochs …")

    for epoch in range(1, config.BASELINE_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          criterion, optimizer)
        val_acc = evaluate(model, val_loader)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.BASELINE_CKPT)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{config.BASELINE_EPOCHS} | "
                  f"Loss {tr_loss:.4f} | "
                  f"Train {tr_acc:.3f} | "
                  f"Val {val_acc:.3f} | "
                  f"Best Val {best_val_acc:.3f}")

    # Final test evaluation using best checkpoint
    model.load_state_dict(torch.load(config.BASELINE_CKPT, map_location=device, weights_only=True))
    test_acc = evaluate(model, test_loader)
    print(f"\n[baseline] Test accuracy: {test_acc:.4f}  "
          f"(checkpoint → {config.BASELINE_CKPT})")


if __name__ == '__main__':
    main()
