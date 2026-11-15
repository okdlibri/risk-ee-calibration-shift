"""
train_sdn.py
Trains an SDN (Shallow-Deep Networks, Kaya et al. ICML 2019) baseline
on CIFAR-10 using the same EarlyExitResNet architecture.

Key differences from our two-phase method
------------------------------------------
  1. Joint training  : all parameters trained end-to-end from epoch 1
  2. Equal loss weights : CE(exit1) + CE(exit2) + CE(final)  (no weighting)
  3. Inference       : standard confidence thresholding

This isolates the contribution of our training procedure and risk function
from architectural differences.  The model uses model.forward_train(x)
which returns (logits1, logits2, logits_final).

Saves: checkpoints/sdn_model.pth
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn

import config
from model import build_model
from data_utils import get_cifar10_loaders

SDN_CKPT   = os.path.join(config.CHECKPOINT_DIR, 'sdn_model.pth')
SDN_EPOCHS = 40
SDN_LR     = 0.01
SDN_WD     = 1e-4


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sdn_loss(logits_tuple, labels, label_smoothing: float = 0.1):
    """Equal-weight cross-entropy at every exit."""
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    losses = [criterion(lg, labels) for lg in logits_tuple]
    return sum(losses) / len(losses)


def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits_tuple = model.forward_train(imgs)
            loss = sdn_loss(logits_tuple, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        preds       = logits_tuple[-1].argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits_tuple = model.forward_train(imgs)
        preds = logits_tuple[-1].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
    return correct / total


def main():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train_sdn] Device: {device}')

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=config.EE_BATCH)

    # Load baseline backbone — all parameters trainable from the start
    model = build_model(device, load_baseline=True)
    for p in model.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[train_sdn] Trainable params: {trainable:,}')

    optimizer = torch.optim.SGD(
        model.parameters(), lr=SDN_LR,
        momentum=0.9, weight_decay=SDN_WD, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SDN_EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val = 0.0
    for epoch in range(1, SDN_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, device, scaler)
        va_acc = eval_epoch(model, val_loader, device)
        scheduler.step()
        print(f'SDN Epoch {epoch:3d}/{SDN_EPOCHS} | '
              f'train {tr_acc:.4f} | val {va_acc:.4f}')
        if va_acc > best_val:
            best_val = va_acc
            # Strip any thop buffers before saving
            for mod in model.modules():
                for attr in ('total_ops', 'total_params'):
                    if hasattr(mod, attr):
                        delattr(mod, attr)
            torch.save(model.state_dict(), SDN_CKPT)
            print(f'  → Saved (val {va_acc:.4f})')

    te_acc = eval_epoch(model, test_loader, device)
    print(f'\n[train_sdn] Test accuracy (final exit): {te_acc:.4f}')
    print(f'[train_sdn] Checkpoint → {SDN_CKPT}')


if __name__ == '__main__':
    main()
