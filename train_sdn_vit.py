"""
train_sdn_vit.py
SDN (Shallow-Deep Networks) baseline for DeiT-Small / CIFAR-100.

Same principle as train_sdn.py:
  - Joint end-to-end training (no two-phase)
  - Equal loss weights at all three exits
  - model.forward(x) returns (logits1, logits2, logits_final)

Saves: checkpoints_vit/sdn_vit.pth
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn

import config_vit as cfg
from model_vit import build_model
from data_utils_vit import get_cifar100_loaders

SDN_CKPT   = os.path.join(cfg.CHECKPOINT_DIR, 'sdn_vit.pth')
SDN_EPOCHS = 30
SDN_LR     = 5e-5
SDN_WD     = 1e-4


def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sdn_loss(logits_tuple, labels, label_smoothing: float = 0.1):
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
            logits_tuple = model(imgs)       # (l1, l2, lfinal)
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
        logits_tuple = model(imgs)
        preds = logits_tuple[-1].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
    return correct / total


def main():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train_sdn_vit] Device: {device}')

    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=cfg.EE_BATCH)

    # Load baseline DeiT checkpoint
    model = build_model(device, load_baseline_ckpt=True)
    for p in model.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[train_sdn_vit] Trainable params: {trainable:,}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SDN_LR, weight_decay=SDN_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SDN_EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val = 0.0
    for epoch in range(1, SDN_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, device, scaler)
        va_acc = eval_epoch(model, val_loader, device)
        scheduler.step()
        print(f'SDN-ViT Epoch {epoch:3d}/{SDN_EPOCHS} | '
              f'train {tr_acc:.4f} | val {va_acc:.4f}')
        if va_acc > best_val:
            best_val = va_acc
            for mod in model.modules():
                for attr in ('total_ops', 'total_params'):
                    if hasattr(mod, attr):
                        delattr(mod, attr)
            torch.save(model.state_dict(), SDN_CKPT)
            print(f'  → Saved (val {va_acc:.4f})')

    te_acc = eval_epoch(model, test_loader, device)
    print(f'\n[train_sdn_vit] Test accuracy (final exit): {te_acc:.4f}')
    print(f'[train_sdn_vit] Checkpoint → {SDN_CKPT}')


if __name__ == '__main__':
    main()
