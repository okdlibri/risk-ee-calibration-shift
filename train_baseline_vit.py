"""
train_baseline_vit.py
Fine-tune a pretrained DeiT-Small on CIFAR-100.
Uses cosine schedule with linear warm-up and saves the best checkpoint.
"""

import random
import torch
import torch.nn as nn
import timm
import numpy as np

import config_vit as cfg
from data_utils_vit import get_cifar100_loaders


def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_deit(num_classes: int = cfg.NUM_CLASSES, device=None):
    """Load pretrained DeiT-Small and replace the head for CIFAR-100."""
    model = timm.create_model(
        'deit_small_patch16_224',
        pretrained=True,
        num_classes=num_classes,
    )
    if device is not None:
        model = model.to(device)
    return model


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def main():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[baseline_vit] Using device: {device}')

    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=cfg.BASELINE_BATCH)

    model     = build_deit(device=device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Separate LR for head vs backbone
    head_params     = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters()
                       if 'head' not in n]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg.BASELINE_LR},
        {'params': head_params,     'lr': cfg.BASELINE_LR * 10},
    ], weight_decay=cfg.BASELINE_WD)

    # Cosine schedule with linear warm-up
    def lr_lambda(epoch):
        if epoch < cfg.BASELINE_WARMUP:
            return (epoch + 1) / cfg.BASELINE_WARMUP
        progress = (epoch - cfg.BASELINE_WARMUP) / max(
            1, cfg.BASELINE_EPOCHS - cfg.BASELINE_WARMUP)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_acc = 0.0
    for epoch in range(1, cfg.BASELINE_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        va_loss, va_acc = eval_epoch(
            model, val_loader, criterion, device)
        scheduler.step()

        print(f'Epoch {epoch:3d}/{cfg.BASELINE_EPOCHS} | '
              f'train loss {tr_loss:.4f} acc {tr_acc:.4f} | '
              f'val loss {va_loss:.4f} acc {va_acc:.4f}')

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), cfg.BASELINE_CKPT)
            print(f'  → Saved best checkpoint (val acc {va_acc:.4f})')

    print(f'\nBest validation accuracy: {best_val_acc:.4f}')

    # Final test evaluation
    model.load_state_dict(
        torch.load(cfg.BASELINE_CKPT, map_location=device, weights_only=True))
    te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)
    print(f'Test accuracy: {te_acc:.4f}')


if __name__ == '__main__':
    main()
