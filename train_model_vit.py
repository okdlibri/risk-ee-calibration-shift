"""
train_model_vit.py
Two-phase training of EarlyExitDeiT on CIFAR-100.

Phase 1: Freeze backbone (patch embed, pos embed, all transformer blocks),
         train only the three exit heads.
Phase 2: End-to-end fine-tune with a small LR.

After training, compute and save per-exit feature statistics (mean CLS vector
and mean L2 distance) on the validation set for the risk function.

Usage:
    python train_model_vit.py              # full two-phase training
    python train_model_vit.py --stats-only # skip training, recompute stats
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn

import config_vit as cfg
from model_vit import EarlyExitDeiT, build_model, save_feature_stats
from data_utils_vit import get_cifar100_loaders


def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── FLOPs profiling ──────────────────────────────────────────────────────────

def profile_flops(model: EarlyExitDeiT, device):
    """Use thop to measure relative FLOPs at each exit."""
    try:
        from thop import profile as thop_profile
        dummy = torch.zeros(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(device)

        # We cannot directly profile partial models with thop on a sequential;
        # instead we measure the *full* model then approximate by block count.
        # Exit 1: 4/12 blocks, Exit 2: 8/12 blocks, Exit 3: 12/12 blocks.
        # (The patch embed and exit heads are small compared to blocks.)
        rel = [4 / 12, 8 / 12, 1.0]
        model.rel_flops = torch.tensor(rel, dtype=torch.float32).to(device)
        print(f'[train_model_vit] Relative FLOPs: {rel}')
    except Exception as e:
        print(f'[train_model_vit] thop unavailable ({e}), using defaults.')

    # Remove any thop-registered buffers
    for mod in model.modules():
        for attr in ('total_ops', 'total_params'):
            if hasattr(mod, attr):
                delattr(mod, attr)


# ─── Loss ─────────────────────────────────────────────────────────────────────

def ee_loss(logits_list, labels):
    """Weighted cross-entropy across exit heads."""
    weights = [cfg.ALPHA1, cfg.ALPHA2, 1.0]
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss = sum(w * criterion(lg, labels)
               for w, lg in zip(weights, logits_list))
    return loss


# ─── Accuracy helper ──────────────────────────────────────────────────────────

@torch.no_grad()
def accuracy_from_logits(logits_list, labels):
    """Returns accuracy of the final exit on this batch."""
    preds = logits_list[-1].argmax(dim=1)
    return (preds == labels).float().mean().item()


# ─── Train / eval ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = total_acc = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits_list = model(imgs)
            loss = ee_loss(logits_list, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        total_acc  += accuracy_from_logits(logits_list, labels) * imgs.size(0)
        total      += imgs.size(0)
    return total_loss / total, total_acc / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = total_acc = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits_list = model(imgs)
            loss = ee_loss(logits_list, labels)
        total_loss += loss.item() * imgs.size(0)
        total_acc  += accuracy_from_logits(logits_list, labels) * imgs.size(0)
        total      += imgs.size(0)
    return total_loss / total, total_acc / total


# ─── Phase helpers ────────────────────────────────────────────────────────────

def freeze_backbone(model: EarlyExitDeiT):
    """Freeze everything except exit heads and final_head."""
    for name, param in model.named_parameters():
        if 'exit1' in name or 'exit2' in name or 'final_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Phase 1] Trainable params: {trainable:,}')


def unfreeze_all(model: EarlyExitDeiT):
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Phase 2] Trainable params: {trainable:,}')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(stats_only: bool = False):
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train_model_vit] Device: {device}')

    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=cfg.EE_BATCH)

    if not stats_only:
        # Build model and load baseline checkpoint
        model = build_model(device, load_baseline_ckpt=True)
        profile_flops(model, device)

        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        # ── Phase 1: train exit heads only ────────────────────────────────────
        print('\n=== Phase 1: Exit heads only ===')
        freeze_backbone(model)
        optimizer1 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.PHASE1_LR, weight_decay=cfg.PHASE1_WD)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer1, T_max=cfg.PHASE1_EPOCHS, eta_min=1e-6)

        best_val = 0.0
        for epoch in range(1, cfg.PHASE1_EPOCHS + 1):
            tr_loss, tr_acc = train_epoch(
                model, train_loader, optimizer1, device, scaler)
            va_loss, va_acc = eval_epoch(model, val_loader, device)
            scheduler1.step()
            print(f'  P1 Epoch {epoch:2d} | '
                  f'train {tr_acc:.4f} | val {va_acc:.4f}')
            if va_acc > best_val:
                best_val = va_acc
                torch.save(model.state_dict(), cfg.EE_CKPT)

        # ── Phase 2: end-to-end fine-tune ─────────────────────────────────────
        print('\n=== Phase 2: End-to-end fine-tune ===')
        model.load_state_dict(
            torch.load(cfg.EE_CKPT, map_location=device, weights_only=True))
        unfreeze_all(model)
        optimizer2 = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.PHASE2_LR, weight_decay=cfg.PHASE2_WD)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=cfg.PHASE2_EPOCHS, eta_min=1e-7)

        best_val = 0.0
        for epoch in range(1, cfg.PHASE2_EPOCHS + 1):
            tr_loss, tr_acc = train_epoch(
                model, train_loader, optimizer2, device, scaler)
            va_loss, va_acc = eval_epoch(model, val_loader, device)
            scheduler2.step()
            print(f'  P2 Epoch {epoch:2d} | '
                  f'train {tr_acc:.4f} | val {va_acc:.4f}')
            if va_acc > best_val:
                best_val = va_acc
                torch.save(model.state_dict(), cfg.EE_CKPT)
                print(f'    → Saved (val {va_acc:.4f})')

        # Clean up thop buffers before final save
        for mod in model.modules():
            for attr in ('total_ops', 'total_params'):
                if hasattr(mod, attr):
                    delattr(mod, attr)

        # Test accuracy
        model.load_state_dict(
            torch.load(cfg.EE_CKPT, map_location=device, weights_only=True))
        _, te_acc = eval_epoch(model, test_loader, device)
        print(f'\nFinal test accuracy (last exit): {te_acc:.4f}')

    # ── Feature statistics ─────────────────────────────────────────────────────
    print('\n=== Computing feature statistics ===')
    model = build_model(device, load_ee_ckpt=True)
    stats = model.compute_feature_stats(val_loader, device)
    save_feature_stats(stats)

    # Re-save checkpoint with updated stat buffers
    torch.save(model.state_dict(), cfg.EE_CKPT)
    print('[train_model_vit] Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats-only', action='store_true',
                        help='Skip training, only recompute feature stats')
    args = parser.parse_args()
    main(stats_only=args.stats_only)
