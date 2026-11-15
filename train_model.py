"""
train_model.py
Train the early-exit heads on top of the frozen/fine-tuned ResNet18 backbone
and compute per-exit feature statistics for the corruption score.

Two-phase training strategy
---------------------------
Phase 1 (freeze backbone):
    Only the three exit classifiers are trained.
    This rapidly adapts the heads to the fixed backbone representations.

Phase 2 (fine-tune end-to-end):
    All parameters are unfrozen and trained jointly with a small LR.
    This allows the backbone to slightly adjust to the multi-exit objective.

Training objective
------------------
L = L_final + alpha1 * L_exit1 + alpha2 * L_exit2
  + alpha_cal * (BrierScore_final + ...)   [optional calibration term]

After training, we make a single pass over the clean training set to compute
the feature statistics (mu_k, sigma_k) needed by the corruption score S_k.

Usage
-----
    python train_model.py

Assumes config.BASELINE_CKPT exists (run train_baseline.py first).
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from model import build_model
from data_utils import get_cifar10_loaders, get_clean_train_loader

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(config.SEED)
random.seed(config.SEED)
np.random.seed(config.SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[train_model] Device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def brier_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Multi-class Brier score (mean squared error over one-hot targets).
    Encourages calibrated softmax probabilities.
    """
    probs   = torch.softmax(logits, dim=1)                  # (B, C)
    one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
    return ((probs - one_hot) ** 2).sum(dim=1).mean()


def multi_exit_loss(logits1, logits2, logits_f, targets):
    """
    Combined training loss over all three exits.
    """
    ce = nn.CrossEntropyLoss()
    L1 = ce(logits1,  targets)
    L2 = ce(logits2,  targets)
    Lf = ce(logits_f, targets)

    loss = Lf + config.ALPHA1 * L1 + config.ALPHA2 * L2

    if config.ALPHA_CAL > 0:
        cal = (brier_score(logits1,  targets) +
               brier_score(logits2,  targets) +
               brier_score(logits_f, targets)) / 3.0
        loss = loss + config.ALPHA_CAL * cal

    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Profiling: compute actual relative FLOPs per exit
# ─────────────────────────────────────────────────────────────────────────────

def profile_flops(model, device):
    """
    Use thop to measure FLOPs at each exit and update config.RELATIVE_FLOPS.
    Falls back to the hard-coded approximation if thop is not installed.
    """
    try:
        from thop import profile as thop_profile

        dummy = torch.randn(1, 3, 32, 32).to(device)
        model.eval()

        # We wrap partial forward passes to measure incremental FLOPs
        class _Exit1(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                x = self.m.stem(x); x = self.m.layer1(x); x = self.m.layer2(x)
                f = self.m.avgpool(x).flatten(1); return self.m.exit1(f)

        class _Exit2(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                x = self.m.stem(x); x = self.m.layer1(x)
                x = self.m.layer2(x); x = self.m.layer3(x)
                f = self.m.avgpool(x).flatten(1); return self.m.exit2(f)

        class _Full(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                x = self.m.stem(x); x = self.m.layer1(x)
                x = self.m.layer2(x); x = self.m.layer3(x)
                x = self.m.layer4(x)
                f = self.m.avgpool(x).flatten(1); return self.m.final_fc(f)

        f1, _ = thop_profile(_Exit1(model), inputs=(dummy,), verbose=False)
        f2, _ = thop_profile(_Exit2(model), inputs=(dummy,), verbose=False)
        ff, _ = thop_profile(_Full(model),  inputs=(dummy,), verbose=False)

        rel = [f1 / ff, f2 / ff, 1.0]
        config.RELATIVE_FLOPS = rel
        model.cost = rel
        print(f"[train_model] FLOPs profiled → exit1={rel[0]:.3f}, "
              f"exit2={rel[1]:.3f}, final=1.000")

        # thop registers total_ops / total_params as buffers on every module.
        # Remove them now so they do not pollute the saved checkpoint.
        for mod in model.modules():
            for attr in ('total_ops', 'total_params'):
                if hasattr(mod, attr):
                    delattr(mod, attr)

    except ImportError:
        print("[train_model] thop not installed — using approximate FLOPs "
              f"{config.RELATIVE_FLOPS}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Use full-network predictions for validation accuracy
        l1, l2, lf = model.forward_train(x)
        correct += lf.argmax(1).eq(y).sum().item()
        total   += x.size(0)
    return correct / total


def run_phase(model, train_loader, val_loader, epochs, lr,
              freeze_backbone=False, phase_name=''):
    """Generic training loop for one phase."""

    if freeze_backbone:
        for name, p in model.named_parameters():
            # Freeze everything except the three exit heads
            if not any(h in name for h in ['exit1', 'exit2', 'final_fc']):
                p.requires_grad_(False)
        print(f"[{phase_name}] Backbone frozen — training exit heads only.")
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        print(f"[{phase_name}] End-to-end fine-tuning.")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=config.EE_MOMENTUM,
                          weight_decay=config.EE_WD, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val  = 0.0
    criterion = multi_exit_loss

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            l1, l2, lf = model.forward_train(x)
            loss = criterion(l1, l2, lf, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        val_acc = evaluate(model, val_loader)
        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), config.EE_CKPT)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{phase_name}] Epoch {epoch:3d}/{epochs} | "
                  f"Loss {total_loss/len(train_loader.dataset):.4f} | "
                  f"Val {val_acc:.4f} | Best {best_val:.4f}")

    # Restore best weights for next phase
    model.load_state_dict(torch.load(config.EE_CKPT, map_location=device, weights_only=True))
    return best_val


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(stats_only: bool = False):
    import os
    if not os.path.exists(config.BASELINE_CKPT):
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {config.BASELINE_CKPT}. "
            "Run train_baseline.py first."
        )

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=config.EE_BATCH
    )

    if stats_only:
        # Skip training — load existing checkpoint and only recompute stats
        if not os.path.exists(config.EE_CKPT):
            raise FileNotFoundError(
                f"EE checkpoint not found at {config.EE_CKPT}. "
                "Run train_model.py without --stats-only first."
            )
        print("[train_model] --stats-only: skipping training, loading existing checkpoint.")
        model = build_model(device, load_ee=True)
    else:
        model = build_model(device, load_baseline=True)
        profile_flops(model, device)

        # ── Phase 1: train exit heads, backbone frozen ────────────────────────
        PHASE1_EPOCHS = max(1, config.EE_EPOCHS // 2)
        run_phase(model, train_loader, val_loader,
                  epochs=PHASE1_EPOCHS, lr=config.EE_LR,
                  freeze_backbone=True, phase_name='Phase1')

        # ── Phase 2: end-to-end fine-tune ─────────────────────────────────────
        PHASE2_EPOCHS = config.EE_EPOCHS - PHASE1_EPOCHS
        run_phase(model, train_loader, val_loader,
                  epochs=PHASE2_EPOCHS, lr=config.EE_LR * 0.1,
                  freeze_backbone=False, phase_name='Phase2')

        # ── Final test accuracy (full network) ────────────────────────────────
        model.load_state_dict(torch.load(config.EE_CKPT, map_location=device, weights_only=True))
        test_acc = evaluate(model, test_loader)
        print(f"\n[train_model] Final test accuracy (full network): {test_acc:.4f}")
        print(f"[train_model] Checkpoint saved → {config.EE_CKPT}")

    # ── Compute feature statistics on clean training data ─────────────────
    print("\n[train_model] Computing feature statistics on clean training set …")
    stats_loader = get_clean_train_loader(batch_size=256)
    model.compute_feature_stats(stats_loader, device)
    print("[train_model] Feature statistics saved. Done.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats-only', action='store_true',
                        help='Skip training; only recompute feature statistics.')
    args = parser.parse_args()
    main(stats_only=args.stats_only)
