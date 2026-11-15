"""
config.py
Central configuration for Risk-Aware Early Exit Networks.
All hyperparameters and paths live here so every other script imports
from one place — no magic numbers scattered across files.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = './data'
CIFAR10C_DIR    = './data/CIFAR-10-C'
CHECKPOINT_DIR  = './checkpoints'
RESULTS_DIR     = './results'
FIGURES_DIR     = './figures'

for _d in [DATA_DIR, CIFAR10C_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Checkpoints ──────────────────────────────────────────────────────────────
BASELINE_CKPT       = os.path.join(CHECKPOINT_DIR, 'resnet18_baseline.pth')
EE_CKPT             = os.path.join(CHECKPOINT_DIR, 'early_exit_model.pth')
FEATURE_STATS_PATH  = os.path.join(CHECKPOINT_DIR, 'feature_stats.npz')

# ─── Model ────────────────────────────────────────────────────────────────────
NUM_CLASSES = 10

# Relative cumulative FLOPs at each exit (exit1, exit2, final), as measured with
# thop by train_model.profile_flops (which also sets this value at training time).
RELATIVE_FLOPS = [0.5483, 0.7743, 1.00]

# ─── Baseline Training ────────────────────────────────────────────────────────
BASELINE_EPOCHS     = 100
BASELINE_LR         = 0.1
BASELINE_MOMENTUM   = 0.9
BASELINE_WD         = 5e-4
BASELINE_BATCH      = 128
LR_MILESTONES       = [50, 75]      # epochs at which LR drops by 0.1
LR_GAMMA            = 0.1

# ─── Early Exit Training ──────────────────────────────────────────────────────
EE_EPOCHS   = 30
EE_LR       = 0.01
EE_MOMENTUM = 0.9
EE_WD       = 1e-4
EE_BATCH    = 128

# Per-exit loss weights in the training objective
ALPHA1 = 0.5    # exit1
ALPHA2 = 0.5    # exit2
# final exit always has weight 1.0

# Optional Brier score calibration term weight (0 = disabled)
ALPHA_CAL = 0.1

# ─── Risk Function ────────────────────────────────────────────────────────────
# R_k(x) = U_k(x) + LAMBDA1 * S_k(x) + LAMBDA2 * C_k
# Default weights of the model's built-in inference routine only.  In the
# experiments (lambda1, lambda2) and tau are selected on the validation split
# by riskee/protocol.py.
LAMBDA1 = 0.5
LAMBDA2 = 0.3

# ─── Corruption Benchmark ─────────────────────────────────────────────────────
CORRUPTION_TYPES = [
    # Noise
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'speckle_noise',
    # Blur
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur',
    # Weather
    'snow', 'frost', 'fog', 'brightness', 'spatter',
    # Digital
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'saturate',
]
SEVERITY_LEVELS = [1, 2, 3, 4, 5]

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ─── DataLoader ───────────────────────────────────────────────────────────────
NUM_WORKERS = 0     # 0 for Windows compatibility; increase on Linux
