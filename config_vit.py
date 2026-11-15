"""
config_vit.py
Configuration for Risk-Aware Early Exit Networks — ViT (DeiT-Small) on CIFAR-100.
Mirrors config.py but uses separate checkpoint / results / figures directories.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR         = './data'
CIFAR100_DIR     = './data/cifar-100-python'
CIFAR100C_DIR    = './data/CIFAR-100-C'
CHECKPOINT_DIR   = './checkpoints_vit'
RESULTS_DIR      = './results_vit'
FIGURES_DIR      = './figures_vit'

for _d in [DATA_DIR, CIFAR100_DIR, CIFAR100C_DIR,
           CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Checkpoints ──────────────────────────────────────────────────────────────
BASELINE_CKPT      = os.path.join(CHECKPOINT_DIR, 'deit_small_baseline.pth')
EE_CKPT            = os.path.join(CHECKPOINT_DIR, 'early_exit_vit.pth')
FEATURE_STATS_PATH = os.path.join(CHECKPOINT_DIR, 'feature_stats_vit.npz')

# ─── Model ────────────────────────────────────────────────────────────────────
NUM_CLASSES    = 100
IMAGE_SIZE     = 224          # DeiT-Small requires 224×224 input
EMBED_DIM      = 384          # DeiT-Small embedding dimension
EXIT_BLOCKS    = [4, 8, 12]   # transformer block indices (0-based: 3, 7, 11)
# Relative cumulative cost at each exit (12 equal-cost blocks; exits after blocks
# 4, 8 and 12).  The analysis uses exactly [1/3, 2/3, 1] (riskee/analysis.COSTS).
RELATIVE_FLOPS = [0.33, 0.67, 1.00]

# ─── Baseline Training (fine-tune pretrained DeiT-Small) ──────────────────────
BASELINE_EPOCHS   = 30
BASELINE_LR       = 1e-4      # small LR for fine-tuning
BASELINE_WD       = 1e-4
BASELINE_BATCH    = 32        # fits P2200 5 GB VRAM
BASELINE_WARMUP   = 5         # cosine schedule with 5-epoch linear warmup

# ─── Early Exit Training ──────────────────────────────────────────────────────
# Phase 1: freeze backbone, train exit heads only
PHASE1_EPOCHS = 10
PHASE1_LR     = 1e-3
PHASE1_WD     = 1e-4

# Phase 2: end-to-end fine-tune with all exits
PHASE2_EPOCHS = 20
PHASE2_LR     = 5e-5
PHASE2_WD     = 1e-4

EE_BATCH = 32

# Per-exit loss weights
ALPHA1     = 0.5   # exit1 (after block 4)
ALPHA2     = 0.5   # exit2 (after block 8)
ALPHA_CAL  = 0.1   # optional calibration term (Brier-style)

# ─── Risk Function ────────────────────────────────────────────────────────────
# Default weights of the model's built-in inference routine only.  In the
# experiments (lambda1, lambda2) and tau are selected on the validation split
# by riskee/protocol.py.
LAMBDA1 = 0.3
LAMBDA2 = 0.3

# ─── Corruption Benchmark ─────────────────────────────────────────────────────
CORRUPTION_TYPES = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur',
    'snow', 'frost', 'fog', 'brightness', 'spatter',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'saturate',
]
SEVERITY_LEVELS = [1, 2, 3, 4, 5]

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ─── DataLoader ───────────────────────────────────────────────────────────────
NUM_WORKERS = 0   # 0 for Windows compatibility
