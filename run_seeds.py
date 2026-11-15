"""
run_seeds.py
Train the additional ResNet-18 / CIFAR-10 models (baseline -> early-exit) with
three random seeds.  Each seed gets its own checkpoint directory:
    checkpoints_seed{N}/
The models are then cached and evaluated by riskee/cache_cifar.py and
riskee/protocol.py (tags cnn_seed42, cnn_seed123, cnn_seed456).

Usage
-----
    python run_seeds.py              # all 3 seeds
    python run_seeds.py --seeds 42   # single seed
"""

import argparse
import importlib
import os
import sys

SEEDS = [42, 123, 456]


def run_one_seed(seed: int):
    """
    Patch the config module and train one seed.
    All modules that import config see the patched values because Python
    caches the module object in sys.modules.
    """
    print('\n' + '=' * 70)
    print(f'SEED = {seed}')
    print('='*70)

    # ── Patch config before importing any training module ─────────────────────
    import config
    config.SEED             = seed
    config.CHECKPOINT_DIR   = f'./checkpoints_seed{seed}'
    config.BASELINE_CKPT    = os.path.join(config.CHECKPOINT_DIR,
                                            'resnet18_baseline.pth')
    config.EE_CKPT          = os.path.join(config.CHECKPOINT_DIR,
                                            'early_exit_model.pth')
    config.FEATURE_STATS_PATH = os.path.join(config.CHECKPOINT_DIR,
                                              'feature_stats.npz')
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ── Reload dependent modules so they re-bind module-level variables ───────
    for mod_name in ['model', 'data_utils', 'train_baseline', 'train_model']:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)

    from train_baseline import main as run_baseline
    from train_model    import main as run_ee
    run_baseline()
    run_ee()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='Seeds to run (default: 42 123 456)')
    args = parser.parse_args()
    for seed in args.seeds:
        run_one_seed(seed)


if __name__ == '__main__':
    main()
