"""
python -m riskee.cache_cifar --tag {cnn,cnn_seed42,cnn_seed123,cnn_seed456,cnn_sdn,vit,vit_sdn}

Caches per-exit logits + feature scores for:
  val   : held-out 10% validation split (used for ALL selection: tau, lambda,
          temperatures, window thresholds)
  test  : clean test set
  c_{corruption}_{severity} : CIFAR-10-C / CIFAR-100-C (95 sets)
Feature-score statistics are fitted on the 90% training split only.
"""
import argparse, os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from riskee.common import (ScoreBank, collect, save_set, exists, log, DEVICE,
                       CORRUPTIONS, SEVERITIES, CACHE_DIR)

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--workers', type=int, default=4)
args = ap.parse_args()
TAG = args.tag
IS_VIT = TAG.startswith('vit')


def main():
    if not IS_VIT:
        import config
        seed = int(TAG.replace('cnn_seed', '')) if 'seed' in TAG else 42
        config.SEED = seed
        import data_utils as du
        from torchvision import datasets
        from model import EarlyExitResNet
        ckpt = {'cnn': 'checkpoints/early_exit_model.pth',
                'cnn_sdn': 'checkpoints/sdn_model.pth'}.get(
            TAG, f'checkpoints_seed{seed}/early_exit_model.pth')
        model = EarlyExitResNet().to(DEVICE)
        sd = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.endswith(('total_ops', 'total_params'))}
        missing, unexp = model.load_state_dict(sd, strict=False)
        log('loaded', ckpt, 'missing', missing, 'unexpected', unexp)
        model.eval()
        num_classes = 10

        def fn(x):
            f1, f2, f3 = model._features(x)
            return [model.exit1(f1), model.exit2(f2), model.final_fc(f3)], [f1, f2, f3]

        # identical split to training code
        full_clean = datasets.CIFAR10(root='./data', train=True, download=False,
                                      transform=du.transform_test)
        n_val = int(len(full_clean) * 0.1)
        perm_split = torch.utils.data.random_split(
            range(len(full_clean)), [len(full_clean) - n_val, n_val],
            generator=torch.Generator().manual_seed(seed))
        tr_idx, va_idx = list(perm_split[0]), list(perm_split[1])
        test_ds = datasets.CIFAR10(root='./data', train=False, download=False,
                                   transform=du.transform_test)
        corrupt_ds = lambda c, s: du.CIFAR10C(c, s)
        bs = 256
    else:
        import config_vit as cfg
        import data_utils_vit as dv
        import torchvision
        from model_vit import EarlyExitDeiT
        ckpt = {'vit': 'checkpoints_vit/early_exit_vit.pth',
                'vit_sdn': 'checkpoints_vit/sdn_vit.pth'}[TAG]
        model = EarlyExitDeiT(pretrained=False).to(DEVICE)
        sd = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.endswith(('total_ops', 'total_params'))}
        missing, unexp = model.load_state_dict(sd, strict=False)
        log('loaded', ckpt, 'missing', missing, 'unexpected', unexp)
        model.eval()
        num_classes = 100

        def fn(x):
            h = model._embed(x)
            feats, logits = [], []
            for i, blk in enumerate(model.blocks):
                h = blk(h)
                if i == model.EXIT_IDX[0]:
                    feats.append(h[:, 0]); logits.append(model.exit1(h[:, 0]))
                elif i == model.EXIT_IDX[1]:
                    feats.append(h[:, 0]); logits.append(model.exit2(h[:, 0]))
            hn = model.norm(h)
            feats.append(hn[:, 0]); logits.append(model.final_head(hn[:, 0]))
            return logits, feats

        full_clean = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                   download=False,
                                                   transform=dv.val_transform())
        # identical split to data_utils_vit.get_cifar100_loaders
        torch.manual_seed(cfg.SEED)
        tr, va = torch.utils.data.random_split(range(len(full_clean)), [45000, 5000])
        tr_idx, va_idx = list(tr), list(va)
        test_ds = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=False,
                                                transform=dv.val_transform())
        corrupt_ds = lambda c, s: dv.CIFAR100C(c, s)
        bs = 128

    mk = lambda ds: DataLoader(ds, batch_size=bs, shuffle=False,
                               num_workers=args.workers, pin_memory=True,
                               persistent_workers=False)
    json.dump({'train_idx_head': tr_idx[:10], 'val_idx_head': va_idx[:10]},
              open(os.path.join(CACHE_DIR, f'{TAG}_split.json'), 'w'))

    # 1) fit score statistics on the TRAINING split only
    log(TAG, 'collecting training-split features', len(tr_idx))
    tr = collect(fn, mk(Subset(full_clean, tr_idx)), want_feats=True)
    bank = ScoreBank()
    bank.fit({k: tr[f'feats{k}'] for k in (1, 2, 3)}, tr['labels'], num_classes)
    ref = bank.ref_json()
    for k in (1, 2, 3):
        ref[str(k)]['energy_mean'] = float(tr[f'energy{k}'].mean())
        ref[str(k)]['energy_std'] = float(tr[f'energy{k}'].std())
    del tr
    os.makedirs(os.path.join(CACHE_DIR, TAG), exist_ok=True)
    json.dump(ref, open(os.path.join(CACHE_DIR, TAG, 'ref.json'), 'w'), indent=1)
    torch.save([bank.stats[k]['mu'].cpu() for k in (1, 2, 3)], os.path.join(CACHE_DIR, TAG, 'centroids.pt'))
    log('score stats fitted')

    # 2) val / test / corruptions
    jobs = [('val', lambda: Subset(full_clean, va_idx)), ('test', lambda: test_ds)]
    for c in CORRUPTIONS:
        for s in SEVERITIES:
            jobs.append((f'c_{c}_{s}', (lambda c=c, s=s: corrupt_ds(c, s))))
    for name, dsf in jobs:
        if exists(TAG, name):
            continue
        res = collect(fn, mk(dsf()), bank=bank)
        save_set(TAG, name, res)
        acc = (res['logits3'].argmax(1) == res['labels']).mean()
        log(TAG, name, f'final-exit acc={acc:.4f}')
    log(TAG, 'DONE')


if __name__ == '__main__':
    main()
