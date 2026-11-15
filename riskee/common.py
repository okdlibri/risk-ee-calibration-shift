"""
riskee/common.py -- shared utilities for caching model outputs.

Design: every model is run ONCE per evaluation set; per-exit logits and
per-exit feature-space scores are cached to disk.  All routing policies,
threshold / lambda selection (on the held-out validation split),
temperature scaling and bootstrap confidence intervals are then computed
offline from the cache (riskee/protocol.py).  Routing decisions depend only on
per-sample quantities, so evaluating from cached outputs is equivalent to
running the early-exit forward pass.

Feature scores cached per exit k:
  dist   : ||f - mu_k||_2 with mu_k fit on the TRAINING split  (Risk-EE's S_k)
  maha   : min_c sqrt((f-mu_kc)^T Sigma_k^-1 (f-mu_kc)), class means and
           tied covariance fit on the TRAINING split   (Lee et al. 2018)
  energy : -logsumexp(logits_k)                        (Liu et al. 2020)

The cache location defaults to <repo>/cache and can be changed with the
environment variable RISKEE_CACHE.
"""
import os, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get('RISKEE_CACHE', os.path.join(ROOT, 'cache'))
os.makedirs(CACHE_DIR, exist_ok=True)

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur',
    'snow', 'frost', 'fog', 'brightness', 'spatter',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'saturate',
]
SEVERITIES = [1, 2, 3, 4, 5]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ScoreBank:
    """Train-split statistics for each feature score at each exit (1..3)."""

    def __init__(self):
        self.stats = {}

    @torch.no_grad()
    def fit(self, feats_per_exit, labels, num_classes):
        labels = torch.as_tensor(labels, device=DEVICE)
        for k, Fk in feats_per_exit.items():
            Fk = Fk.to(DEVICE).float()
            mu = Fk.mean(0)
            d = (Fk - mu).norm(dim=1)
            cls_means = torch.stack([Fk[labels == c].mean(0) for c in range(num_classes)])
            Xc = Fk - cls_means[labels]
            cov = (Xc.T @ Xc / Fk.shape[0]).double()
            cov = cov + 1e-6 * torch.trace(cov) / cov.shape[0] * torch.eye(
                cov.shape[0], device=DEVICE, dtype=torch.float64)
            prec = torch.linalg.inv(cov).float()
            st = dict(mu=mu, dbar=d.mean(), cls_means=cls_means, prec=prec)
            self.stats[k] = st
            maha_tr = torch.cat([self._maha(Fk[i:i + 2048], st)
                                 for i in range(0, Fk.shape[0], 2048)])
            st['ref'] = {'dist': (float(d.mean()), float(d.std())),
                         'maha': (float(maha_tr.mean()), float(maha_tr.std()))}

    @staticmethod
    def _maha(f, st):
        diff = f[:, None, :] - st['cls_means'][None]
        m = torch.einsum('bcd,de,bce->bc', diff, st['prec'], diff)
        return m.clamp(min=0).min(dim=1).values.sqrt()

    @torch.no_grad()
    def scores(self, k, f):
        f = f.float()
        out = {}
        if k in self.stats:
            st = self.stats[k]
            out['dist'] = (f - st['mu']).norm(dim=1)
            out['maha'] = self._maha(f, st)
        return out

    def ref_json(self):
        return {str(k): dict(dbar=float(st['dbar']),
                             **{f'{n}_mean': v[0] for n, v in st['ref'].items()},
                             **{f'{n}_std': v[1] for n, v in st['ref'].items()})
                for k, st in self.stats.items()}


@torch.no_grad()
def collect(model_fn, loader, bank=None, want_feats=False):
    """Run model_fn(x) -> (list_of_logits, list_of_feats) over a loader."""
    logits_acc = feats_acc = scores_acc = None
    labels = []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits, feats = model_fn(x)
        K = len(logits)
        if logits_acc is None:
            logits_acc = [[] for _ in range(K)]
            feats_acc = [[] for _ in range(K)]
            scores_acc = [dict() for _ in range(K)]
        for k in range(K):
            lk = logits[k].float()
            logits_acc[k].append(lk.cpu())
            scores_acc[k].setdefault('energy', []).append((-torch.logsumexp(lk, 1)).cpu())
            if want_feats:
                feats_acc[k].append(feats[k].float().cpu())
            if bank is not None:
                for n, v in bank.scores(k + 1, feats[k]).items():
                    scores_acc[k].setdefault(n, []).append(v.cpu())
        labels.append(torch.as_tensor(y))
    res = {'labels': torch.cat(labels).numpy().astype(np.int64)}
    for k in range(len(logits_acc)):
        res[f'logits{k+1}'] = torch.cat(logits_acc[k]).numpy().astype(np.float32)
        for n, lst in scores_acc[k].items():
            res[f'{n}{k+1}'] = torch.cat(lst).numpy().astype(np.float32)
        if want_feats:
            res[f'feats{k+1}'] = torch.cat(feats_acc[k])
    return res


def save_set(tag, name, res):
    d = os.path.join(CACHE_DIR, tag)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f'{name}.npz'),
                        **{k: v for k, v in res.items() if not k.startswith('feats')})


def exists(tag, name):
    return os.path.exists(os.path.join(CACHE_DIR, tag, f'{name}.npz'))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)
