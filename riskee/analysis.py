"""
riskee/analysis.py -- shared library for the offline evaluation.

Loads the cached per-exit logits and feature scores written by
riskee/cache_cifar.py and riskee/tiny.py, and implements the metrics (ECE with
15 equal-width bins, bootstrap ECE, Brier score), temperature fitting and the
first-exit routing rules of every method.  The evaluation protocol itself is in
riskee/protocol.py.
"""
import os, sys, json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, logsumexp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from riskee.common import CACHE_DIR, CORRUPTIONS, SEVERITIES

N_BINS = 15
# Relative cumulative FLOPs per exit.  ResNet-18 values measured with thop
# (train_model.profile_flops; checked by riskee/flops_handcount.py); DeiT-S:
# exits after blocks 4, 8 and 12 of 12.
COSTS = {'cnn': [0.5483486023990316, 0.7742860680092439, 1.00],
         'vit': [1 / 3, 2 / 3, 1.00],
         'tiny': [0.5481876050470714, 0.7741380554820393, 1.00]}


def arch_of(tag):
    return 'vit' if tag.startswith('vit') else ('tiny' if tag.startswith('tiny') else 'cnn')


# ───────────────────────────── data ─────────────────────────────────────────
class DS:
    """One cached evaluation set with derived per-exit quantities."""

    def __init__(self, path, ref, T_full=None, T_exit=None):
        z = np.load(path)
        self.y = z['labels']
        self.K = 3
        self.logits = [z[f'logits{k}'].astype(np.float64) for k in (1, 2, 3)]
        self.raw = {n: [z[f'{n}{k}'].astype(np.float64) for k in (1, 2, 3)]
                    for n in ('dist', 'maha', 'energy') if f'{n}1' in z}
        self.ref = ref
        self.C = self.logits[0].shape[1]
        self.probs = [softmax(l, 1) for l in self.logits]
        self.conf = [p.max(1) for p in self.probs]
        self.ent = [-(p * np.log(p + 1e-12)).sum(1) / np.log(self.C) for p in self.probs]
        self.pred = [p.argmax(1) for p in self.probs]
        # normalised deviation scores, same hinge form as the paper's S_k
        self.S = {}
        for n, vals in self.raw.items():
            out = []
            for k in range(3):
                r = ref[str(k + 1)]
                if n == 'energy':
                    # energy is signed: standardise w.r.t. training split
                    out.append(np.maximum(0, (vals[k] - r['energy_mean']) / r['energy_std']))
                else:
                    out.append(np.maximum(0, vals[k] / r[f'{n}_mean'] - 1.0))
            self.S[n] = out
        self.n = len(self.y)

    def scaled_probs(self, k, T):
        return softmax(self.logits[k] / T, 1)


def load_ref(tag):
    ref = json.load(open(os.path.join(CACHE_DIR, tag, 'ref.json')))
    # all score references (centroid mean distance d_bar, Mahalanobis mean, energy
    # mean/std) are fitted on the TRAINING split in riskee/cache_cifar.py / riskee/tiny.py
    return ref


# ───────────────────────────── metrics ──────────────────────────────────────
def ece(conf, correct, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges, right=True) - 1, 0, n_bins - 1)
    cnt = np.bincount(b, minlength=n_bins)
    sa = np.bincount(b, weights=correct, minlength=n_bins)
    sc = np.bincount(b, weights=conf, minlength=n_bins)
    m = cnt > 0
    return float(np.sum(np.abs(sa[m] - sc[m])) / len(conf))


def ece_boot(conf, correct, idx_mat, n_bins=N_BINS):
    """ECE for many bootstrap index sets at once. idx_mat: (B, n)."""
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges, right=True) - 1, 0, n_bins - 1)
    B, n = idx_mat.shape
    bb = b[idx_mat] + n_bins * np.arange(B)[:, None]
    cnt = np.bincount(bb.ravel(), minlength=B * n_bins).reshape(B, n_bins)
    sa = np.bincount(bb.ravel(), weights=correct[idx_mat].ravel(), minlength=B * n_bins).reshape(B, n_bins)
    sc = np.bincount(bb.ravel(), weights=conf[idx_mat].ravel(), minlength=B * n_bins).reshape(B, n_bins)
    return np.abs(sa - sc).sum(1) / n


def brier(probs, y):
    oh = np.zeros_like(probs); oh[np.arange(len(y)), y] = 1
    return float(((probs - oh) ** 2).sum(1).mean())


def summarize(out, y, costs):
    probs, ex = out
    pred = probs.argmax(1); conf = probs.max(1); cor = (pred == y).astype(float)
    return dict(acc=float(cor.mean()), ece=ece(conf, cor), brier=brier(probs, y),
                flops=float(np.asarray(costs)[ex].mean()),
                exit_dist=[float((ex == k).mean()) for k in range(3)])


# ───────────────────────────── temperature ──────────────────────────────────
def fit_T(logits, y):
    def nll(t):
        z = logits / t
        return float(np.mean(logsumexp(z, 1) - z[np.arange(len(y)), y]))
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method='bounded').x)


# ───────────────────────────── routing ──────────────────────────────────────
def route(ds, scores, tau, accept_low=True, probs_k=None):
    """Generic first-exit routing. scores[k] per sample; exit at first k with
    score <= tau (accept_low) or >= tau. Returns (probs, exit_idx)."""
    n = ds.n
    ex = np.full(n, 2)
    done = np.zeros(n, bool)
    for k in range(2):
        ok = (scores[k] <= tau) if accept_low else (scores[k] >= tau)
        sel = ok & ~done
        ex[sel] = k; done |= sel
    P = probs_k if probs_k is not None else ds.probs
    probs = np.where((ex == 0)[:, None], P[0], np.where((ex == 1)[:, None], P[1], P[2]))
    return probs, ex


def method_scores(ds, m, p):
    """Return (scores list, accept_low, probs list) for method m, params p."""
    if m == 'conf':
        return ds.conf, False, None
    if m == 'entropy':
        return ds.ent, True, None
    if m == 'pets_conf':
        P = [ds.scaled_probs(k, p['T'][k]) for k in range(3)]
        return [q.max(1) for q in P], False, P
    if m.startswith('risk'):
        score = p.get('score', 'dist')
        costs = p['costs']
        if p.get('T') is not None:     # Risk-EE on per-exit-TS logits
            P = [ds.scaled_probs(k, p['T'][k]) for k in range(3)]
            U = [-(q * np.log(q + 1e-12)).sum(1) / np.log(ds.C) for q in P]
        else:
            P, U = None, ds.ent
        R = [U[k] + p['l1'] * ds.S[score][k] + p['l2'] * costs[k] for k in range(3)]
        return R, True, P
    raise ValueError(m)


def run(ds, m, p, tau, costs):
    if m == 'full':
        ex = np.full(ds.n, 2); return ds.probs[2], ex
    if m == 'ts_full':
        ex = np.full(ds.n, 2); return ds.scaled_probs(2, p['T']), ex
    sc, low, P = method_scores(ds, m, p)
    return route(ds, sc, tau, low, P)


def ens_probs(ds):
    """Exit-ensemble probabilities (mean softmax of exits 1..k), Xia & Bouganis."""
    return [np.mean(ds.probs[:k + 1], axis=0) for k in range(3)]
