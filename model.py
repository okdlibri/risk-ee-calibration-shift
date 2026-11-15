"""
model.py
Risk-Aware Early Exit ResNet18.

Architecture
------------
Backbone : ResNet18 (pretrained weights loaded externally)
Exit 1   : after layer2  -> 128-d pooled features
Exit 2   : after layer3  -> 256-d pooled features
Final    : after layer4  -> 512-d pooled features

Risk Function (our method)
--------------------------
R_k(x) = U_k(x) + lambda1 * S_k(x) + lambda2 * C_k

  U_k  : normalised entropy of softmax output at exit k  ∈ [0, 1]
  S_k  : feature-deviation score (z-score distance from training mean) ∈ R+
  C_k  : normalised cumulative FLOPs at exit k  ∈ [0, 1]

Feature statistics (mu_k, sigma_k) are computed once on the clean training
set and saved to disk.  At test time S_k requires no second forward pass.

Note: the experiments in the paper do not use the built-in inference routine
below.  They cache per-exit logits and features (riskee/cache_cifar.py) and
compute S_k = max(0, ||f_k - mu_k|| / d_bar_k - 1), with mu_k and d_bar_k
fitted on the training split, together with all routing rules, in
riskee/analysis.py and riskee/protocol.py.

Inference modes supported
-------------------------
  'risk'        : our method — exit when R_k(x) ≤ tau
  'entropy'     : baseline — exit when U_k(x) ≤ tau  (raw entropy)
  'confidence'  : BranchyNet baseline — exit when max-softmax ≥ tau
  'full'        : no early exit, always runs to the final exit
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalised_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy of softmax output, normalised to [0, 1].
    Shape: (B,) -> (B,)
    """
    probs = F.softmax(logits, dim=1)
    raw   = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return raw / math.log(config.NUM_CLASSES)


def _max_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Max softmax probability.  Shape: (B,) -> (B,)"""
    return F.softmax(logits, dim=1).max(dim=1).values


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class EarlyExitResNet(nn.Module):
    """
    ResNet18 with two intermediate exit heads.

    Parameters
    ----------
    num_classes : int
    relative_flops : list[float]
        Normalised cumulative FLOPs at [exit1, exit2, final].
        Defaults to config.RELATIVE_FLOPS; overwritten after profiling.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 relative_flops: list = None):
        super().__init__()

        base = models.resnet18(weights=None)

        # ── Backbone blocks ──────────────────────────────────────────────────
        self.stem    = nn.Sequential(base.conv1, base.bn1, base.relu,
                                     base.maxpool)
        self.layer1  = base.layer1   # out: 64-d
        self.layer2  = base.layer2   # out: 128-d
        self.layer3  = base.layer3   # out: 256-d
        self.layer4  = base.layer4   # out: 512-d
        self.avgpool = base.avgpool  # global avg pool -> (B, C, 1, 1)

        # ── Exit classifiers ─────────────────────────────────────────────────
        self.exit1    = nn.Linear(128, num_classes)
        self.exit2    = nn.Linear(256, num_classes)
        self.final_fc = nn.Linear(512, num_classes)

        nn.init.normal_(self.exit1.weight,    std=0.01)
        nn.init.zeros_(self.exit1.bias)
        nn.init.normal_(self.exit2.weight,    std=0.01)
        nn.init.zeros_(self.exit2.bias)

        # ── FLOPs cost vector ─────────────────────────────────────────────────
        self.cost = relative_flops or config.RELATIVE_FLOPS[:]

        # ── Feature statistics (populated by compute_feature_stats) ───────────
        # mu_k  : mean feature vector for each exit  [ndarray, shape (D_k,)]
        # sig_k : scalar std of L2-distance from mean across training set
        self.register_buffer('mu1',  torch.zeros(128))
        self.register_buffer('mu2',  torch.zeros(256))
        self.register_buffer('mu3',  torch.zeros(512))
        self.register_buffer('sig1', torch.ones(1))
        self.register_buffer('sig2', torch.ones(1))
        self.register_buffer('sig3', torch.ones(1))
        self._stats_ready = False

    # ── Internal feature extraction ──────────────────────────────────────────

    def _features(self, x):
        """Return pooled features at all three exits."""
        x = self.stem(x)
        x = self.layer1(x)

        x = self.layer2(x)
        f1 = self.avgpool(x).flatten(1)   # (B, 128)

        x = self.layer3(x)
        f2 = self.avgpool(x).flatten(1)   # (B, 256)

        x = self.layer4(x)
        f3 = self.avgpool(x).flatten(1)   # (B, 512)

        return f1, f2, f3

    # ── Training forward ─────────────────────────────────────────────────────

    def forward(self, x):
        """Standard full-network forward (used by baseline training)."""
        f1, f2, f3 = self._features(x)
        return self.final_fc(f3)

    def forward_train(self, x):
        """
        Returns logits at all three exits for multi-exit training loss.
        Returns: (logits1, logits2, logits_final)
        """
        f1, f2, f3 = self._features(x)
        return self.exit1(f1), self.exit2(f2), self.final_fc(f3)

    # ── Feature statistics ───────────────────────────────────────────────────

    @torch.no_grad()
    def compute_feature_stats(self, dataloader, device):
        """
        Pass the entire (clean) training set through the network once and
        compute per-exit mean feature vector and the scalar std of the
        L2-distance distribution.  Results are stored in registered buffers
        and saved to config.FEATURE_STATS_PATH.
        """
        self.eval()
        all_f1, all_f2, all_f3 = [], [], []

        for x, _ in dataloader:
            x = x.to(device)
            f1, f2, f3 = self._features(x)
            all_f1.append(f1.cpu())
            all_f2.append(f2.cpu())
            all_f3.append(f3.cpu())

        F1 = torch.cat(all_f1)   # (N, 128)
        F2 = torch.cat(all_f2)   # (N, 256)
        F3 = torch.cat(all_f3)   # (N, 512)

        def _stats(F):
            mu  = F.mean(0)                          # (D,)
            d   = (F - mu).norm(dim=1)               # (N,) L2 distances
            # Store the MEAN distance as the reference scale.
            # S_k = max(0, dist/ref - 1): ≈0 for clean samples,
            # grows positive as the input deviates from the training distribution.
            ref = d.mean().clamp(min=1e-6)           # scalar
            return mu, ref

        mu1, sig1 = _stats(F1)
        mu2, sig2 = _stats(F2)
        mu3, sig3 = _stats(F3)

        self.mu1.copy_(mu1.to(device))
        self.mu2.copy_(mu2.to(device))
        self.mu3.copy_(mu3.to(device))
        self.sig1.copy_(sig1.to(device).unsqueeze(0))
        self.sig2.copy_(sig2.to(device).unsqueeze(0))
        self.sig3.copy_(sig3.to(device).unsqueeze(0))
        self._stats_ready = True

        np.savez(config.FEATURE_STATS_PATH,
                 mu1=mu1.numpy(),  sig1=sig1.numpy(),
                 mu2=mu2.numpy(),  sig2=sig2.numpy(),
                 mu3=mu3.numpy(),  sig3=sig3.numpy())
        print(f"[model] Feature stats saved -> {config.FEATURE_STATS_PATH}")

    def load_feature_stats(self, device):
        """Load previously computed feature statistics from disk."""
        d = np.load(config.FEATURE_STATS_PATH)
        self.mu1.copy_(torch.from_numpy(d['mu1']).to(device))
        self.mu2.copy_(torch.from_numpy(d['mu2']).to(device))
        self.mu3.copy_(torch.from_numpy(d['mu3']).to(device))
        self.sig1.copy_(torch.tensor([float(d['sig1'])]).to(device))
        self.sig2.copy_(torch.tensor([float(d['sig2'])]).to(device))
        self.sig3.copy_(torch.tensor([float(d['sig3'])]).to(device))
        self._stats_ready = True

    # ── Per-sample risk components ────────────────────────────────────────────

    def _corruption_score(self, feat, mu, sig):
        """
        S_k(x) = max(0, ||feat - mu_k||_2 / ref_k  -  1)

        sig_k here stores the mean L2 distance from the training mean (ref_k).
        S_k = 0  if the distance is at most the mean training distance ref_k
        S_k > 0  if it is larger (this includes many clean inputs, since
                 ref_k is a mean; S_k is a distance score, not an OOD label)
        Shape: (B,)
        """
        dist = (feat - mu).norm(dim=1)              # (B,)
        return ((dist / sig.clamp(min=1e-6)) - 1.0).clamp(min=0.0)

    def _risk(self, logits, feat, mu, sig, cost_k, lambda1, lambda2):
        """
        R_k(x) = U_k + lambda1 * S_k + lambda2 * C_k
        Shape: (B,)
        """
        U = _normalised_entropy(logits)
        S = self._corruption_score(feat, mu, sig)
        return U + lambda1 * S + lambda2 * cost_k

    # ── Unified inference ────────────────────────────────────────────────────

    @torch.no_grad()
    def forward_inference(self, x: torch.Tensor,
                          method: str  = 'risk',
                          tau:    float = 0.5,
                          lambda1: float = config.LAMBDA1,
                          lambda2: float = config.LAMBDA2):
        """
        Run inference with early exit.

        Parameters
        ----------
        x        : input batch (B, C, H, W)
        method   : 'risk' | 'entropy' | 'confidence' | 'full'
        tau      : exit threshold
        lambda1  : weight for S_k  (only used when method='risk')
        lambda2  : weight for C_k  (only used when method='risk')

        Returns
        -------
        predictions : (B,) int64 — predicted class indices
        exit_idx    : (B,) int64 — exit taken (1, 2, or 3)
        info        : list of dicts with per-sample diagnostic values
                      {'entropy', 'confidence', 'corruption_score', 'risk',
                       'exit', 'logits'}
        """
        self.eval()
        B = x.size(0)
        predictions = torch.zeros(B, dtype=torch.long,  device=x.device)
        exit_idx    = torch.zeros(B, dtype=torch.long,  device=x.device)
        info        = [None] * B

        # Process sample-by-sample to allow different exit points
        for i in range(B):
            xi = x[i].unsqueeze(0)   # (1, C, H, W)

            xi = self.stem(xi)
            xi = self.layer1(xi)

            # ── Exit 1 ───────────────────────────────────────────────────────
            h2 = self.layer2(xi)
            f1 = self.avgpool(h2).flatten(1)         # (1, 128)
            l1 = self.exit1(f1)
            should_exit = self._check_exit(l1, f1, self.mu1, self.sig1,
                                           self.cost[0], method, tau,
                                           lambda1, lambda2)
            if should_exit:
                predictions[i], exit_idx[i], info[i] = \
                    self._make_record(l1, f1, self.mu1, self.sig1,
                                     self.cost[0], 1, lambda1, lambda2)
                continue

            # ── Exit 2 ───────────────────────────────────────────────────────
            h3 = self.layer3(h2)
            f2 = self.avgpool(h3).flatten(1)         # (1, 256)
            l2 = self.exit2(f2)
            should_exit = self._check_exit(l2, f2, self.mu2, self.sig2,
                                           self.cost[1], method, tau,
                                           lambda1, lambda2)
            if should_exit:
                predictions[i], exit_idx[i], info[i] = \
                    self._make_record(l2, f2, self.mu2, self.sig2,
                                     self.cost[1], 2, lambda1, lambda2)
                continue

            # ── Final exit ───────────────────────────────────────────────────
            h4 = self.layer4(h3)
            f3 = self.avgpool(h4).flatten(1)         # (1, 512)
            l3 = self.final_fc(f3)
            predictions[i], exit_idx[i], info[i] = \
                self._make_record(l3, f3, self.mu3, self.sig3,
                                  self.cost[2], 3, lambda1, lambda2)

        return predictions, exit_idx, info

    def _check_exit(self, logits, feat, mu, sig, cost_k,
                    method, tau, lambda1, lambda2):
        """Return True if this sample should exit here."""
        if method == 'full':
            return False
        if method == 'confidence':
            return _max_confidence(logits).item() >= tau
        if method == 'entropy':
            return _normalised_entropy(logits).item() <= tau
        if method == 'risk':
            r = self._risk(logits, feat, mu, sig, cost_k, lambda1, lambda2)
            return r.item() <= tau
        raise ValueError(f"Unknown method: {method}")

    def _make_record(self, logits, feat, mu, sig, cost_k, exit_k,
                     lambda1, lambda2):
        """Package prediction and diagnostics for one sample."""
        pred = logits.argmax(dim=1).item()
        U    = _normalised_entropy(logits).item()
        conf = _max_confidence(logits).item()
        S    = self._corruption_score(feat, mu, sig).item()
        R    = U + lambda1 * S + lambda2 * cost_k

        record = {
            'entropy':           U,
            'confidence':        conf,
            'corruption_score':  S,
            'risk':              R,
            'cost':              cost_k,
            'exit':              exit_k,
            'logits':            logits.squeeze(0).cpu(),
        }
        return pred, exit_k, record

    # ── Convenience: full-model logits (no early exit) ───────────────────────

    @torch.no_grad()
    def predict_full(self, x: torch.Tensor):
        """Full-network predictions without any early exit."""
        self.eval()
        f1, f2, f3 = self._features(x)
        return self.final_fc(f3)


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_model(device, load_baseline: bool = False,
                load_ee: bool = False,
                load_stats: bool = False) -> EarlyExitResNet:
    """
    Build and optionally load checkpoints.

    Parameters
    ----------
    load_baseline : load the clean ResNet18 baseline weights into the backbone
    load_ee       : load the full early-exit model checkpoint
    load_stats    : load pre-computed feature statistics
    """
    model = EarlyExitResNet().to(device)

    if load_baseline and not load_ee:
        ckpt = torch.load(config.BASELINE_CKPT, map_location=device, weights_only=True)

        # The baseline ResNet18 uses flat naming (conv1, bn1, fc).
        # EarlyExitResNet packs conv1+bn1+relu+maxpool into self.stem (a Sequential),
        # and renames fc -> final_fc.  We remap explicitly so no weights are dropped.
        remap = {}
        for k, v in ckpt.items():
            if k.startswith('conv1.'):
                remap['stem.0.' + k[len('conv1.'):]] = v
            elif k.startswith('bn1.'):
                remap['stem.1.' + k[len('bn1.'):]] = v
            elif k.startswith('fc.'):
                remap['final_fc.' + k[len('fc.'):]] = v
            else:
                remap[k] = v   # layer1-4, avgpool — names match directly

        missing, unexpected = model.load_state_dict(remap, strict=False)
        # Only exit1 and exit2 heads should be missing (they are new)
        print(f"[model] Loaded baseline weights. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"         Missing keys (expected = exit heads only): "
                  f"{[k for k in missing]}")

    if load_ee:
        ckpt = torch.load(config.EE_CKPT, map_location=device, weights_only=True)
        # Strip thop profiling buffers (total_ops / total_params) if present
        ckpt = {k: v for k, v in ckpt.items()
                if not k.endswith('total_ops') and not k.endswith('total_params')}
        model.load_state_dict(ckpt)
        print(f"[model] Loaded early-exit checkpoint -> {config.EE_CKPT}")

    if load_stats:
        model.load_feature_stats(device)
        print(f"[model] Feature stats loaded -> {config.FEATURE_STATS_PATH}")

    return model
