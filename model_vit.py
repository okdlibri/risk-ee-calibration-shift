"""
model_vit.py
EarlyExitDeiT: DeiT-Small with 3 exit points (after transformer blocks 4, 8, 12).
Each exit uses the CLS token and the same risk function as the CNN model.

Risk:  R_k(x) = U_k(x) + λ1·S_k(x) + λ2·C_k
  U_k = normalised entropy
  S_k = max(0, ||cls_k - μ_k|| / mean_dist_k  - 1)
  C_k = relative FLOPs at exit k
"""

import math
import torch
import torch.nn as nn
import numpy as np
import timm

import config_vit as cfg


# ─── Exit head ────────────────────────────────────────────────────────────────

class ExitHead(nn.Module):
    """Single-layer classifier on the CLS token."""
    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.norm  = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(embed_dim, num_classes)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.norm(cls_token)))


# ─── EarlyExitDeiT ────────────────────────────────────────────────────────────

class EarlyExitDeiT(nn.Module):
    """
    Wraps DeiT-Small and inserts exit heads after transformer blocks 4, 8, 12.
    Block indexing is 0-based: exits after blocks[3], blocks[7], blocks[11].
    The final exit uses the standard timm head.
    """

    EXIT_IDX = [3, 7, 11]   # 0-based indices of the blocks *after which* we exit

    def __init__(self, num_classes: int = cfg.NUM_CLASSES,
                 embed_dim: int = cfg.EMBED_DIM,
                 pretrained: bool = False):
        super().__init__()
        # Load backbone (without classification head — we attach our own)
        backbone = timm.create_model(
            'deit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0,           # removes the original head
        )
        self.patch_embed = backbone.patch_embed
        self.cls_token   = backbone.cls_token
        self.dist_token  = getattr(backbone, 'dist_token', None)
        self.pos_embed   = backbone.pos_embed
        self.pos_drop    = backbone.pos_drop
        self.blocks      = backbone.blocks    # nn.Sequential of 12 blocks
        self.norm        = backbone.norm      # final LN before original head

        self.embed_dim   = embed_dim
        self.num_classes = num_classes

        # Exit heads (after block 4, 8, 12)
        self.exit1 = ExitHead(embed_dim, num_classes)
        self.exit2 = ExitHead(embed_dim, num_classes)
        self.final_head = nn.Linear(embed_dim, num_classes)

        # Feature statistics buffers (filled by compute_feature_stats)
        for i in range(1, 4):
            self.register_buffer(f'mu_{i}',  torch.zeros(embed_dim))
            self.register_buffer(f'sig_{i}', torch.tensor(1.0))   # scalar

        # Relative FLOPs (updated after profiling)
        self.register_buffer('rel_flops',
                             torch.tensor(cfg.RELATIVE_FLOPS, dtype=torch.float32))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _corruption_score(self, feat: torch.Tensor,
                          mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        dist = (feat - mu).norm(dim=1)
        return ((dist / sig.clamp(min=1e-6)) - 1.0).clamp(min=0.0)

    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1).clamp(min=1e-9)
        raw   = -(probs * probs.log()).sum(dim=1)
        return raw / math.log(self.num_classes)   # normalise to [0,1]

    def _risk(self, logits: torch.Tensor, feat: torch.Tensor,
              mu: torch.Tensor, sig: torch.Tensor,
              c_k: float) -> torch.Tensor:
        u = self._entropy(logits)
        s = self._corruption_score(feat, mu, sig)
        return u + cfg.LAMBDA1 * s + cfg.LAMBDA2 * c_k

    # ── forward passes ────────────────────────────────────────────────────────

    def _embed(self, x: torch.Tensor):
        """Patch embed + CLS + pos embed."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        if self.dist_token is not None:
            dist = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls, dist, x], dim=1)
        else:
            x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

    def forward(self, x: torch.Tensor):
        """Training forward — returns (logits1, logits2, logits_final)."""
        x = self._embed(x)
        logits_list = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.EXIT_IDX[0]:
                logits_list.append(self.exit1(x[:, 0]))
            elif i == self.EXIT_IDX[1]:
                logits_list.append(self.exit2(x[:, 0]))
        x_norm = self.norm(x)
        logits_list.append(self.final_head(x_norm[:, 0]))
        return tuple(logits_list)

    @torch.no_grad()
    def forward_inference(self, x: torch.Tensor, method: str,
                          threshold: float):
        """
        Batch-aware inference with per-sample early exit.
        Each sample independently decides which exit to use.
        Returns (logits, exit_indices) where:
            logits       : (B, num_classes) float32
            exit_indices : (B,) int64, values in {1, 2, 3}
        """
        B      = x.shape[0]
        x_seq  = self._embed(x)
        dev    = x.device

        out_logits = torch.zeros(B, self.num_classes, device=dev,
                                 dtype=torch.float32)
        out_exits  = torch.zeros(B, dtype=torch.long, device=dev)
        done       = torch.zeros(B, dtype=torch.bool, device=dev)

        mus    = [self.mu_1, self.mu_2, self.mu_3]
        sigs   = [self.sig_1, self.sig_2, self.sig_3]
        c_vals = self.rel_flops.tolist()
        heads  = [self.exit1, self.exit2]

        for i, blk in enumerate(self.blocks):
            x_seq = blk(x_seq)

            for exit_pos, block_idx in enumerate(self.EXIT_IDX[:2]):
                if i != block_idx or method == 'full':
                    continue

                active_idx = (~done).nonzero(as_tuple=True)[0]
                if active_idx.numel() == 0:
                    break

                cls    = x_seq[active_idx, 0]
                logits = heads[exit_pos](cls).float()

                if method == 'confidence':
                    score = torch.softmax(logits, dim=1).max(dim=1).values
                    mask  = score >= threshold
                elif method == 'entropy':
                    score = self._entropy(logits)
                    mask  = score <= threshold
                elif method == 'risk':
                    score = self._risk(logits, cls,
                                       mus[exit_pos], sigs[exit_pos],
                                       c_vals[exit_pos])
                    mask  = score <= threshold
                else:
                    mask = torch.zeros(active_idx.numel(),
                                       dtype=torch.bool, device=dev)

                leaving = active_idx[mask]
                if leaving.numel() > 0:
                    out_logits[leaving] = logits[mask]
                    out_exits[leaving]  = exit_pos + 1
                    done[leaving]       = True

        # Remaining samples (or all samples for 'full') → final exit
        remaining = (~done).nonzero(as_tuple=True)[0]
        if remaining.numel() > 0:
            x_norm = self.norm(x_seq)
            cls    = x_norm[remaining, 0]
            logits = self.final_head(cls).float()
            out_logits[remaining] = logits
            out_exits[remaining]  = 3

        return out_logits, out_exits

    @torch.no_grad()
    def compute_feature_stats(self, loader, device):
        """
        Compute per-exit CLS-token statistics on a clean validation set.
        Stores mean vector μ_k and mean L2-distance sig_k for each exit.
        """
        feats = {1: [], 2: [], 3: []}

        self.eval()
        for imgs, _ in loader:
            imgs = imgs.to(device)
            x = self._embed(imgs)
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i == self.EXIT_IDX[0]:
                    feats[1].append(x[:, 0].cpu())
                elif i == self.EXIT_IDX[1]:
                    feats[2].append(x[:, 0].cpu())
            x_norm = self.norm(x)
            feats[3].append(x_norm[:, 0].cpu())

        def _stats(F):
            mu  = F.mean(0)
            d   = (F - mu).norm(dim=1)
            ref = d.mean().clamp(min=1e-6)
            return mu, ref

        stats = {}
        for k in [1, 2, 3]:
            F = torch.cat(feats[k], dim=0)
            mu, sig = _stats(F)
            setattr(self, f'mu_{k}',  mu.to(device))
            setattr(self, f'sig_{k}', sig.to(device))
            stats[f'mu_{k}']  = mu.numpy()
            stats[f'sig_{k}'] = sig.numpy()

        return stats


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_model(device, load_ee_ckpt: bool = False,
                load_baseline_ckpt: bool = False) -> EarlyExitDeiT:
    model = EarlyExitDeiT(
        num_classes=cfg.NUM_CLASSES,
        embed_dim=cfg.EMBED_DIM,
        pretrained=False,
    ).to(device)

    if load_ee_ckpt:
        ckpt = torch.load(cfg.EE_CKPT, map_location=device, weights_only=True)
        ckpt = {k: v for k, v in ckpt.items()
                if not k.endswith('total_ops') and not k.endswith('total_params')}
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f'[model_vit] Loaded EE checkpoint | '
              f'Missing: {len(missing)}, Unexpected: {len(unexpected)}')

    elif load_baseline_ckpt:
        ckpt = torch.load(cfg.BASELINE_CKPT, map_location=device, weights_only=True)
        ckpt = {k: v for k, v in ckpt.items()
                if not k.endswith('total_ops') and not k.endswith('total_params')}
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f'[model_vit] Loaded baseline checkpoint | '
              f'Missing: {len(missing)}, Unexpected: {len(unexpected)}')

    return model


def save_feature_stats(stats: dict):
    np.savez(cfg.FEATURE_STATS_PATH, **stats)
    print(f'[model_vit] Feature stats saved → {cfg.FEATURE_STATS_PATH}')


def load_feature_stats(model: EarlyExitDeiT, device):
    data = np.load(cfg.FEATURE_STATS_PATH)
    for k in [1, 2, 3]:
        setattr(model, f'mu_{k}',
                torch.tensor(data[f'mu_{k}'], dtype=torch.float32).to(device))
        setattr(model, f'sig_{k}',
                torch.tensor(data[f'sig_{k}'], dtype=torch.float32).to(device))
    print(f'[model_vit] Feature stats loaded from {cfg.FEATURE_STATS_PATH}')
