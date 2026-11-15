"""
python -m riskee.figures_sk -- S_k distribution figures (both architectures),
generated only from the cached model outputs.

Output: figures/sk_distribution_{vit,cnn}.pdf and .png
Palette: reference categorical slots 1-2 (blue #2a78d6, orange #eb6834; a
validated adjacent pair) for the two corruption families, neutral gray for clean.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from riskee.analysis import DS, load_ref
from riskee.common import CACHE_DIR

PNG = IMG = os.path.join(ROOT, 'figures')
os.makedirs(IMG, exist_ok=True)

INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#8a8984'
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
})


def save(fig, name):
    fig.savefig(os.path.join(IMG, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(PNG, name + '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('wrote', name)


def sk_figure(tag, name, title):
    """Share of inputs with S_k > 0 (top) and mean S_k (bottom), exits 1-2,
    for clean test data and Gaussian noise / fog at severities 1, 3, 5."""
    ref = load_ref(tag); d = os.path.join(CACHE_DIR, tag)
    conds = [('clean', 'test', GRAY)] + \
            [(f'noise {s}', f'c_gaussian_noise_{s}', BLUE) for s in (1, 3, 5)] + \
            [(f'fog {s}', f'c_fog_{s}', ORANGE) for s in (1, 3, 5)]
    S = {lab: DS(os.path.join(d, f + '.npz'), ref).S['dist'] for lab, f, _ in conds}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), sharex=True)
    x = np.arange(len(conds))
    for k in range(2):
        pos = [100 * (S[lab][k] > 0).mean() for lab, _, _ in conds]
        mean = [S[lab][k].mean() for lab, _, _ in conds]
        for row, vals, ylab in ((0, pos, r'inputs with $S_k>0$ (%)'), (1, mean, r'mean $S_k$')):
            ax = axes[row, k]
            ax.bar(x, vals, width=0.62, color=[c for _, _, c in conds], edgecolor='white', linewidth=1.0, zorder=2)
            ax.axhline(vals[0], color=GRAY, lw=0.8, ls='--', zorder=1)   # clean reference
            ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
            ax.set_ylabel(ylab if k == 0 else '')
            if row == 0:
                ax.set_ylim(0, 100); ax.set_title(f'exit {k + 1}', color=INK)
            else:
                ax.set_ylim(0, max(0.05, 1.15 * max(mean)))
            ax.set_xticks(x); ax.set_xticklabels([lab for lab, _, _ in conds], rotation=0)
    for row in range(2):
        lo = min(axes[row, 0].get_ylim()[0], axes[row, 1].get_ylim()[0])
        hi = max(axes[row, 0].get_ylim()[1], axes[row, 1].get_ylim()[1])
        for k in range(2): axes[row, k].set_ylim(lo, hi)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GRAY, BLUE, ORANGE)]
    fig.legend(handles, ['clean', 'Gaussian noise (severity)', 'fog (severity)'], loc='upper center',
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(title, y=1.09, fontsize=8.5, color=INK)
    fig.tight_layout()
    save(fig, name)


if __name__ == '__main__':
    what = sys.argv[1:] or ['sk']
    if 'sk' in what:
        sk_figure('vit', 'sk_distribution_vit', 'DeiT-S / CIFAR-100')
        sk_figure('cnn', 'sk_distribution_cnn', 'ResNet-18 / CIFAR-10')
