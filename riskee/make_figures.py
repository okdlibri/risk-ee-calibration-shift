"""Generate the results figures of the paper from results/*_main.json and the caches.

Every figure is drawn from the JSON files in results/ (the reliability diagrams from
results/cache_stats.json, written by riskee/cache_stats.py).
Output: figures/*.pdf|png
Colours: fixed categorical order (method -> colour never changes between figures), plus
distinct markers / line styles so that identity does not rely on colour alone.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

R = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)
L = lambda f: json.load(open(os.path.join(R, f)))

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
STYLE = {  # method: (label, colour, marker, linestyle)
    'risk':      ('Risk-EE (ours)', '#2a78d6', 'o', '-'),
    'conf':      ('Confidence', '#eb6834', 's', '-'),
    'entropy':   ('Entropy', '#1baf7a', '^', '-'),
    'pets_conf': ('Per-exit TS + conf.', '#eda100', 'D', '-'),
    'ts_full':   ('Full network + TS', '#e87ba4', 'v', '--'),
    'full':      ('Full network', '#8a8984', 'x', ':'),
}
ARCH = {'cnn': 'ResNet-18 / CIFAR-10', 'vit': 'DeiT-S / CIFAR-100', 'tiny': 'ResNet-18 / Tiny-ImageNet'}
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def grid(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('figure', name)


def line(ax, x, y, m, **kw):
    lab, col, mk, ls = STYLE[m]
    ax.plot(x, y, color=col, marker=mk, ls=ls, lw=1.6 if m == 'risk' else 1.2, ms=4.5, label=lab,
            zorder=3 if m == 'risk' else 2, **kw)


# ── 1. budget curves ────────────────────────────────────────────────────────
def fig_budget():
    B = L('budget_sweep_main.json')
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    for row, tag in enumerate(['cnn', 'vit']):
        v = B[tag]['budgets']; bs = sorted(v, key=float); x = [float(b) for b in bs]
        ax = axes[row, 0]
        for m in ['conf', 'entropy', 'pets_conf', 'risk']:
            line(ax, x, [v[b][m]['ece'] for b in bs], m)
        ax.set_ylabel('Clean ECE'); ax.set_title(f'{ARCH[tag]}: ECE at each budget'); grid(ax)
        ax = axes[row, 1]
        for key, m in [('d_conf', 'conf'), ('d_entropy', 'entropy')]:
            d = np.array([v[b]['risk'][key] for b in bs])
            lab, col, mk, _ = STYLE[m]
            ax.fill_between(x, d[:, 1], d[:, 2], color=col, alpha=0.18, lw=0)
            ax.plot(x, d[:, 0], color=col, marker=mk, lw=1.4, ms=4.5, label=f'Risk-EE $-$ {lab}')
        ax.axhline(0, color=INK2, lw=0.8)
        sel = 0.80 if tag == 'cnn' else 0.70
        for a in axes[row]:
            a.axvline(sel, color=INK2, lw=0.8, ls='--')
        ax.set_ylabel('Paired ECE difference'); ax.set_title(f'{ARCH[tag]}: difference, 95% CI'); grid(ax)
        for a in axes[row]:
            a.set_xlabel('FLOPs budget (fraction of full network)')
    axes[0, 0].legend(frameon=False, loc='upper right'); axes[0, 1].legend(frameon=False, loc='lower left')
    fig.tight_layout(); save(fig, 'fig_budget')


def fig_budget_seeds():
    B = L('budget_sweep_main.json')
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    for ax, tag, run in zip(axes, ['cnn_seed42', 'cnn_seed123', 'cnn_seed456'], (2, 3, 4)):
        v = B[tag]['budgets']; bs = sorted(v, key=float); x = [float(b) for b in bs]
        for key, m in [('d_conf', 'conf'), ('d_entropy', 'entropy'), ('d_pets_conf', 'pets_conf')]:
            d = np.array([v[b]['risk'][key] for b in bs]); lab, col, mk, _ = STYLE[m]
            ax.fill_between(x, d[:, 1], d[:, 2], color=col, alpha=0.18, lw=0)
            ax.plot(x, d[:, 0], color=col, marker=mk, lw=1.3, ms=4, label=f'Risk-EE $-$ {lab}')
        ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0.80, color=INK2, lw=0.8, ls='--')
        ax.set_title(f'ResNet-18, run {run}'); ax.set_xlabel('FLOPs budget'); grid(ax)
    axes[0].set_ylabel('Paired clean ECE difference'); axes[0].legend(frameon=False, loc='upper left', fontsize=6.5)
    fig.tight_layout(); save(fig, 'fig_budget_seeds')


# ── 2. severity: ECE, FLOPs, final-exit share ───────────────────────────────
def fig_severity():
    tags = ['cnn', 'vit', 'tiny']
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.2))
    for j, tag in enumerate(tags):
        d = L(f'{tag}_main.json'); P = d['corrupt_per_set']; types = sorted(P['risk'])
        sev = [0, 1, 2, 3, 4, 5]
        def series(m, f):
            return [f(d['clean'][m])] + [np.mean([f(P[m][t][str(s)]) for t in types]) for s in sev[1:]]
        for m in ['full', 'ts_full', 'conf', 'entropy', 'pets_conf', 'risk']:
            line(axes[0, j], sev, series(m, lambda r: r['ece']), m)
        for m in ['conf', 'entropy', 'pets_conf', 'risk']:
            line(axes[1, j], sev, series(m, lambda r: r['flops']), m)
            line(axes[2, j], sev, series(m, lambda r: r['exit_dist'][2]), m)
        axes[0, j].set_title(ARCH[tag])
        for i in range(3):
            grid(axes[i, j]); axes[i, j].set_xticks(sev); axes[i, j].set_xticklabels(['clean', 1, 2, 3, 4, 5])
        axes[2, j].set_xlabel('Corruption severity')
    axes[0, 0].set_ylabel('ECE'); axes[1, 0].set_ylabel('FLOPs (fraction)'); axes[2, 0].set_ylabel('Share at final exit')
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, 'fig_severity')


# ── 3. per-type mCE relative to the full network ────────────────────────────
def fig_pertype():
    tags = ['cnn', 'vit', 'tiny']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.6))
    for ax, tag in zip(axes, tags):
        d = L(f'{tag}_main.json'); P = d['corrupt_per_set']; types = sorted(P['risk'])
        mce = lambda m, t: np.mean([1 - P[m][t][s]['acc'] for s in P[m][t]])
        y = np.arange(len(types)); h = 0.38
        for off, m in [(-h / 2, 'conf'), (h / 2, 'risk')]:
            lab, col, _, _ = STYLE[m]
            ax.barh(y + off, [100 * (mce(m, t) - mce('full', t)) for t in types], height=h - 0.04, color=col, label=lab)
        ax.axvline(0, color=INK2, lw=0.8); ax.axvline(0.5, color=INK2, lw=0.6, ls=':')
        ax.set_yticks(y); ax.set_yticklabels([t.replace('_', ' ') for t in types], fontsize=6.5); ax.invert_yaxis()
        ax.set_title(ARCH[tag]); ax.set_xlabel('mCE $-$ full-network mCE (pp)'); grid(ax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, 'fig_pertype_mce')


# ── 4. lambda grid on validation ────────────────────────────────────────────
def fig_lambda():
    tags = ['cnn', 'vit', 'tiny']; G = [0.1, 0.3, 0.5, 0.7, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, tag in zip(axes, tags):
        d = L(f'{tag}_main.json'); M = np.full((5, 5), np.nan)
        for q in d['lambda_grid_val']:
            M[G.index(q['l2']), G.index(q['l1'])] = q['val']['ece']
        im = ax.imshow(M, cmap='Blues_r', origin='lower')
        for i in range(5):
            for k in range(5):
                v = M[i, k]; ax.text(k, i, f'{v:.3f}', ha='center', va='center', fontsize=5.6,
                                     color='white' if v < np.nanmin(M) + 0.4 * (np.nanmax(M) - np.nanmin(M)) else INK)
        s = d['selection']['risk']
        ax.add_patch(plt.Rectangle((G.index(s['l1']) - 0.5, G.index(s['l2']) - 0.5), 1, 1, fill=False, ec='#eb6834', lw=1.8))
        ax.set_xticks(range(5)); ax.set_xticklabels(G); ax.set_yticks(range(5)); ax.set_yticklabels(G)
        ax.set_xlabel(r'$\lambda_1$'); ax.set_ylabel(r'$\lambda_2$'); ax.set_title(ARCH[tag])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    fig.tight_layout(); save(fig, 'fig_lambda_val')


# ── 5. reliability diagrams (recomputed from caches, checked against JSON) ──
def fig_reliability():
    tags = ['cnn', 'vit']; ms = ['full', 'conf', 'pets_conf', 'risk']; nb = 15
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.7), sharex=True, sharey=True)
    for i, tag in enumerate(tags):
        d = L(f'{tag}_main.json'); rel = L('cache_stats.json')['reliability'][tag]
        for j, m in enumerate(ms):
            e = rel[m]['ece']
            assert abs(e - d['clean'][m]['ece']) < 1e-9, (tag, m, e, d['clean'][m]['ece'])
            edges = np.linspace(0, 1, nb + 1)
            ax = axes[i, j]; lab, col, _, _ = STYLE[m]
            accs = [np.nan if a is None else a for a in rel[m]['acc']]
            ax.bar(edges[:-1], np.nan_to_num(accs), width=1 / nb, align='edge', color=col, alpha=0.85, ec='white', lw=0.5)
            ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls='--')
            ax.set_title(f'{lab}\nECE {e:.4f}', fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f'{ARCH[tag].split(" /")[0]}\nAccuracy')
            if i == 1:
                ax.set_xlabel('Confidence')
    fig.tight_layout(); save(fig, 'fig_reliability')


if __name__ == '__main__':
    fig_budget(); fig_budget_seeds(); fig_severity(); fig_pertype(); fig_lambda(); fig_reliability()
