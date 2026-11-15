"""
python -m riskee.protocol --tag {cnn,cnn_seed42,cnn_seed123,cnn_seed456,vit,tiny} [--boot 2000]

Main evaluation protocol.  Every choice is made on the held-out validation
split and then frozen:

  * temperatures (full-network TS and per-exit TS) are fit by NLL on validation;
  * matched compute: for each threshold-based method, tau is chosen on a fine
    grid as the value whose VALIDATION average FLOPs is the largest value not
    exceeding the budget (ResNet-18: 0.80, DeiT-S: 0.70 of full-network FLOPs);
    ties -> higher validation accuracy;
  * Risk-EE weights (lambda1, lambda2) are chosen on the grid
    {0.1, 0.3, 0.5, 0.7, 1.0}^2 to minimise validation ECE at the selected tau
    (ties -> higher validation accuracy, then smaller (l1, l2)); every Risk-EE
    variant selects its free weight(s) by the same rule;
  * the selected configuration is applied unchanged to the clean test set and
    to every corruption set; FLOPs are measured per set from the actual exit
    indices;
  * 95% CIs: percentile bootstrap over test images (2000 resamples; the same
    resampled image indices in all corruption sets, since each is a corrupted
    copy of the same images); paired differences against Risk-EE use the same
    resamples.

Output: results/<tag>_main.json
"""
import os, sys, json, argparse, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from riskee.analysis import (DS, load_ref, run, summarize, ece, ece_boot, fit_T,
                         ens_probs, arch_of, COSTS, method_scores)
from riskee.common import CACHE_DIR, CORRUPTIONS, SEVERITIES, ROOT

CONF_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
ENT_GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
RISK_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
WIN_GRID = [round(x, 3) for x in np.linspace(0.02, 0.50, 15)]
LAM1_GRID = [0.1, 0.3, 0.5, 0.7, 1.0]
LAM2_GRID = [0.1, 0.3, 0.5, 0.7, 1.0]
# matched-compute budget (fraction of full-network FLOPs), set per architecture in evaluate()
BUDGETS = {'cnn': 0.80, 'tiny': 0.80, 'vit': 0.70}
BUDGET = 0.70


# ── window-based cascade (Xia & Bouganis, ICCV 2023, Alg. 2) ────────────────
def _ens(ds, T=None):
    """exit-ensemble probabilities; optionally of per-exit temperature-scaled exits"""
    if T is None:
        return ens_probs(ds)
    P = [ds.scaled_probs(k, T[k]) for k in range(3)]
    return [np.mean(P[:k + 1], axis=0) for k in range(3)]


def window_fit(val, coverage=0.8, T=None):
    """theta_k = MSP-uncertainty quantile at the target coverage on val;
    window = [q(cov-w), q(cov+w)] of val uncertainties at stage k."""
    P = _ens(val, T)
    return [np.sort(-P[k].max(1)) for k in range(2)], coverage, T


def run_window(ds, wfit, w):
    Us, cov, T = wfit
    P = _ens(ds, T)
    n = ds.n; ex = np.full(n, 2); done = np.zeros(n, bool)
    for k in range(2):
        t1 = np.quantile(Us[k], max(0.0, cov - w)); t2 = np.quantile(Us[k], min(1.0, cov + w))
        U = -P[k].max(1)
        out = ((U < t1) | (U > t2)) & ~done
        ex[out] = k; done |= out
    probs = np.where((ex == 0)[:, None], P[0], np.where((ex == 1)[:, None], P[1], P[2]))
    return probs, ex


# ── method table ────────────────────────────────────────────────────────────
def select_lambda(val, costs, score='dist', T=None, g1=None, g2=None):
    """Validation lambda rule: argmin val ECE of Risk-EE at its val-selected tau (matched rule);
    ties -> higher val acc, then smaller (l1, l2)."""
    g1 = LAM1_GRID if g1 is None else g1
    g2 = LAM2_GRID if g2 is None else g2
    best = None
    for a, b in itertools.product(g1, g2):
        p = {'l1': a, 'l2': b, 'score': score, 'costs': costs, 'T': T}
        t, _ = select_tau(val, 'risk', p, RISK_GRID, costs)
        s = summarize(apply(val, 'risk', p, t, costs), val.y, costs)
        key = (round(s['ece'], 6), -s['acc'], a, b)
        if best is None or key < best[0]:
            best = (key, p)
    return best[1]


def methods(arch, T_full, T_exit, wfit, val):
    c = COSTS[arch]
    # every Risk-EE variant selects its free lambda(s) on validation by the same rule
    sl = lambda score, T=None, g1=None, g2=None: select_lambda(val, c, score, T, g1, g2)
    return {
        'full':        ('full', {}, [0.0]),
        'ts_full':     ('ts_full', {'T': T_full}, [0.0]),
        'conf':        ('conf', {}, CONF_GRID),
        'entropy':     ('entropy', {}, ENT_GRID),
        'pets_conf':   ('pets_conf', {'T': T_exit}, CONF_GRID),
        'window':      ('window', {'wfit': wfit[0]}, WIN_GRID),
        'window_ts':   ('window', {'wfit': wfit[1]}, WIN_GRID),
        'risk':        ('risk', sl('dist'), RISK_GRID),
        'risk_maha':   ('risk', sl('maha'), RISK_GRID),
        'risk_energy': ('risk', sl('energy'), RISK_GRID),
        'risk_noS':    ('risk', sl('dist', g1=[0.0]), RISK_GRID),
        'risk_noC':    ('risk', sl('dist', g2=[0.0]), RISK_GRID),
        'risk_pets':   ('risk', sl('dist', T=T_exit), RISK_GRID),
    }


def apply(ds, m, p, t, costs):
    if m == 'window':
        return run_window(ds, p['wfit'], t)
    return run(ds, m, p, t, costs)


def fine_grid(val, m, p, grid):
    if m in ('full', 'ts_full'):
        return [0.0]
    if m == 'window':
        return sorted(set(np.linspace(0.0, 1.0, 401).tolist()) | set(grid))
    sc, _, _ = method_scores(val, m, p)
    allv = np.concatenate(sc[:2])
    qs = np.quantile(allv, np.linspace(0, 1, 1601))
    return sorted(set(qs.tolist()) | set(grid) | {float(allv.min()) - 1e-9, float(allv.max()) + 1e-9})


def select_tau(val, m, p, grid, costs):
    """Returns (tau, rows on the coarse reporting grid).
    tau is chosen on a fine grid as the value whose VALIDATION avg FLOPs is the
    largest value <= BUDGET (ties -> higher val accuracy): matched compute."""
    rows = [(t, summarize(apply(val, m, p, t, costs), val.y, costs)) for t in grid]
    if m in ('full', 'ts_full'):
        cand = [r for r in rows if r[1]['flops'] <= BUDGET + 1e-9] or rows
        return max(cand, key=lambda r: r[1]['acc'])[0], rows
    best = None
    for t in fine_grid(val, m, p, grid):
        s = summarize(apply(val, m, p, t, costs), val.y, costs)
        if s['flops'] <= BUDGET + 1e-9:
            key = (round(s['flops'], 6), s['acc'])
            if best is None or key > best[0]:
                best = (key, t)
    return best[1], rows


def load_sets(tag):
    d = os.path.join(CACHE_DIR, tag); ref = load_ref(tag)
    return d, ref, DS(os.path.join(d, 'val.npz'), ref), DS(os.path.join(d, 'test.npz'), ref)


def boot_ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def evaluate(tag, n_boot=2000, seed=0, corrupt=True, extra_tags=()):
    global BUDGET
    arch = arch_of(tag); costs = COSTS[arch]; BUDGET = BUDGETS[arch]
    d, ref, val, test = load_sets(tag)
    T_full = fit_T(val.logits[2], val.y)
    T_exit = [fit_T(val.logits[k], val.y) for k in range(3)]
    wfit = (window_fit(val), window_fit(val, T=T_exit))
    M = methods(arch, T_full, T_exit, wfit, val)
    sel, sweeps = {}, {}
    for name, (m, p, grid) in M.items():
        t, rows = select_tau(val, m, p, grid, costs)
        sel[name] = (m, p, t)
        sweeps[name] = {'val': [(float(tt), s) for tt, s in rows],
                        'test': [(float(tt), summarize(apply(test, m, p, tt, costs), test.y, costs)) for tt in grid]}
    # separately trained SDN model (confidence routing), if cached
    sdn_tag = {'cnn': 'cnn_sdn', 'vit': 'vit_sdn'}.get(tag)
    sdn = None
    # use SDN only when its cache is complete (last corruption set written)
    if sdn_tag and os.path.exists(os.path.join(CACHE_DIR, sdn_tag, 'c_saturate_5.npz')):
        _, sref, sval, stest = load_sets(sdn_tag)
        t, rows = select_tau(sval, 'conf', {}, CONF_GRID, costs)
        sdn = (sdn_tag, sref, stest, t)
        sweeps['sdn'] = {'val': [(float(tt), s) for tt, s in rows],
                         'test': [(float(tt), summarize(apply(stest, 'conf', {}, tt, costs), stest.y, costs)) for tt in CONF_GRID]}

    res = {'tag': tag, 'protocol': f'val-selected, rule=matched, budget {BUDGET}, lambda=val',
           'T_full': T_full, 'T_exit': T_exit,
           'selection': {n: {'method': m, 'tau': float(t), **{k: v for k, v in p.items() if k in ('l1', 'l2', 'score')}}
                         for n, (m, p, t) in sel.items()},
           'sweeps': sweeps, 'clean': {}, 'clean_ci': {}, 'clean_paired': {}}
    if sdn:
        res['selection']['sdn'] = {'method': 'conf', 'tau': float(sdn[3]), 'model': sdn[0]}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, test.n, size=(n_boot, test.n))
    outs = {n: apply(test, m, p, t, costs) for n, (m, p, t) in sel.items()}
    if sdn:
        outs['sdn'] = apply(sdn[2], 'conf', {}, sdn[3], costs)
    B = {}
    for n, (probs, ex) in outs.items():
        res['clean'][n] = summarize((probs, ex), test.y, costs)
        conf = probs.max(1); cor = (probs.argmax(1) == test.y).astype(float)
        B[n] = {'ece': ece_boot(conf, cor, idx), 'acc': cor[idx].mean(1),
                'flops': np.asarray(costs)[ex][idx].mean(1)}
        res['clean_ci'][n] = {k: boot_ci(v) for k, v in B[n].items()}
    for n in B:
        if n == 'risk': continue
        res['clean_paired'][n] = {k: {'diff': float((B['risk'][k] - B[n][k]).mean()),
                                      'ci': boot_ci(B['risk'][k] - B[n][k])} for k in ('ece', 'acc')}

    if corrupt:
        names = list(outs)
        agg = {n: [] for n in names}
        BB = {n: {'ece': np.zeros(n_boot), 'acc': np.zeros(n_boot), 'flops': np.zeros(n_boot)} for n in names}
        per = {}
        nset = 0
        y_ref = None
        for c in CORRUPTIONS:
            for sv in SEVERITIES:
                path = os.path.join(d, f'c_{c}_{sv}.npz')
                if not os.path.exists(path): continue
                nset += 1
                cs = DS(path, ref)
                if arch != 'tiny':
                    # CIFAR-C sets are corrupted copies of the test images, in order
                    assert (cs.y == test.y).all(), 'corrupted set not aligned with test images'
                    ii = idx
                else:
                    # Tiny-ImageNet-C: all 75 sets are corrupted copies of the SAME
                    # 10k source images in the same order (verified by file names and
                    # pixel correlation), but these are not the clean Tiny-ImageNet val
                    # files.  One shared resample of the source images across all sets.
                    if y_ref is None:
                        y_ref = cs.y
                        idx_c = np.random.default_rng(seed + 1).integers(0, cs.n, size=(n_boot, cs.n))
                    assert (cs.y == y_ref).all(), 'Tiny-ImageNet-C sets not aligned'
                    ii = idx_c
                css = DS(os.path.join(CACHE_DIR, sdn[0], f'c_{c}_{sv}.npz'), sdn[1]) if sdn else None
                for n in names:
                    if n == 'sdn':
                        o = apply(css, 'conf', {}, sdn[3], costs)
                    else:
                        m, p, t = sel[n]; o = apply(cs, m, p, t, costs)
                    s = summarize(o, cs.y, costs)
                    agg[n].append(s); per.setdefault(n, {}).setdefault(c, {})[sv] = s
                    conf = o[0].max(1); cor = (o[0].argmax(1) == cs.y).astype(float)
                    BB[n]['ece'] += ece_boot(conf, cor, ii)
                    BB[n]['acc'] += cor[ii].mean(1)
                    BB[n]['flops'] += np.asarray(costs)[o[1]][ii].mean(1)
        res['n_corrupt_sets'] = nset
        res['corrupt_per_set'] = per
        res['corrupt_mean'] = {n: {k: float(np.mean([s[k] for s in agg[n]])) for k in ('acc', 'ece', 'brier', 'flops')}
                               for n in names}
        for n in names:
            res['corrupt_mean'][n]['mce'] = 1 - res['corrupt_mean'][n]['acc']
            for k in BB[n]: BB[n][k] /= nset
        res['corrupt_ci'] = {n: {k: boot_ci(v) for k, v in BB[n].items()} for n in names}
        res['corrupt_paired'] = {n: {k: {'diff': float((BB['risk'][k] - BB[n][k]).mean()),
                                         'ci': boot_ci(BB['risk'][k] - BB[n][k])} for k in ('ece', 'acc')}
                                 for n in names if n != 'risk'}

    # validation ECE of Risk-EE over the full lambda grid (for the lambda-selection figure)
    lg = []
    for a, b in itertools.product(LAM1_GRID, LAM2_GRID):
        p = {'l1': a, 'l2': b, 'score': 'dist', 'costs': costs, 'T': None}
        t, rows = select_tau(val, 'risk', p, RISK_GRID, costs)
        sv_ = summarize(apply(val, 'risk', p, t, costs), val.y, costs)
        lg.append({'l1': a, 'l2': b, 'tau': t, 'val': sv_})
    res['lambda_grid_val'] = lg
    return res


def show(res):
    print(f"\n=== {res['tag']}  ({res['protocol']}) ===")
    print(f"T_full={res['T_full']:.3f}  T_exit={[round(x, 3) for x in res['T_exit']]}")
    hdr = f"{'method':13s} {'tau':>6s} {'acc':>6s} {'ECE':>7s} {'ECE 95% CI':>17s} {'FLOPs':>6s}"
    if 'corrupt_mean' in res:
        hdr += f" | {'c.acc':>6s} {'c.ECE':>6s} {'c.ECE 95% CI':>15s} {'c.FLOPs':>7s}"
    print(hdr)
    for n, s in res['clean'].items():
        ci = res['clean_ci'][n]['ece']
        line = (f"{n:13s} {res['selection'][n]['tau']:6.3f} {100*s['acc']:6.2f} {s['ece']:7.4f} "
                f"[{ci[0]:.4f},{ci[1]:.4f}] {s['flops']:6.3f}")
        if 'corrupt_mean' in res:
            cm = res['corrupt_mean'][n]; cci = res['corrupt_ci'][n]['ece']
            line += f" | {100*cm['acc']:6.2f} {cm['ece']:6.4f} [{cci[0]:.4f},{cci[1]:.4f}] {cm['flops']:7.3f}"
        print(line)
    print('paired (risk - X), clean ECE:')
    for n, v in res['clean_paired'].items():
        print(f"   {n:13s} {v['ece']['diff']:+.4f} [{v['ece']['ci'][0]:+.4f},{v['ece']['ci'][1]:+.4f}]   acc {100*v['acc']['diff']:+.2f}pp")
    if 'corrupt_paired' in res:
        print('paired (risk - X), corrupted ECE:')
        for n, v in res['corrupt_paired'].items():
            print(f"   {n:13s} {v['ece']['diff']:+.4f} [{v['ece']['ci'][0]:+.4f},{v['ece']['ci'][1]:+.4f}]   acc {100*v['acc']['diff']:+.2f}pp")
    best = sorted(res['lambda_grid_val'], key=lambda r: r['val']['ece'])[:3]
    print('val lambda grid, lowest val ECE at selected tau:', [(r['l1'], r['l2'], r['tau'], round(r['val']['ece'], 4)) for r in best])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--no-corrupt', action='store_true')
    a = ap.parse_args()
    r = evaluate(a.tag, a.boot, corrupt=not a.no_corrupt)
    r['rule'] = 'matched'; r['lam_mode'] = 'val'
    out = os.path.join(ROOT, 'results'); os.makedirs(out, exist_ok=True)
    json.dump(r, open(os.path.join(out, f'{a.tag}_main.json'), 'w'), indent=1, default=float)
    show(r)
