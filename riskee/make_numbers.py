r"""
python -m riskee.make_numbers -- single source of truth for every number quoted in
the paper.

Reads results/*.json only and writes
tex/numbers.tex, which defines
    \R{<key>}   e.g. \R{cnn.risk.clean.ece} -> 0.0187
An undefined key typesets as a bold red ??key?? and emits a LaTeX warning, so
a typo can never silently print a wrong number.  Also writes
tex/numbers_index.txt (key = value, sorted).

Key scheme
  {tag}.{method}.{clean|corr}.{acc|ece|ecelo|ecehi|brier|flops|mce}
  {tag}.{method}.clean.exit{1,2,3}             exit distribution (percent)
  {tag}.{method}.{l1|l2|tau}                   selected parameters
  {tag}.d.{method}.{clean|corr}[.lo|.hi|.sig]  ECE(risk) - ECE(method): point diff,
                                               percentile-bootstrap CI, sig = yes/no
  {tag}.red.{method}.{clean|corr}              ECE reduction of Risk-EE vs X: 100*(1 - ECE_risk/ECE_X)
  {tag}.saving.{clean|corr}                    100*(1 - FLOPs_risk)  (percent)
  theory.{tag}.{S1|S2|pass1|final|flops}.{mono|ge}, theory.{tag}.n
  lat.{arch}.{bs1|bs128}.{clean|corrupt}.full  (ms/img); .{method} speed-up (x)
  sweep.{tag}.{budget}.{method}.ece ; sweep.{tag}.{budget}.d.{method}[.lo|.hi|.sig]
  pertype.{tag}.vs{full|conf}.{max|maxtype|n}  per-corruption-type accuracy gap
  costs.{cnn|tiny|vit}.{e1|e2}
  sev.{tag}.{method}.{0..5}.{ece|flops|final}  per-severity means (0 = clean)
  thex.{cnn|vit|tiny}.*                         exceptions to monotone deferral
  exitacc.{tag}.{1,2,3}                         clean accuracy of each exit (percent)
Tags: cnn, vit, tiny, seedA (42), seedB (123), seedC (456).
"""
import json, os, glob, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'tex'); os.makedirs(OUT, exist_ok=True)
TAGS = {'cnn': 'cnn', 'vit': 'vit', 'tiny': 'tiny', 'seedA': 'cnn_seed42', 'seedB': 'cnn_seed123', 'seedC': 'cnn_seed456'}
ARCH = {'cnn': 'cnn', 'seedA': 'cnn', 'seedB': 'cnn', 'seedC': 'cnn', 'vit': 'vit', 'tiny': 'tiny'}
N = {}


def put(k, v):
    assert k not in N, f'duplicate key {k}'
    N[k] = v


def f4(x): return f'{x:.4f}'
def f3(x): return f'{x:.3f}'
def pct(x, d=2): return f'{100 * x:.{d}f}'
def sgn(x, d=4):
    s = f'{abs(x):.{d}f}'
    return ('+' if x >= 0 else '$-$') + s
def sig(lo, hi): return 'yes' if (lo > 0 or hi < 0) else 'no'


def protocol_file(short, tag):
    for t in tag:
        p = os.path.join(R, t)
        if os.path.exists(p):
            return json.load(open(p))
    return None


for short, tag in TAGS.items():
    r = protocol_file(short, [f'{tag}_main.json'])
    if r is None:
        print('missing', tag); continue
    put(f'{short}.n', str(r.get('n_corrupt_sets', 0)))
    for m, s in r['clean'].items():
        b = f'{short}.{m}'
        put(f'{b}.clean.acc', pct(s['acc'])); put(f'{b}.clean.ece', f4(s['ece']))
        put(f'{b}.clean.brier', f4(s['brier'])); put(f'{b}.clean.flops', f3(s['flops']))
        lo, hi = r['clean_ci'][m]['ece']; put(f'{b}.clean.ecelo', f4(lo)); put(f'{b}.clean.ecehi', f4(hi))
        for i, e in enumerate(s['exit_dist']): put(f'{b}.clean.exit{i + 1}', pct(e, 1))
        sel = r['selection'].get(m, {})
        put(f'{b}.tau', f'{sel.get("tau", 0.0):.3f}')
        if 'l1' in sel: put(f'{b}.l1', f'{sel["l1"]:.1f}'); put(f'{b}.l2', f'{sel["l2"]:.1f}')
        if 'corrupt_mean' in r and m in r['corrupt_mean']:
            c = r['corrupt_mean'][m]
            put(f'{b}.corr.acc', pct(c['acc'])); put(f'{b}.corr.ece', f4(c['ece']))
            put(f'{b}.corr.brier', f4(c['brier'])); put(f'{b}.corr.flops', f3(c['flops']))
            put(f'{b}.corr.mce', f4(c['mce']))
            lo, hi = r['corrupt_ci'][m]['ece']; put(f'{b}.corr.ecelo', f4(lo)); put(f'{b}.corr.ecehi', f4(hi))
    put(f'{short}.T.full', f'{r["T_full"]:.2f}')
    for i, t_ in enumerate(r['T_exit']): put(f'{short}.T.e{i + 1}', f'{t_:.2f}')
    rk = r['clean']['risk']
    put(f'{short}.saving.clean', pct(1 - rk['flops'], 0))
    if 'corrupt_mean' in r:
        put(f'{short}.saving.corr', pct(1 - r['corrupt_mean']['risk']['flops'], 0))
    for split, base, pair in [('clean', r['clean'], r['clean_paired']),
                              ('corr', r.get('corrupt_mean', {}), r.get('corrupt_paired', {}))]:
        for m, v in pair.items():
            d = base['risk']['ece'] - base[m]['ece']; lo, hi = v['ece']['ci']
            assert lo - 1e-12 <= d <= hi + 1e-12, f'point diff outside CI: {tag} {split} {m}'
            k = f'{short}.d.{m}.{split}'
            put(k, sgn(d)); put(k + '.lo', sgn(lo)); put(k + '.hi', sgn(hi)); put(k + '.sig', sig(lo, hi))
            put(f'{short}.dacc.{m}.{split}', sgn(100 * (base['risk']['acc'] - base[m]['acc']), 2))
    for split, base in [('clean', r['clean']), ('corr', r.get('corrupt_mean', {}))]:
        for m, s in base.items():
            if m != 'risk':   # ECE reduction of Risk-EE relative to X, percent (negative = Risk-EE worse)
                red = 100 * (1 - base['risk']['ece'] / s['ece'])
                put(f'{short}.red.{m}.{split}', f'{red:.0f}' if red >= 0 else f'$-${abs(red):.0f}')
    # per-type accuracy gaps
    if 'corrupt_per_set' in r:
        P = r['corrupt_per_set']
        for bse in ('full', 'conf'):
            g = {t: np.mean([P[bse][t][sv]['acc'] - P['risk'][t][sv]['acc'] for sv in P['risk'][t]]) for t in P['risk']}
            mt = max(g, key=g.get)
            put(f'pertype.{short}.vs{bse}.max', f'{g[mt]:.4f}')
            put(f'pertype.{short}.vs{bse}.maxtype', mt.replace('_', ' '))
            put(f'pertype.{short}.vs{bse}.n', str(sum(v > 0.005 for v in g.values())))
            put(f'pertype.{short}.ntypes', str(len(g))) if bse == 'full' else None

for short in ('cnn', 'vit', 'tiny'):
    p = os.path.join(R, f'{TAGS[short]}_theory_check_main.json')
    if not os.path.exists(p): continue
    t = json.load(open(p)); put(f'theory.{short}.n', str(t['n_types']))
    for k in ('S1', 'S2', 'pass1', 'final', 'flops'):
        put(f'theory.{short}.{k}.mono', str(t['counts'][k])); put(f'theory.{short}.{k}.ge', str(t['all_severities_ge_clean'][k]))

LATD = {}
for arch in ('cnn', 'vit'):
    p = os.path.join(R, f'latency_{arch}_main.json')
    if not os.path.exists(p): continue
    L = json.load(open(p))
    for bs in ('bs1', 'bs128'):
        for c in ('clean', 'corrupt'):
            f = L[f'{bs}/{c}/full']
            put(f'lat.{arch}.{bs}.{c}.full', f'{f:.2f}' if f >= 1 else f'{f:.3f}')
            for m in ('conf', 'entropy', 'pets_conf', 'risk'):
                put(f'lat.{arch}.{bs}.{c}.{m}', f'{f / L[f"{bs}/{c}/{m}"]:.2f}')
            dm = L[f'{bs}/{c}/risk'] - L[f'{bs}/{c}/conf']   # ms/img, Risk-EE minus confidence routing
            put(f'lat.{arch}.{bs}.{c}.riskminusconf', f'{dm:.2f}' if abs(dm) >= 0.01 else f'{dm:.3f}')
            LATD.setdefault(bs, []).append(dm)
if LATD:
    for bs, v in LATD.items():
        put(f'lat.max.{bs}.riskminusconf', f'{max(v):.3f}')

p = os.path.join(R, 'budget_sweep_main.json')
if os.path.exists(p):
    B = json.load(open(p))
    inv = {v: k for k, v in TAGS.items()}
    for tag, v in B.items():
        short = inv[tag]
        for b, row in v['budgets'].items():
            bb = f'{float(b):.2f}'
            for m, s in row.items():
                put(f'sweep.{short}.{bb}.{m}.ece', f4(s['ece']))
            for k, d in row['risk'].items():
                if k.startswith('d_'):   # point difference (same convention as the main tables) + bootstrap CI
                    m = k[2:]; kk = f'sweep.{short}.{bb}.d.{m}'
                    pd = row['risk']['ece'] - row[m]['ece']
                    assert d[1] - 1e-12 <= pd <= d[2] + 1e-12, f'sweep point diff outside CI {tag} {b} {m}'
                    put(kk, sgn(pd)); put(kk + '.lo', sgn(d[1])); put(kk + '.hi', sgn(d[2])); put(kk + '.sig', sig(d[1], d[2]))

p = os.path.join(R, 's_stats.json')
if os.path.exists(p):   # S_k / U_k statistics (model properties; independent of lambda, tau)
    SS = json.load(open(p)); inv = {v: k for k, v in TAGS.items()}
    for tag, sets in SS.items():
        short = inv[tag]
        for st in ('test', 'c_gaussian_noise_5', 'c_fog_5', 'c_contrast_5', 'c_gaussian_noise_1'):
            if st not in sets: continue
            nm = {'test': 'clean', 'c_gaussian_noise_5': 'gnfive', 'c_fog_5': 'fogfive',
                  'c_contrast_5': 'contrastfive', 'c_gaussian_noise_1': 'gnone'}[st]
            for k, v in sets[st].items():
                put(f'sstat.{short}.{nm}.{k}', pct(v, 0) if k.endswith('_pos') else f'{v:.3f}')
        cor = [v for s_, v in sets.items() if s_.startswith('c_')]
        for k in ('S1_mean', 'S2_mean', 'U1_mean'):
            put(f'sstat.{short}.corrmean.{k}', f'{np.mean([v[k] for v in cor]):.3f}')
            put(f'sstat.{short}.nbelowclean.{k}', str(sum(v[k] < sets['test'][k] for v in cor)))
            put(f'sstat.{short}.naboveclean.{k}', str(sum(v[k] > sets['test'][k] for v in cor)))
        put(f'sstat.{short}.nsets', str(len(cor)))

import sys
sys.path.insert(0, ROOT)
from riskee.analysis import COSTS
for a in ('cnn', 'tiny', 'vit'):
    put(f'costs.{a}.e1', f3(COSTS[a][0])); put(f'costs.{a}.e2', f3(COSTS[a][1]))

# per-severity means over corruption types (severity 0 = clean): ECE, FLOPs, share at the final exit
for short, tag in TAGS.items():
    r = json.load(open(os.path.join(R, f'{tag}_main.json'))); P = r['corrupt_per_set']
    for m in P:
        types = sorted(P[m])
        for sv in range(6):
            rows = [r['clean'][m]] if sv == 0 else [P[m][t][str(sv)] for t in types]
            put(f'sev.{short}.{m}.{sv}.ece', f4(np.mean([x['ece'] for x in rows])))
            put(f'sev.{short}.{m}.{sv}.flops', f3(np.mean([x['flops'] for x in rows])))
            put(f'sev.{short}.{m}.{sv}.final', pct(np.mean([x['exit_dist'][2] for x in rows]), 1))

# exceptions to monotone deferral (Risk-EE FLOPs, severity 0..5) and their relation to the
# full network's own accuracy being non-monotone in severity
for short in ('cnn', 'vit', 'tiny'):
    r = json.load(open(os.path.join(R, f'{TAGS[short]}_main.json'))); P = r['corrupt_per_set']; types = sorted(P['risk'])
    fl = {t: [r['clean']['risk']['flops']] + [P['risk'][t][str(s)]['flops'] for s in range(1, 6)] for t in types}
    ac = {t: [r['clean']['full']['acc']] + [P['full'][t][str(s)]['acc'] for s in range(1, 6)] for t in types}
    nonmono = [t for t in types if any(a > b + 1e-12 for a, b in zip(fl[t], fl[t][1:]))]
    accnm = [t for t in types if any(a < b for a, b in zip(ac[t], ac[t][1:]))]
    other = [t for t in nonmono if t not in accnm]
    put(f'thex.{short}.nonmono', str(len(nonmono)))
    put(f'thex.{short}.accexplained', str(len(nonmono) - len(other)))
    put(f'thex.{short}.other', str(len(other)))
    dev = max([max(max(fl[t][:i + 1]) - fl[t][i] for i in range(6)) for t in other], default=0.0)
    put(f'thex.{short}.othermaxdev', f'{dev:.4f}')
    fin0 = r['clean']['risk']['exit_dist'][2]
    for t in ('gaussian_noise', 'shot_noise', 'impulse_noise'):
        below = [s for s in range(1, 6) if P['risk'][t][str(s)]['exit_dist'][2] < fin0]
        put(f'thex.{short}.{t.split("_")[0]}.firstbelow', str(below[0]) if below else 'none')

# accuracy of each exit on the clean test set (model property), and accuracy of the
# inputs that Risk-EE lets exit at exit k (clean test, main operating point);
# computed from the cached outputs by riskee/cache_stats.py
CS = json.load(open(os.path.join(R, 'cache_stats.json')))
for short, tag in TAGS.items():
    for k in range(3):
        put(f'exitacc.{short}.{k + 1}', pct(CS['exitacc'][tag][k], 1))
    for k in range(2):
        put(f'exitsub.{short}.{k + 1}', pct(CS['exitsub'][tag][k], 1))

# Risk-EE on per-exit-TS outputs vs per-exit TS + confidence routing (riskee/pair_risk_pets.py)
p_ = os.path.join(R, 'pair_risk_pets.json')
if os.path.exists(p_):
    inv = {v: k for k, v in TAGS.items()}
    PP = json.load(open(p_))
    for tag, v in PP.items():
        short = inv[tag]
        for sp in ('clean', 'corr'):
            r_ = v[sp]; k = f'pairpets.{short}.{sp}'
            put(k, sgn(r_['d_ece'])); put(k + '.lo', sgn(r_['ci'][0])); put(k + '.hi', sgn(r_['ci'][1]))
            put(k + '.sig', sig(*r_['ci'])); put(k + '.dacc', sgn(100 * r_['d_acc'], 2))
    for sp in ('clean', 'corr'):
        put(f'pairpets.nlower.{sp}', str(sum(v[sp]['ci'][1] < 0 for v in PP.values())))
        put(f'pairpets.nhigher.{sp}', str(sum(v[sp]['ci'][0] > 0 for v in PP.values())))

# ── write ────────────────────────────────────────────────────────────────────
with open(os.path.join(OUT, 'numbers.tex'), 'w', encoding='utf-8', newline='\n') as f:
    f.write('% AUTO-GENERATED by make_numbers.py from the result files -- do not edit by hand.\n')
    f.write('\\makeatletter\n')
    f.write('\\newcommand{\\R}[1]{\\ifcsname rnum@#1\\endcsname\\csname rnum@#1\\endcsname'
            '\\else\\textbf{\\textcolor{red}{??#1??}}\\PackageWarning{numbers}{Undefined number key #1}\\fi}\n')
    for k in sorted(N):
        f.write(f'\\expandafter\\def\\csname rnum@{k}\\endcsname{{{N[k]}}}\n')
    f.write('\\makeatother\n')
with open(os.path.join(OUT, 'numbers_index.txt'), 'w', encoding='utf-8', newline='\n') as f:
    for k in sorted(N):
        f.write(f'{k} = {N[k]}\n')
print(f'{len(N)} keys -> tex/numbers.tex')
