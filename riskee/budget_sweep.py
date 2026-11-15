"""python -m riskee.budget_sweep <tags...>
ECE-vs-compute curve: for each budget, every method's tau is selected on the
validation split by the same matched rule as protocol.select_tau (fine grid;
largest val FLOPs <= budget; ties -> higher val acc). Clean test metrics plus
paired-bootstrap 95% CIs (1000 resamples over test images) of
ECE(risk) - ECE(X). The val sweep is computed once per method.
Every Risk-EE variant uses the lambda selected on validation at the main
operating budget (read from results/<tag>_main.json), fixed across budgets.
Output: results/budget_sweep_main.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import riskee.protocol as P
from riskee.analysis import fit_T, summarize, ece_boot, COSTS, arch_of
BUDGETS = {'resnet': [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
           'vit': [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]}
NAMES = ['conf', 'entropy', 'pets_conf', 'risk', 'risk_noS', 'risk_noC']
N_BOOT = 1000
args = sys.argv[1:]
SRC = 'main'
path = os.path.join(P.ROOT, 'results', 'budget_sweep_main.json')
out = json.load(open(path)) if os.path.exists(path) else {}
for tag in args:
    arch = arch_of(tag); costs = COSTS[arch]
    d, ref, val, test = P.load_sets(tag)
    T_exit = [fit_T(val.logits[k], val.y) for k in range(3)]
    selm = json.load(open(os.path.join(P.ROOT, 'results', f'{tag}_main.json')))['selection']
    c_ = COSTS[arch]
    M = {'conf': ('conf', {}, P.CONF_GRID), 'entropy': ('entropy', {}, P.ENT_GRID),
         'pets_conf': ('pets_conf', {'T': T_exit}, P.CONF_GRID)}
    for n in NAMES:
        if n.startswith('risk'):
            M[n] = ('risk', {'l1': selm[n]['l1'], 'l2': selm[n]['l2'], 'score': 'dist', 'costs': c_, 'T': None},
                    P.RISK_GRID)
    vs = {n: [(t, summarize(P.apply(val, M[n][0], M[n][1], t, costs), val.y, costs))
              for t in P.fine_grid(val, M[n][0], M[n][1], M[n][2])] for n in NAMES}
    idx = np.random.default_rng(0).integers(0, test.n, size=(N_BOOT, test.n))
    out[tag] = {'costs': costs, 'src': SRC, 'lambda': {n: [M[n][1]['l1'], M[n][1]['l2']] for n in NAMES if n.startswith('risk')}, 'budgets': {}}
    print(f'== {tag} costs={[round(c, 3) for c in costs]}: clean test ECE (acc%) [FLOPs]; paired risk-X ECE [95% CI]', flush=True)
    for b in BUDGETS['vit' if arch == 'vit' else 'resnet']:
        row, eb = {}, {}
        for n in NAMES:
            m, p, _ = M[n]
            cand = [((round(s['flops'], 6), s['acc']), t) for t, s in vs[n] if s['flops'] <= b + 1e-9]
            t = max(cand)[1]
            probs, ex = P.apply(test, m, p, t, costs)
            s = summarize((probs, ex), test.y, costs)
            row[n] = {'tau': float(t), **{k: float(s[k]) for k in ('acc', 'ece', 'flops')}}
            eb[n] = ece_boot(probs.max(1), (probs.argmax(1) == test.y).astype(float), idx)
        for n in NAMES:
            if n != 'risk':
                dlt = eb['risk'] - eb[n]
                row['risk'][f'd_{n}'] = [float(row['risk']['ece'] - row[n]['ece']),
                                        float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5))]
        out[tag]['budgets'][str(b)] = row
        r = row['risk']
        print(f"{b:.2f} " + '  '.join(f"{n} {row[n]['ece']:.4f} ({100*row[n]['acc']:.2f}) [{row[n]['flops']:.3f}]" for n in NAMES), flush=True)
        print('      ' + '  '.join(f"risk-{n} {r['d_'+n][0]:+.4f} [{r['d_'+n][1]:+.4f},{r['d_'+n][2]:+.4f}]" for n in ('conf', 'entropy', 'pets_conf', 'risk_noS')), flush=True)
    json.dump(out, open(path, 'w'), indent=1)
