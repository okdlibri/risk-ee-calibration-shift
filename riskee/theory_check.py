"""
python -m riskee.theory_check --tag {cnn,vit,tiny}

Empirical test of the consequences of Proposition 1 / Corollary 1 for the
selected Risk-EE configuration (from results/<tag>_main.json):
for every corruption type, over severities 0 (clean test) .. 5:
  * mean S_k at exits 1 and 2 (training-split centroid score);
  * P(pass exit 1), P(reach final exit), and average FLOPs.
A sequence is counted as monotone if it is non-decreasing over 0..5.
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from riskee.analysis import DS, load_ref, run, summarize, COSTS, arch_of
from riskee.common import CACHE_DIR, CORRUPTIONS, ROOT

ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='cnn')
a = ap.parse_args()
SFX = '_main'
tag = a.tag; arch = arch_of(tag); costs = COSTS[arch]
d = os.path.join(CACHE_DIR, tag); ref = load_ref(tag)
R = json.load(open(os.path.join(ROOT, 'results', f'{tag}_main.json')))
s = R['selection']['risk']
p = {'l1': s['l1'], 'l2': s['l2'], 'score': s['score'], 'costs': costs, 'T': None}
tau = s['tau']

clean = DS(os.path.join(d, 'test.npz'), ref)


def stats(ds):
    probs, ex = run(ds, 'risk', p, tau, costs)
    return dict(S1=float(ds.S['dist'][0].mean()), S2=float(ds.S['dist'][1].mean()),
                pass1=float((ex >= 1).mean()), final=float((ex == 2).mean()),
                flops=float(np.asarray(costs)[ex].mean()))


c0 = stats(clean)
out = {'clean': c0, 'per_corruption': {}, 'counts': {}}
keys = ['S1', 'S2', 'pass1', 'final', 'flops']
mono = {k: [] for k in keys}; above_clean = {k: 0 for k in keys}; n = 0
for c in CORRUPTIONS:
    if not os.path.exists(os.path.join(d, f'c_{c}_1.npz')):
        continue
    seq = [c0] + [stats(DS(os.path.join(d, f'c_{c}_{sv}.npz'), ref)) for sv in range(1, 6)]
    out['per_corruption'][c] = seq
    n += 1
    for k in keys:
        v = [x[k] for x in seq]
        if all(v[i + 1] >= v[i] - 1e-12 for i in range(5)):
            mono[k].append(c)
        if all(x >= v[0] - 1e-12 for x in v[1:]):
            above_clean[k] += 1
out['n_types'] = n
out['monotone_types'] = mono
out['counts'] = {k: len(v) for k, v in mono.items()}
out['all_severities_ge_clean'] = above_clean
json.dump(out, open(os.path.join(ROOT, 'results', f'{tag}_theory_check{SFX}.json'), 'w'), indent=1)
print(f'{tag}: {n} corruption types; Risk-EE tau={tau:.4f} l1={p["l1"]} l2={p["l2"]}')
for k in keys:
    nm = sorted(set(out['per_corruption']) - set(mono[k]))
    print(f'  {k:6s} monotone in severity for {len(mono[k])}/{n}; all severities >= clean for {above_clean[k]}/{n}; non-monotone: {nm}')
