"""
python -m riskee.cache_stats -- the few per-sample summaries that the numbers and
figures need from the cached model outputs, so that make_numbers.py and
make_figures.py run from results/ alone.

Output: results/cache_stats.json
  exitacc[tag]          clean test accuracy of exits 1, 2, 3 (all test images)
  exitsub[tag]          clean test accuracy at exits 1, 2 of the images that Risk-EE
                        lets exit there (main operating point)
  reliability[tag][m]   15-bin reliability diagram on the clean test set (same bins as
                        analysis.ece): per-bin accuracy (null for empty bins) and ECE
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from riskee.analysis import DS, run, load_ref, COSTS, CACHE_DIR, arch_of, ece, N_BINS  # noqa: E402

R = os.path.join(ROOT, 'results')
TAGS = ['cnn', 'vit', 'tiny', 'cnn_seed42', 'cnn_seed123', 'cnn_seed456']
REL_TAGS = ['cnn', 'vit']
REL_METHODS = ['full', 'conf', 'pets_conf', 'risk']


def params(d, m, c):
    s = d['selection'][m]
    if m == 'risk':
        return {'l1': s['l1'], 'l2': s['l2'], 'score': 'dist', 'costs': c, 'T': None}, s['tau']
    if m == 'pets_conf':
        return {'T': d['T_exit']}, s['tau']
    if m == 'ts_full':
        return {'T': d['T_full']}, 0.0
    return {}, s.get('tau', 0.0)


out = {'exitacc': {}, 'exitsub': {}, 'reliability': {}}
for tag in TAGS:
    d = json.load(open(os.path.join(R, f'{tag}_main.json')))
    c = COSTS[arch_of(tag)]
    ds = DS(os.path.join(CACHE_DIR, tag, 'test.npz'), load_ref(tag))
    out['exitacc'][tag] = [float(np.mean(ds.pred[k] == ds.y)) for k in range(3)]
    p, tau = params(d, 'risk', c)
    _, ex = run(ds, 'risk', p, tau, c)
    out['exitsub'][tag] = [float(np.mean(ds.pred[k][ex == k] == ds.y[ex == k])) for k in range(2)]
    if tag in REL_TAGS:
        out['reliability'][tag] = {}
        for m in REL_METHODS:
            p, tau = params(d, m, c)
            probs, _ = run(ds, m, p, tau, c)
            conf = probs.max(1); ok = probs.argmax(1) == ds.y
            e = ece(conf, ok.astype(float))
            assert abs(e - d['clean'][m]['ece']) < 1e-9, (tag, m, e, d['clean'][m]['ece'])
            edges = np.linspace(0, 1, N_BINS + 1)
            idx = np.clip(np.digitize(conf, edges, right=True) - 1, 0, N_BINS - 1)
            acc = [float(ok[idx == b].mean()) if (idx == b).any() else None for b in range(N_BINS)]
            out['reliability'][tag][m] = {'acc': acc, 'ece': e}
    print(tag, 'done', flush=True)
json.dump(out, open(os.path.join(R, 'cache_stats.json'), 'w', newline='\n'), indent=1)
print('wrote results/cache_stats.json')
