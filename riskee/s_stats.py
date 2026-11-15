"""
python -m riskee.s_stats -- distribution statistics of the deviation score S_k (centroid,
'dist') and of the normalised entropy U_k, per model and per test / corrupted
set.  S_k and U_k do not depend on lambda or tau, so these are properties of
the trained model only.  Output: results/s_stats.json
  {tag: {set: {S1_mean, S1_median, S1_pos, S2_mean, S2_median, S2_pos, U1_mean, U2_mean}}}
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from riskee.analysis import DS, load_ref
from riskee.common import CACHE_DIR, ROOT

out = {}
for tag in ['cnn', 'vit', 'tiny', 'cnn_seed42', 'cnn_seed123', 'cnn_seed456']:
    d = os.path.join(CACHE_DIR, tag); ref = load_ref(tag); out[tag] = {}
    for f in sorted(os.listdir(d)):
        if not (f == 'test.npz' or f.startswith('c_')):
            continue
        ds = DS(os.path.join(d, f), ref); s = {}
        for k in (0, 1):
            v = ds.S['dist'][k]
            s[f'S{k + 1}_mean'] = float(v.mean()); s[f'S{k + 1}_median'] = float(np.median(v))
            s[f'S{k + 1}_pos'] = float((v > 0).mean()); s[f'U{k + 1}_mean'] = float(ds.ent[k].mean())
        out[tag][f[:-4]] = s
    print(tag, len(out[tag]), 'sets', flush=True)
json.dump(out, open(os.path.join(ROOT, 'results', 's_stats.json'), 'w'), indent=1)
