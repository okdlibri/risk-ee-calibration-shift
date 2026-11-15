"""Paired comparison: Risk-EE on per-exit temperature-scaled outputs (risk_pets)
versus per-exit temperature scaling with confidence routing (pets_conf).

Uses the operating points, temperatures and weights stored in results/<tag>_main.json
(nothing is re-selected) and EXACTLY the bootstrap resamples of protocol.py
(seed 0 for the test images; seed 1 for the shared Tiny-ImageNet-C resample), so the
intervals are directly comparable with all other paired intervals in the paper.
Before computing anything, it checks that both methods reproduce the stored ECE values.

Output: results/pair_risk_pets.json
  {tag: {clean|corr: {d_ece, ci, d_acc, ci_acc}}}, d = ECE(risk_pets) - ECE(pets_conf), point difference.
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from riskee.analysis import DS, run, load_ref, COSTS, CACHE_DIR, ece_boot, summarize, arch_of  # noqa: E402
from riskee.common import CORRUPTIONS, SEVERITIES  # noqa: E402

R = os.path.join(ROOT, 'results')
N_BOOT, SEED = 2000, 0
TAGS = ['cnn', 'cnn_seed42', 'cnn_seed123', 'cnn_seed456', 'vit', 'tiny']
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def policies(d, costs):
    s = d['selection']['risk_pets']; p = d['selection']['pets_conf']
    return {'risk_pets': ('risk', {'l1': s['l1'], 'l2': s['l2'], 'score': s.get('score', 'dist'),
                                   'costs': costs, 'T': d['T_exit']}, s['tau']),
            'pets_conf': ('pets_conf', {'T': d['T_exit']}, p['tau'])}


def main():
    out = {}
    for tag in TAGS:
        d = json.load(open(os.path.join(R, f'{tag}_main.json')))
        arch = arch_of(tag); costs = COSTS[arch]; ref = load_ref(tag)
        test = DS(os.path.join(CACHE_DIR, tag, 'test.npz'), ref)
        P = policies(d, costs)
        idx = np.random.default_rng(SEED).integers(0, test.n, size=(N_BOOT, test.n))
        B, res = {}, {}
        for n, (m, p, t) in P.items():
            o = run(test, m, p, t, costs); s = summarize(o, test.y, costs)
            assert abs(s['ece'] - d['clean'][n]['ece']) < 1e-9, (tag, n, s['ece'], d['clean'][n]['ece'])
            conf = o[0].max(1); cor = (o[0].argmax(1) == test.y).astype(float)
            B[n] = (ece_boot(conf, cor, idx), cor[idx].mean(1), s)
        de = B['risk_pets'][0] - B['pets_conf'][0]; da = B['risk_pets'][1] - B['pets_conf'][1]
        pd = B['risk_pets'][2]['ece'] - B['pets_conf'][2]['ece']
        pa = B['risk_pets'][2]['acc'] - B['pets_conf'][2]['acc']
        res['clean'] = {'d_ece': pd, 'ci': ci(de), 'd_acc': pa, 'ci_acc': ci(da)}

        BB = {n: [np.zeros(N_BOOT), np.zeros(N_BOOT), []] for n in P}
        nset, y_ref = 0, None
        for c in CORRUPTIONS:
            for sv in SEVERITIES:
                path = os.path.join(CACHE_DIR, tag, f'c_{c}_{sv}.npz')
                if not os.path.exists(path):
                    continue
                nset += 1; cs = DS(path, ref)
                if arch != 'tiny':
                    assert (cs.y == test.y).all(); ii = idx
                else:
                    if y_ref is None:
                        y_ref = cs.y
                        idx_c = np.random.default_rng(SEED + 1).integers(0, cs.n, size=(N_BOOT, cs.n))
                    assert (cs.y == y_ref).all(); ii = idx_c
                for n, (m, p, t) in P.items():
                    o = run(cs, m, p, t, costs); s = summarize(o, cs.y, costs)
                    assert abs(s['ece'] - d['corrupt_per_set'][n][c][str(sv)]['ece']) < 1e-9, (tag, n, c, sv)
                    conf = o[0].max(1); cor = (o[0].argmax(1) == cs.y).astype(float)
                    BB[n][0] += ece_boot(conf, cor, ii); BB[n][1] += cor[ii].mean(1); BB[n][2].append(s)
        for n in P:
            BB[n][0] /= nset; BB[n][1] /= nset
        me = {n: float(np.mean([s['ece'] for s in BB[n][2]])) for n in P}
        ma = {n: float(np.mean([s['acc'] for s in BB[n][2]])) for n in P}
        for n in P:
            assert abs(me[n] - d['corrupt_mean'][n]['ece']) < 1e-9
        res['corr'] = {'d_ece': me['risk_pets'] - me['pets_conf'], 'ci': ci(BB['risk_pets'][0] - BB['pets_conf'][0]),
                       'd_acc': ma['risk_pets'] - ma['pets_conf'], 'ci_acc': ci(BB['risk_pets'][1] - BB['pets_conf'][1]),
                       'n_sets': nset}
        for sp in ('clean', 'corr'):
            r = res[sp]; assert r['ci'][0] - 1e-12 <= r['d_ece'] <= r['ci'][1] + 1e-12, (tag, sp, r)
        out[tag] = res
        print(tag, {sp: (round(res[sp]['d_ece'], 4), [round(x, 4) for x in res[sp]['ci']]) for sp in ('clean', 'corr')},
              flush=True)
    json.dump(out, open(os.path.join(R, 'pair_risk_pets.json'), 'w'), indent=1)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
