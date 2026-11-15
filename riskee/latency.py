"""
python -m riskee.latency --arch {cnn,vit}

Wall-clock latency of each routing policy at its validation-selected
operating point (from results/<tag>_main.json), measured two ways on the
same GPU (NVIDIA Quadro P2200):

  bs1   : batch size 1, sequential (edge / streaming setting). Execution
          really stops at the exit, so saved FLOPs can become saved time.
  bs128 : batch of 128 with compaction: after each exit the samples that
          exit are removed and only the remaining ones continue.

Both are reported on clean test images and on a corrupted mix (1000
images drawn uniformly from the 95 CIFAR-C sets), because the exit
distribution - and hence latency - changes under shift.
"""
import os, sys, json, time, argparse, math
import numpy as np
import torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from riskee.common import CORRUPTIONS, SEVERITIES, CACHE_DIR
DEV = torch.device('cuda')
ap = argparse.ArgumentParser(); ap.add_argument('--arch', default='cnn')
ap.add_argument('--n', type=int, default=1000)
args = ap.parse_args()
torch.backends.cudnn.benchmark = False


def build():
    if args.arch == 'cnn':
        import data_utils as du
        from model import EarlyExitResNet
        from torchvision import datasets
        m = EarlyExitResNet().to(DEV)
        sd = torch.load('checkpoints/early_exit_model.pth', map_location=DEV, weights_only=True)
        m.load_state_dict({k: v for k, v in sd.items() if not k.endswith(('total_ops', 'total_params'))}, strict=False)
        m.eval()
        stages = [lambda x: m.layer2(m.layer1(m.stem(x))), m.layer3, m.layer4]
        heads = [lambda h: (m.exit1(f := m.avgpool(h).flatten(1)), f),
                 lambda h: (m.exit2(f := m.avgpool(h).flatten(1)), f),
                 lambda h: (m.final_fc(f := m.avgpool(h).flatten(1)), f)]
        test = datasets.CIFAR10('./data', train=False, transform=du.transform_test)
        clean = torch.stack([test[i][0] for i in range(args.n)])
        rng = np.random.default_rng(0); cor = []
        cache = {}
        for i in range(args.n):
            c = CORRUPTIONS[rng.integers(19)]; s = int(rng.integers(1, 6))
            if (c, s) not in cache: cache[(c, s)] = du.CIFAR10C(c, s)
            cor.append(cache[(c, s)][int(rng.integers(10000))][0])
        tag = 'cnn'; C = 10
    else:
        import data_utils_vit as dv, torchvision
        from model_vit import EarlyExitDeiT
        m = EarlyExitDeiT().to(DEV)
        sd = torch.load('checkpoints_vit/early_exit_vit.pth', map_location=DEV, weights_only=True)
        m.load_state_dict({k: v for k, v in sd.items() if not k.endswith(('total_ops', 'total_params'))}, strict=False)
        m.eval()
        stages = [lambda x: m.blocks[0:4](m._embed(x)), m.blocks[4:8], lambda h: m.norm(m.blocks[8:12](h))]
        heads = [lambda h: (m.exit1(h[:, 0]), h[:, 0]), lambda h: (m.exit2(h[:, 0]), h[:, 0]),
                 lambda h: (m.final_head(h[:, 0]), h[:, 0])]
        test = torchvision.datasets.CIFAR100('./data', train=False, transform=dv.val_transform())
        clean = torch.stack([test[i][0] for i in range(args.n)])
        rng = np.random.default_rng(0); cor = []; cache = {}
        for i in range(args.n):
            c = CORRUPTIONS[rng.integers(19)]; s = int(rng.integers(1, 6))
            if (c, s) not in cache: cache[(c, s)] = dv.CIFAR100C(c, s)
            cor.append(cache[(c, s)][int(rng.integers(10000))][0])
        tag = 'vit'; C = 100
    return m, stages, heads, clean, torch.stack(cor), tag, C


m, stages, heads, clean, cor, tag, C = build()
res_json = json.load(open(os.path.join(ROOT, 'results', f'{tag}_main.json')))
T_exit = res_json['T_exit']
sel = res_json['selection']
ref = json.load(open(os.path.join(CACHE_DIR, tag, 'ref.json')))
from riskee.analysis import COSTS
costs = COSTS[tag]   # measured costs (only enter C_k here)
# S_k needs the training-split centroids (written by the caching step) and d_bar (ref.json)


def centroids():
    path = os.path.join(CACHE_DIR, tag, 'centroids.pt')
    if os.path.exists(path): return [c.to(DEV) for c in torch.load(path)]
    raise SystemExit(f'missing {path}: run the caching step first')


cents = centroids()
dbar = [ref[str(k)]['dbar'] for k in (1, 2, 3)]
logC = math.log(C)


def decide(k, logits, feat, name):
    s = sel[name]; m_ = s['method']
    p = logits.softmax(1)
    if m_ == 'full' or m_ == 'ts_full': return torch.zeros(len(p), dtype=torch.bool, device=DEV)
    if m_ == 'conf': return p.max(1).values >= s['tau']
    if m_ == 'pets_conf': return (logits / T_exit[k]).softmax(1).max(1).values >= s['tau']
    if m_ == 'entropy':
        U = -(p * (p + 1e-12).log()).sum(1) / logC
        return U <= s['tau']
    if m_ == 'risk':
        U = -(p * (p + 1e-12).log()).sum(1) / logC
        S = ((feat - cents[k]).norm(dim=1) / dbar[k] - 1).clamp(min=0)
        return U + s['l1'] * S + s['l2'] * costs[k] <= s['tau']
    raise ValueError


@torch.no_grad()
def infer_bs1(x, name):
    h = x
    for k in range(3):
        h = stages[k](h)
        if name == 'full' and k < 2: continue          # plain network: no exit heads
        lg, f = heads[k](h)
        if k < 2 and bool(decide(k, lg, f, name)[0]):   # .item(): host sync, as in deployment
            return k
    return 2


@torch.no_grad()
def infer_batch(x, name):
    h = x; active = torch.arange(len(x), device=DEV)
    out = torch.empty(len(x), C, device=DEV)
    for k in range(3):
        h = stages[k](h)
        if name == 'full' and k < 2: continue          # plain network: no exit heads
        lg, f = heads[k](h)
        if k < 2:
            ex = decide(k, lg, f, name)
            if COUNT is not None: COUNT[k] += int(ex.sum())
            out[active[ex]] = lg[ex]
            keep = ~ex
            if not bool(keep.any()): return out
            h = h[keep]; active = active[keep]
        else:
            out[active] = lg
    return out


def timeit(X, name, mode, reps=3):
    X = X.to(DEV)
    for _ in range(2):  # warm-up
        if mode == 'bs1':
            for i in range(50): infer_bs1(X[i:i + 1], name)
        else:
            infer_batch(X[:128], name)
    torch.cuda.synchronize(); ts = []
    for r in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if mode == 'bs1':
            for i in range(len(X)): infer_bs1(X[i:i + 1], name)
        else:
            for i in range(0, len(X) - 127, 128): infer_batch(X[i:i + 128], name)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    nb = len(X) if mode == 'bs1' else (len(X) // 128) * 128
    return 1000 * np.median(ts) / nb   # ms per image


methods = ['full', 'conf', 'entropy', 'pets_conf', 'risk']
out = {'selection': {n: sel[n] for n in methods}}
# sanity check: exit fractions on these clean images vs the cached analysis
COUNT = None
for name in methods[1:]:
    COUNT = [0, 0]
    for i in range(0, len(clean) - 127, 128): infer_batch(clean[i:i + 128].to(DEV), name)
    nb = (len(clean) // 128) * 128
    out[f'exitfrac/clean/{name}'] = [COUNT[0] / nb, COUNT[1] / nb]
    print('exit fractions', name, [round(c / nb, 3) for c in COUNT], 'analysis (full test):',
          [round(x, 3) for x in res_json['clean'][name]['exit_dist'][:2]], flush=True)
COUNT = None
for mode in ('bs1', 'bs128'):
    for dname, X in (('clean', clean), ('corrupt', cor)):
        for name in methods:
            out[f'{mode}/{dname}/{name}'] = timeit(X, name, mode)
            print(mode, dname, name, f"{out[f'{mode}/{dname}/{name}']:.3f} ms/img", flush=True)
json.dump(out, open(os.path.join(ROOT, 'results', f'latency_{tag}_main.json'), 'w'), indent=1)
