"""
python -m riskee.tiny {prepare | baseline | ee | cache}

Larger-scale benchmark: Tiny-ImageNet (200 classes,
64x64) with Tiny-ImageNet-C (Hendrycks & Dietterich 2019; 15 corruptions x
5 severities applied to the official 10k validation images).

Protocol mirrors the CIFAR-10 / ResNet-18 experiment exactly:
  * same EarlyExitResNet (torchvision ResNet-18, exits after layer2/layer3),
    trained FROM SCRATCH (no ImageNet weights: Tiny-ImageNet images are
    drawn from ImageNet, so ImageNet pre-training would leak test images);
  * baseline: SGD(0.1, nesterov, wd 5e-4), cosine, batch 128;
  * early-exit: Phase 1 (backbone frozen, lr 0.01) + Phase 2 (end-to-end,
    lr 0.001), loss = CE_final + 0.5 CE_1 + 0.5 CE_2 + 0.1 * Brier;
  * split: 90% of the official train set for training, 10% (50 img/class)
    held out as validation for ALL selection; the official val set (which
    Tiny-ImageNet-C corrupts) is the test set.
"""
import os, sys, glob, json, math, time, zipfile, tarfile
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
TD = os.path.join(ROOT, 'data', 'tiny')
CK = os.path.join(ROOT, 'checkpoints_tiny'); os.makedirs(CK, exist_ok=True)
DEV = torch.device('cuda')
TINY_C = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
          'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
          'brightness', 'contrast', 'elastic_transform', 'pixelate',
          'jpeg_compression']
SEED = 42
BASE_EPOCHS = int(os.environ.get('TINY_BASE_EPOCHS', 100))
EE_EPOCHS = 30


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_img(p):
    return np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)


def prepare():
    src = os.path.join(TD, 'tiny-imagenet-200')
    if not os.path.isdir(src):
        zipfile.ZipFile(os.path.join(TD, 'tiny-imagenet-200.zip')).extractall(TD)
    wnids = [l.strip() for l in open(os.path.join(src, 'wnids.txt'))]
    cid = {w: i for i, w in enumerate(wnids)}
    X, Y = [], []
    for w in wnids:
        for p in sorted(glob.glob(os.path.join(src, 'train', w, 'images', '*.JPEG'))):
            X.append(load_img(p)); Y.append(cid[w])
    X = np.stack(X); Y = np.array(Y, np.int64)
    ann = [l.split('\t') for l in open(os.path.join(src, 'val', 'val_annotations.txt'))]
    vX = np.stack([load_img(os.path.join(src, 'val', 'images', a[0])) for a in ann])
    vY = np.array([cid[a[1]] for a in ann], np.int64)
    # stratified 90/10 split of the training set
    rng = np.random.default_rng(SEED)
    va = np.concatenate([rng.permutation(np.where(Y == c)[0])[:50] for c in range(200)])
    tr = np.setdiff1d(np.arange(len(Y)), va)
    np.savez(os.path.join(TD, 'tiny_arrays.npz'), Xtr=X[tr], Ytr=Y[tr], Xva=X[va],
             Yva=Y[va], Xte=vX, Yte=vY)
    json.dump(wnids, open(os.path.join(TD, 'wnids.json'), 'w'))
    log('train', X[tr].shape, 'val', X[va].shape, 'test', vX.shape)
    # Tiny-ImageNet-C
    tarp = os.path.join(TD, 'Tiny-ImageNet-C.tar')
    cdir = os.path.join(TD, 'Tiny-ImageNet-C')
    if not os.path.isdir(cdir):
        tarfile.open(tarp).extractall(TD)
    # Tiny-ImageNet-C files are NOT pixel-aligned with the Tiny-ImageNet val
    # files (verified: names test_N do not map to val_N; content does not
    # match).  Each corruption set is therefore stored with its OWN labels,
    # read from the class folder of every image (standard protocol).
    for c in TINY_C:
        for s in range(1, 6):
            out = os.path.join(TD, 'c', f'{c}_{s}.npz')
            if os.path.exists(out): continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            X, Yc = [], []
            for w in wnids:
                for p in sorted(glob.glob(os.path.join(cdir, c, str(s), w, '*.JPEG'))):
                    X.append(load_img(p)); Yc.append(cid[w])
            X = np.stack(X); Yc = np.array(Yc, np.int64)
            assert len(Yc) == 10000 and (np.bincount(Yc, minlength=200) == 50).all(), (c, s)
            np.savez(out, X=X, Y=Yc)
            log('corruption', c, s, X.shape)


def arrays():
    z = np.load(os.path.join(TD, 'tiny_arrays.npz'))
    return {k: z[k] for k in z.files}


class GPUData:
    def __init__(self, X, Y, mean, std):
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()  # uint8 cpu
        self.Y = torch.from_numpy(Y)
        self.mean = torch.tensor(mean, device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=DEV).view(1, 3, 1, 1)

    def norm(self, xb):
        return (xb.to(DEV, non_blocking=True).float() / 255 - self.mean) / self.std

    def batches(self, bs, train=False, gen=None):
        n = len(self.Y)
        order = torch.randperm(n, generator=gen) if train else torch.arange(n)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            x = self.norm(self.X[idx]); y = self.Y[idx].to(DEV)
            if train:   # random crop (pad 8) + horizontal flip, on GPU
                B = x.shape[0]
                x = F.pad(x, (8, 8, 8, 8), mode='reflect')
                ox = torch.randint(0, 17, (B,), generator=gen).to(DEV)
                oy = torch.randint(0, 17, (B,), generator=gen).to(DEV)
                base = torch.arange(64, device=DEV)
                rows = (oy[:, None] + base)[:, None, :, None]
                cols = (ox[:, None] + base)[:, None, None, :]
                x = x[torch.arange(B, device=DEV)[:, None, None, None],
                      torch.arange(3, device=DEV)[None, :, None, None], rows, cols]
                flip = (torch.rand(B, generator=gen) < 0.5).to(DEV)
                x[flip] = x[flip].flip(3)
            yield x, y


def stats(A):
    m = (A['Xtr'] / 255.0).mean((0, 1, 2)); s = (A['Xtr'] / 255.0).std((0, 1, 2))
    return m.tolist(), s.tolist()


def make_model():
    from model import EarlyExitResNet
    return EarlyExitResNet(num_classes=200, relative_flops=[0.5482, 0.7741, 1.0]).to(DEV)


@torch.no_grad()
def evaluate(model, data, which='final'):
    model.eval(); c = n = 0
    for x, y in data.batches(500):
        l = model.forward_train(x)[2]
        c += (l.argmax(1) == y).sum().item(); n += len(y)
    return c / n


def train_baseline():
    torch.manual_seed(SEED); np.random.seed(SEED)
    A = arrays(); m, s = stats(A)
    tr = GPUData(A['Xtr'], A['Ytr'], m, s); va = GPUData(A['Xva'], A['Yva'], m, s)
    model = make_model()
    params = [p for n, p in model.named_parameters() if not n.startswith(('exit1', 'exit2'))]
    opt = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, BASE_EPOCHS)
    gen = torch.Generator().manual_seed(SEED); best = 0
    for ep in range(1, BASE_EPOCHS + 1):
        model.train(); t0 = time.time(); tl = 0
        for x, y in tr.batches(128, True, gen):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.forward_train(x)[2], y)
            loss.backward(); opt.step(); tl += loss.item() * len(y)
        sch.step(); acc = evaluate(model, va)
        if acc > best:
            best = acc; torch.save(model.state_dict(), os.path.join(CK, 'baseline.pth'))
        log(f'baseline ep {ep}/{BASE_EPOCHS} loss {tl/len(tr.Y):.3f} val {acc:.4f} best {best:.4f} ({time.time()-t0:.0f}s)')


def brier(l, y):
    p = l.softmax(1); oh = F.one_hot(y, p.shape[1]).float()
    return ((p - oh) ** 2).sum(1).mean()


EXIT_HEAD_LR = 1.0   # chosen from {0.01, 0.1, 1.0} by exit-head val cross-entropy


def train_ee():
    torch.manual_seed(SEED)
    A = arrays(); m, s = stats(A)
    tr = GPUData(A['Xtr'], A['Ytr'], m, s); va = GPUData(A['Xva'], A['Yva'], m, s)
    model = make_model()
    model.load_state_dict(torch.load(os.path.join(CK, 'baseline.pth'), map_location=DEV))
    gen = torch.Generator().manual_seed(SEED)
    for phase, (ep_n, lr, frozen) in enumerate([(EE_EPOCHS // 2, 0.01, True),
                                                 (EE_EPOCHS - EE_EPOCHS // 2, 0.001, False)], 1):
        for n, p in model.named_parameters():
            p.requires_grad_(not frozen or any(h in n for h in ('exit1', 'exit2', 'final_fc')))
        # Phase 1: the new exit heads use EXIT_HEAD_LR (selected on the validation
        # split by a frozen-backbone probe of the exit heads: at the CIFAR rate
        # 0.01 the 200-way heads stay near uniform); final_fc
        # keeps the CIFAR rate. Phase 2: all parameters at the CIFAR rate.
        heads = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(('exit1', 'exit2'))]
        rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(('exit1', 'exit2'))]
        groups = [{'params': heads, 'lr': EXIT_HEAD_LR if frozen else lr}, {'params': rest, 'lr': lr}]
        opt = torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep_n); best = 0
        for ep in range(1, ep_n + 1):
            model.train(); tl = 0
            for x, y in tr.batches(128, True, gen):
                opt.zero_grad(set_to_none=True)
                l1, l2, lf = model.forward_train(x)
                loss = (F.cross_entropy(lf, y) + 0.5 * F.cross_entropy(l1, y) + 0.5 * F.cross_entropy(l2, y)
                        + 0.1 * (brier(l1, y) + brier(l2, y) + brier(lf, y)) / 3)
                loss.backward(); opt.step(); tl += loss.item() * len(y)
            sch.step(); acc = evaluate(model, va)
            if acc > best:
                best = acc; torch.save(model.state_dict(), os.path.join(CK, 'early_exit.pth'))
            log(f'EE phase{phase} ep {ep}/{ep_n} loss {tl/len(tr.Y):.3f} val {acc:.4f} best {best:.4f}')
        model.load_state_dict(torch.load(os.path.join(CK, 'early_exit.pth'), map_location=DEV))


def cache():
    from riskee.common import ScoreBank, save_set, exists
    import riskee.common as C
    A = arrays(); m, s = stats(A)
    model = make_model()
    model.load_state_dict(torch.load(os.path.join(CK, 'early_exit.pth'), map_location=DEV))
    model.eval()

    def fn(x):
        f1, f2, f3 = model._features(x)
        return [model.exit1(f1), model.exit2(f2), model.final_fc(f3)], [f1, f2, f3]

    def loader(X, Y):
        d = GPUData(X, Y, m, s)
        for x, y in d.batches(500):
            yield x, y.cpu()
    tr = C.collect(fn, loader(A['Xtr'], A['Ytr']), want_feats=True)
    bank = ScoreBank(); bank.fit({k: tr[f'feats{k}'] for k in (1, 2, 3)}, tr['labels'], 200)
    ref = bank.ref_json()
    for k in (1, 2, 3):
        ref[str(k)]['energy_mean'] = float(tr[f'energy{k}'].mean())
        ref[str(k)]['energy_std'] = float(tr[f'energy{k}'].std())
    del tr
    os.makedirs(os.path.join(C.CACHE_DIR, 'tiny'), exist_ok=True)
    json.dump(ref, open(os.path.join(C.CACHE_DIR, 'tiny', 'ref.json'), 'w'), indent=1)
    torch.save([bank.stats[k]['mu'].cpu() for k in (1, 2, 3)], os.path.join(C.CACHE_DIR, 'tiny', 'centroids.pt'))
    sets = [('val', A['Xva'], A['Yva']), ('test', A['Xte'], A['Yte'])]
    for c in TINY_C:
        for sv in range(1, 6):
            sets.append((f'c_{c}_{sv}', None, None))
    for name, X, Y in sets:
        if exists('tiny', name): continue
        if X is None:
            z = np.load(os.path.join(TD, 'c', name[2:] + '.npz')); X, Y = z['X'], z['Y']
        res = C.collect(fn, loader(X, Y), bank=bank)
        save_set('tiny', name, res)
        log('tiny', name, 'acc', (res['logits3'].argmax(1) == res['labels']).mean())


if __name__ == '__main__':
    {'prepare': prepare, 'baseline': train_baseline, 'ee': train_ee, 'cache': cache}[sys.argv[1]]()
