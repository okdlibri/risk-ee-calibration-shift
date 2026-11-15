"""
python -m riskee.flops_handcount -- independent check of the relative exit costs.

Counts multiply-accumulates (MACs) of every Conv2d and Linear layer with the
textbook formula, using forward hooks only to read layer output shapes
(no thop):
    Conv2d: C_out * (C_in / groups) * k_h * k_w * H_out * W_out
    Linear: in_features * out_features
BatchNorm/ReLU/pooling are ignored (they are <1% here).  Prints cumulative MACs
at exit 1 (after layer2 + exit head), exit 2 (after layer3 + exit head) and the
final exit, relative to the full network, next to the thop-measured vector used
in the analysis (riskee/analysis.COSTS).
The ratios do not depend on the input resolution (each stage halves H, W and
doubles C), which is why CIFAR (32x32) and Tiny-ImageNet (64x64) agree.
"""
import os, sys
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import EarlyExitResNet
from riskee.analysis import COSTS


def count(model, x):
    macs = {}

    def hook(name):
        def f(mod, inp, out):
            if isinstance(mod, nn.Conv2d):
                k = mod.kernel_size[0] * mod.kernel_size[1]
                macs[name] = mod.out_channels * (mod.in_channels // mod.groups) * k * out.shape[2] * out.shape[3]
            elif isinstance(mod, nn.Linear):
                macs[name] = mod.in_features * mod.out_features
        return f

    hs = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        h = model.stem(x); h = model.layer1(h); h = model.layer2(h)
        model.exit1(model.avgpool(h).flatten(1))
        h = model.layer3(h); model.exit2(model.avgpool(h).flatten(1))
        h = model.layer4(h); model.final_fc(model.avgpool(h).flatten(1))
    for hh in hs: hh.remove()
    part = lambda *pre: sum(v for n, v in macs.items() if n.startswith(pre))
    backbone = {s: part(s) for s in ('stem', 'layer1', 'layer2', 'layer3', 'layer4')}
    e1 = backbone['stem'] + backbone['layer1'] + backbone['layer2'] + part('exit1')
    e2 = e1 + backbone['layer3'] + part('exit2')
    full = e2 + backbone['layer4'] + part('final_fc')
    return backbone, [e1 / full, e2 / full, 1.0], full


if __name__ == '__main__':
    for res, ncls, tag in [(32, 10, 'cnn'), (64, 200, 'tiny')]:
        m = EarlyExitResNet(num_classes=ncls).eval()
        bb, rel, full = count(m, torch.zeros(1, 3, res, res))
        print(f'{tag} input {res}x{res}: per-stage MACs (M) ' + ', '.join(f'{k} {v / 1e6:.2f}' for k, v in bb.items()))
        print(f'   cumulative relative cost (conv+linear only): {[round(r, 4) for r in rel]}   total {full / 1e6:.1f} M MACs')
        print(f'   thop-measured (used in the analysis):       {[round(r, 4) for r in COSTS[tag]]}')
