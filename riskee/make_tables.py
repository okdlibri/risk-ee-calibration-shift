r"""Generate every results table of the paper from results/*_main.json (no hand-typed numbers).

Output: tex/tables/*.tex
Formatting matches make_numbers.py: accuracy in % (2 dp), ECE / Brier / mCE 4 dp, FLOPs 3 dp.
Markers on ECE entries compare the entry with Risk-EE (paired 95% bootstrap CI):
  \sigb = Risk-EE significantly LOWER ECE than this method; \sigw = Risk-EE significantly HIGHER.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'tex', 'tables')
os.makedirs(OUT, exist_ok=True)

L = lambda f: json.load(open(os.path.join(R, f)))
NAME = {'full': 'Full network', 'ts_full': 'Full network + TS', 'conf': 'Confidence', 'entropy': 'Entropy',
        'pets_conf': 'Per-exit TS + conf.', 'sdn': 'SDN', 'window': 'Window cascade',
        'window_ts': 'Window cascade + TS', 'risk': r'\our{} (ours)', 'risk_pets': r'\our{} + per-exit TS',
        'risk_maha': r'$S_k$ = Mahalanobis', 'risk_energy': r'$S_k$ = energy',
        'risk_noS': r'No $S_k$ ($\lambda_1 = 0$)', 'risk_noC': r'No $C_k$ ($\lambda_2 = 0$)'}
MAIN_ROWS = ['full', 'ts_full', 'conf', 'entropy', 'pets_conf', 'sdn', 'window', 'window_ts', 'risk', 'risk_pets']


def sig(p):
    """marker for method X given paired (risk - X) ECE entry."""
    if p is None:
        return ''
    lo, hi = p['ece']['ci']
    if hi < 0:
        return r'\sigb'
    if lo > 0:
        return r'\sigw'
    return ''


def f_acc(x): return f'{100 * x:.2f}'
def f_ece(x): return f'{x:.4f}'
def f_fl(x): return f'{x:.3f}'


def main_table(tag, label, caption):
    d = L(f'{tag}_main.json')
    c, k, ci, kci = d['clean'], d['corrupt_mean'], d['clean_ci'], d['corrupt_ci']
    lines = [r'\begin{table*}', r'  \centering', f'  \\caption{{{caption}}}', f'  \\label{{{label}}}',
             r'  \setlength{\tabcolsep}{4pt}', r'  \resizebox{\textwidth}{!}{%',
             r'  \begin{tabular}{lcccc@{\hskip 12pt}ccccc}', r'    \toprule',
             r'    & \multicolumn{4}{c}{Clean test set} & \multicolumn{5}{c}{Corrupted test sets (mean)} \\',
             r'    \cmidrule(lr){2-5}\cmidrule(lr){6-10}',
             r'    Method & Acc.\ (\%) & ECE$\downarrow$ [95\% CI] & Brier$\downarrow$ & FLOPs'
             r' & Acc.\ (\%) & ECE$\downarrow$ [95\% CI] & Brier$\downarrow$ & FLOPs & mCE$\downarrow$ \\',
             r'    \midrule']
    for m in MAIN_ROWS:
        if m not in c:
            continue
        if m == 'conf' or m == 'risk':
            lines.append(r'    \midrule')
        pc = None if m == 'risk' else d['clean_paired'].get(m)
        pk = None if m == 'risk' else d['corrupt_paired'].get(m)
        name = NAME[m]
        ec = f'{f_ece(c[m]["ece"])}{sig(pc)} [{f_ece(ci[m]["ece"][0])}, {f_ece(ci[m]["ece"][1])}]'
        ek = f'{f_ece(k[m]["ece"])}{sig(pk)} [{f_ece(kci[m]["ece"][0])}, {f_ece(kci[m]["ece"][1])}]'
        row = [name, f_acc(c[m]['acc']), ec, f_ece(c[m]['brier']), f_fl(c[m]['flops']),
               f_acc(k[m]['acc']), ek, f_ece(k[m]['brier']), f_fl(k[m]['flops']), f_ece(k[m]['mce'])]
        if m == 'risk':
            row = [r'\textbf{' + x + '}' if i else x for i, x in enumerate(row)]
            row[0] = r'\textbf{' + name + '}'
        lines.append('    ' + ' & '.join(row) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}}', r'\end{table*}', '']
    return '\n'.join(lines)


def seeds_table():
    tags = ['cnn', 'cnn_seed42', 'cnn_seed123', 'cnn_seed456']
    D = {t: L(f'{t}_main.json') for t in tags}
    head = ['Main', 'Seed 42', 'Seed 123', 'Seed 456']
    lines = [r'\begin{table*}', r'  \centering',
             r'  \caption{Four independently trained ResNet-18 early-exit models on CIFAR-10 / CIFAR-10-C'
             r' at budget 0.80, each with its own validation-selected $(\lambda_1, \lambda_2)$.'
             r' Upper block: \our{} itself.  Lower block: paired ECE difference \our{} $-$ method'
             r' (negative: \our{} better calibrated); $^{*}$ the 95\% paired bootstrap interval'
             r' excludes zero.  Full network + TS uses 100\% FLOPs.}',
             r'  \label{tab:seeds}', r'  \setlength{\tabcolsep}{4pt}', r'  \resizebox{\textwidth}{!}{%',
             r'  \begin{tabular}{l' + 'cc' * 4 + '}', r'    \toprule',
             '    & ' + ' & '.join(r'\multicolumn{2}{c}{' + h + '}' for h in head) + r' \\',
             '    ' + ''.join(rf'\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(4)),
             '    & ' + ' & '.join(['Clean & Corr.'] * 4) + r' \\', r'    \midrule']
    lines.append(r'    $(\lambda_1, \lambda_2)$ & ' + ' & '.join(
        rf'\multicolumn{{2}}{{c}}{{({D[t]["selection"]["risk"]["l1"]}, {D[t]["selection"]["risk"]["l2"]})}}' for t in tags) + r' \\')
    lines.append(r'    \our{} accuracy (\%) & ' + ' & '.join(
        f'{f_acc(D[t]["clean"]["risk"]["acc"])} & {f_acc(D[t]["corrupt_mean"]["risk"]["acc"])}' for t in tags) + r' \\')
    lines.append(r'    \our{} ECE & ' + ' & '.join(
        f'{f_ece(D[t]["clean"]["risk"]["ece"])} & {f_ece(D[t]["corrupt_mean"]["risk"]["ece"])}' for t in tags) + r' \\')
    lines.append(r'    \our{} FLOPs & ' + ' & '.join(
        f'{f_fl(D[t]["clean"]["risk"]["flops"])} & {f_fl(D[t]["corrupt_mean"]["risk"]["flops"])}' for t in tags) + r' \\')
    lines.append(r'    \midrule')
    lines.append(r'    \multicolumn{9}{l}{\emph{ECE difference, \our{} $-$ method}} \\')
    for m in ['conf', 'entropy', 'pets_conf', 'ts_full', 'window', 'window_ts', 'risk_maha', 'risk_energy', 'risk_noS', 'risk_noC']:
        cells = []
        for t in tags:
            for sp in ['clean', 'corrupt']:
                p = D[t][f'{sp}_paired'][m]['ece']
                base = D[t]['clean'] if sp == 'clean' else D[t]['corrupt_mean']
                pd = base['risk']['ece'] - base[m]['ece']   # point difference, as in the text
                assert p['ci'][0] - 1e-12 <= pd <= p['ci'][1] + 1e-12
                s = '^{*}' if (p['ci'][0] > 0 or p['ci'][1] < 0) else r'^{\phantom{*}}'
                cells.append(f'${pd:+.4f}{s}$')
        lines.append(f'    {NAME[m]} & ' + ' & '.join(cells) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}}', r'\end{table*}', '']
    return '\n'.join(lines)


def variants_table():
    tags = [('cnn', 'ResNet-18 / CIFAR-10'), ('vit', 'DeiT-S / CIFAR-100'), ('tiny', 'ResNet-18 / Tiny-ImageNet')]
    D = {t: L(f'{t}_main.json') for t, _ in tags}
    rows = ['risk', 'risk_maha', 'risk_energy', 'risk_noS', 'risk_noC', 'risk_pets']
    lines = [r'\begin{table*}', r'  \centering',
             r'  \caption{Design of the risk score: \our{} with the centroid deviation $S_k$, with $S_k$ replaced by'
             r' standard feature-based OOD scores, and with one term removed.  ECE at the matched operating'
             r' point; each variant selects its own free weights on validation by the same rule.'
             r' \sigb: \our{} has significantly lower ECE (95\% paired bootstrap interval);'
             r' \sigw: significantly higher.}',
             r'  \label{tab:variants}', r'  \setlength{\tabcolsep}{4pt}', r'  \resizebox{\textwidth}{!}{%',
             r'  \begin{tabular}{l' + 'ccc' * 3 + '}', r'    \toprule',
             '    & ' + ' & '.join(r'\multicolumn{3}{c}{' + h + '}' for _, h in tags) + r' \\',
             '    ' + ''.join(rf'\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(3)),
             '    Variant & ' + ' & '.join([r'Clean ECE & Corr.\ ECE & $(\lambda_1, \lambda_2)$'] * 3) + r' \\',
             r'    \midrule']
    for m in rows:
        cells = []
        for t, _ in tags:
            d = D[t]
            pc = None if m == 'risk' else d['clean_paired'].get(m)
            pk = None if m == 'risk' else d['corrupt_paired'].get(m)
            s = d['selection'][m]
            l1 = s.get('l1', 0.0); l2 = s.get('l2', 0.0)
            cells += [f'{f_ece(d["clean"][m]["ece"])}{sig(pc)}', f'{f_ece(d["corrupt_mean"][m]["ece"])}{sig(pk)}',
                      f'({l1:g}, {l2:g})']
        name = NAME[m] if m != 'risk' else r'\our{} (centroid $S_k$)'
        if m == 'risk':
            cells = [r'\textbf{' + x + '}' for x in cells]
            name = r'\textbf{' + name + '}'
        if m == 'risk_pets':
            lines.append(r'    \midrule')
        lines.append(f'    {name} & ' + ' & '.join(cells) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}}', r'\end{table*}', '']
    return '\n'.join(lines)


def latency_table():
    lines = [r'\begin{table}', r'  \centering',
             r'  \caption{Measured wall-clock time at the operating points of Tables~\ref{tab:cnn_main}'
             r' and~\ref{tab:vit_main} (one NVIDIA Quadro P2200).  Full: milliseconds per image for the'
             r' full network.  Other columns: speed-up over the full network (values below 1 are slower).'
             r' Batch 128 removes exited samples from the batch.  $\Delta$: \our{} minus confidence'
             r' routing, ms per image.}',
             r'  \label{tab:latency}', r'  \setlength{\tabcolsep}{3.5pt}', r'  \resizebox{\columnwidth}{!}{%',
             r'  \begin{tabular}{lllcccccc}', r'    \toprule',
             r'    Model & Batch & Data & Full (ms) & Conf. & Entropy & PETS+conf. & \our{} & $\Delta$ (ms) \\',
             r'    \midrule']
    for arch, nm in [('cnn', 'ResNet-18'), ('vit', 'DeiT-S')]:
        lat = L(f'latency_{arch}_main.json')
        for bs, bn in [('bs1', '1'), ('bs128', '128')]:
            for cdn, cn in [('clean', 'clean'), ('corrupt', 'corr.')]:
                f = lat[f'{bs}/{cdn}/full']
                sp = [f'{f / lat[f"{bs}/{cdn}/{m}"]:.2f}' for m in ['conf', 'entropy', 'pets_conf', 'risk']]
                dm = lat[f'{bs}/{cdn}/risk'] - lat[f'{bs}/{cdn}/conf']
                fs = f'{f:.2f}' if f >= 1 else f'{f:.3f}'
                ds = f'{dm:+.2f}' if abs(dm) >= 0.01 else f'{dm:+.3f}'
                lines.append(f'    {nm} & {bn} & {cn} & {fs} & ' + ' & '.join(sp) + f' & {ds} \\\\')
        if arch == 'cnn':
            lines.append(r'    \midrule')
    lines += [r'    \bottomrule', r'  \end{tabular}}', r'\end{table}', '']
    return '\n'.join(lines)


SIGNOTE = (r' \sigb: \our{} has significantly lower ECE than this method (95\% paired bootstrap'
           r' interval); \sigw: significantly higher; no mark: interval contains zero.')
tables = {
    'tab_cnn_main.tex': main_table(
        'cnn', 'tab:cnn_main',
        r'ResNet-18 on CIFAR-10 (clean) and CIFAR-10-C (mean over 95 corrupted sets) at a budget of 0.80'
        r' of full-network FLOPs.  Thresholds, temperatures and weights are selected on the clean validation'
        r' split and kept fixed on corrupted data; FLOPs are measured on each evaluation set.' + SIGNOTE),
    'tab_vit_main.tex': main_table(
        'vit', 'tab:vit_main',
        r'DeiT-Small on CIFAR-100 (clean) and CIFAR-100-C (mean over 95 corrupted sets) at a budget of 0.70'
        r' of full-network FLOPs.  Protocol as in Table~\ref{tab:cnn_main}.' + SIGNOTE),
    'tab_tiny_main.tex': main_table(
        'tiny', 'tab:tiny_main',
        r'ResNet-18 on Tiny-ImageNet (clean) and Tiny-ImageNet-C (mean over 75 corrupted sets) at a budget'
        r' of 0.80 of full-network FLOPs.  Protocol as in Table~\ref{tab:cnn_main}; no SDN model was'
        r' trained for this dataset.' + SIGNOTE),
    'tab_seeds.tex': seeds_table(),
    'tab_variants.tex': variants_table(),
    'tab_latency.tex': latency_table(),
}
for fn, txt in tables.items():
    open(os.path.join(OUT, fn), 'w', encoding='utf8').write(txt)
print('wrote', ', '.join(tables))
