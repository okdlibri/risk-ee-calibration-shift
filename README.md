# Risk-Aware Early Exit Networks: calibration under distribution shift

Code and result files for the paper *Risk-Aware Early Exit Networks: Calibrated
Adaptive Inference under Distribution Shift* (Omar Dib, Changrui Wu).

Corresponding author: Omar Dib.

Risk-EE routes each input to the first exit k whose risk score

    R_k(x) = U_k(x) + lambda1 * S_k(x) + lambda2 * C_k

is at most a threshold tau, where U_k is the normalised predictive entropy,
S_k = max(0, ||f_k(x) - mu_k|| / d_bar_k - 1) is the deviation of the exit
features from their training-split centroid, and C_k is the relative compute
spent up to exit k.

## Repository layout

| Path | Content |
|---|---|
| `config.py`, `model.py`, `data_utils.py` | ResNet-18 early-exit model, CIFAR-10 / CIFAR-10-C data |
| `config_vit.py`, `model_vit.py`, `data_utils_vit.py` | DeiT-Small early-exit model, CIFAR-100 / CIFAR-100-C data |
| `train_*.py`, `run_seeds.py` | training (baselines, early-exit models, SDN, extra ResNet-18 seeds) |
| `riskee/` | caching of model outputs, evaluation protocol, analyses, tables, figures, numbers |
| `results/` | result files behind every number, table and figure in the paper |

## Installation

Python 3.10 with a CUDA GPU (experiments were run on one NVIDIA Quadro P2200).

    pip install -r requirements.txt

## Data

Put the datasets under `./data`:

| Dataset | Location | Source |
|---|---|---|
| CIFAR-10 | downloaded automatically | https://www.cs.toronto.edu/~kriz/cifar.html |
| CIFAR-100 | `data/cifar-100-python/` | https://www.cs.toronto.edu/~kriz/cifar.html |
| CIFAR-10-C | `data/CIFAR-10-C/` | https://github.com/hendrycks/robustness |
| CIFAR-100-C | `data/CIFAR-100-C/` | https://github.com/hendrycks/robustness |
| Tiny-ImageNet | `data/tiny/tiny-imagenet-200.zip` | http://cs231n.stanford.edu/tiny-imagenet-200.zip |
| Tiny-ImageNet-C | `data/tiny/Tiny-ImageNet-C.tar` | https://github.com/hendrycks/robustness |

## Reproducing the results

All commands are run from the repository root.

**1. Train the models**

    # ResNet-18 / CIFAR-10 (main model, seed 42) and SDN baseline
    python train_baseline.py
    python train_model.py
    python train_sdn.py
    # three additional ResNet-18 seeds (42, 123, 456)
    python run_seeds.py
    # DeiT-Small / CIFAR-100 and SDN baseline
    python train_baseline_vit.py
    python train_model_vit.py
    python train_sdn_vit.py
    # ResNet-18 / Tiny-ImageNet
    python -m riskee.tiny prepare
    python -m riskee.tiny baseline
    python -m riskee.tiny ee

**2. Cache the model outputs.**  Each model is run once on the validation
split, the clean test set and every corrupted set; per-exit logits and feature
scores are stored in `./cache` (set `RISKEE_CACHE` to use another location).
The feature statistics (centroids, d_bar) are fitted on the training split.

    for t in cnn cnn_sdn cnn_seed42 cnn_seed123 cnn_seed456 vit vit_sdn; do
        python -m riskee.cache_cifar --tag $t
    done
    python -m riskee.tiny cache

**3. Evaluate.**  `riskee/protocol.py` selects everything on the validation
split (temperatures; tau at matched compute, i.e. the largest validation FLOPs
not above the budget, 0.80 for ResNet-18 and 0.70 for DeiT-S; and
(lambda1, lambda2) from {0.1, 0.3, 0.5, 0.7, 1.0}^2 by minimum validation ECE),
then applies the frozen configuration to the clean and corrupted test sets,
with 2000 paired bootstrap resamples over test images.

    for t in cnn vit cnn_seed42 cnn_seed123 cnn_seed456 tiny; do
        python -m riskee.protocol --tag $t --boot 2000
    done
    for t in cnn vit tiny; do python -m riskee.theory_check --tag $t; done
    python -m riskee.budget_sweep cnn cnn_seed42 cnn_seed123 cnn_seed456 vit
    python -m riskee.latency --arch cnn
    python -m riskee.latency --arch vit
    python -m riskee.s_stats
    python -m riskee.pair_risk_pets
    python -m riskee.flops_handcount      # check of the ResNet-18 exit costs

**4. Tables, figures and numbers**

    python -m riskee.make_tables     # tex/tables/*.tex
    python -m riskee.make_numbers    # tex/numbers.tex (every number quoted in the text)
    python -m riskee.make_figures    # figures/*.pdf
    python -m riskee.figures_sk      # figures/sk_distribution_*.pdf

`make_tables` reads only `results/`.  `make_numbers` also reads the clean test
caches (per-exit accuracies), and `make_figures` (reliability diagrams) and
`figures_sk` read the caches as well.

## Result files

| File | Produced by | Used for |
|---|---|---|
| `results/<tag>_main.json` | `riskee.protocol` | main tables, seeds table, variants table, severity and per-corruption figures |
| `results/budget_sweep_main.json` | `riskee.budget_sweep` | ECE-vs-budget figures |
| `results/latency_{cnn,vit}_main.json` | `riskee.latency` | latency table |
| `results/<tag>_theory_check_main.json` | `riskee.theory_check` | empirical check of the theoretical results |
| `results/s_stats.json` | `riskee.s_stats` | S_k statistics |
| `results/pair_risk_pets.json` | `riskee.pair_risk_pets` | Risk-EE on per-exit temperature-scaled outputs vs. per-exit TS |

Tags: `cnn` (ResNet-18 / CIFAR-10, main model), `cnn_seed42`, `cnn_seed123`,
`cnn_seed456` (additional seeds), `vit` (DeiT-Small / CIFAR-100), `tiny`
(ResNet-18 / Tiny-ImageNet).

## Licence

MIT (see `LICENSE`).
