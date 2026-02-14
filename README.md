# TC-Regularized Learnable Discrete Diffusion

This repository contains our implementation for the ETH Deep Learning course (2025). The project extends Forward-Learned Discrete Diffusion (FLDD) with three contributions: (1) a total correlation (TC) regularization term that penalizes statistical dependencies in the aggregated forward posterior, encouraging target distributions that a factorized reverse process can match more effectively, (2) Sinkhorn optimal transport with a learnable cost matrix as an alternative coupling mechanism, and (3) variance-reduction baselines for the REINFORCE gradient estimator.

Based on the paper *"Forward-Learned Discrete Diffusion: Learning How to Noise to Denoise Faster"* (Anonymous, ICLR 2026 submission).

## Project Structure

```
├── fldd.py                          # Core model, training loop, and sampling
├── SinkhornTransport.py             # Sinkhorn optimal transport coupling
├── two_gaussians.py                 # Mixture-of-two-Gaussians toy dataset
├── requirements.txt                 # Python dependencies
├── cluster.md                       # ETH student cluster setup guide
├── run_tc_weights.sh                # Local experiment launcher
├── DL_report_diffusion.pdf          # Project report
├── Tutorial_..._(DDPM).ipynb        # Educational DDPM notebook
└── scripts/
    ├── fldd_baseline_none.slurm     # Grid search: no baseline
    ├── fldd_baseline_09.slurm       # Grid search: baseline beta=0.9
    └── fldd_baseline_099.slurm      # Grid search: baseline beta=0.99
```

## Method Overview

### Theoretical Motivation

The KL-divergence between the aggregated forward posterior and the factorized reverse process decomposes into two terms (Eq. 7 in the report):

1. **Total Correlation (TC)** of the forward posterior -- measures statistical dependence across coordinates. This term depends solely on the forward process and *cannot* be reduced by learning the reverse model.
2. **Dimension-wise KL** -- a sum of per-coordinate KL divergences that the reverse network can minimize.

This decomposition motivates explicitly penalizing the TC during training, encouraging the learned forward process to produce targets that are closer to factorized. The modified objective is: `L = L_ELBO + lambda * E[TC(z_s | z_t)]`.

### Coupling Methods

Three transport plan methods are implemented for constructing the posterior q(z_s | z_t, x):

| Method | Flag | Description |
|---|---|---|
| **Maximum Coupling** | `maximum` | Analytical solution (Eq. 11 from FLDD) that minimizes the probability of state change |
| **Sinkhorn OT (fixed cost)** | `sinkhorn` | Entropic optimal transport with fixed cost matrix C_{u,v} = 1_{u!=v} |
| **Sinkhorn OT (learnable cost)** | `sinkhorn_learnable` | Entropic OT with a cost matrix learned end-to-end during training |

### Training Phases

1. **Warm-up phase** -- Concrete/Gumbel-Softmax relaxation with temperature annealing (1.0 to 1e-3) to enable gradient flow through discrete sampling.
2. **REINFORCE phase** -- Switches to discrete samples with REINFORCE policy gradients. An optional EMA baseline reduces gradient variance.

### Datasets

- **Binary MNIST** (D=784, vocab_size=2, T=4 timesteps) -- evaluated with FID.
- **Mixture of Two Gaussians** (D=2, vocab_size=50, T=2 timesteps) -- evaluated with MMD. A toy dataset where the factorization mismatch manifests as incorrectly mixed samples across modes.

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, torchvision, torchmetrics (FID), scipy, POT (Python Optimal Transport), matplotlib.

## Usage

```bash
python fldd.py \
    --transport sinkhorn_learnable \
    --tc_weight 0.001 \
    --use_baseline True \
    --baseline_beta 0.9
```

| Argument | Description | Default |
|---|---|---|
| `--transport` | Coupling method (`maximum`, `sinkhorn`, `sinkhorn_learnable`) | `maximum` |
| `--tc_weight` | TC regularization weight (lambda) | `1e-4` |
| `--use_baseline` | Enable EMA baseline for REINFORCE (`True`/`False`) | `True` |
| `--baseline_beta` | EMA decay rate for the baseline | `0.9` |

## Experiment Scripts

The `scripts/` folder contains three SLURM job scripts used for our hyperparameter grid search on the ETH student cluster. Each script sweeps over TC weights `[10, 1, 0.1, 0.01, 0.001, 0.0001, 0]` using **Sinkhorn with a learnable cost matrix**, varying the baseline configuration:

| Script | Baseline | Beta |
|---|---|---|
| `fldd_baseline_none.slurm` | Disabled | -- |
| `fldd_baseline_09.slurm` | Enabled | 0.9 |
| `fldd_baseline_099.slurm` | Enabled | 0.99 |

Submit on the ETH cluster with:

```bash
sbatch scripts/fldd_baseline_099.slurm
```

See `cluster.md` for cluster setup instructions.

## Results

Best FID scores on MNIST (T=4, mean over multiple runs, lower is better):

| Coupling | Best FID | TC Weight | Baseline |
|---|---|---|---|
| Maximum Coupling (FLDD baseline) | 40.80 | 0 | None |
| Maximum Coupling + TC + baseline | 31.55 | 1e-4 | beta=0.99 |
| Sinkhorn (fixed cost) | 27.88 | 1e-3 | None |
| Sinkhorn (learnable cost) + TC + baseline | **22.31** | 1e-3 | beta=0.9 |

Our best configuration achieves a **45% FID improvement** over the FLDD baseline. Full results for all combinations of transport plan, TC weight, and baseline are reported in Tables 4--6 of the project report.

## Authors

Luca Vignola, Jonathan Unger, Benson Lee, Benedikt Jung, Isabel Haas
