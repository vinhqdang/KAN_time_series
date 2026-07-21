# CD-KAN: Causal Discovery Kolmogorov-Arnold Networks

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

> **An interpretable forecasting artifact that unifies time-series prediction with lag-resolved causal structure learning.**

CD-KAN combines Kolmogorov-Arnold (B-spline) edge functions with differentiable,
lag-aware DAG learning. It produces a one-step forecast and an interpretable
candidate causal graph from a single differentiable model.

## Key Results (honest, reproducible)

> **Correction (2026).** Earlier versions of this README and the manuscript
> reported causal-discovery F1 = 0.8971 and forecasting MSE = 0.0008 (a "20×"
> gain). **Both are retracted.** Under a fair, leakage-free re-evaluation (see
> `scripts/honest_causal_benchmark.py` and `scripts/honest_forecast_benchmark.py`):
>
> - **Causal discovery:** CD-KAN's adjacency recovers ground-truth structure only
>   *near chance* (AUROC ≈ 0.51 non-linear, 0.58 linear; F1 ≈ 0.33–0.47) and is
>   **significantly worse than every baseline** (VAR-Lasso, NOTEARS, GOLEM, PCMCI,
>   VAR-LiNGAM, and even GC-KAN), Wilcoxon *p* < 0.001. The 0.8971 figure was not
>   reproducible.
> - **Forecasting:** the 0.0008 MSE was an artifact of test-leaking normalization.
>   The correct, leakage-free rolling-origin MSE is on the order of 10⁻². CD-KAN is
>   *competitive* with strong deep baselines but not dominant.
> - **What holds up:** the learned real-market graph is temporally stable ~6× above
>   a permutation chance level (*p* < 0.0001), and the spline edge functions provide
>   genuine interpretability.

See the manuscript in [`manuscript/`](manuscript/) and
[`experimental_results/NUMBERS.json`](experimental_results/) for the numbers.

## 📋 Features

### Core Capabilities
- **Unified model**: one differentiable model yields a forecast and a candidate causal graph
- **Differentiable DAG Learning**: Augmented Lagrangian Method (ALM); acyclicity enforced on the contemporaneous block only
- **Lag-aware**: separate adjacency per lag, with expected-lag attribution
- **Interpretable**: learnable B-spline edge functions expose the shape of each dependency
- **Complexity**: O(N·d²·G) per epoch (linear in samples N); O(d³) acyclicity term — suited to moderate d

### Technical Innovations
- **KAN Function Approximation**: Learnable spline-based activation functions
- **CausalStructure Module**: Learnable adjacency matrix with Gumbel-Sigmoid sampling
- **Reversible Instance Normalization (RevIN)**: Handles non-stationary data
- **Residual Architecture**: Deep network with skip connections for stability

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/vinhqdang/KAN_time_series.git
cd KAN_time_series

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualizations)

### Basic Usage

```python
import torch
from src.cdkan.model import CDKANForecaster
from src.cdkan.trainer import CDKANTrainer

# Load your time series data
X = torch.randn(1000, 10, 5)  # [batch, window, features]
y = torch.randn(1000, 5)      # [batch, features]

# Initialize model
model = CDKANForecaster(
    in_features=5,
    hidden_dim=128,
    out_features=5,
    max_lag=5,
    n_layers=4
)

# Train
trainer = CDKANTrainer(model, device='cuda')
history = trainer.train(
    train_loader={'X': X, 'y': y},
    test_loader={'X': X, 'y': y},
    epochs=100
)

# Get learned causal structure
adjacency = model.cd_layer.get_adjacency()
print("Causal Adjacency Matrix:", adjacency.shape)
```

## 📊 Benchmark Results

### Causal discovery, synthetic ground truth (AUROC; mean over 4 configs × 3 seeds)

| Method | Non-linear AUROC | Linear AUROC |
|--------|:---:|:---:|
| GOLEM | 0.98 | 1.00 |
| PCMCI | 0.98 | 1.00 |
| NOTEARS | 0.97 | 1.00 |
| VAR-Lasso | 0.97 | 1.00 |
| VAR-LiNGAM | 0.94 | 1.00 |
| GC-KAN | 0.89 | 0.97 |
| GC-KAN + ALM | 0.90 | 0.97 |
| **CD-KAN (ours)** | **0.51** | **0.58** |

CD-KAN is the weakest method — its forecaster-coupled adjacency does not reliably
recover structure. Reproduce with `python scripts/honest_causal_benchmark.py`.

Forecasting is evaluated under a leakage-free rolling-origin protocol
(`scripts/honest_forecast_benchmark.py`), which also demonstrates the
normalization leak behind the retracted 0.0008 figure.

## 📁 Project Structure

```
KAN_time_series/
├── src/
│   ├── cdkan/              # CD-KAN implementation
│   │   ├── model.py        # CDKANForecaster
│   │   ├── layers.py       # CausalStructure, CDKANLayer
│   │   ├── trainer.py      # Augmented Lagrangian Method
│   │   ├── losses.py       # DAG constraint, sparsity penalties
│   │   └── utils.py        # Gumbel-Sigmoid, attention modules
│   ├── benchmarks/         # Benchmark infrastructure
│   │   ├── datasets.py     # Data loaders
│   │   └── baselines.py    # Baseline algorithms
│   ├── data.py             # Data preprocessing
│   └── synthetic.py        # Synthetic data generation
├── scripts/                # Utility scripts
│   ├── run_benchmark.py    # Main benchmark script
│   └── create_visualizations.py
├── experimental_results/   # Benchmark results and reports
│   ├── REPORT_CDKAN.md     # Comprehensive evaluation report
│   ├── benchmark_comprehensive_sota.csv
│   └── *.png               # Visualizations
└── tests/                  # Unit tests
```

## 🔬 Methodology

### Architecture

CD-KAN v2 employs a multi-stage architecture:

1. **Input Normalization**: RevIN for handling distribution shifts
2. **Causal Discovery Layer**: Learns sparse adjacency matrix with temporal lags
3. **Deep KAN Backbone**: Residual blocks with learnable activation functions
4. **Output Projection**: Final KAN layer for predictions

### Training

Augmented Lagrangian Method (ALM) for DAG constraint enforcement:

$$\mathcal{L} = \text{MSE} + \lambda_{\text{sparse}} \sum A_{ij} + \alpha h(A) + \frac{\rho}{2} h(A)^2$$

Where $h(A) = \text{tr}(e^{A \circ A}) - d$ ensures acyclicity, applied **only to
the contemporaneous adjacency block** (lagged edges are acyclic by temporal order).

**Schedule** (see `src/cdkan/trainer.py`):
- Inner loop: minimize the augmented Lagrangian (Adam, gradient clip)
- Outer loop: every 10 epochs, if $h > 10^{-8}$ then $\rho \leftarrow 2\rho$, $\alpha \leftarrow \alpha + \rho h$
- Gumbel temperature annealed $1.0 \rightarrow 0.1$

## 📖 Citation

If you use CD-KAN in your research, please cite:

```bibtex
@software{cdkan2025,
  title = {CD-KAN: Causal Discovery Kolmogorov-Arnold Networks},
  author = {Dang, Vinh Quang},
  year = {2025},
  url = {https://github.com/vinhqdang/KAN_time_series}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
- [NOTEARS: DAG Learning via Continuous Optimization](https://arxiv.org/abs/1803.01422)
- [NTiCD: Neural Time-invariant Causal Discovery](https://openreview.net/forum?id=N8tggW5dWQ)

## 📞 Contact

**Author**: Dr. Dang Quang Vinh  
**Email**: vinh.dq4@buv.edu.vn  
**Repository**: https://github.com/vinhqdang/KAN_time_series

---

**Status**: Research prototype. Headline claims from earlier versions have been
retracted; see the correction note above and the honest benchmark scripts.
