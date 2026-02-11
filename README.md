# CD-KAN: Causal Discovery Kolmogorov-Arnold Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

> **State-of-the-Art (SOTA) time series causal discovery and forecasting**

CD-KAN combines Kolmogorov-Arnold Networks with differentiable DAG learning to achieve unprecedented performance in both prediction accuracy and causal structure recovery.

## 🏆 Key Results

**Causal Discovery (F1 Score)**:
- CD-KAN v2: **0.8971** (SOTA)
- Best Baseline (PCMCI): 0.6550
- **+37% improvement**, 81% fewer structural errors

**Forecasting (MSE)**:
- CD-KAN: **0.0008** (SOTA)
- TSMixer: 0.0178
- **20× better performance**

Evaluated on **10 diverse datasets** (500-5000 samples) against **8 baseline algorithms**.

## 📋 Features

### Core Capabilities
- **Dual SOTA**: Best-in-class for both forecasting and causal discovery
- **Differentiable DAG Learning**: Augmented Lagrangian Method (ALM) for strict acyclicity
- **Scalable**: Linear complexity O(n·d), handles large datasets efficiently
- **Interpretable**: Learns sparse causal graphs with temporal lag information
- **Robust**: Handles linear, nonlinear, and time-varying dynamics

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

### Comprehensive Evaluation

We evaluated CD-KAN v2 against 7 baseline algorithms on 10 datasets:

| Algorithm | Mean F1 | Mean SHD | Runtime (s) |
|-----------|---------|----------|-------------|
| **CD-KAN v2** | **0.8971** | **1.4** | 10.32 |
| PCMCI | 0.6550 | 10.6 | 32.56 |
| VAR-Lasso | 0.6434 | 7.6 | 4.76 |
| NTiCD | 0.5939 | 10.4 | 19.26 |
| VAR-GrangerTest | 0.5585 | 13.6 | 6.81 |
| NOTEARS | 0.5075 | 15.4 | 37.09 |
| DYNOTEARS | 0.5069 | 13.0 | 53.13 |
| GOLEM | 0.4954 | 15.0 | 45.22 |

**Datasets Tested**:
- Synthetic: Linear/Nonlinear/TimeVarying (500-2000 samples)
- Real-world: Financial, Crypto, Macro, Energy Grid, Climate Sensors

See [`experimental_results/REPORT_CDKAN.md`](experimental_results/REPORT_CDKAN.md) for full details.

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

Where $h(A) = \text{tr}(e^A) - d$ ensures acyclicity.

**Schedule**:
- Inner loop: Minimize augmented Lagrangian (10 epochs)
- Outer loop: Update dual variables $\rho$, $\alpha$ if $h(A) > 10^{-8}$

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
- [NOTEARS: DAG Learning via Continuous Optimization](https://arxiv.org/abs/1803.01422)
- [NTiCD: Neural Time-invariant Causal Discovery](https://openreview.net/forum?id=N8tggW5dWQ)

## 📞 Contact

**Author**: Dr. Dang Quang Vinh  
**Email**: vinh.dq4@buv.edu.vn  
**Repository**: https://github.com/vinhqdang/KAN_time_series

---

**Status**: Production-ready SOTA implementation with comprehensive benchmarks. Last updated: February 2025.
