"""
run_sensitivity.py — Sensitivity analysis for CD-KAN hyperparameters.

Sweeps over:
  - tau_init      : Gumbel-Sigmoid temperature
  - lambda_sparse : edge sparsity regularization
  - rho_init      : ALM penalty coefficient
  - grid_size     : B-spline grid size

Usage:
    python scripts/run_sensitivity.py          # full sweep
    python scripts/run_sensitivity.py --quick  # small grid (for testing)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cdkan.model import CDKANForecaster
from src.cdkan.trainer import CDKANTrainer, CDKANTrainerConfig
from src.synthetic import generate_nonlinear_scm


# ---------------------------------------------------------------------------
# Dataset preparation helper
# ---------------------------------------------------------------------------

def make_synthetic_loaders(n_samples=600, n_nodes=5, window=16, seed=0):
    """Generate a synthetic SCM dataset and return train/test dict loaders."""
    data, adj, _ = generate_nonlinear_scm(
        n_samples=n_samples, n_nodes=n_nodes, density=0.3,
        max_lag=3, seed=seed,
    )
    # Normalize
    mean = data.mean(0, keepdims=True)
    std  = data.std(0, keepdims=True) + 1e-8
    data = (data - mean) / std

    # Sliding windows
    horizon = 1
    X, y = [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i : i + window])
        y.append(data[i + window])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    split = int(len(X) * 0.8)
    dev = torch.device('cpu')

    train_loader = {
        'X': torch.from_numpy(X[:split]).to(dev),
        'y': torch.from_numpy(y[:split]).to(dev),
    }
    test_loader = {
        'X': torch.from_numpy(X[split:]).to(dev),
        'y': torch.from_numpy(y[split:]).to(dev),
    }
    return train_loader, test_loader, n_nodes, adj


# ---------------------------------------------------------------------------
# Sensitivity sweep
# ---------------------------------------------------------------------------

def run_sensitivity(quick: bool = False):
    print("=" * 60)
    print("CD-KAN Sensitivity Analysis")
    print("=" * 60)

    if quick:
        param_grid = {
            'tau_init':      [1.0, 0.5],
            'lambda_sparse': [0.01, 0.1],
            'rho_init':      [1.0],
            'grid_size':     [5],
        }
        epochs = 20
    else:
        param_grid = {
            'tau_init':      [1.0, 0.5, 0.2],
            'lambda_sparse': [0.001, 0.01, 0.1],
            'rho_init':      [0.5, 1.0, 5.0],
            'grid_size':     [5, 10, 20],
        }
        epochs = 50

    train_loader, test_loader, n_nodes, _ = make_synthetic_loaders(seed=42)

    results = []

    from itertools import product as iproduct
    keys  = list(param_grid.keys())
    combos = list(iproduct(*[param_grid[k] for k in keys]))
    print(f"Total runs: {len(combos)} × epochs={epochs}")

    for combo in combos:
        params = dict(zip(keys, combo))
        grid_size = int(params.pop('grid_size'))

        def model_fn():
            return CDKANForecaster(
                in_features=n_nodes,
                hidden_dim=32,
                out_features=n_nodes,
                max_lag=5,
                n_layers=2,
                grid_size=grid_size,
            )

        cfg = CDKANTrainerConfig(
            tau_init      = params['tau_init'],
            tau_final     = 0.05,
            tau_decay     = 0.97,
            lambda_sparse = params['lambda_sparse'],
            rho_init      = params['rho_init'],
            rho_max       = 1e10,
            seed          = 42,
        )
        params['grid_size'] = grid_size  # restore

        model   = model_fn()
        trainer = CDKANTrainer(model, device='cpu', config=cfg)
        hist    = trainer.train(train_loader, test_loader,
                                epochs=epochs, patience=epochs)

        row = {
            **params,
            'grid_size':     grid_size,
            'final_val_mse': hist['test_loss'][-1],
            'min_val_mse':   min(hist['test_loss']),
            'final_h':       hist['h_residuals'][-1],
            'min_h':         min(hist['h_residuals']),
        }
        results.append(row)
        print(f"  [OK] {params} => val_mse={row['final_val_mse']:.5f}  h(W)={row['final_h']:.4e}")

    df = pd.DataFrame(results)
    out_path = 'sensitivity_results.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSensitivity results saved to {out_path}")
    print(df.sort_values('final_val_mse').to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Run a quick 2×2 grid for testing')
    args = parser.parse_args()
    run_sensitivity(quick=args.quick)
