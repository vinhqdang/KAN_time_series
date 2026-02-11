import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.benchmarks.datasets import get_synthetic_linear
from sklearn.linear_model import Lasso

def verify():
    print("Generating Synthetic Linear Data...")
    X, adj = get_synthetic_linear(n_samples=1000, n_nodes=5)
    
    print(f"X shape: {X.shape}")
    print(f"X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"X mean: {X.mean():.4f}, std: {X.std():.4f}")
    
    print("Adj (Ground Truth):")
    print(adj)
    
    # Check Spectral Radius of underlying process (need to regenerate manually to verify coeffs)
    # We can't access coeffs from get_synthetic_linear directly as it returns X, adj.
    # But we can estimate it with Least Squares.
    
    print("\nFitting Least Squares (OLS) to estimate coeffs...")
    # X_t = A X_{t-1}
    Inp = X[:-1]
    Tgt = X[1:]
    
    # OLS: A = (X'X)^-1 X'Y
    # Or per component
    coeffs_est = np.zeros_like(adj, dtype=float)
    for i in range(5):
        # target column i
        y = Tgt[:, i]
        # fit on all previous
        res = np.linalg.lstsq(Inp, y, rcond=None)
        coeffs_est[i] = res[0]
        
    print("Estimated Coeffs (OLS):")
    print(coeffs_est.round(2))
    
    print("\nRunning Lasso (alpha=0.01)...")
    lasso_adj = np.zeros_like(adj)
    for i in range(5):
        model = Lasso(alpha=0.01)
        model.fit(Inp, Tgt[:, i])
        lasso_adj[i] = model.coef_
        
    print("Lasso Coeffs:")
    print(lasso_adj.round(2))
    
    # Threshold
    thresh = 0.05
    adj_pred = (np.abs(lasso_adj) > thresh).astype(int)
    print("\nPredicted Adj (Lasso > 0.05):")
    print(adj_pred)
    
    # Metrics
    tp = (adj_pred * adj).sum()
    fp = (adj_pred * (1-adj)).sum()
    fn = ((1-adj_pred) * adj).sum()
    
    prec = tp / (tp + fp) if (tp+fp) > 0 else 0
    rec = tp / (tp + fn) if (tp+fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec+rec) > 0 else 0
    
    print(f"\nMetrics: P={prec:.2f}, R={rec:.2f}, F1={f1:.2f}")

if __name__ == "__main__":
    verify()
