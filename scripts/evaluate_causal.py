
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.synthetic import generate_nonlinear_scm, visualize_ground_truth
from src.benchmarks.baselines import VARLasso, CorrelationThresholdBaseline
from src.cdkan.model import CDKANForecaster
from src.cdkan.trainer import CDKANTrainer
from src.config import DEVICE

def shd(target, pred):
    """Structural Hamming Distance: count of edge differences."""
    diff = np.abs(target - pred)
    return np.sum(diff)

def evaluate_graph(target, pred):
    """Compute F1, Precision, Recall, SHD ignoring diagonal self-loops."""
    # Mask diagonals
    mask = ~np.eye(target.shape[0], dtype=bool)
    
    target_flat = target[mask]
    pred_flat = pred[mask]
    
    f1 = f1_score(target_flat, pred_flat, zero_division=0)
    prec = precision_score(target_flat, pred_flat, zero_division=0)
    rec = recall_score(target_flat, pred_flat, zero_division=0)
    s_dist = shd(target[mask], pred[mask])
    
    return {"F1": f1, "Precision": prec, "Recall": rec, "SHD": s_dist}

def plot_adjs(truth, pred, title="Predicted vs Truth", filename="causal_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(truth, annot=True, ax=axes[0], cmap="Blues", cbar=False)
    axes[0].set_title("Ground Truth Graph")
    
    sns.heatmap(pred, annot=True, ax=axes[1], cmap="Reds", cbar=False)
    axes[1].set_title(f"Predicted Graph ({title})")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Comparison plot saved to {filename}")

def main():
    print("Generating Synthetic Data (Non-Linear SCM)...")
    n_nodes = 5
    n_samples = 2000
    data, adj_true, lags_true = generate_nonlinear_scm(n_samples=n_samples, n_nodes=n_nodes, density=0.3, max_lag=2)
    
    # Save ground truth
    visualize_ground_truth(adj_true, labels=[f"X{i}" for i in range(n_nodes)], filename="synthetic_truth.png")
    
    # Prepare Data for CD-KAN
    # [Samples, Window, Nodes]
    window = 10
    X, Y = [], []
    for i in range(len(data) - window - 1):
        X.append(data[i : i+window])
        Y.append(data[i+window : i+window+1]) # Horizon 1
        
    X = np.array(X, dtype=np.float32)
    Y = np.reshape(np.array(Y, dtype=np.float32), (-1, n_nodes))
    
    # Split
    split = int(0.8 * len(X))
    train_loader = {
        'X': torch.from_numpy(X[:split]).to(DEVICE),
        'y': torch.from_numpy(Y[:split]).to(DEVICE)
    }
    test_loader = {
        'X': torch.from_numpy(X[split:]).to(DEVICE),
        'y': torch.from_numpy(Y[split:]).to(DEVICE)
    }
    
    # --- 1. VAR-Lasso Baseline ---
    print("\nRunning VAR-Lasso...")
    lasso = VARLasso(max_lag=2, alpha=0.05)
    lasso.fit(data)
    adj_lasso = lasso.get_adjacency()
    
    # Threshold Baseline (MAGNITUDE PRUNING)
    adj_lasso_binary = (adj_lasso > 0.01).astype(int)
    metrics_lasso = evaluate_graph(adj_true, adj_lasso_binary)
    print(f"VAR-Lasso Results: {metrics_lasso}")
    
    # --- 2. Correlation Baseline ---
    print("\nRunning Correlation...")
    corr = CorrelationThresholdBaseline(threshold=0.3)
    corr.fit(data)
    adj_corr = corr.get_adjacency()
    
    adj_corr_binary = (adj_corr > 0.3).astype(int)
    metrics_corr = evaluate_graph(adj_true, adj_corr_binary)
    print(f"Correlation Results: {metrics_corr}")
    
    # --- 3. CD-KAN ---
    print("\nTraining CD-KAN on Synthetic Data...")
    
    model = CDKANForecaster(
        in_features=n_nodes,
        hidden_dim=32,
        out_features=n_nodes,
        max_lag=2,
        n_layers=2,
        learn_structure=True # Enabled true structural Gumbel gating
    ).to(DEVICE)
    
    trainer = CDKANTrainer(model, device=DEVICE)
    trainer.train(train_loader, test_loader, epochs=60, patience=10)
    
    # Extract Graph using the natively learned probabilities
    adj_prob = model.get_summary_adjacency()
    if isinstance(adj_prob, torch.Tensor):
        adj_prob = adj_prob.cpu().numpy()
        
    print(f"Edge Probability Range: Min {adj_prob.min():.4f}, Max {adj_prob.max():.4f}, Mean {adj_prob.mean():.4f}")
        
    # Thresholding directly on learned binary confidence probabilities
    thresh = 0.5
    print(f"Confidence Threshold: {thresh}")
    
    adj_cdkan = (adj_prob > thresh).astype(int) 
    
    metrics_cdkan = evaluate_graph(adj_true, adj_cdkan)
    print(f"CD-KAN Results (Thresh {thresh}): {metrics_cdkan}")
    
    plot_adjs(adj_true, adj_cdkan, title="CD-KAN Prediction", filename="synthetic_cdkan_comparison.png")
    plot_adjs(adj_true, adj_lasso, title="VAR-Lasso Prediction", filename="synthetic_lasso_comparison.png")

    # Save Results
    results = pd.DataFrame([
        {"Model": "VAR-Lasso", **metrics_lasso},
        {"Model": "Correlation", **metrics_corr},
        {"Model": "CD-KAN", **metrics_cdkan}
    ])
    print("\n=== Causal Discovery Benchmark ===")
    print(results)
    results.to_csv("causal_benchmark_results.csv", index=False)

if __name__ == "__main__":
    main()
