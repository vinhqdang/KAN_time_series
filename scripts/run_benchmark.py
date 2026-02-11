import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tabulate import tabulate

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.benchmarks.datasets import get_dataset
from src.benchmarks.baselines import VARLasso, PCMCIProxy, NTiCDProxy, GOLEMProxy, CDKANWrapper
from src.cdkan.model import CDKANForecaster
from src.cdkan.trainer import CDKANTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_graph(true_adj, pred_adj):
    """
    Compute Precision, Recall, F1, SHD.
    Assumes true_adj and pred_adj are binary 0/1 matrices.
    """
    # Flatten
    true_flat = true_adj.flatten()
    pred_flat = pred_adj.flatten()
    
    prec = precision_score(true_flat, pred_flat, zero_division=0)
    rec = recall_score(true_flat, pred_flat, zero_division=0)
    f1 = f1_score(true_flat, pred_flat, zero_division=0)
    
    # SHD: Just Hamming distance for binary
    shd = np.abs(true_flat - pred_flat).sum()
    
    return {'Precision': prec, 'Recall': rec, 'F1': f1, 'SHD': shd}

def run_benchmark():
    datasets = ['synthetic_linear', 'synthetic_nonlinear', 'financial', 'crypto', 'macro']
    # Limited for testing 
    # datasets = ['synthetic_linear'] 
    
    results = []
    
    print(f"Starting Benchmark on Device: {device}")
    
    for ds_name in datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        try:
            X, true_adj = get_dataset(ds_name)
        except Exception as e:
            print(f"Skipping {ds_name}: {e}")
            continue
            
        print(f"Data Shape: {X.shape}")
        
        # Split Data
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        
        # Prepare Baselines
        baselines = {
            'VAR-Lasso': VARLasso(max_lag=5),
            'PCMCI*': PCMCIProxy(max_lag=5),
            'NTiCD*': NTiCDProxy(epochs=50, device=device),
            'GOLEM*': GOLEMProxy(epochs=100, device=device),
        }
        
        # Add CD-KAN
        # Init Model
        n_features = X.shape[1]
        # in_features, hidden, out_features, max_lag
        cdkan_model = CDKANForecaster(n_features, hidden_dim=64, out_features=n_features, max_lag=5).to(device)
        cdkan_trainer = CDKANTrainer(cdkan_model, device=device)
        baselines['CD-KAN v2'] = CDKANWrapper(cdkan_model, cdkan_trainer, epochs=50) # Reduced epochs for benchmark speed
        
        for algo_name, model in baselines.items():
            print(f"  > Running {algo_name}...")
            
            # FIT
            try:
                if algo_name == 'CD-KAN v2':
                    # Prepare Loader
                    # Creating simple Batched Loader or Dictionary
                    # Helper to slice windows
                    def make_windows(data, in_seq, out_seq):
                        L = len(data)
                        X_w, y_w = [], []
                        for i in range(L - in_seq - out_seq + 1):
                            X_w.append(data[i:i+in_seq])
                            y_w.append(data[i+in_seq:i+in_seq+out_seq])
                        return np.array(X_w), np.array(y_w)
                    
                    X_w, y_w = make_windows(X_train, 6, 1) # Lag 5 + 1
                    X_torch = torch.tensor(X_w, dtype=torch.float32).to(device)
                    y_torch = torch.tensor(y_w[:, 0, :], dtype=torch.float32).to(device) # Horizon 1
                    
                    train_loader = {'X': X_torch, 'y': y_torch}
                    model.fit(train_loader)
                    
                else:
                    model.fit(X_train)
            except Exception as e:
                print(f"    FAILED: {e}")
                continue
                
            # EVALUATE
            adj = model.get_adjacency()
            
            # Thresholding
            thresh = 0.05
            # For CD-KAN and Neural Baselines, adjacency is soft [0,1].
            # VARLasso gives Coeffs (unbounded). Normalize or threshold magnitude?
            # VARLasso outputs sum of absolute coeffs. 
            if algo_name == 'VAR-Lasso':
                thresh = 0.01 # Lasso is sparse already
            
            adj_binary = (np.abs(adj) > thresh).astype(int)
            
            metrics = {}
            # Structural Metrics (if ground truth exists)
            if true_adj is not None:
                # Need to match dimensions?
                if adj_binary.shape == true_adj.shape:
                    g_metrics = evaluate_graph(true_adj, adj_binary)
                    metrics.update(g_metrics)
                else:
                    metrics.update({'F1': -1, 'SHD': -1})
            else:
                 metrics.update({'F1': 'N/A', 'SHD': 'N/A'})
            
            # Predictive Metrics (MSE on Test)
            # Need 'predict' method? 
            # Baselines don't all have predict. We skip MSE for now or implement generic predict AR(1).
            # We focus on Causal Discovery (F1) as primary objective Phase 2.
            
            results.append({
                'Dataset': ds_name,
                'Algorithm': algo_name,
                **metrics
            })
            print(f"    Results: {metrics}")

    # Report
    df_res = pd.DataFrame(results)
    print("\nBenchmark Comparison:")
    print(tabulate(df_res, headers='keys', tablefmt='github'))
    
    # Save
    df_res.to_csv("benchmark_results_phase2.csv", index=False)

if __name__ == "__main__":
    run_benchmark()
