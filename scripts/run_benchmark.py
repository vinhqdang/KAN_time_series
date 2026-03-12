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
    # datasets = ['synthetic_linear', 'synthetic_nonlinear', 'financial', 'crypto', 'macro']
    # Limited for testing 
    datasets = ['synthetic_linear'] 
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
        
        n_samples = len(X)
        n_train = int(n_samples * 0.6)
        n_test = int(n_samples * 0.2)
        stride = int(n_samples * 0.2)
        
        n_features = X.shape[1]
        
        # Baselines factories
        def get_baselines():
            cdkan_model = CDKANForecaster(n_features, hidden_dim=64, out_features=n_features, max_lag=5).to(device)
            cdkan_trainer = CDKANTrainer(cdkan_model, device=device)
            return {
                'VAR-Lasso': VARLasso(max_lag=5),
                'PCMCI*': PCMCIProxy(max_lag=5),
                'NTiCD*': NTiCDProxy(epochs=50, device=device),
                'GOLEM*': GOLEMProxy(epochs=100, device=device),
                'CD-KAN v2': CDKANWrapper(cdkan_model, cdkan_trainer, epochs=50)
            }
        
        seeds = [42, 43, 44]
        algo_metrics = {k: [] for k in get_baselines().keys()}
        
        # Walk-forward loop
        start = 0
        fold_idx = 0
        while start + n_train + n_test <= n_samples:
            print(f"  --- Fold {fold_idx + 1} ---")
            train_raw = X[start : start + n_train]
            
            # Normalize on train ONLY
            mean = train_raw.mean(axis=0, keepdims=True)
            std = train_raw.std(axis=0, keepdims=True) + 1e-8
            X_train = (train_raw - mean) / std
            
            # Note: the test fold isn't evaluated effectively since these proxies lack predict capabilities,
            # but we run training on isolated train fold rigorously to harvest causal maps properly.
            
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                baselines = get_baselines()
                
                for algo_name, model in baselines.items():
                    try:
                        if algo_name == 'CD-KAN v2':
                            def make_windows(data, in_seq, out_seq):
                                L = len(data)
                                X_w, y_w = [], []
                                for i in range(L - in_seq - out_seq + 1):
                                    X_w.append(data[i:i+in_seq])
                                    y_w.append(data[i+in_seq:i+in_seq+out_seq])
                                return np.array(X_w), np.array(y_w)
                            
                            # Create an internal train/val split for CD-KAN early stopping (to avoid testing leakage)
                            n_inner_train = int(len(X_train) * 0.8)
                            X_inner_train = X_train[:n_inner_train]
                            X_inner_val = X_train[n_inner_train:]
                            
                            X_w_t, y_w_t = make_windows(X_inner_train, 6, 1) # Lag 5 + 1
                            X_torch_t = torch.tensor(X_w_t, dtype=torch.float32).to(device)
                            y_torch_t = torch.tensor(y_w_t[:, 0, :], dtype=torch.float32).to(device) # Horizon 1
                            train_loader = {'X': X_torch_t, 'y': y_torch_t}

                            X_w_v, y_w_v = make_windows(X_inner_val, 6, 1)
                            X_torch_v = torch.tensor(X_w_v, dtype=torch.float32).to(device)
                            y_torch_v = torch.tensor(y_w_v[:, 0, :], dtype=torch.float32).to(device)
                            val_loader = {'X': X_torch_v, 'y': y_torch_v}
                            
                            model.fit(train_loader, val_loader)
                        else:
                            model.fit(X_train)
                            
                        adj = model.get_adjacency()
                        thresh = 0.01 if algo_name == 'VAR-Lasso' else 0.05
                        adj_binary = (np.abs(adj) > thresh).astype(int)
                        
                        if true_adj is not None and adj_binary.shape == true_adj.shape:
                            g_metrics = evaluate_graph(true_adj, adj_binary)
                            algo_metrics[algo_name].append(g_metrics)
                            
                    except Exception as e:
                        print(f"    {algo_name} FAILED: {e}")
            
            start += stride
            fold_idx += 1
            
        for algo_name, m_list in algo_metrics.items():
            if not m_list:
                continue
            avg_metrics = {k: np.mean([m[k] for m in m_list]) for k in m_list[0].keys()}
            results.append({
                'Dataset': ds_name,
                'Algorithm': algo_name,
                **avg_metrics
            })
            print(f"    {algo_name} Avg Results: {avg_metrics}")

    # Report
    df_res = pd.DataFrame(results)
    print("\nBenchmark Comparison:")
    print(tabulate(df_res, headers='keys', tablefmt='github'))
    
    # Save
    df_res.to_csv("benchmark_results_phase2.csv", index=False)

if __name__ == "__main__":
    run_benchmark()
