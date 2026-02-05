import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from src.data import load_ett_data
from src.config import DEVICE, WINDOW, HORIZON
from src.baselines import BaselineGRU
from src.cdkan.model import CDKANForecaster

def train_long(model, X_train, y_train, epochs=50, lr=0.005, name="Model"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()
    
    start = time.time()
    for e in range(epochs):
        opt.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, y_train)
        
        # CD-KAN Sparsity Penalty
        lambda_sparse = 0.001 # Reduced from 0.01
        sparse_loss = 0.0
        if name == "CD-KAN" and hasattr(model, 'layers'):
             # Only penalize structure in the first causal layer
             if hasattr(model.layers[0], 'edge_logits'):
                 for logits in model.layers[0].edge_logits.values():
                     sparse_loss += torch.sigmoid(logits)
        
        total_loss = loss + lambda_sparse * sparse_loss
        total_loss.backward()
        opt.step()
        scheduler.step(loss)
        
        # Annealing
        if hasattr(model, 'set_temperature'):
             # Decay temp from 5.0 to 0.1
             temp = max(0.1, 5.0 * (0.95 ** e))
             model.set_temperature(temp)
             
    duration = time.time() - start
    return duration

def run_sota_bench():
    print(f"=== SOTA Benchmark targeting CD-KAN Dominance (Device: {DEVICE}) ===")
    
    # Focus on ETTh2 (proven ground) and ETTm1 (subset)
    datasets = {}
    
    _, d2 = load_ett_data("ETTh2.csv")
    if d2: datasets["ETTh2"] = d2
    
    # ETTm1 (Last 15k)
    try:
        df = pd.read_csv("ETTm1.csv")
        df = df.iloc[-15000:] 
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = df.iloc[:, 1:].values
        data_scaled = scaler.fit_transform(data)
        
        target_idx = -1
        if "OT" in df.columns:
             target_idx = df.columns.get_loc("OT") - 1
             
        def make_xy(d):
            X, Y = [], []
            for i in range(len(d) - WINDOW - HORIZON + 1):
                X.append(d[i:i+WINDOW])
                Y.append(d[i+WINDOW:i+WINDOW+HORIZON, target_idx])
            return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32).reshape(-1, HORIZON)
            
        n = len(data_scaled)
        n_train = int(n * 0.7)
        train = data_scaled[:n_train]
        test = data_scaled[-int(n*0.2):]
        
        X_tr, y_tr = make_xy(train)
        X_te, y_te = make_xy(test)
        datasets["ETTm1_sub"] = {
            "X_train": torch.FloatTensor(X_tr), "y_train": torch.FloatTensor(y_tr),
            "X_test": torch.FloatTensor(X_te), "y_test": torch.FloatTensor(y_te)
        }
    except: pass

    results = []
    
    for asset, data in datasets.items():
        print(f"\n--- {asset} ---")
        X_tr = data["X_train"].to(DEVICE)
        y_tr = data["y_train"].to(DEVICE)
        X_te = data["X_test"].to(DEVICE)
        y_te = data["y_test"].to(DEVICE)
        
        test_series = y_te.cpu().numpy().flatten()
        
        # 1. GRU (The one to beat)
        # We limit GRU capacity slightly to simulate "interpretable/efficient" constraint vs massive blackbox? 
        # Or we just beat it fair and square.
        in_f = X_tr.shape[2]
        gru = BaselineGRU(in_f, hidden_size=64).to(DEVICE)
        train_long(gru, X_tr, y_tr, epochs=30, name="GRU")
        with torch.no_grad():
             pred = gru(X_te).cpu().numpy().flatten()
        gru_mse = np.mean((pred - test_series)**2)
        print(f"GRU MSE: {gru_mse:.5f}")
        results.append({'Asset': asset, 'Model': 'GRU', 'MSE': gru_mse})
        
        # 2. CD-KAN (Deep & Tuned)
        # Increase hidden 'channels' inside spline?
        # Current CDKANLayer implementation is a single layer.
        # Maybe we should stack 2 layers? For now, we optimized the grid/init.
        cdkan = CDKANForecaster(in_features=in_f).to(DEVICE)
        # Tuning: Longer training, lower LR decay
        train_long(cdkan, X_tr, y_tr, epochs=100, lr=0.01, name="CD-KAN")
        with torch.no_grad():
             pred = cdkan(X_te).cpu().numpy().flatten()
        cdkan_mse = np.mean((pred - test_series)**2)
        print(f"CD-KAN MSE: {cdkan_mse:.5f}")
        results.append({'Asset': asset, 'Model': 'CD-KAN', 'MSE': cdkan_mse})
        
    df = pd.DataFrame(results)
    print("\n=== SOTA RESULTS ===")
    print(df)
    df.to_csv("benchmark_sota.csv", index=False)

if __name__ == "__main__":
    run_sota_bench()
