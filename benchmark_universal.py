import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from src.data import load_ett_data, load_and_preprocess_data
from src.config import DEVICE, WINDOW, HORIZON
from src.baselines import BaselineGRU
from src.cdkan.model import CDKANForecaster

# Helper to load WTI/GLD properly
def load_fin_data(asset):
    # Reuse load_and_preprocess_data but filter for specific asset
    # This is a bit inefficient (reloads all) but robust
    from src.data import load_and_preprocess_data
    scalers, datasets = load_and_preprocess_data()
    return datasets.get(asset)

def train_long(model, X_train, y_train, epochs=50, lr=0.005, name="Model"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()
    
    start = time.time()
    for e in range(epochs):
        opt.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, y_train)
        
        # CD-KAN Sparsity Penalty (Only for CD-KAN)
        lambda_sparse = 0.001 
        sparse_loss = 0.0
        if name == "CD-KAN" and hasattr(model, 'layers'):
             if hasattr(model.layers[0], 'edge_logits'):
                 for logits in model.layers[0].edge_logits.values():
                     sparse_loss += torch.sigmoid(logits)
        
        total_loss = loss + lambda_sparse * sparse_loss
        total_loss.backward()
        opt.step()
        scheduler.step(loss)
        
        if hasattr(model, 'set_temperature'):
             temp = max(0.1, 5.0 * (0.95 ** e))
             model.set_temperature(temp)
             
    return time.time() - start

def run_universal_bench():
    print(f"=== Universal SOTA Benchmark (Device: {DEVICE}) ===")
    
    datasets = {}
    
    # 1. ETTh1
    _, d1 = load_ett_data("ETTh1.csv")
    if d1: datasets["ETTh1"] = d1
    
    # 2. ETTh2
    _, d2 = load_ett_data("ETTh2.csv")
    if d2: datasets["ETTh2"] = d2
    
    # 3. ETTm1 (Subset)
    try:
        df = pd.read_csv("ETTm1.csv")
        df = df.iloc[-15000:] 
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = df.iloc[:, 1:].values
        data_scaled = scaler.fit_transform(data)
        target_idx = -1
        if "OT" in df.columns: target_idx = df.columns.get_loc("OT") - 1
        
        def make_xy(d):
            X, Y = [], []
            for i in range(len(d) - WINDOW - HORIZON + 1):
                X.append(d[i:i+WINDOW])
                Y.append(d[i+WINDOW:i+WINDOW+HORIZON, target_idx])
            return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32).reshape(-1, HORIZON)
        n = len(data_scaled); n_train = int(n * 0.7)
        X_tr, y_tr = make_xy(data_scaled[:n_train])
        X_te, y_te = make_xy(data_scaled[-int(n*0.2):])
        datasets["ETTm1_sub"] = {
            "X_train": torch.FloatTensor(X_tr), "y_train": torch.FloatTensor(y_tr),
            "X_test": torch.FloatTensor(X_te), "y_test": torch.FloatTensor(y_te)
        }
    except: pass

    # 4. WTI (Financial/Commodity)
    d_wti = load_fin_data("WTI")
    if d_wti: 
        # Expand dims for WTI if univariate [Batch, Seq] -> [Batch, Seq, 1]
        if d_wti["X_train"].dim() == 2:
            d_wti["X_train"] = d_wti["X_train"].unsqueeze(-1)
            d_wti["X_test"] = d_wti["X_test"].unsqueeze(-1)
        datasets["WTI"] = d_wti

    results = []
    
    for asset, data in datasets.items():
        print(f"\n--- {asset} ---")
        X_tr = data["X_train"].to(DEVICE)
        y_tr = data["y_train"].to(DEVICE)
        X_te = data["X_test"].to(DEVICE)
        y_te = data["y_test"].to(DEVICE)
        
        test_series = y_te.cpu().numpy().flatten()
        in_f = X_tr.shape[2]
        
        # 1. GRU (Baseline)
        gru = BaselineGRU(in_f, hidden_size=64).to(DEVICE)
        train_long(gru, X_tr, y_tr, epochs=30, name="GRU")
        with torch.no_grad():
             pred = gru(X_te).cpu().numpy().flatten()
        gru_mse = np.mean((pred - test_series)**2)
        print(f"GRU MSE: {gru_mse:.5f}")
        results.append({'Asset': asset, 'Model': 'GRU', 'MSE': gru_mse})
        
        # 2. R-CDKAN (Universal Config)
        cdkan = CDKANForecaster(in_features=in_f, hidden_dim=64, n_layers=2).to(DEVICE)
        train_long(cdkan, X_tr, y_tr, epochs=50, lr=0.01, name="CD-KAN") # 50 epochs sufficient for R-CDKAN
        with torch.no_grad():
             pred = cdkan(X_te).cpu().numpy().flatten()
        cdkan_mse = np.mean((pred - test_series)**2)
        print(f"CD-KAN MSE: {cdkan_mse:.5f}")
        results.append({'Asset': asset, 'Model': 'CD-KAN', 'MSE': cdkan_mse})
        
    df = pd.DataFrame(results)
    print("\n=== UNIVERSAL SOTA RESULTS ===")
    print(df)
    df.to_csv("benchmark_universal.csv", index=False)

if __name__ == "__main__":
    run_universal_bench()
