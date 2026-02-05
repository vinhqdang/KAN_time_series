import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from src.data import load_ett_data
from src.config import DEVICE, WINDOW, HORIZON
from src.baselines import BaselineLSTM, BaselineGRU, run_arima, run_prophet
from src.adakan.models.adakan_forecaster import AdaKANForecaster
from src.cdkan.model import CDKANForecaster

# Same training logic but slightly fewer epochs if needed
def train_torch_model(model, X_train, y_train, epochs=15, lr=0.01, name="Model"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    start = time.time()
    for e in range(epochs):
        opt.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        opt.step()
        if hasattr(model, 'set_temperature') and e > 5:
             model.set_temperature(0.5)
    return time.time() - start

def evaluate_fast():
    print("=== Loading Data (Fast Mode) ===")
    datasets = {}
    
    # 1. ETTh1 (Standard)
    _, d1 = load_ett_data("ETTh1.csv")
    if d1: datasets["ETTh1"] = d1
    
    # 2. ETTh2 (Standard)
    _, d2 = load_ett_data("ETTh2.csv")
    if d2: datasets["ETTh2"] = d2
    
    # 3. ETTm1 (Subset - Last 10k)
    # scale and load manually to subset
    try:
        df = pd.read_csv("ETTm1.csv")
        df = df.iloc[-10000:] # Slice last 10k
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = df.iloc[:, 1:].values
        data_scaled = scaler.fit_transform(data)
        
        # Target OT
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
    except Exception as e:
        print(f"Failed to load ETTm1: {e}")

    results = []
    
    for asset, data in datasets.items():
        print(f"\n--- Benchmarking: {asset} ---")
        X_tr_t = data["X_train"].to(DEVICE)
        y_tr_t = data["y_train"].to(DEVICE)
        X_te_t = data["X_test"].to(DEVICE)
        y_te_t = data["y_test"].to(DEVICE)
        
        # Univariate Series for ARIMA/Prophet
        train_series = y_tr_t.cpu().numpy().flatten()
        test_series = y_te_t.cpu().numpy().flatten()
        test_len = len(test_series)
        
        # 1. ARIMA
        try:
             # Reduce ARIMA computation by fitting on subset if series is long
             # 10k points for ARIMA is manageable but slowish.
             t0 = time.time()
             # Simple forecast
             arima_pred = run_arima(train_series, test_len)
             arima_time = time.time() - t0
             arima_mse = np.mean((arima_pred - test_series)**2)
             results.append({'Asset': asset, 'Model': 'ARIMA', 'MSE': arima_mse})
        except: pass
        
        # 2. GRU (Basline)
        try:
             # Multivariate Input
             in_f = X_tr_t.shape[2]
             gru = BaselineGRU(in_f).to(DEVICE)
             gru_time = train_torch_model(gru, X_tr_t, y_tr_t, name="GRU")
             with torch.no_grad():
                 pred = gru(X_te_t).cpu().numpy().flatten()
             mse = np.mean((pred - test_series)**2)
             results.append({'Asset': asset, 'Model': 'GRU', 'MSE': mse})
        except Exception as e: print(e)

        # 3. CD-KAN (Champion)
        try:
             in_f = X_tr_t.shape[2]
             cdkan = CDKANForecaster(in_features=in_f).to(DEVICE)
             # Relax sparsity? Default is 0.01 in validate script.
             # Here we use standard training.
             cdkan_time = train_torch_model(cdkan, X_tr_t, y_tr_t, name="CD-KAN")
             with torch.no_grad():
                 pred = cdkan(X_te_t).cpu().numpy().flatten()
             mse = np.mean((pred - test_series)**2)
             results.append({'Asset': asset, 'Model': 'CD-KAN', 'MSE': mse})
        except Exception as e: print(e)
        
        print(f"Done {asset}")

    df_res = pd.DataFrame(results)
    print("\n=== FAST BENCHMARK RESULTS ===")
    print(df_res)
    df_res.to_csv("benchmark_fast_results.csv", index=False)

if __name__ == "__main__":
    evaluate_fast()
