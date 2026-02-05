import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from src.data import load_and_preprocess_data, load_ett_data
from src.config import DEVICE, WINDOW, HORIZON
from src.baselines import BaselineLSTM, BaselineGRU, run_arima, run_prophet
from src.adakan.models.adakan_forecaster import AdaKANForecaster
from src.cdkan.model import CDKANForecaster

def train_torch_model(model, X_train, y_train, epochs=20, lr=0.01, name="Model"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    start = time.time()
    for e in range(epochs):
        opt.zero_grad()
        out = model(X_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        opt.step()
        # Callbacks
        if hasattr(model, 'on_epoch_end'):
             model.on_epoch_end(e)
        if hasattr(model, 'set_temperature'): # For CD-KAN annealing
             if e > 10: model.set_temperature(0.5)
             
    duration = time.time() - start
    return duration

def evaluate_models():
    print("=== Loading Data ===")
    
    datasets = {}
    
    # Load ETT Datasets (Multivariate)
    for asset in ["ETTh1", "ETTm1", "ETTh2"]:
        filename = f"{asset}.csv"
        print(f"Loading {asset}...")
        scaler, data_dict = load_ett_data(file_path=filename)
        if data_dict:
            datasets[asset] = data_dict
            
    # Load GLD/WTI/DXY (Univariate)
    # scalers_uni, datasets_uni = load_and_preprocess_data()
    # datasets.update(datasets_uni)
    
    results = []
    assets = list(datasets.keys())
    
    for asset in assets:
        print(f"\n--- Benchmarking Asset: {asset} ---")
        data = datasets[asset]
        # Torch Tensors
        X_tr_t, y_tr_t = data["X_train"].to(DEVICE), data["y_train"].to(DEVICE)
        X_te_t, y_te_t = data["X_test"].to(DEVICE), data["y_test"].to(DEVICE)
        
        # Numpy for ARIMA/Prophet
        # Reconstruct "history" from X_train (just the close prices?)
        # data.py returns scaled windows. For ARIMA we ideally want unscaled or simple series.
        # But to be fair, we should evaluate on the *same target* (y_test).
        # We can just feed the 'y' series to ARIMA? 
        # y_train is [Batch, 1]. Effectively a time series.
        train_series = y_tr_t.cpu().numpy().flatten()
        test_series = y_te_t.cpu().numpy().flatten()
        test_len = len(test_series)
        
        # 1. ARIMA
        try:
            t0 = time.time()
            arima_pred = run_arima(train_series, test_len)
            arima_time = time.time() - t0
            arima_mse = np.mean((arima_pred - test_series)**2)
            results.append({'Asset': asset, 'Model': 'ARIMA', 'MSE': arima_mse, 'Time': arima_time})
        except Exception as e:
            print(f"ARIMA failed: {e}")

        # 2. Prophet
        try:
            t0 = time.time()
            # Create DF
            dates = pd.date_range(start='2020-01-01', periods=len(train_series))
            df = pd.DataFrame({'ds': dates, 'y': train_series})
            prophet_pred = run_prophet(df, test_len)
            prophet_time = time.time() - t0
            prophet_mse = np.mean((prophet_pred - test_series)**2)
            results.append({'Asset': asset, 'Model': 'Prophet', 'MSE': prophet_mse, 'Time': prophet_time})
        except Exception as e:
            print(f"Prophet failed: {e}")

        
        # Determine Input Dimension
        if X_tr_t.dim() == 3:
            # [Batch, Window, Features] -> Multivariate
            in_features = X_tr_t.shape[2]
            X_tr_rnn, X_te_rnn = X_tr_t, X_te_t
            # Flatten for AdaKAN (Univariate/Flattened input)
            X_tr_flat = X_tr_t.reshape(X_tr_t.size(0), -1)
            X_te_flat = X_te_t.reshape(X_te_t.size(0), -1)
        else:
            # [Batch, Window] -> Univariate
            in_features = 1
            X_tr_rnn = X_tr_t.unsqueeze(-1)
            X_te_rnn = X_te_t.unsqueeze(-1)
            X_tr_flat, X_te_flat = X_tr_t, X_te_t

        # 1. ARIMA (Univariate Baseline)
        try:
             # ARIMA only sees history of the TARGET variable
             # y_tr_t is [Batch, 1], flatten it to get time series
             train_series = y_tr_t.cpu().numpy().flatten()
             test_series = y_te_t.cpu().numpy().flatten()
             test_len = len(test_series)
             
             t0 = time.time()
             arima_pred = run_arima(train_series, test_len)
             arima_time = time.time() - t0
             arima_mse = np.mean((arima_pred - test_series)**2)
             results.append({'Asset': asset, 'Model': 'ARIMA', 'MSE': arima_mse, 'Time': arima_time})
        except Exception as e:
            print(f"ARIMA failed: {e}") 

        # 3. LSTM
        try:
            lstm = BaselineLSTM(input_size=X_tr_rnn.shape[-1]).to(DEVICE)
            lstm_time = train_torch_model(lstm, X_tr_rnn, y_tr_t, name="LSTM")
            with torch.no_grad():
                pred = lstm(X_te_rnn).cpu().numpy().flatten()
            lstm_mse = np.mean((pred - test_series)**2)
            results.append({'Asset': asset, 'Model': 'LSTM', 'MSE': lstm_mse, 'Time': lstm_time})
        except Exception as e:
             print(f"LSTM failed: {e}")

        # 4. GRU
        try:
            gru = BaselineGRU(input_size=X_tr_rnn.shape[-1]).to(DEVICE)
            gru_time = train_torch_model(gru, X_tr_rnn, y_tr_t, name="GRU")
            with torch.no_grad():
                pred = gru(X_te_rnn).cpu().numpy().flatten()
            gru_mse = np.mean((pred - test_series)**2)
            results.append({'Asset': asset, 'Model': 'GRU', 'MSE': gru_mse, 'Time': gru_time})
        except Exception as e:
             print(f"GRU failed: {e}")

        # 5. AdaKAN
        try:
            # AdaKAN takes flattened input [Batch, Window*Features]
            ad_win = X_tr_flat.shape[1]
            adakan = AdaKANForecaster(window=ad_win, hidden=32).to(DEVICE)
            adakan_time = train_torch_model(adakan, X_tr_flat, y_tr_t, name="AdaKAN")
            with torch.no_grad():
                pred = adakan(X_te_flat).cpu().numpy().flatten()
            if len(pred) != len(test_series):
                 pred = pred[:len(test_series)]
            adakan_mse = np.mean((pred - test_series)**2)
            results.append({'Asset': asset, 'Model': 'AdaKAN', 'MSE': adakan_mse, 'Time': adakan_time})
        except Exception as e:
             print(f"AdaKAN failed: {e}")

        # 6. CD-KAN
        try:
            # CD-KAN takes [Batch, Window, Features]
            # It discovers structure between Features.
            # IMPORTANT: Lower sparsity penalty?
            cdkan = CDKANForecaster(in_features=in_features).to(DEVICE)
            # We can tweak internal temperature schedule in train_torch_model call if needed
            cdkan_time = train_torch_model(cdkan, X_tr_rnn, y_tr_t, name="CD-KAN")
            with torch.no_grad():
                pred = cdkan(X_te_rnn).cpu().numpy().flatten()
            cdkan_mse = np.mean((pred - test_series)**2)
            results.append({'Asset': asset, 'Model': 'CD-KAN', 'MSE': cdkan_mse, 'Time': cdkan_time})
        except Exception as e:
             print(f"CD-KAN failed: {e}")

    # Summary
    df_res = pd.DataFrame(results)
    print("\n=== FINAL BENCHMARK RESULTS ===")
    print(df_res.sort_values(by=['Asset', 'MSE']))
    
    # Save CSV
    df_res.to_csv("benchmark_results.csv", index=False)

if __name__ == "__main__":
    evaluate_models()
