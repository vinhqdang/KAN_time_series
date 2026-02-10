
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from src.data import load_multivariate_data
from src.cdkan.model import CDKANForecaster
from src.baselines import BaselineLSTM, TSMixer
from src.config import DEVICE, WINDOW, HORIZON
import time

def train_and_evaluate(model, name, train_loader, test_X, test_y, epochs=30):
    print(f"\n--- Training {name} ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        # Simple full batch or mini-batch
        # Using full batch for simplicity as in validate_cdkan
        optimizer.zero_grad()
        
        # CD-KAN specific Phase logic
        if "CD-KAN" in name:
            if epoch < 10:
                if hasattr(model, 'set_temperature'): model.set_temperature(5.0)
            elif epoch < 20: 
                # sparsity
                pass
            else:
                if hasattr(model, 'set_temperature'): model.set_temperature(0.5)
                
        pred = model(train_loader['X'])
        loss = criterion(pred, train_loader['y'])
        
        # Add sparsity loss for CD-KAN
        if "CD-KAN" in name and epoch > 10:
             # simple sparsity proxy
             pass
             
        loss.backward()
        optimizer.step()
        epoch_loss = loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.6f}")
            
    train_time = time.time() - start_time
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(test_X)
        mse = criterion(preds, test_y).item()
        mae = torch.mean(torch.abs(preds - test_y)).item()
        
    print(f"{name} Results - MSE: {mse:.6f}, MAE: {mae:.6f}, Time: {train_time:.2f}s")
    return {"Model": name, "MSE": mse, "MAE": mae, "Time": train_time}

def run_benchmark():
    # 1. Load Data
    scaler, dataset = load_multivariate_data()
    X_train, y_train = dataset['X_train'], dataset['y_train']
    X_test, y_test = dataset['X_test'], dataset['y_test']
    
    # X: [Batch, Window, Feats]
    # y: [Batch, Horizon, Feats] -> We predict Horizon=1 for now based on config
    # If Horizon > 1, some models might need adjustment (e.g. LSTM outputs sequence?)
    # Baselines currently output [Batch, 1] or [Batch, Feats] (if we adjust output dim)
    
    print(f"Features: {dataset['asset_names']}")
    n_feats = X_train.shape[2]
    n_out = y_train.shape[2] # Predict all assets
    
    # Flatten y for models that expect 2D output if multi-horizon?
    # Our y is [Batch, 1, Feats]. Squeeze dim 1.
    y_train = y_train.squeeze(1)
    y_test = y_test.squeeze(1)
    
    train_data = {'X': X_train, 'y': y_train}
    
    results = []
    
    # 2. Define Models
    
    # A. LSTM
    lstm = BaselineLSTM(input_size=n_feats, hidden_size=64, num_layers=2).to(DEVICE)
    # BaselineLSTM output is [Batch, 1]. We need [Batch, n_out].
    # Modify Linear layer
    lstm.fc = nn.Linear(64, n_out).to(DEVICE)
    results.append(train_and_evaluate(lstm, "LSTM", train_data, X_test, y_test))
    
    # B. TSMixer
    # Seq Len = WINDOW, Pred Len = 1 usually (squeezed)
    # TSMixer returns [Batch, Pred, Feat].
    # We need to wrap it to match [Batch, Feat] output format or unsqueeze data?
    # Let's wrap it.
    class TSMixerWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m
        def forward(self, x):
            out = self.model(x) # [Batch, 1, Feat]
            return out.squeeze(1)
            
    tsmixer = TSMixer(n_series=n_feats, seq_len=WINDOW, pred_len=1).to(DEVICE)
    tsmixer_wrapped = TSMixerWrapper(tsmixer).to(DEVICE)
    results.append(train_and_evaluate(tsmixer_wrapped, "TSMixer", train_data, X_test, y_test))
    
    # C. CD-KAN
    # CDKANForecaster output is [Batch, Out]. Perfect.
    cdkan = CDKANForecaster(in_features=n_feats, out_features=n_out, max_lag=min(10, WINDOW-1), n_layers=2).to(DEVICE)
    results.append(train_and_evaluate(cdkan, "CD-KAN", train_data, X_test, y_test, epochs=50))
    
    # D. Naive KAN (CD-KAN without structure learning / lags)
    # We can simulate this by fixing lags to 0 or 1 and structure to full.
    # Or just use same class with LearnStructure=False?
    # CDKANLayer has learn_structure arg.
    # But CDKANForecaster doesn't expose it easily.
    # Let's skip explicit 'Naive KAN' class for this script simplicity, or add it if requested.
    # The user asked for "naive KAN".
    # I'll create a KAN-only model using KANLayer directly.
    from src.cdkan.layers import KANLayer
    class NaiveKAN(nn.Module):
        def __init__(self, in_dim, hidden, out_dim):
            super().__init__()
            self.k1 = KANLayer(in_dim, hidden)
            self.k2 = KANLayer(hidden, out_dim)
        def forward(self, x):
            # Flatten or max pool over time?
            # Naive KAN usually takes flat input.
            # Flatten window: [Batch, Window*Feat]
            b, w, f = x.shape
            x_flat = x.reshape(b, -1)
            # Re-init k1 with correct input dim
            # Wait, local KANLayer takes [Batch, In].
            # So we need new init.
            pass
            return self.k2(self.k1(x_flat))
            
    naive_kan = NaiveKAN(n_feats*WINDOW, 64, n_out).to(DEVICE)
    # Re-init correct dims
    naive_kan.k1 = KANLayer(n_feats*WINDOW, 64).to(DEVICE) 
    results.append(train_and_evaluate(naive_kan, "Naive KAN", train_data, X_test, y_test))
    
    # 3. Report
    df_res = pd.DataFrame(results)
    print("\n=== Benchmark Results ===")
    print(df_res)
    df_res.to_csv("benchmark_results_predictive.csv", index=False)

if __name__ == "__main__":
    run_benchmark()
