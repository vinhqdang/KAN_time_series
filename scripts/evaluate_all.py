
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_multivariate_data
from src.config import DEVICE, WINDOW, HORIZON
from src.cdkan.model import CDKANForecaster
from src.cdkan.trainer import CDKANTrainer
from src.cdkan.layers import KANLayer, CDKANLayer
from src.baselines import BaselineLSTM, TSMixer
from src.adakan.models.adakan_forecaster import AdaKANForecaster

def evaluate_model(model, X_test, y_test, name="Model"):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        mse = nn.MSELoss()(preds, y_test).item()
        mae = torch.mean(torch.abs(preds - y_test)).item()
    return {"Model": name, "MSE": mse, "MAE": mae}

class NaiveKAN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.k1 = KANLayer(in_dim, hidden)
        self.k2 = KANLayer(hidden, out_dim)
    def forward(self, x):
        # x: [Batch, Window, Feats] -> Flatten -> [Batch, In]
        b = x.shape[0]
        x_flat = x.reshape(b, -1)
        return self.k2(self.k1(x_flat))

class AdaKANWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        b = x.shape[0]
        x_flat = x.reshape(b, -1)
        return self.model(x_flat)

class TSMixerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # TSMixer: [Batch, 1, Feat] -> [Batch, Feat]
        out = self.model(x)
        return out.squeeze(1)
        
class LSTMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)




def plot_adjacency(adj, names, filename="cdkan_adjacency.png", threshold=0.1):
    # Hard Thresholding for "Best" Graph
    adj[adj < threshold] = 0.0
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj, xticklabels=names, yticklabels=names, annot=True, fmt=".2f", cmap="viridis")
    plt.title("CD-KAN Learned Causal Adjacency (Pruned)")
    plt.xlabel("Cause (t-tau)")
    plt.ylabel("Effect (t)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Adjacency plot saved to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100) # SOTA Training Length
    parser.add_argument("--fast", action="store_true", help="Run fast mode (few epochs)")
    args = parser.parse_args()
    
    epochs = 5 if args.fast else args.epochs
    
    # 1. Load Data
    print(f"Loading Data... (Epochs: {epochs})")
    
    data_path = 'data/financial_2020_2025.csv'
    dataset = None
    
    if os.path.exists(data_path):
        print(f"Found local data at {data_path}. Loading...")
        try:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            
            # Helper for creating windows
            def make_dataset(d):
                X, Y = [], []
                if len(d) < WINDOW + HORIZON:
                    return np.zeros((0, WINDOW, d.shape[1])), np.zeros((0, HORIZON, d.shape[1]))
                for i in range(len(d) - WINDOW - HORIZON + 1):
                    X.append(d[i : i+WINDOW])
                    Y.append(d[i+WINDOW : i+WINDOW+HORIZON])
                return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            data_values = df.values.astype(np.float32)
            data_scaled = scaler.fit_transform(data_values)
            
            n = len(data_scaled)
            n_train = int(n * 0.8)
            
            train_data = data_scaled[:n_train]
            test_data = data_scaled[n_train:]
            
            X_train_np, y_train_np = make_dataset(train_data)
            X_test_np, y_test_np = make_dataset(test_data)
            
            dataset = {
                "X_train": torch.from_numpy(X_train_np).to(DEVICE),
                "y_train": torch.from_numpy(y_train_np).to(DEVICE),
                "X_test": torch.from_numpy(X_test_np).to(DEVICE),
                "y_test": torch.from_numpy(y_test_np).to(DEVICE),
                "asset_names": df.columns.tolist()
            }
            print("Local data loaded successfully.")
            
        except Exception as e:
            print(f"Failed to load local CSV: {e}. Fallback to API.")
            
    if dataset is None:
        try:
            _, dataset = load_multivariate_data()
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {e}")
    
    X_train, y_train = dataset['X_train'], dataset['y_train']
    X_test, y_test = dataset['X_test'], dataset['y_test']
    asset_names = dataset['asset_names']
    
    n_assets = len(asset_names)
    n_timesteps = X_train.shape[1] # WINDOW
    input_dim_flat = n_assets * n_timesteps
    
    # Squeeze y for [Batch, Out]
    # y_train is [Batch, Horizon, Feats]
    y_train = y_train.squeeze(1)
    y_test = y_test.squeeze(1)
    
    train_loader = {'X': X_train, 'y': y_train}
    test_loader = {'X': X_test, 'y': y_test}
    
    results = []
    
    # --- 2. Baselines ---
    
    print("\nTraining LSTM...")
    lstm = BaselineLSTM(input_size=n_assets, hidden_size=64).to(DEVICE)
    lstm.fc = nn.Linear(64, n_assets).to(DEVICE)
    trainer_lstm = CDKANTrainer(LSTMWrapper(lstm), device=DEVICE)
    # Give baselines decent chance but prioritize CD-KAN optimization
    trainer_lstm.train(train_loader, test_loader, epochs=epochs, patience=10)
    results.append(evaluate_model(trainer_lstm.model, X_test, y_test, "LSTM"))
    
    print("\nTraining TSMixer...")
    tsmixer = TSMixer(n_series=n_assets, seq_len=n_timesteps, pred_len=1).to(DEVICE)
    trainer_tsmixer = CDKANTrainer(TSMixerWrapper(tsmixer), device=DEVICE)
    trainer_tsmixer.train(train_loader, test_loader, epochs=epochs, patience=10)
    results.append(evaluate_model(trainer_tsmixer.model, X_test, y_test, "TSMixer"))
    
    print("\nTraining Naive KAN...")
    naive_kan = NaiveKAN(input_dim_flat, 64, n_assets).to(DEVICE)
    trainer_naive = CDKANTrainer(naive_kan, device=DEVICE)
    trainer_naive.train(train_loader, test_loader, epochs=epochs, patience=10)
    results.append(evaluate_model(naive_kan, X_test, y_test, "Naive KAN"))
    
    print("\nTraining ADA-KAN...")
    adakan = AdaKANForecaster(window=input_dim_flat, hidden=32, horizon=n_assets).to(DEVICE)
    trainer_ada = CDKANTrainer(AdaKANWrapper(adakan), device=DEVICE)
    trainer_ada.train(train_loader, test_loader, epochs=epochs, patience=10)
    results.append(evaluate_model(trainer_ada.model, X_test, y_test, "ADA-KAN"))
    
    # --- 3. CD-KAN (SOTA Config) ---
    
    print("\nTraining CD-KAN (SOTA Config)...")
    # New Architecture: RevIN + Residual Deep KAN
    cdkan_model = CDKANForecaster(
        in_features=n_assets, 
        hidden_dim=64, 
        out_features=n_assets, 
        max_lag=min(10, n_timesteps-1),
        n_layers=3,     # Deeper
        dropout=0.1
    ).to(DEVICE)
    
    trainer_cdkan = CDKANTrainer(cdkan_model, device=DEVICE)
    
    # Ensure CD-KAN runs full course for structure discovery
    # Increased patience to avoid premature stopping during phase transitions
    trainer_cdkan.train(train_loader, test_loader, epochs=epochs, patience=20)
    
    results.append(evaluate_model(cdkan_model, X_test, y_test, "CD-KAN"))
    
    # Save Results
    df_res = pd.DataFrame(results)
    print("\n=== Comprehensive Benchmark Results ===")
    print(df_res)
    df_res.to_csv("benchmark_comprehensive.csv", index=False)
    
    # Save Graph
    if hasattr(cdkan_model.cd_layer, 'get_causal_graph'):
        adj, lags = cdkan_model.cd_layer.get_causal_graph()
        if isinstance(adj, torch.Tensor): adj = adj.cpu().numpy()
        if isinstance(lags, torch.Tensor): lags = lags.cpu().numpy()
        
        # Save Adjacency CSV
        adj_df = pd.DataFrame(adj, index=asset_names, columns=asset_names)
        adj_df.to_csv("cdkan_adjacency.csv")
        
        # Plot with Thresholding
        try:
           plot_adjacency(adj, asset_names, threshold=0.1) # Aggressive pruning for visuals
        except Exception as e:
           print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
