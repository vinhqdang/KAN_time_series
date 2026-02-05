from src.data import load_and_preprocess_data
from src.models import KANForecaster, LSTMForecaster, TransformerForecaster, TSMixer
from src.train import train_once
from src.config import WINDOW, DEVICE
import torch

print("Device:", DEVICE)

print("Loading data...")
scalers, datasets = load_and_preprocess_data()

print("Testing Model Training...")
for asset, d in datasets.items():
    print(f"--- Asset: {asset} ---")
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    
    # KAN
    print("Training KAN...")
    m = KANForecaster(WINDOW).to(DEVICE)
    train_once(m, Xtr, ytr, Xte, yte, steps=10, lr=1e-2, name="KAN") # Reduced steps for speed
    
    # LSTM
    print("Training LSTM...")
    m = LSTMForecaster().to(DEVICE)
    train_once(m, Xtr, ytr, Xte, yte, steps=10, lr=1e-3, name="LSTM")
    
    break # Test one asset only

print("\nSuccess! Models architecture and training loop verified.")
