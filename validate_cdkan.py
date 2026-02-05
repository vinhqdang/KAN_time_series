import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from src.data import load_and_preprocess_data
from src.cdkan.model import CDKANForecaster
from src.config import DEVICE, WINDOW

def train_cdkan():
    print("=== Loading Real Data ===")
    scalers, datasets = load_and_preprocess_data()
    
    # Use GLD data
    asset = "GLD"
    if asset not in datasets:
        print(f"Asset {asset} not found, picking first available.")
        asset = list(datasets.keys())[0]
        
    data = datasets[asset]
    X_train, y_train = data["X_train"].to(DEVICE), data["y_train"].to(DEVICE)
    X_test, y_test = data["X_test"].to(DEVICE), data["y_test"].to(DEVICE)
    
    # Reshape for CD-KAN constraints: [Batch, Seq, Features]
    if X_train.dim() == 2:
        X_train = X_train.unsqueeze(-1)
        X_test = X_test.unsqueeze(-1)
        
    print(f"Training on {asset}: {X_train.shape} -> {y_train.shape}")
    
    # CD-KAN Config
    in_features = X_train.shape[2]
    out_features = y_train.shape[1] # Typically 1 (Close price)
    max_lag = min(10, WINDOW-1)
    
    model = CDKANForecaster(in_features, out_features=out_features, max_lag=max_lag).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training Loop with Phases
    epochs = 60
    history = {'loss': [], 'sparsity': []}
    
    print("\n=== Phase 1: Warm-up (Epochs 1-20) ===")
    model.set_temperature(5.0)
    lambda_sparse = 0.0
    
    for epoch in range(1, epochs + 1):
        # Phase transitions
        if epoch == 21:
            print("\n=== Phase 2: Structure Discovery (Epochs 21-40) ===")
            lambda_sparse = 0.01 # Activate sparsity penalty
        if epoch == 41:
            print("\n=== Phase 3: Fine-tuning (Epochs 41-60) ===")
            model.set_temperature(0.5) # Sharpen discrete sampling
            
        optimizer.zero_grad()
        indices = torch.randperm(X_train.size(0))
        batch_X = X_train[indices] # Full batch for prototype
        batch_y = y_train[indices]
        
        # Forward
        pred = model(batch_X)
        mse_loss = F.mse_loss(pred, batch_y)
        
        # Sparsity Loss
        sparsity_loss = 0.0
        for logits in model.layer.edge_logits.values():
            sparsity_loss += torch.sigmoid(logits)
            
        total_loss = mse_loss + lambda_sparse * sparsity_loss
        
        total_loss.backward()
        optimizer.step()
        
        history['loss'].append(mse_loss.item())
        history['sparsity'].append(sparsity_loss.item())
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: MSE={mse_loss.item():.6f}, SparseLoss={sparsity_loss.item():.4f}")
            
    # Evaluation
    print("\n=== Evaluation on Test Set ===")
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_mse = F.mse_loss(test_pred, y_test).item()
        print(f"Test MSE: {test_mse:.6f}")
        
    # Interpretability: Extract Graph
    print("\n=== Discovered Causal Structure ===")
    adj, lags = model.layer.get_causal_graph()
    
    # Features names (approximate based on data.py logic)
    print("Top Causal Edges (Probability > 0.5):")
    # Assuming Feature 0 is Close lag, etc.
    # In data.py features are lagged values if supervised, BUT
    # Wait: CD-KAN expects *historical sequence* of features, not flattened lagged features.
    # checking data.py...
    
    # The `data.py` `create_dataset` returns [Batch, Window, Features].
    # So Feature 0 is likely the asset price itself? 
    # Actually `data.py` typically scales the Close price.
    # If `create_dataset` just slides a window over the single column, then Features=1.
    # If it includes other macro variables, Features > 1.
    
    # Let's inspect shapes to interpret
    print(f"Input Features Dim: {in_features}")
    
    for i in range(out_features):
        for j in range(in_features):
            prob = adj[i, j].item()
            lag = lags[i, j].item()
            if prob > 0.5:
                print(f"Feature {j} -> Output {i}: Prob={prob:.2f}, Expected Lag={lag:.2f} steps")

if __name__ == "__main__":
    train_cdkan()
