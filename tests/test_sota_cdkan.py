
import torch
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cdkan.model import CDKANForecaster

def test_sota_cdkan_forward():
    batch_size = 4
    window = 16
    n_assets = 5
    
    # Random input
    x = torch.randn(batch_size, window, n_assets)
    
    # Create model
    model = CDKANForecaster(
        in_features=n_assets,
        hidden_dim=32,
        out_features=n_assets,
        max_lag=5,
        n_layers=2
    )
    
    print("Model created successfully.")
    
    # Forward pass
    y = model(x)
    
    print(f"Output shape: {y.shape}")
    
    assert y.shape == (batch_size, n_assets), f"Expected ({batch_size}, {n_assets}), got {y.shape}"
    print("Forward pass successful.")
    
    # Test gradients
    loss = y.sum()
    loss.backward()
    print("Backward pass successful.")
    
    # Test Graph Extraction
    adj, lags = model.cd_layer.get_causal_graph()
    print(f"Adjacency shape: {adj.shape}")
    assert adj.shape == (n_assets, n_assets)
    print("Graph extraction successful.")

if __name__ == "__main__":
    test_sota_cdkan_forward()
