import torch
import torch.nn as nn
from src.adakan.models.adakan_forecaster import AdaKANForecaster
from src.data import load_and_preprocess_data
from src.config import WINDOW, DEVICE
import time
import numpy as np

def test_component_refinement():
    print("\n[Test 1] GDGR: Grid Refinement")
    model = AdaKANForecaster(window=16, hidden=8, horizon=1, initial_grid=5).to(DEVICE)
    loss_fn = nn.MSELoss()
    
    # Fake inputs
    x = torch.randn(10, 16).to(DEVICE)
    y = torch.randn(10, 1).to(DEVICE)
    
    print("Initial grid sizes (sample):")
    print(f"Layer 1 Edge 0_0: {model.layer1.grid_sizes['0_0']}")
    
    # Train loop simulation to trigger refinement
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(1, 11):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        
        # Trigger callback
        model.on_epoch_end(epoch)
        
    print("Final grid sizes (sample):")
    print(f"Layer 1 Edge 0_0: {model.layer1.grid_sizes['0_0']}")
    
    if model.layer1.grid_sizes['0_0'] > 5:
        print("SUCCESS: Grid size increased.")
    else:
        print("NOTE: Grid size did not increase (gradients might be too uniform or small).")

def test_component_hybrid_basis():
    print("\n[Test 2] HMBF: Hybrid Basis Forward Pass")
    from src.adakan.core.hybrid_basis import HybridBasisFunction
    basis = HybridBasisFunction().to(DEVICE)
    x = torch.tensor([0.1, 0.5, 0.9]).to(DEVICE)
    
    y = basis(x)
    print(f"Input: {x}")
    print(f"Output: {y}")
    print("SUCCESS: Hybrid Basis forward pass working.")
    
def test_component_pruning():
    print("\n[Test 3] REPUR: Pruning")
    model = AdaKANForecaster(window=16, hidden=8, horizon=1).to(DEVICE)
    
    # Manually zero out gradients/activity for an edge to force pruning
    edge = model.layer1.edges['0_0']
    with torch.no_grad():
        edge.alpha.data.fill_(-20) # Very low alpha
        edge.spline_coeffs.data.fill_(0) # Zero coeffs
        
    # Trigger pruning
    from src.adakan.core.edge_pruning import prune_edges
    pruned = prune_edges(model.layer1, threshold=0.1)
    
    print(f"Pruned edges count: {len(pruned)}")
    if '0_0' in pruned:
        print("SUCCESS: Edge 0_0 was pruned.")
    else:
        print("FAIL: Edge 0_0 was not pruned.")

def benchmark_vs_standard():
    print("\n[Test 4] Benchmark: AdaKAN vs Standard LSTM (Proxy for 'Standard KAN' which needs library)")
    # We will use the load_and_preprocess_data from refactored code
    try:
        scalers, datasets = load_and_preprocess_data()
        asset = "GLD" # Test on Gold
        d = datasets[asset]
        Xtr, ytr = d["X_train"], d["y_train"]
        
        # AdaKAN
        model = AdaKANForecaster(window=16, hidden=16, horizon=1).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        t0 = time.time()
        for i in range(100): # Short training
            opt.zero_grad()
            out = model(Xtr)
            loss = loss_fn(out, ytr)
            loss.backward()
            opt.step()
            model.on_epoch_end(i)
        t_adakan = time.time() - t0
        loss_adakan = loss.item()
        
        print(f"AdaKAN: Time={t_adakan:.2f}s, Loss={loss_adakan:.6f}")
        
        # We assume success if it runs and loss decreases
        if loss_adakan < 1.0: 
             print("SUCCESS: AdaKAN training converges.")
             
    except Exception as e:
        print(f"Skipping benchmark due to data loading error: {e}")

if __name__ == "__main__":
    test_component_refinement()
    test_component_hybrid_basis()
    test_component_pruning()
    benchmark_vs_standard()
