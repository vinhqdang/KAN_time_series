
import sys
import os
import torch
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cdkan.layers import CDKANLayer, KANLayer
from src.cdkan.trainer import CDKANTrainer

class ExplicitCDKAN(nn.Module):
    def __init__(self, n_feats, hidden, out_feats, max_lag):
        super().__init__()
        self.cd_layer = CDKANLayer(n_feats, n_feats, max_lag=max_lag, grid_size=5)
        self.deep_kan = KANLayer(n_feats, hidden, grid_size=5)
        self.out_kan = KANLayer(hidden, out_feats, grid_size=5)
        
    def forward(self, x):
        x_causal = self.cd_layer(x) 
        x_deep = self.deep_kan(x_causal)
        return self.out_kan(x_deep)
        
    def set_temperature(self, t):
         if hasattr(self.cd_layer, 'temperature'):
            self.cd_layer.temperature.data.fill_(t)

def test_minimal():
    print("Testing CD-KAN Minimal...")
    device = 'cpu' # Force CPU for test
    B, T, D = 10, 20, 5
    max_lag = 5
    
    x = torch.randn(B, T, D).to(device)
    y = torch.randn(B, D).to(device)
    
    model = ExplicitCDKAN(D, 8, D, max_lag).to(device)
    trainer = CDKANTrainer(model, device=device)
    
    loader = {'X': x, 'y': y}
    
    print("Starting Training Loop (2 epochs)...")
    try:
        trainer.train(loader, loader, epochs=2)
        print("SUCCESS: Training completed without error.")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal()
