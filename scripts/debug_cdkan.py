import torch
import torch.nn as nn
from src.cdkan.model import CDKANForecaster
from src.cdkan.trainer import CDKANTrainer

def test_cdkan():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Random Data
    n_features = 5
    X = torch.randn(100, 10, n_features).to(device)
    y = torch.randn(100, 1, n_features).to(device)
    
    loader = {'X': X, 'y': y}
    
    # Model
    model = CDKANForecaster(n_features, hidden_dim=32, out_features=n_features, max_lag=5).to(device)
    trainer = CDKANTrainer(model, device=device)
    
    print("Starting Train...")
    try:
        trainer.train(loader, loader, epochs=5)
        print("Training Success!")
    except Exception as e:
        print(f"Training Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cdkan()
