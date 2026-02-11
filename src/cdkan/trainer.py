import torch
import torch.nn as nn
import time
import numpy as np
from .losses import structural_sparsity_loss, causal_consistency_loss, group_lasso_loss

class CDKANTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
        # Split parameters
        # We need to optimize structure and weights together or separately.
        # ALM typically optimizes them together in the primal step.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        self.criterion = nn.MSELoss()
        
    def train(self, train_loader, test_loader, epochs=100, patience=10):
        # ALM Parameters
        rho = 1.0
        alpha = 0.0
        h_tol = 1e-8
        rho_max = 1e20
        
        history = {'train_loss': [], 'test_loss': []}
        best_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting ALM Training ({epochs} max epochs)...")
        
        # We break epochs into "ALM Steps".
        # Let's assume we do 10-20 ALM steps.
        # inner_epochs = epochs // 10? Or dynamic?
        # Let's try continuous updates schedule for simplicity and stability in deep learning context
        # "DAG-GNN" style: update rho/alpha every N epochs.
        
        update_freq = 10 # Update dual vars every 10 epochs
        
        for epoch in range(epochs):
            start_time = time.time()
            self.model.train()
            
            # --- Training Loop ---
            epoch_mse = 0
            epoch_sparse = 0
            epoch_dag = 0
            epoch_h = 0
            
            if isinstance(train_loader, dict):
                X, y = train_loader['X'], train_loader['y']
                loss_val, log_dict = self._step(X, y, rho, alpha)
                epoch_mse = log_dict['mse']
                epoch_sparse = log_dict['sparse']
                epoch_dag = log_dict['dag']
                epoch_h = log_dict['h']
            else:
                count = 0
                for batch_X, batch_y in train_loader:
                   loss_val, log_dict = self._step(batch_X, batch_y, rho, alpha)
                   epoch_mse += log_dict['mse']
                   epoch_sparse += log_dict['sparse']
                   epoch_dag += log_dict['dag']
                   epoch_h += log_dict['h']
                   count += 1
                epoch_mse /= max(1, count)
                epoch_sparse /= max(1, count)
                epoch_dag /= max(1, count)
                epoch_h /= max(1, count)

            # --- Validation ---
            val_mse = self.evaluate(test_loader)
            
            history['train_loss'].append(epoch_mse)
            history['test_loss'].append(val_mse)
            
            elapsed = time.time() - start_time
            
            if (epoch+1) % 1 == 0:
                print(f"Epoch {epoch+1:03d} | Rho: {rho:.1f} | Alpha: {alpha:.2f} | "
                      f"MSE: {epoch_mse:.5f} | Val MSE: {val_mse:.5f} | "
                      f"DAG Loss: {epoch_dag:.5f} | h(A): {epoch_h:.5e} | Sparse: {epoch_sparse:.5f}")

            # --- ALM Dual Update ---
            # Increase rho if constraint not satisfied
            if (epoch + 1) % update_freq == 0:
                with torch.no_grad():
                    # Calculate h(A) on the whole model (approx from last batch avg)
                    # Ideally we recompute h(A) exactly.
                    # Let's trust epoch_h approx or recompute.
                    h_val = epoch_h 
                    
                    if h_val > h_tol:
                        rho = min(rho * 2.0, rho_max)
                        alpha = alpha + rho * h_val
                        print(f"ALM Update: Rho -> {rho:.1f}, Alpha -> {alpha:.4f}")
            
            # Early Stopping (on val mse)
            if val_mse < best_loss:
                best_loss = val_mse
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience and rho > 100: # Only stop if we have sufficient penalty
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        return history

    def _step(self, X, y, rho, alpha):
        self.optimizer.zero_grad()
        
        pred = self.model(X)
        mse_loss = self.criterion(pred, y)
        
        # Sparsity (L1 on structure)
        lambda_sparse = 0.01 # Tuning knob
        sparse_loss = structural_sparsity_loss(self.model, lambda_sparse)
        
        # DAG Constraint h(A)
        # causal_consistency_loss returns lambda * (tr(e^A) - d)
        # We need the raw value 'h(A)' first for ALM.
        # But `causal_consistency_loss` computes it. 
        # Let's modify usage: calculate h(A) manually or reuse function with lambda=1?
        # Let's look at `losses.py`. `causal_consistency_loss` returns `lambda * diff`.
        
        # Using helper function from losses but treating lambda as 1.0 to get h(A) estimate
        h_val_loss = causal_consistency_loss(self.model, lambda_dag=1.0) 
        h_val = h_val_loss.item() # This is h(A) assuming single DAG module
        
        # ALM Terms
        # alpha * h(A) + rho/2 * h(A)^2
        alm_term = alpha * h_val_loss + (rho / 2.0) * h_val_loss * h_val_loss
        
        total_loss = mse_loss + sparse_loss + alm_term
        total_loss.backward()
        
        self.optimizer.step()
        
        return total_loss.item(), {
            'mse': mse_loss.item(), 
            'sparse': sparse_loss.item(), 
            'dag': alm_term.item(),
            'h': h_val
        }

    def evaluate(self, loader):
        self.model.eval()
        total_mse = 0
        count = 0
        with torch.no_grad():
            if isinstance(loader, dict):
                pred = self.model(loader['X'])
                total_mse = self.criterion(pred, loader['y']).item()
            else:
                for X, y in loader:
                    pred = self.model(X)
                    loss = self.criterion(pred, y)
                    total_mse += loss.item()
                    count += 1
                total_mse /= max(1, count)
        return total_mse
