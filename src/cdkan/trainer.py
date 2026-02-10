
import torch
import torch.nn as nn
import time
import numpy as np
from .losses import structural_sparsity_loss, causal_consistency_loss

class CDKANTrainer:
    def __init__(self, model, optimizer=None, device='cuda'):
        self.model = model
        self.device = device
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def train(self, train_loader, test_loader, epochs=100, patience=10):
        # 3-Phase Schedule
        # Phase 1: Warm-up (0-20)
        # Phase 2: Discovery (20-60)
        # Phase 3: Fine-tuning (60+)
        
        history = {'train_loss': [], 'test_loss': []}
        best_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting Training ({epochs} epochs)...")
        
        for epoch in range(epochs):
            start_time = time.time()
            self.model.train()
            
            # --- Schedule Update ---
            if epoch < 20:
                phase = "Warm-up"
                temp = 5.0
                lambda_sparse = 0.0
                lambda_dag = 0.0
            elif epoch < 60:
                phase = "Discovery"
                # Linear anneal 5.0 -> 0.5
                progress = (epoch - 20) / 40
                temp = 5.0 - progress * (5.0 - 0.5)
                lambda_sparse = 0.001 + progress * 0.009 # 0.001 -> 0.01
                lambda_dag = 0.01
            else:
                phase = "Fine-tuning"
                temp = 0.1
                lambda_sparse = 0.01
                lambda_dag = 0.1
                # Optional: Lock structure? (Not implemented here, soft lock via temp)
                
            if hasattr(self.model, 'set_temperature'):
                self.model.set_temperature(temp)
                
            # --- Training Loop ---
            epoch_mse = 0
            epoch_sparse = 0
            epoch_dag = 0
            
            # Assuming train_loader gives dictionary {'X': ..., 'y': ...} (Full batch)
            # or is a DataLoader
            
            if isinstance(train_loader, dict):
                # Full Batch
                X, y = train_loader['X'], train_loader['y']
                loss_val, log_dict = self._step(X, y, lambda_sparse, lambda_dag)
                epoch_mse = log_dict['mse']
                epoch_sparse = log_dict['sparse']
                
            else:
                # Mini-batch
                count = 0
                for batch_X, batch_y in train_loader:
                   loss_val, log_dict = self._step(batch_X, batch_y, lambda_sparse, lambda_dag)
                   epoch_mse += log_dict['mse']
                   count += 1
                epoch_mse /= max(1, count)

            # --- Validation ---
            val_mse = self.evaluate(test_loader)
            
            history['train_loss'].append(epoch_mse)
            history['test_loss'].append(val_mse)
            
            # Early Stopping
            if val_mse < best_loss:
                best_loss = val_mse
                patience_counter = 0
                # Save best model state?
            else:
                patience_counter += 1
                
            elapsed = time.time() - start_time
            
            if (epoch+1) % 10 == 0 or epochs <= 10:
                print(f"Epoch {epoch+1:03d} [{phase}] | Temp: {temp:.2f} | "
                      f"Train MSE: {epoch_mse:.5f} | Val MSE: {val_mse:.5f} | "
                      f"Sparse: {epoch_sparse:.5f} | Time: {elapsed:.1f}s")
                      
            if patience_counter >= patience and epoch > 60:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        return history

    def _step(self, X, y, lambda_sparse, lambda_dag):
        self.optimizer.zero_grad()
        
        pred = self.model(X)
        mse_loss = self.criterion(pred, y)
        
        # Aux Losses
        sparse_loss = structural_sparsity_loss(self.model, lambda_sparse)
        
        # DAG Loss (requires retrieving adjacency)
        dag_loss = torch.tensor(0.0, device=self.device)
        
        # Safety check for DAG loss compatibility
        if lambda_dag > 0:
            if hasattr(self.model, 'layers'):
                 # Try to find a CDKANLayer
                 # Could be first layer, or iterate?
                 # Assuming first layer for now as per architecture
                 try:
                     first_layer = self.model.layers[0]
                     if hasattr(first_layer, 'edge_logits'):
                           layer = first_layer
                           in_dim = layer.in_features
                           out_dim = layer.out_features
                           if in_dim == out_dim:
                                adj = torch.zeros(out_dim, in_dim, device=self.device)
                                for i in range(out_dim):
                                    for j in range(in_dim):
                                        edge_id = f"{i}_{j}"
                                        if edge_id in layer.edge_logits:
                                            adj[i, j] = torch.sigmoid(layer.edge_logits[edge_id])
                                dag_loss = causal_consistency_loss(adj, None, lambda_dag)
                 except Exception:
                     pass # metrics shouldn't crash training

        total_loss = mse_loss + sparse_loss + dag_loss
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), {'mse': mse_loss.item(), 'sparse': sparse_loss.item()}

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
