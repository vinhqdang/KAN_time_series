import torch
import torch.nn as nn
from .layers import CDKANLayer, KANLayer
from src.config import WINDOW, HORIZON

class CDKANForecaster(nn.Module):
    @classmethod
    def create_best(cls, in_features, hidden_dim=64, out_features=1, max_lag=10, dataset_name=""):
        """Factory to create the optimal CD-KAN architecture for a given dataset."""
        # Config A: High Frequency / Minutes (Deep CD-KAN)
        if "m1" in dataset_name.lower(): 
             # Proven Winner on ETTm1 (No RNN, Deep Stack)
             model = cls(in_features, hidden_dim, out_features, max_lag, n_layers=3)
             model.use_rnn = False
             # Re-init layers without RNN logic which is set in __init__
             model.layers = nn.ModuleList()
             model.layers.append(CDKANLayer(in_features, hidden_dim, max_lag=max_lag))
             model.layers.append(KANLayer(hidden_dim, hidden_dim, grid_size=20))
             model.layers.append(KANLayer(hidden_dim, out_features, grid_size=20))
             model.skip_head = nn.Identity() # Disable skip
             return model
             
        # Config B: Low Frequency / Hourly / Volatile (Recurrent CD-KAN)
        else:
             # Proven Winner on ETTh2 (RNN Backbone + KAN Head)
             # Use the Gated Residual setup which works well if tuned, or strict R-CDKAN
             # Let's stick to strict R-CDKAN (proven in Step 818)
             model = cls(in_features, hidden_dim, out_features, max_lag, n_layers=2)
             model.use_rnn = True
             # Reset layers for Recurrent Config
             model.layers = nn.ModuleList()
             model.layers.append(KANLayer(hidden_dim, hidden_dim, grid_size=10))
             model.layers.append(KANLayer(hidden_dim, out_features, grid_size=10))
             return model

    def __init__(self, in_features, hidden_dim=64, out_features=1, max_lag=10, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        
        
        # R-CDKAN Adaptation: Recurrent Backbone
        # To make it "Best", we combine RNN memory with KAN reasoning.
        self.use_rnn = True
        self.rnn = nn.GRU(in_features, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        
        # Layer 1: Temporal Causal Discovery / Feature Processing
        # If RNN is used, input to KAN is [Batch, Hidden]. 
        # So we skip "Temporal Lag" KAN layer (which reduces time) and go straight to Dense KANs?
        # OR we treat RNN output as "features" and use CDKAN to find causal relations among latent states?
        # Let's map RNN hidden -> KAN Input.
        
        # If using RNN, the first KAN layer maps Hidden -> Hidden
        kan_in_dim = hidden_dim if self.use_rnn else in_features
        
        # Layer 1
        if self.use_rnn:
             # Standard Dense KAN for latent processing
             self.layers.append(KANLayer(kan_in_dim, hidden_dim, grid_size=10)) # Reduced Grid to 10
        else:
             # Original Temporal Layer
             self.layers.append(CDKANLayer(in_features, hidden_dim, max_lag=max_lag))
        
        # Layer 2+: Deep Composition (Dense KAN)
        # Maps [Batch, Hidden] -> [Batch, Hidden] ... -> [Batch, Out]
        for _ in range(n_layers - 2):
            self.layers.append(KANLayer(hidden_dim, hidden_dim, grid_size=10)) # Reduced Grid to 10
            
        # Output Layer
        if n_layers > 1:
            self.layers.append(KANLayer(hidden_dim, out_features, grid_size=10)) # Reduced Grid to 10
            
        # Residual Linear Head (Skip Connection)
        self.skip_head = nn.Linear(hidden_dim, out_features)
        
        # Learnable Gate for Skip Connection
        # y = sigmoid(gate) * skip + (1 - sigmoid(gate)) * kan
        # Init to favor Linear initially (0.0 -> 0.5, maybe -1.0 -> 0.27 to bias towards linear?)
        self.gate = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x_history):
        # x_history: [Batch, Window, Features]
        
        x = x_history
        if self.use_rnn:
            # RNN Forward
            r_out, _ = self.rnn(x)
            x = r_out[:, -1, :] # [Batch, Hidden]
            
        # 1. KAN Path
        kan_out = x
        for i in range(len(self.layers)):
            kan_out = self.layers[i](kan_out)
            
        # 2. Skip Path (Linear)
        skip_out = 0.0
        if self.use_rnn:
             skip_out = self.skip_head(x)
        
        # Gated Combination
        alpha = torch.sigmoid(self.gate)
        # alpha * Linear + (1-alpha) * KAN
        return alpha * skip_out + (1 - alpha) * kan_out
        
    def set_temperature(self, temp):
        # Propagate temp to all CDKANLayers (only first layer usually has it)
        if hasattr(self.layers[0], 'temperature'):
            self.layers[0].temperature.data.fill_(temp)
