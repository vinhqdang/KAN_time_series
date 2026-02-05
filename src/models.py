import torch
import torch.nn as nn
from kan import KAN
from .config import WINDOW, HORIZON, DEVICE

# --- KAN wrapper ---
class KANForecaster(nn.Module):
    def __init__(self, window=WINDOW, hidden=32, horizon=HORIZON, seed=42):
        super().__init__()
        # grid=2, k=3 from notebook
        self.kan = KAN(width=[window, hidden, horizon], grid=2, k=3, seed=seed, device=DEVICE.type)

    def forward(self, x):                  # x: [B, W]
        return self.kan(x)

# --- LSTM baseline ---
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden=64, num_layers=1, horizon=HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden, horizon)
    def forward(self, x):                  # x: [B, W]
        x = x.unsqueeze(-1)               # [B, W, 1]
        out, _ = self.lstm(x)
        last = out[:, -1, :]              # [B, H]
        return self.fc(last)              # [B, horizon]

# --- Simple Transformer forecaster (Encoder-only) ---
class TransformerForecaster(nn.Module):
    def __init__(self, window=WINDOW, d_model=32, nhead=4, num_layers=2, dim_ff=64, horizon=HORIZON):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        self.fc = nn.Linear(d_model, horizon)
    def forward(self, x):                  # x: [B, W]
        x = x.unsqueeze(-1)               # [B, W, 1]
        z = self.embed(x) + self.pos      # [B, W, d_model]
        z = self.encoder(z)               # [B, W, d_model]
        last = z[:, -1, :]                # [B, d_model]
        return self.fc(last)              # [B, horizon]

# --- Simple TS-Mixer forecaster ---
class TSMixer(nn.Module):
    def __init__(self, window=WINDOW, d_feat=16, time_hidden=64, chan_hidden=32, num_blocks=2, horizon=HORIZON):
        super().__init__()
        self.proj_in  = nn.Linear(1, d_feat)
        self.time_mlps = nn.ModuleList([nn.Sequential(
            nn.LayerNorm([window, d_feat]),
            nn.Linear(window, time_hidden), nn.GELU(),
            nn.Linear(time_hidden, window)
        ) for _ in range(num_blocks)])
        self.chan_mlps = nn.ModuleList([nn.Sequential(
            nn.LayerNorm([window, d_feat]),
            nn.Linear(d_feat, chan_hidden), nn.GELU(),
            nn.Linear(chan_hidden, d_feat)
        ) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(window*d_feat, 64), nn.GELU(), nn.Linear(64, horizon)
        )
    def forward(self, x):                  # x: [B, W]
        x = x.unsqueeze(-1)               # [B, W, 1]
        z = self.proj_in(x)               # [B, W, F]
        for tmlp, cmlp in zip(self.time_mlps, self.chan_mlps):
            z = z + tmlp(z.transpose(1,2)).transpose(1,2)  # mix over time
            z = z + cmlp(z)                                # mix over channels
        return self.head(z)               # [B, horizon]
