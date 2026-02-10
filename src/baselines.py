import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
from src.config import DEVICE, WINDOW, HORIZON

class BaselineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # Predict scalar
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # Last step

class BaselineGRU(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def run_arima(train_series, test_len, order=(5,1,0)):
    # Simple recursive forecast or one-step
    history = [x for x in train_series]
    predictions = []
    # For speed, we might fit once or fit incrementally.
    # Fitting ARIMA every step is slow. Fitting once is less accurate.
    # Let's fit once on history and forecast horizon.
    # Wait, test set is often sequential.
    # Let's do a simple fit-predict on test set via sliding window?
    # Or just `forecast` if horizon matches.
    
    # Ideally: Walk-forward validation (Retrain or Re-filter)
    # Re-filtering (using new inputs without re-estimating params) is faster.
    model = ARIMA(history, order=order).fit()
    preds = model.forecast(steps=test_len)
    return preds

def run_prophet(train_df, test_len):
    if Prophet is None:
        print("Prophet not installed. Skipping.")
        return np.zeros(test_len)
    # train_df expects ['ds', 'y']
    m = Prophet(daily_seasonality=True)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=test_len)
    forecast = m.predict(future)
    return forecast['yhat'].iloc[-test_len:].values
    return forecast['yhat'].iloc[-test_len:].values

class TSMixer(nn.Module):
    def __init__(self, n_series, seq_len, pred_len, d_model=64, num_layers=4, dropout=0.1):
        super().__init__()
        self.n_series = n_series
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.layers = nn.ModuleList([
            TSMixerLayer(n_series, seq_len, d_model, dropout) 
            for _ in range(num_layers)
        ])
        
        self.temporal_projection = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        # x: [Batch, Seq, Feat]
        # TSMixer expects [Batch, Feat, Seq] usually?
        # Standard implementation mixes Time then Channel.
        
        x = x.transpose(1, 2) # [Batch, Feat, Seq]
        
        for layer in self.layers:
            x = layer(x)
            
        # Projection: x is [Batch, Feat, Seq] -> [Batch, Feat, Pred]
        x_out = self.temporal_projection(x) 
        
        return x_out.transpose(1, 2) # [Batch, Pred, Feat]

class TSMixerLayer(nn.Module):
    def __init__(self, n_series, seq_len, d_model, dropout):
        super().__init__()
        # Time Mixing
        self.time_norm = nn.BatchNorm1d(n_series)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout)
        )
        
        # Feature Mixing
        self.feat_norm = nn.BatchNorm1d(n_series)
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_series, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_series),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: [Batch, Feat, Seq]
        
        # Time mixing: apply to last dim (Seq)
        # Input to MLP should be last dim? Linear applies to last dim.
        # Yes.
        res = x
        x_norm = self.time_norm(x)
        x_time = self.time_mlp(x_norm)
        x = x + x_time
        
        # Feature mixing: apply to Feat dim
        # Transpose to [Batch, Seq, Feat]
        res = x
        x_t = x.transpose(1, 2) # [Batch, Seq, Feat]
        # BUT BatchNorm1d expects [Batch, Feat, Seq] if we treat Feat as Channel.
        # Here we mix features.
        
        # We need to mix along dimension 1 (Feat).
        # MLP applies to last dim. So transpose.
        x_feat = self.feat_mlp(x_t) # [Batch, Seq, Feat] -> Linear(Feat->Model->Feat)
        x_feat = x_feat.transpose(1, 2) # [Batch, Feat, Seq]
        
        x = res + x_feat
        return x
