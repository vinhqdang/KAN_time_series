import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
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
    # train_df expects ['ds', 'y']
    m = Prophet(daily_seasonality=True)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=test_len)
    forecast = m.predict(future)
    return forecast['yhat'].iloc[-test_len:].values
