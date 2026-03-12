import numpy as np
import pandas as pd
import yfinance as yf
from ..synthetic import generate_nonlinear_scm 
from ..data import load_multivariate_data

def get_dataset(name):
    if name == 'synthetic_linear':
        return get_synthetic_linear()
    elif name == 'synthetic_nonlinear':
        return get_synthetic_nonlinear()
    elif name == 'financial':
        return get_financial_dataset()
    elif name == 'crypto':
        return get_crypto_dataset()
    elif name == 'macro':
        return get_macro_dataset()
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_synthetic_linear(n_samples=1000, n_nodes=5):
    # Using existing synthetic non-linear generator but forcing linear functions?
    # Or simple VAR generation here.
    np.random.seed(42)
    # Generate A (DAG)
    adj = np.random.binomial(1, 0.2, size=(n_nodes, n_nodes))
    np.fill_diagonal(adj, 0)
    # Ensure acyclicity
    adj = np.tril(adj, k=-1) 
    
    # Generate Data
    X = np.zeros((n_samples, n_nodes))
    coeffs = np.random.uniform(0.5, 0.9, size=(n_nodes, n_nodes)) * adj
    
    for t in range(1, n_samples):
        noise = np.random.normal(0, 0.1, size=n_nodes)
        X[t] = coeffs @ X[t-1] + noise
        
    return X, adj

def get_synthetic_nonlinear(n_samples=1000, n_nodes=5):
    # Wrapper around src.synthetic
    # returns data, adj, lags
    X, adj_true, _ = generate_nonlinear_scm(n_samples, n_nodes, density=0.2)
    return X, adj_true

def get_financial_dataset():
    # Helper to load the standard 8-asset dataset
    # Returns X (numpy), and 'None' for adj (unknown)
    # load_multivariate_data returns (scaler, dataset_dict)
    scaler, dataset = load_multivariate_data() 
    df = dataset['raw_df']
    # Walk-forward handles the scaling internally
    data_raw = df.values.astype(np.float32)
    return data_raw, None # No ground truth

def get_crypto_dataset():
    # Download BTC, ETH, SOL, ADA, XRP
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD']
    data = yf.download(tickers, start="2022-01-01", end="2025-01-01")['Close']
    data = data.ffill().dropna() # no backward fill
    # Log returns
    returns = np.log(data / data.shift(1)).dropna()
    return returns.values.astype(np.float32), None

def get_macro_dataset():
    # Download Macro indicators: 10Y Treasury, Oil, Gold, SP500, DXY
    tickers = ['^TNX', 'CL=F', 'GC=F', '^GSPC', 'DX-Y.NYB']
    data = yf.download(tickers, start="2020-01-01", end="2025-01-01")['Close']
    data = data.ffill().dropna() # no backward fill
    # Differencing for stationarity
    diff_data = data.diff().dropna()
    return diff_data.values.astype(np.float32), None
