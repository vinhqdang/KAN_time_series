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
    try:
        # load_multivariate_data returns (scaler, dataset_dict)
        scaler, dataset = load_multivariate_data() 
        df = dataset['raw_df']
        # Transform using the scaler fitted in load_multivariate_data
        data_scaled = scaler.transform(df.values.astype(np.float32))
        return data_scaled, None # No ground truth
    except Exception as e:
        print(f"Error loading financial data: {e}")
        # Fallback to random if failed (for testing)
        return np.random.randn(1000, 8), None

def get_crypto_dataset():
    # Download BTC, ETH, SOL, ADA, XRP
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD']
    try:
        data = yf.download(tickers, start="2022-01-01", end="2025-01-01")['Close']
        data = data.ffill().bfill()
        # Log returns
        returns = np.log(data / data.shift(1)).dropna()
        # Normalize
        norm_data = (returns - returns.mean()) / returns.std()
        return norm_data.values, None
    except Exception as e:
         print(f"Error loading crypto data: {e}")
         return np.random.randn(1000, 5), None

def get_macro_dataset():
    # Download Macro indicators: 10Y Treasury, Oil, Gold, SP500, DXY
    tickers = ['^TNX', 'CL=F', 'GC=F', '^GSPC', 'DX-Y.NYB']
    try:
        data = yf.download(tickers, start="2020-01-01", end="2025-01-01")['Close']
        data = data.ffill().bfill()
        # Differencing for stationarity
        diff_data = data.diff().dropna()
        # Normalize
        norm_data = (diff_data - diff_data.mean()) / diff_data.std()
        return norm_data.values, None
    except Exception as e:
         print(f"Error loading macro data: {e}")
         return np.random.randn(1000, 5), None
