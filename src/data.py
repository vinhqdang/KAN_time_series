import os
import numpy as np
import pandas as pd
import yfinance as yf
try:
    from pandas_datareader import data as web
except (ImportError, TypeError):
    web = None
from sklearn.preprocessing import MinMaxScaler
import torch

from .config import START_DATE, END_DATE, ASSETS, FRED_SERIES, WINDOW, HORIZON, DEVICE

# Check for FRED key
FRED_API_KEY = os.environ.get("FRED_API_KEY", "").strip()
_fred = None
if FRED_API_KEY:
    try:
        from fredapi import Fred
        _fred = Fred(api_key=FRED_API_KEY)
    except Exception:
        pass

def _try_yf(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(df, pd.DataFrame) and "Close" in df.columns and not df["Close"].dropna().empty:
            out = df[["Close"]].dropna().copy()
            out.index.name = "Date"
            out.sort_index(inplace=True)
            print(f"[YF] {ticker}: {len(out)} rows")
            return out
    except Exception as e:
        print(f"[YF FAIL] {ticker}: {e}")
    return None

def _try_stooq(ticker, start, end):
    if web is None:
        return None
    try:
        df = web.DataReader(ticker, "stooq", start=start, end=end)
        df = df.sort_index()
        cols = {c: c.title() for c in df.columns}
        df.rename(columns=cols, inplace=True)
        if "Close" in df.columns and not df["Close"].dropna().empty:
            out = df[["Close"]].dropna().copy()
            out.index.name = "Date"
            print(f"[STQ] {ticker}: {len(out)} rows")
            return out
    except Exception as e:
        print(f"[STQ FAIL] {ticker}: {e}")
    return None

def _try_fred(ticker, start, end):
    if not _fred: 
        return None
    sid = FRED_SERIES.get(ticker)
    if not sid:
        return None
    try:
        s = _fred.get_series(sid, observation_start=start, observation_end=end)
        s = s.dropna()
        if s.empty: 
            return None
        df = pd.DataFrame({"Close": s})
        df.index = pd.to_datetime(df.index)
        df = df.resample("D").ffill().dropna()
        df.index.name = "Date"
        print(f"[FRED] {ticker}->{sid}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[FRED FAIL] {ticker}: {e}")
        return None

def load_close_series(ticker, start=START_DATE, end=END_DATE):
    # 1) yfinance
    out = _try_yf(ticker, start, end)
    if out is not None: 
        return out
    # 2) stooq fallback
    out = _try_stooq(ticker, start, end)
    if out is not None:
        return out
    # 3) FRED
    out = _try_fred(ticker, start, end)
    if out is not None:
        return out
    raise RuntimeError(f"No real data available for {ticker}. Provide FRED_API_KEY or adjust tickers/date range.")

def build_supervised(series, window=WINDOW, horizon=HORIZON):
    s = np.asarray(series, dtype=np.float32)
    X, y = [], []
    for i in range(len(s) - window - horizon + 1):
        X.append(s[i:i+window])
        y.append(s[i+window:i+window+horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, horizon)

def create_dataset(series, window=WINDOW, horizon=HORIZON):
    s = np.asarray(series, dtype=np.float32)
    X, y = [], []
    for i in range(len(s) - window - horizon + 1):
        X.append(s[i:i+window])
        y.append(s[i+window:i+window+horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, horizon)

def load_ett_data(file_path="ETTh1.csv", target_col="OT", window=WINDOW, horizon=HORIZON):
    """
    Load ETT dataset (Electricity Transformer Temperature)
    Columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    Target: OT
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None, None
        
    # Scale all columns
    scaler = MinMaxScaler()
    data = df.iloc[:, 1:].values # Skip date, take all cols (7 features)
    data_scaled = scaler.fit_transform(data)
    
    # Identify target index (OT is usually last)
    target_idx = -1
    if target_col in df.columns:
        target_idx = df.columns.get_loc(target_col) - 1 # -1 because we skipped date
        
    n = len(data_scaled)
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    
    train = data_scaled[:n_train]
    test = data_scaled[-n_test:]
    
    # create_dataset returns (X, y) where y is full horizon for all features
    # We want X=[Window, Feats], y=[Horizon, Target]
    
    def make_xy(d):
        X, Y = [], []
        for i in range(len(d) - window - horizon + 1):
            X.append(d[i:i+window])
            # Target is only the OT column
            Y.append(d[i+window:i+window+horizon, target_idx])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32).reshape(-1, horizon)

    X_train, y_train = make_xy(train)
    X_test, y_test = make_xy(test)
    
    dataset = {
        "X_train": torch.FloatTensor(X_train),
        "y_train": torch.FloatTensor(y_train),
        "X_test": torch.FloatTensor(X_test),
        "y_test": torch.FloatTensor(y_test)
    }
    return scaler, dataset

def load_and_preprocess_data():
    raw = {}
    for name, cfg in ASSETS.items():
        raw[name] = load_close_series(cfg["ticker"], start=START_DATE, end=END_DATE)

    scalers, datasets = {}, {}
    for asset, df in raw.items():
        sc = MinMaxScaler()
        scaled = sc.fit_transform(df[["Close"]]).astype(np.float32).reshape(-1)
        scalers[asset] = sc
        X, y = build_supervised(scaled, WINDOW, HORIZON)
        split = int(len(X)*0.8)
        datasets[asset] = {
            "df": df,
            "scaled": scaled,
            "X_train": torch.tensor(X[:split], dtype=torch.float32, device=DEVICE),
            "y_train": torch.tensor(y[:split], dtype=torch.float32, device=DEVICE),
            "X_test":  torch.tensor(X[split:], dtype=torch.float32, device=DEVICE),
            "y_test":  torch.tensor(y[split:], dtype=torch.float32, device=DEVICE),
            "split":   split,
            "X_all":   X, "y_all": y
        }
        print(f"{asset}: train {X[:split].shape}, test {X[split:].shape}")
    return scalers, datasets
