import asyncio
import contextlib
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import WINDOW, ASSETS, MAX_SCORED, KAFKA_BOOTSTRAP, TOPIC, DEVICE
from src.data import load_and_preprocess_data
from src.models import KANForecaster, LSTMForecaster, TransformerForecaster, TSMixer
from src.train import train_once
from src.kafka_client import ensure_topic, kafka_smoke, producer, consumer

# Suppress warnings
warnings.filterwarnings("ignore")

# Define available models
MODELS = ["KAN", "LSTM", "Transformer", "TSMixer"]

def summarize(truth, pred, infer_ms, e2e_ms):
    def _mape(y, yhat):
        y, yhat = np.asarray(y), np.asarray(yhat)
        if len(y) == 0: return np.nan
        return float(np.mean(np.abs((y - yhat) / (np.abs(y) + 1e-8))) * 100.0)

    for asset in ASSETS:
        y_true = truth[asset]
        print(f"\n=== {asset} ===")
        print(f"scored truths: {len(y_true)}")
        rows = []
        for m in MODELS:
            yhat = pred[m][asset]
            mae  = mean_absolute_error(y_true, yhat) if len(yhat) else np.nan
            rmse = mean_squared_error(y_true, yhat, squared=False) if len(yhat) else np.nan
            mp   = _mape(y_true, yhat)
            infs = infer_ms[m][asset]
            rows.append({
                "Model": m,
                "MAPE (%)": mp,
                "RMSE": rmse,
                "MAE": mae,
                "Avg Inference (ms)": float(np.mean(infs)) if infs else np.nan,
                "p50 Inference (ms)": float(np.quantile(infs, 0.50)) if infs else np.nan,
                "p95 Inference (ms)": float(np.quantile(infs, 0.95)) if infs else np.nan,
                "Samples": len(yhat)
            })
        df = pd.DataFrame(rows).sort_values("MAPE (%)")
        print(df.to_string(index=False))

        lat = e2e_ms[asset]
        if len(lat):
            print(f"E2E Latency  avg={np.mean(lat):.2f} ms | p50={np.quantile(lat,0.5):.2f} ms | p95={np.quantile(lat,0.95):.2f} ms")
        else:
            print("E2E Latency  (no samples)")

async def main():
    print("=== Starting Extension KAN Time Series Pipeline ===")
    
    # 1. Load Data
    print("\n--- Data Loading ---")
    scalers, datasets = load_and_preprocess_data()
    
    # 2. Train Models
    print("\n--- Model Training ---")
    models = {m: {} for m in MODELS}
    
    for asset, d in datasets.items():
        Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
        print(f"\n=== Train on {asset} ===")
        
        m_kan  = KANForecaster(WINDOW).to(DEVICE)
        m_lstm = LSTMForecaster().to(DEVICE)
        m_trf  = TransformerForecaster(WINDOW).to(DEVICE)
        m_mix  = TSMixer(WINDOW).to(DEVICE)

        train_once(m_kan,  Xtr, ytr, Xte, yte, steps=150, lr=1e-2, name=f"[{asset}] KAN")
        train_once(m_lstm, Xtr, ytr, Xte, yte, steps=150, lr=1e-3, name=f"[{asset}] LSTM")
        train_once(m_trf,  Xtr, ytr, Xte, yte, steps=150, lr=1e-3, name=f"[{asset}] Transformer")
        train_once(m_mix,  Xtr, ytr, Xte, yte, steps=150, lr=2e-3, name=f"[{asset}] TSMixer")

        models["KAN"][asset]          = m_kan.eval()
        models["LSTM"][asset]         = m_lstm.eval()
        models["Transformer"][asset]  = m_trf.eval()
        models["TSMixer"][asset]      = m_mix.eval()

    # 3. Streaming Evaluation
    print("\n--- Streaming Evaluation ---")
    
    # Setup Kafka
    await ensure_topic(TOPIC)
    await kafka_smoke()
    print("Topic ready:", TOPIC)
    
    TARGET_SCORES = {
        asset: int(min(MAX_SCORED, len(datasets[asset]["X_test"])))
        for asset in ASSETS
    }
    print("TARGET_SCORES:", TARGET_SCORES)
    
    # Buffers
    PRED     = {m: {a: [] for a in ASSETS} for m in MODELS}
    TRUTH    = {a: [] for a in ASSETS}
    INFER_MS = {m: {a: [] for a in ASSETS} for m in MODELS}
    E2E_MS   = {a: [] for a in ASSETS}
    
    # Run Producer/Consumer
    cons_task = asyncio.create_task(consumer(
        models, scalers, datasets, TARGET_SCORES,
        TRUTH, PRED, INFER_MS, E2E_MS
    ))
    await asyncio.sleep(0.7)
    
    prod_task = asyncio.create_task(producer(datasets, TARGET_SCORES))
    
    try:
        await asyncio.wait_for(asyncio.gather(prod_task, cons_task), timeout=120.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        print(f"[WARN] Timed out or cancelled; using partial results.")
        with contextlib.suppress(Exception): await prod_task
        with contextlib.suppress(Exception): await cons_task
        
    # 4. Summarize
    summarize(TRUTH, PRED, INFER_MS, E2E_MS)

if __name__ == "__main__":
    try:
        # Check for nest_asyncio just in case this is run in a notebook-like env
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
        
    asyncio.run(main())
