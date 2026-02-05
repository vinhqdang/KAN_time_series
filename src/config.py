import os
import time
import uuid
import torch

# ---------- Runtime ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "127.0.0.1:9094")
RUN_ID = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
TOPIC  = f"mln_stream_{RUN_ID}"

# Sliding-window + horizon
WINDOW  = 16
HORIZON = 1  # one-step ahead

# Evaluation caps
MAX_SCORED = 50     # max scored predictions per asset before stopping

# Date Range
START_DATE = "2023-01-01"
END_DATE   = "2024-01-01"

# Map for optional FRED daily series
FRED_SERIES = {
    "CL=F":     "DCOILWTICO",   # WTI crude oil spot/daily
    "DX-Y.NYB": "DTWEXBGS",     # Broad Dollar Index daily
    "GLD":      "GOLDAMGBD228NLBM",  # London morning fix (daily)
}

ASSETS = {
    "GLD":     {"ticker": "GLD"},
    "WTI":     {"ticker": "CL=F"},
    "DXY":     {"ticker": "DX-Y.NYB"},
}
