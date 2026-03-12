import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
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
START_DATE = "2020-01-01"
END_DATE   = "2025-01-01"

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
    "SPX":     {"ticker": "^GSPC"}, # S&P 500
    "BTC":     {"ticker": "BTC-USD"}, # Bitcoin
    "SLV":     {"ticker": "SLV"},   # Silver
    "COPPER":  {"ticker": "HG=F"},  # Copper
    "NG":      {"ticker": "NG=F"},  # Natural Gas
}


# ---------------------------------------------------------------------------
# CD-KAN Hyperparameter Config (centralised for reproducibility)
# ---------------------------------------------------------------------------

@dataclass
class CDKANConfig:
    """
    Single source of truth for all CD-KAN tunable parameters.

    Architecture
    ------------
    hidden_dim   : width of the residual KAN backbone
    n_layers     : number of ResidualKANBlock layers
    grid_size    : number of B-spline intervals per edge function
    max_lag      : maximum lag considered in CDKANLayer

    Regularisation / ALM
    --------------------
    lambda_sparse : L1 sparsity penalty weight on edge probabilities
    lambda_dag    : NOTEARS DAG penalty weight (multiplied with ALM terms)
    rho_init      : initial ALM penalty coefficient ρ
    rho_max       : ceiling for ρ
    alpha_init    : initial ALM dual variable α
    h_tol         : convergence threshold on h(W)
    update_freq   : epochs between ALM dual-variable updates

    Temperature annealing (Gumbel-Sigmoid)
    ----------------------------------------
    tau_init      : starting temperature
    tau_final     : minimum temperature
    tau_decay     : multiplicative decay per epoch

    Training
    --------
    lr            : Adam learning rate
    epochs        : default training epochs
    patience      : early stopping patience
    dropout       : dropout rate in backbone

    Reproducibility
    ---------------
    seed          : random seed (None = no seeding)
    n_seeds       : number of seeds to average over for CI reporting
    """
    # Architecture
    hidden_dim:    int   = 64
    n_layers:      int   = 3
    grid_size:     int   = 10
    max_lag:       int   = 10

    # Regularisation
    lambda_sparse: float = 0.01
    lambda_dag:    float = 1.0
    rho_init:      float = 1.0
    rho_max:       float = 1e16
    alpha_init:    float = 0.0
    h_tol:         float = 1e-8
    update_freq:   int   = 10

    # Temperature
    tau_init:      float = 1.0
    tau_final:     float = 0.1
    tau_decay:     float = 0.99

    # Training
    lr:            float = 1e-3
    epochs:        int   = 100
    patience:      int   = 20
    dropout:       float = 0.1

    # Reproducibility
    seed:          Optional[int] = 42
    n_seeds:       int   = 3


