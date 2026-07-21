# Vendored third-party baselines

Vendored verbatim for reproducibility of the causal-discovery benchmark.

- `neural_gc/` — Neural Granger Causality (cMLP/cLSTM), Tank et al., IEEE TPAMI 2021.
  Source: https://github.com/iancovert/Neural-GC (MIT License). We use the cMLP
  architecture and its `GC()` read-out; training uses a standard Adam + group-lasso
  loop (the repo's GISTA/Adam trainers are incompatible with this environment's
  PyTorch), which is the method's prescribed group-sparse objective.
- `nts_notears/` — NTS-NOTEARS, Sun et al., AISTATS 2023.
  Source: https://github.com/xiangyu-sun-789/NTS-NOTEARS (Apache-2.0 License).
  Used unmodified via `NTS_NOTEARS` + `train_NTS_NOTEARS`.

Rhino (Gong et al., ICLR 2023) could not be added: its `causica` package requires
Python <3.11, incompatible with this environment (Python 3.11).
