"""
evaluation.py — Reviewer-driven evaluation utilities for CD-KAN.

Provides:
  - walk_forward_split : strict temporal (rolling-origin) train/test splits
  - evaluate_with_ci  : multi-seed CI-aware forecasting evaluation
  - evaluate_causal_synthetic : F1 / SHD / AUROC restricted to datasets with ground truth
  - evaluate_causal_real      : stability-based indirect validation for real data
  - compute_graph_metrics     : helper for threshold-independent AUROC / PR curves
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Walk-forward (rolling-origin) temporal splitting
# ---------------------------------------------------------------------------

def walk_forward_split(
    data: np.ndarray,
    n_train: int,
    n_test:  int,
    stride:  int = 1,
    window:  int = 16,
    horizon: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Rolling-origin (walk-forward) dataset generator.

    Unlike a simple 80/20 split, this guarantees strict temporal order and
    prevents leakage: normalization stats are computed per-fold from the
    training slice only.

    Args:
        data     : [T, D] array of raw (un-scaled) time series
        n_train  : number of time steps in each training window
        n_test   : number of time steps in each test window
        stride   : how many steps to advance the window each fold
        window   : look-back window for sliding windows within each fold
        horizon  : forecast horizon

    Yields:
        (X_train, y_train, X_test, y_test) as np.float32 arrays,
        all normalized with training-fold statistics only.
    """
    T, D = data.shape
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    start = 0
    while start + n_train + n_test <= T:
        train_raw = data[start : start + n_train]
        test_raw  = data[start + n_train : start + n_train + n_test]

        # Normalize using TRAIN statistics only (no leakage)
        mean = train_raw.mean(axis=0, keepdims=True)
        std  = train_raw.std(axis=0, keepdims=True) + 1e-8
        train_scaled = (train_raw - mean) / std
        test_scaled  = (test_raw  - mean) / std

        X_tr, y_tr = _make_windows(train_scaled, window, horizon)
        X_te, y_te = _make_windows(test_scaled,  window, horizon)

        if X_tr.shape[0] > 0 and X_te.shape[0] > 0:
            folds.append((X_tr, y_tr, X_te, y_te))

        start += stride

    return folds


def _make_windows(
    data: np.ndarray, window: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) sliding windows from a scaled array."""
    X, y = [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i : i + window])
        y.append(data[i + window : i + window + horizon])
    if not X:
        return np.zeros((0, window, data.shape[1]), dtype=np.float32), \
               np.zeros((0, horizon, data.shape[1]), dtype=np.float32)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32))


# ---------------------------------------------------------------------------
# Multi-seed CI evaluation for forecasting
# ---------------------------------------------------------------------------

def evaluate_with_ci(
    model_fn:    Callable[[], nn.Module],
    trainer_fn:  Callable[[nn.Module], Any],
    folds:       List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seeds:       List[int] = (42, 43, 44),
    device:      str = 'cpu',
    epochs:      int = 100,
    patience:    int = 20,
) -> Dict[str, float]:
    """
    Evaluate a model over multiple seeds and folds, returning mean ± std.

    Args:
        model_fn   : factory that returns a fresh (untrained) model
        trainer_fn : factory that takes a model and returns a Trainer
        folds      : list of (X_tr, y_tr, X_te, y_te) from walk_forward_split
        seeds      : random seeds to use
        device     : compute device
        epochs     : max training epochs
        patience   : early stopping patience

    Returns:
        dict with 'mse_mean', 'mse_std', 'mae_mean', 'mae_std'
    """
    all_mse: List[float] = []
    all_mae: List[float] = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        fold_mse: List[float] = []
        fold_mae: List[float] = []

        for X_tr, y_tr, X_te, y_te in folds:
            # To tensor
            dev = torch.device(device)
            X_tr_t = torch.from_numpy(X_tr).to(dev)
            y_tr_t = torch.from_numpy(y_tr.squeeze(1) if y_tr.ndim == 3 and y_tr.shape[1] == 1
                                       else y_tr).to(dev)
            X_te_t = torch.from_numpy(X_te).to(dev)
            y_te_t = torch.from_numpy(y_te.squeeze(1) if y_te.ndim == 3 and y_te.shape[1] == 1
                                       else y_te).to(dev)

            model   = model_fn().to(dev)
            trainer = trainer_fn(model)

            train_loader = {'X': X_tr_t, 'y': y_tr_t}
            test_loader  = {'X': X_te_t, 'y': y_te_t}

            trainer.train(train_loader, test_loader, epochs=epochs, patience=patience)

            model.eval()
            with torch.no_grad():
                preds = model(X_te_t)
                mse   = nn.MSELoss()(preds, y_te_t).item()
                mae   = torch.mean(torch.abs(preds - y_te_t)).item()

            fold_mse.append(mse)
            fold_mae.append(mae)

        all_mse.extend(fold_mse)
        all_mae.extend(fold_mae)

    return {
        'mse_mean': float(np.mean(all_mse)),
        'mse_std':  float(np.std(all_mse)),
        'mae_mean': float(np.mean(all_mae)),
        'mae_std':  float(np.std(all_mae)),
        'n_runs':   len(all_mse),
    }


# ---------------------------------------------------------------------------
# Causal graph metrics (for synthetic datasets with ground truth only)
# ---------------------------------------------------------------------------

def binarise_adjacency(adj: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a soft adjacency to binary using a probability threshold."""
    return (adj >= threshold).astype(int)


def compute_graph_metrics(
    pred_adj:   np.ndarray,
    true_adj:   np.ndarray,
    threshold:  float = 0.5,
) -> Dict[str, float]:
    """
    Compute causal graph evaluation metrics.

    Threshold-based: F1, SHD (Structural Hamming Distance), Precision, Recall.
    Threshold-free:  AUROC, AUPRC (area under PR curve).

    NOTE: Only call this on *synthetic* datasets where the ground-truth adjacency
    is known. Applying these metrics to real-world datasets without a known DAG
    is methodologically invalid (as the reviewer noted).

    Args:
        pred_adj  : [D, D] float array of predicted edge probabilities
        true_adj  : [D, D] binary array of ground-truth edges
        threshold : probability threshold for binary classification

    Returns:
        dict of metric names → values
    """
    assert pred_adj.shape == true_adj.shape, "Shape mismatch"

    # Remove diagonal (no self-loops)
    d = pred_adj.shape[0]
    mask = ~np.eye(d, dtype=bool)

    pred_flat  = pred_adj[mask].astype(float)
    true_flat  = true_adj[mask].astype(int)
    pred_binary = (pred_flat >= threshold).astype(int)

    # Threshold-based
    f1   = f1_score(true_flat, pred_binary, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(true_flat, pred_binary, labels=[0, 1]).ravel() \
        if true_flat.sum() > 0 and (1 - true_flat).sum() > 0 else (0, 0, 0, 0)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    shd       = int(np.sum(np.abs(pred_binary - true_flat)))

    # Threshold-free (only if both classes present)
    auroc, auprc = float('nan'), float('nan')
    if true_flat.sum() > 0 and (1 - true_flat).sum() > 0:
        try:
            auroc = roc_auc_score(true_flat, pred_flat)
            auprc = average_precision_score(true_flat, pred_flat)
        except Exception:
            pass

    return {
        'f1':        f1,
        'shd':       shd,
        'precision': precision,
        'recall':    recall,
        'auroc':     auroc,
        'auprc':     auprc,
    }


def evaluate_causal_synthetic(
    pred_adj:  np.ndarray,
    true_adj:  np.ndarray,
    dataset_name: str = "synthetic",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Full causal evaluation for a *synthetic* dataset with known ground truth.

    Wraps compute_graph_metrics with clear labelling so results are never
    inadvertently mixed with real-world datasets.
    """
    metrics = compute_graph_metrics(pred_adj, true_adj, threshold=threshold)
    return {'dataset': dataset_name, 'has_ground_truth': True, **metrics}


def evaluate_causal_real(
    pred_adj_runs: List[np.ndarray],
    dataset_name:  str = "real",
    threshold:     float = 0.5,
) -> Dict[str, Any]:
    """
    Indirect validation for *real-world* datasets without ground truth.

    Instead of F1/SHD (which require ground truth), we report:
      - Graph stability across runs (mean pairwise Jaccard index)
      - Mean edge density

    Args:
        pred_adj_runs : list of [D, D] adjacency matrices from multiple runs/windows
        dataset_name  : name for labelling results
        threshold     : threshold for binarising soft adjacency

    Returns:
        dict with stability and density metrics
    """
    binary_runs = [binarise_adjacency(a, threshold) for a in pred_adj_runs]
    n = len(binary_runs)

    jaccard_scores: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b   = binary_runs[i].flatten(), binary_runs[j].flatten()
            inter  = (a & b).sum()
            union  = (a | b).sum()
            jaccard_scores.append(inter / (union + 1e-8))

    mean_density = np.mean([b.mean() for b in binary_runs])

    return {
        'dataset':           dataset_name,
        'has_ground_truth':  False,
        'stability_jaccard': float(np.mean(jaccard_scores)) if jaccard_scores else float('nan'),
        'mean_edge_density': float(mean_density),
        'n_runs':            n,
    }
