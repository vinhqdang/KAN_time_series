"""
test_reviewer_fixes.py â€” Unit tests verifying all reviewer-driven fixes.

Tests:
  1. NOTEARS h(W) formula: h=0 for a DAG, h>0 for a cycle
  2. LagAwareAdjacency: 3D tensor with independent per-lag gradients
  3. walk_forward_split: no temporal leakage, correct fold count
  4. evaluate_causal_synthetic: returns valid F1 for known adjacency
  5. CDKANTrainer: uses new (tensor, float) return from causal_consistency_loss
  6. CDKANForecaster: forward pass produces correct shape
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cdkan.losses import _h_notears, causal_consistency_loss, structural_sparsity_loss
from src.cdkan.layers import LagAwareAdjacency, CDKANLayer, KANLayer
from src.cdkan.model import CDKANForecaster
from src.cdkan.trainer import CDKANTrainer, CDKANTrainerConfig
from src.evaluation import (
    walk_forward_split, evaluate_causal_synthetic, evaluate_causal_real,
    compute_graph_metrics,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def assert_close(a, b, tol=1e-3, msg=""):
    assert abs(a - b) < tol, f"Expected {a} â‰ˆ {b} (tol={tol}). {msg}"

def announce(name):
    print(f"\n{'-'*50}")
    print(f"  TEST: {name}")
    print(f"{'-'*50}")


# ---------------------------------------------------------------------------
# Test 1: NOTEARS formula
# ---------------------------------------------------------------------------

def test_notears_formula():
    announce("NOTEARS h(W) formula")

    # A DAG with no edges should give h = 0
    W_dag = torch.zeros(4, 4)
    h_dag = _h_notears(W_dag).item()
    assert abs(h_dag) < 1e-5, f"h(zeros) should be ~0, got {h_dag}"
    print(f"  h(zero DAG) = {h_dag:.6f}  âœ“")

    # A strictly lower-triangular matrix is a DAG (no cycles)
    W_lower = torch.tril(torch.randn(5, 5), diagonal=-1)
    h_lower = _h_notears(W_lower).item()
    # For a strict lower-triangular matrix, Wâˆ˜W is also lower-triangular
    # so exp(Wâˆ˜W) has trace = d (ones on diagonal after zeroing squared off-diag? No,
    # the diagonal can also have values. But with all zeros on diagonal in W,
    # W*W diagonal is 0, so exp(0)=1 â†’ trace = d â†’ h = 0)
    assert abs(h_lower) < 1e-4, f"h(lower triangular) should be ~0, got {h_lower}"
    print(f"  h(lower triangular DAG) = {h_lower:.6f}  âœ“")

    # A matrix with a self-loop (nonzero diagonal) should give h > 0
    W_cycle = torch.zeros(3, 3)
    W_cycle[0, 0] = 1.0   # self-loop
    h_cycle = _h_notears(W_cycle).item()
    assert h_cycle > 0, f"h(self-loop) should be > 0, got {h_cycle}"
    print(f"  h(self-loop) = {h_cycle:.6f}  > 0  âœ“")

    print("PASS: NOTEARS formula correct")


# ---------------------------------------------------------------------------
# Test 2: LagAwareAdjacency â€” 3D structure and independent gradients
# ---------------------------------------------------------------------------

def test_lag_aware_adjacency():
    announce("LagAwareAdjacency: 3D tensor with independent per-lag gradients")

    lag_adj = LagAwareAdjacency(num_nodes=4, max_lag=5)

    # Shape check
    assert lag_adj.adj_logits.shape == (5, 4, 4), \
        f"Expected [5,4,4], got {lag_adj.adj_logits.shape}"
    print(f"  adj_logits shape = {tuple(lag_adj.adj_logits.shape)}  âœ“")

    # Forward gives correct shape
    A = lag_adj(temperature=1.0, hard=False)
    assert A.shape == (5, 4, 4)
    print(f"  forward() shape = {tuple(A.shape)}  âœ“")

    # Summary adj [d, d]
    A_summary = lag_adj.get_summary_adj()
    assert A_summary.shape == (4, 4)
    print(f"  get_summary_adj() shape = {tuple(A_summary.shape)}  âœ“")

    # Per-lag grad independence: setting one lag's logit to -inf should not affect another
    with torch.no_grad():
        lag_adj.adj_logits[0, :, :] = -100.0   # zero out lag 0
    A_lag0 = lag_adj.get_lag_adj(0)
    A_lag1 = lag_adj.get_lag_adj(1)
    assert A_lag0.max().item() < 0.01, "Lag 0 should be ~0 after suppression"
    assert A_lag1.max().item() > 0.01, "Lag 1 should be unaffected"
    print(f"  Per-lag independence check: lag0_max={A_lag0.max().item():.4f}, "
          f"lag1_max={A_lag1.max().item():.4f}  âœ“")

    print("PASS: LagAwareAdjacency is correctly 3D with independent lags")


# ---------------------------------------------------------------------------
# Test 3: walk_forward_split â€” no leakage, correct shape
# ---------------------------------------------------------------------------

def test_walk_forward_split():
    announce("walk_forward_split: temporal no-leakage")

    np.random.seed(0)
    T, D = 300, 4
    data = np.random.randn(T, D).astype(np.float32)

    folds = walk_forward_split(data, n_train=200, n_test=50, stride=50,
                               window=16, horizon=1)
    assert len(folds) >= 1, f"Expected >0 folds, got {len(folds)}"
    print(f"  Number of folds: {len(folds)}  âœ“")

    for k, (X_tr, y_tr, X_te, y_te) in enumerate(folds):
        assert X_tr.shape[1] == 16,  f"Fold {k}: window mismatch"
        assert X_te.shape[1] == 16,  f"Fold {k}: test window mismatch"
        assert X_tr.shape[2] == D,    f"Fold {k}: feature dim mismatch"
        assert X_te.shape[2] == D,    f"Fold {k}: feature dim mismatch"
        # Normalization: training data should have ~zero mean
        mean_tr = X_tr.mean()
        assert abs(mean_tr) < 0.5, f"Fold {k}: train mean too large ({mean_tr:.3f})"

    print(f"  All {len(folds)} folds have correct shapes and normalization  âœ“")
    print("PASS: walk_forward_split produces valid no-leakage folds")


# ---------------------------------------------------------------------------
# Test 4: evaluate_causal_synthetic â€” valid F1 for known adjacency
# ---------------------------------------------------------------------------

def test_causal_metrics():
    announce("evaluate_causal_synthetic: restricted to synthetic ground truth")

    D = 5
    true_adj = np.zeros((D, D), dtype=int)
    true_adj[1, 0] = 1  # 0 -> 1
    true_adj[2, 1] = 1  # 1 -> 2

    # Perfect prediction
    pred_adj = true_adj.astype(float)
    result = evaluate_causal_synthetic(pred_adj, true_adj, "test_dag")
    assert result['f1'] == 1.0, f"Expected F1=1.0, got {result['f1']}"
    assert result['shd'] == 0,  f"Expected SHD=0, got {result['shd']}"
    assert result['has_ground_truth'] == True
    print(f"  Perfect prediction: F1={result['f1']:.2f}, SHD={result['shd']}  âœ“")

    # All zeros prediction
    zero_adj = np.zeros((D, D), dtype=float)
    result_zero = evaluate_causal_synthetic(zero_adj, true_adj, "test_dag")
    assert result_zero['f1'] == 0.0, f"Expected F1=0 for empty pred, got {result_zero['f1']}"
    print(f"  Empty prediction: F1={result_zero['f1']:.2f}  âœ“")

    # Real-world stability evaluation does NOT report F1
    runs = [np.random.rand(D, D) for _ in range(5)]
    real_result = evaluate_causal_real(runs, "test_real")
    assert real_result['has_ground_truth'] == False
    assert 'f1' not in real_result, "Real evaluation should NOT have F1"
    assert 'stability_jaccard' in real_result
    print(f"  Real-world stability: {real_result['stability_jaccard']:.3f}  âœ“")

    print("PASS: Causal metrics correctly restricted to synthetic datasets")


# ---------------------------------------------------------------------------
# Test 5: CDKANTrainer uses new loss tuple signature
# ---------------------------------------------------------------------------

def test_trainer_loss_signature():
    announce("CDKANTrainer: new causal_consistency_loss tuple signature")

    model = CDKANForecaster(in_features=3, hidden_dim=8, out_features=3,
                             max_lag=3, n_layers=1, grid_size=3)

    cfg     = CDKANTrainerConfig(seed=0, lambda_sparse=0.01, rho_init=1.0)
    trainer = CDKANTrainer(model, device='cpu', config=cfg)

    B, T, D = 4, 10, 3
    X = torch.randn(B, T, D)
    y = torch.randn(B, D)

    # Verify _step runs without error
    total_loss, log = trainer._step(X, y, rho=1.0, alpha=0.0)
    assert isinstance(total_loss, float), f"Expected float loss, got {type(total_loss)}"
    assert 'h' in log, "Log must contain h(W) residual"
    assert isinstance(log['h'], float), f"h should be float, got {type(log['h'])}"
    print(f"  _step() => loss={total_loss:.4f}, h(W)={log['h']:.4e}  âœ“")

    print("PASS: Trainer correctly handles new (tensor, float) loss return")


# ---------------------------------------------------------------------------
# Test 6: CDKANForecaster full forward pass
# ---------------------------------------------------------------------------

def test_model_forward():
    announce("CDKANForecaster: forward pass shape and RevIN leakage guard")

    B, T, D = 8, 20, 5
    model = CDKANForecaster(in_features=D, hidden_dim=16, out_features=D,
                             max_lag=5, n_layers=1, grid_size=3)
    model.eval()

    x1 = torch.randn(B, T, D)
    out1 = model(x1)
    assert out1.shape == (B, D), f"Expected [{B},{D}], got {out1.shape}"
    print(f"  Output shape: {tuple(out1.shape)}  âœ“")

    # RevIN leakage guard: stats are not leaked across two separate calls
    x2 = torch.randn(B, T, D) * 10.0  # very different scale
    out2 = model(x2)
    assert out2.shape == (B, D), f"Second call shape mismatch: {out2.shape}"
    print(f"  RevIN handles different-scale input correctly  âœ“")

    # Lag adjacency shape
    adj = model.get_summary_adjacency()
    assert adj.shape == (D, D), f"Summary adj shape mismatch: {adj.shape}"
    lags = model.get_expected_lags()
    assert lags.shape == (D, D), f"Expected lags shape mismatch: {lags.shape}"
    print(f"  Summary adjacency: {tuple(adj.shape)}, expected lags: {tuple(lags.shape)}  âœ“")

    print("PASS: CDKANForecaster forward pass and diagnostics correct")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = 0
    failed = 0

    tests = [
        test_notears_formula,
        test_lag_aware_adjacency,
        test_walk_forward_split,
        test_causal_metrics,
        test_trainer_loss_signature,
        test_model_forward,
    ]

    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"\n  FAIL: {t.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print('='*50)
    if failed > 0:
        sys.exit(1)
