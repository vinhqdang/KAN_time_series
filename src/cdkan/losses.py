
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# NOTEARS acyclicity helpers
# ---------------------------------------------------------------------------

def _h_notears(W: torch.Tensor) -> torch.Tensor:
    """
    Correct NOTEARS acyclicity constraint (Zheng et al., 2018):
        h(W) = tr(exp(W ∘ W)) − d = 0
    where W is a real-valued d×d weight matrix and ∘ denotes element-wise product.

    Args:
        W: [d, d] real-valued adjacency weights (NOT sigmoid-bounded probabilities).
    Returns:
        Scalar tensor h(W) ≥ 0; equals 0 iff W is a DAG.
    """
    assert W.dim() == 2 and W.shape[0] == W.shape[1], "W must be square"
    d = W.shape[0]
    # Element-wise square keeps gradient flow; avoids computing W^T W
    expm_W = torch.matrix_exp(W * W)
    return torch.trace(expm_W) - d


def compute_dag_residual(model) -> float:
    """
    Compute and return the raw h(W) value (as a Python float) for diagnostic logging.
    Iterates over all LagAwareAdjacency modules; for plain CausalStructure uses raw logits.
    """
    total_h = 0.0
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if module.__class__.__name__ == 'LagAwareAdjacency':
                # 3D lag tensor: use summary (max-over-lags) for constraint
                W = module.adj_logits  # [max_lag, d, d]
                W_summary = W.abs().max(dim=0).values  # [d, d]
                h = _h_notears(W_summary)
                total_h += h.item()
                count += 1
            elif module.__class__.__name__ == 'CausalStructure':
                W = module.adj_logits  # [d, d]
                if W.shape[0] == W.shape[1]:
                    h = _h_notears(W)
                    total_h += h.item()
                    count += 1
    return total_h


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def structural_sparsity_loss(model, lambda_sparse: float = 0.01) -> torch.Tensor:
    """
    L1 sparsity penalty on soft edge probabilities.
    Works with both LagAwareAdjacency (3D) and CausalStructure (2D).
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    for module in model.modules():
        if module.__class__.__name__ == 'LagAwareAdjacency':
            probs = torch.sigmoid(module.adj_logits)  # [max_lag, d, d]
            loss = loss + probs.sum()
            count += probs.numel()
        elif module.__class__.__name__ == 'CausalStructure':
            probs = torch.sigmoid(module.adj_logits)  # [d, d]
            loss = loss + probs.sum()
            count += probs.numel()
    if count == 0:
        return loss
    return lambda_sparse * loss


def group_lasso_loss(model, lambda_sparse: float = 0.01) -> torch.Tensor:
    """
    Group Lasso on KAN spline coefficients — encourages entire edges to vanish.
    L2 norm of each edge's coefficient block summed and scaled.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    counter = 0
    for _, module in model.named_modules():
        if hasattr(module, 'edge_functions'):
            for _, func in module.edge_functions.items():
                l2_norm = torch.norm(func.coef, p=2)
                loss = loss + l2_norm
                counter += 1
    if counter == 0:
        return loss
    return lambda_sparse * loss


def causal_consistency_loss(model, lambda_dag: float = 0.1) -> tuple:
    """
    NOTEARS DAG penalty using the correct h(W) = tr(exp(W ∘ W)) − d formulation.

    Applies the constraint over:
      - LagAwareAdjacency modules:  summary graph (max-pool over lag dim)
      - CausalStructure modules:    raw logits directly

    Returns:
        (alm_loss, h_val) where:
            alm_loss = lambda_dag * h(W)  (for backward pass)
            h_val    = float diagnostic value
    """
    dag_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    h_val = 0.0
    count = 0

    for module in model.modules():
        if module.__class__.__name__ == 'LagAwareAdjacency':
            # Use raw logits so gradients flow unrestricted
            W = module.adj_logits          # [max_lag, d, d]
            W_summary = W.abs().max(dim=0).values  # [d, d]  summary graph
            h = _h_notears(W_summary)
            dag_loss = dag_loss + h
            h_val += h.item()
            count += 1
        elif module.__class__.__name__ == 'CausalStructure':
            W = module.adj_logits          # [d, d]
            if W.shape[0] == W.shape[1]:
                h = _h_notears(W)
                dag_loss = dag_loss + h
                h_val += h.item()
                count += 1

    if count == 0:
        return dag_loss, 0.0

    return lambda_dag * dag_loss, h_val


def intervention_loss(model, x, do_idx, do_val):
    """Placeholder for future do-calculus intervention penalty."""
    return torch.tensor(0.0, device=x.device)


def granger_regularization(adjacency, x, y, threshold=0.1):
    """Placeholder for future Granger regularization."""
    return torch.tensor(0.0, device=x.device)

