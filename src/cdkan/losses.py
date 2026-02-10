
import torch
import torch.nn.functional as F

def structural_sparsity_loss(model, lambda_sparse=0.01):
    """
    Penalize the sum of edge probabilities to encourage sparsity.
    """
    loss = 0.0
    count = 0
    
    # Iterate over all CDKANLayers
    for layer in model.modules():
        if hasattr(layer, 'edge_logits'):
            # edge_logits is ParameterDict. values() is iterator.
            logits = list(layer.edge_logits.values())
            if logits:
                logits_tensor = torch.stack(logits)
                probs = torch.sigmoid(logits_tensor)
                loss = loss + torch.sum(probs)
                count += len(probs)
            
    if count == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
        
    return lambda_sparse * loss

def causal_consistency_loss(adjacency, lag_matrix, lambda_dag=0.1):
    """
    Penalize cycles in the adjacency matrix (DAG constraint).
    For time series, causality must be forward in time (lag >= 0).
    Cycles are only problematic if they are instantaneous (lag=0).
    A cycle like A->B (lag 1) -> A (lag 1) is valid (A(t-2) causes A(t)).
    But A->B (lag 0) -> A (lag 0) is invalid.
    
    So we penalize cycles in the 'instantaneous' adjacency matrix where lag is effectively 0.
    However, CD-KAN learns continuous lags.
    
    Simplified DAGness on the binary adjacency matrix A:
    minimize tr(e^A) - d  (approx for Directed Acyclic Graph)
    But we only care about cycles if the total lag around the cycle is 0?
    Actually, generally we assume a DAG structure for causal discovery.
    """
    
    # DAG regularization (tr(e^A) - d)
    # A is [N, N].
    d = adjacency.shape[0]
    
    # matrix exponential trace
    # exps = torch.matrix_exp(adjacency)
    # dag_loss = torch.trace(exps) - d
    
    # Or polynomial approximation: tr((I + A/d)^d) - d
    # This is often more stable.
    # adj_normalized = adjacency / d
    # eye = torch.eye(d, device=adjacency.device)
    # poly = torch.matrix_power(eye + adjacency, d) 
    # This computes (I+A)^d
    # dag_loss = torch.trace(poly) - d
    
    # Using the standard NOTEARS constraint:
    # h(A) = tr(e^A) - d = 0
    
    expm_A = torch.matrix_exp(adjacency)
    dag_loss = torch.trace(expm_A) - d
    
    return lambda_dag * dag_loss

def intervention_loss(model, x, do_idx, do_val):
    """
    Experimental: Intervention loss.
    Simulate intervention do(X[do_idx] = do_val) and compare with observational data
    where X[do_idx] is naturally close to do_val.
    """
    # 1. Find samples where feature `do_idx` is naturally close to `do_val`
    # x shape: [Batch, Window, Feats]
    # We look at the LAST time step in the window for intervention? 
    # Or strict intervention on the whole window? 
    # Usually we intervene on a cause to see effect. 
    # Let's assume intervention at t-1 affecting t.
    
    # Simple approach: 
    # If we have a causal model, P(Y|do(X)) should match P(Y|X) if confounding is handled.
    # In CD-KAN, we want the learned structural equation to hold under distribution shift.
    # This is hard to enforce without interventional data.
    
    # Placeholder for now until we have true interventional data or a specific protocol.
    return torch.tensor(0.0, device=x.device)

def granger_regularization(adjacency, x, y, threshold=0.1):
    """
    Penalize edges (i <- j) if x_j does not Granger-cause y_i.
    This is computationally expensive if done every step.
    Use sparingly or on small batches.
    
    adjacency: [Out, In] (Soft edge weights)
    x: [Batch, Window, In] (Past)
    y: [Batch, Horizon, Out] (Future)
    """
    # Simplified: We don't run full Granger test here (too slow).
    # We assume 'Granger causality' simply means predictive power.
    # The CD-KAN model ALREADY optimizes predictive power (MSE).
    # So if an edge has high weight but contributes nothing, MSE gradient should kill it?
    # Not necessarily if it overfits.
    
    return torch.tensor(0.0, device=x.device)

