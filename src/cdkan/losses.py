
import torch
import torch.nn.functional as F

def structural_sparsity_loss(model, lambda_sparse=0.01):
    """
    Penalize the sum of edge probabilities to encourage sparsity.
    Now looks for CausalStructure modules.
    """
    loss = 0.0
    count = 0
    
    # Iterate over all modules to find CausalStructure
    for module in model.modules():
        if module.__class__.__name__ == 'CausalStructure':
            # Get Probabilities
            probs = torch.sigmoid(module.adj_logits)
            loss = loss + torch.sum(probs)
            count += probs.numel()
        elif module.__class__.__name__ == 'CausalStructureMatrix':
             probs = torch.sigmoid(module.adj_logits)
             loss = loss + torch.sum(probs)
             count += probs.numel()
            
    if count == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
        
    return lambda_sparse * loss

def group_lasso_loss(model, lambda_sparse=0.01):
    """
    Apply Group Lasso (L1-norm of L2-norms) to KAN edges.
    Encourages entire edges (groups of coefficients) to go to zero.
    """
    loss = 0.0
    counter = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'edge_functions'):
             # module is likely a CDKANLayer
             for edge_id, func in module.edge_functions.items():
                 # functional coeffs: func.coef
                 # L2 norm of this edge's coefficients
                 l2_norm = torch.norm(func.coef, p=2)
                 loss = loss + l2_norm
                 counter += 1
                 
    if counter == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
        
    return lambda_sparse * loss

def causal_consistency_loss(model, lambda_dag=0.1):
    """
    Penalize cycles in the adjacency matrix (DAG constraint).
    h(A) = tr(e^A) - d = 0
    """
    dag_loss = 0.0
    count = 0
    
    for module in model.modules():
        if module.__class__.__name__ in ['CausalStructure', 'CausalStructureMatrix']:
            adj = torch.sigmoid(module.adj_logits)
            d = adj.shape[0]
            
            # Matrix Exponential Trace
            # Ensure A is square. If M x N and M != N, we can't compute trace directly for cycles.
            # Usually strict DAG is for square matrices.
            if adj.shape[0] == adj.shape[1]:
                expm_A = torch.matrix_exp(adj)
                dag_loss = dag_loss + (torch.trace(expm_A) - d)
                count += 1
            else:
                # Rectangular case (e.g. bipartite)? 
                # cycles not defined in the same way. 
                pass
                
    if count == 0:
         return torch.tensor(0.0, device=next(model.parameters()).device)

    return lambda_dag * dag_loss

def intervention_loss(model, x, do_idx, do_val):
    return torch.tensor(0.0, device=x.device)

def granger_regularization(adjacency, x, y, threshold=0.1):
    return torch.tensor(0.0, device=x.device)

