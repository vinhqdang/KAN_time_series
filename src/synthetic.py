
import numpy as np
import torch
import pandas as pd
import networkx as nx

def generate_nonlinear_scm(n_samples=1000, n_nodes=5, density=0.2, max_lag=3, seed=42):
    """
    Generates synthetic time series data based on a non-linear Structural Causal Model (SCM).
    
    X_t^j = sum_{i in PA(j)} f_{ij}(X_{t-tau}^i) + eta_t
    
    where f_{ij} are random non-linear functions (Linear, Sine, Tanh).
    
    Returns:
        data (np.ndarray): [n_samples, n_nodes]
        adj_matrix (np.ndarray): [n_nodes, n_nodes] Binary adjacency (Summary graph)
        lags_matrix (np.ndarray): [n_nodes, n_nodes] Lag values (0 if no edge)
    """
    np.random.seed(seed)
    
    # 1. Generate DAG structure (Summary Graph)
    # We want a DAG for the summary graph to ensure acyclicity in contemporaneous connections if any.
    # But here we focus on lagged connections, so cycles are allowed in summary graph if lags > 0.
    # Let's generate a random adjacency matrix.
    
    adj_matrix = (np.random.rand(n_nodes, n_nodes) < density).astype(int)
    np.fill_diagonal(adj_matrix, 0) # No self-loops for simplicity
    
    # 2. Assign Lags and Functions
    lags_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    functions = {} # (j, i) -> function
    
    nonlinearities = [
        lambda x: 0.8 * x,              # Linear
        lambda x: np.sin(x),            # Sine
        lambda x: np.tanh(x),           # Tanh
        lambda x: 0.5 * x**2 * np.sign(x) # Signed Square
    ]
    
    for j in range(n_nodes): # Effect
        for i in range(n_nodes): # Cause
            if adj_matrix[j, i] == 1:
                lag = np.random.randint(1, max_lag + 1)
                lags_matrix[j, i] = lag
                func_idx = np.random.randint(0, len(nonlinearities))
                functions[(j, i)] = nonlinearities[func_idx]
                
    # 3. Simulate Data
    # Burn-in period
    burn_in = 200
    total_samples = n_samples + burn_in
    data = np.zeros((total_samples, n_nodes))
    
    # Noise
    noise = np.random.normal(0, 0.1, size=(total_samples, n_nodes))
    
    for t in range(max_lag, total_samples):
        for j in range(n_nodes):
            val = noise[t, j]
            # Add parents influence
            for i in range(n_nodes):
                if adj_matrix[j, i] == 1:
                    lag = lags_matrix[j, i]
                    func = functions[(j, i)]
                    val += func(data[t - lag, i])
            data[t, j] = val
            
    # Remove burn-in
    data = data[burn_in:]
    
    return data, adj_matrix, lags_matrix

def visualize_ground_truth(adj, labels=None, filename="ground_truth_graph.png"):
    import matplotlib.pyplot as plt
    
    G = nx.DiGraph(adj)
    if labels:
        G = nx.relabel_nodes(G, {i: l for i, l in enumerate(labels)})
        
    plt.figure(figsize=(6, 6))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_weight='bold')
    plt.title("Ground Truth Causal Graph")
    plt.savefig(filename)
    plt.close()
