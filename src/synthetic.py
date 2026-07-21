
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

def generate_svar_contemp(n_samples=2000, n_nodes=6, contemp_density=0.3,
                          lag_density=0.15, max_lag=2, seed=0):
    """
    Generate a non-linear SVAR with a KNOWN instantaneous (contemporaneous) DAG
    plus lagged edges. Used to validate CD-KAN's acyclicity constraint.

    x_{t,i} = sum_{j: j precedes i} f0_ij(x_{t,j})            # instantaneous DAG
              + sum_{h>=1} sum_j fh_ij(x_{t-h,j}) + eps        # lagged
    The contemporaneous graph B0 is acyclic by construction (random topological
    order); functions are contractive to keep the recursion stable.

    Returns: data [n_samples, d], B0 [d,d] contemporaneous adjacency (effect<-cause),
             BL [d,d] summary lagged adjacency.
    """
    rng = np.random.RandomState(seed)
    d = n_nodes
    perm = rng.permutation(d)
    B0 = np.zeros((d, d), int)
    for a in range(d):
        for b in range(a):
            if rng.rand() < contemp_density:
                B0[perm[a], perm[b]] = 1          # perm[b] -> perm[a]
    BL_lags = [(rng.rand(d, d) < lag_density).astype(int) for _ in range(max_lag)]
    for L in BL_lags:
        np.fill_diagonal(L, 0)
    fns = [lambda x: 0.7 * x, lambda x: 0.6 * np.tanh(x), lambda x: 0.5 * np.sin(2 * x)]
    f0, fl = {}, {}
    for i in range(d):
        for j in range(d):
            if B0[i, j]:
                f0[(i, j)] = fns[rng.randint(len(fns))]
            for h in range(max_lag):
                if BL_lags[h][i, j]:
                    fl[(i, j, h)] = fns[rng.randint(len(fns))]
    burn = 100
    X = np.zeros((n_samples + burn, d))
    for t in range(max_lag, n_samples + burn):
        for i in perm:                             # topological order
            v = rng.normal(0, 0.3)
            for j in range(d):
                if B0[i, j]:
                    v += f0[(i, j)](X[t, j])
                for h in range(max_lag):
                    if BL_lags[h][i, j]:
                        v += fl[(i, j, h)](X[t - 1 - h, j])
            X[t, i] = v
    return X[burn:], B0, np.clip(sum(BL_lags), 0, 1)


def generate_instantaneous_anm(n_samples=1500, n_nodes=6, density=None, seed=0):
    """
    Pure i.i.d. non-linear Additive-Noise-Model DAG (no time structure).
        x_e = sum_{c in pa(e)} f_ce(x_c) + noise,  generated in topological order.
    Returns: X [n, d], A [d, d] with A[c, e]=1 iff cause c -> effect e (a DAG).
    Mechanisms are strongly non-linear (tanh, sin, quadratic, rational) and noise
    is non-Gaussian (uniform) so the model is ANM-identifiable.
    """
    r = np.random.RandomState(seed)
    d = n_nodes
    density = density if density is not None else 2.0 / (d - 1)
    perm = r.permutation(d)
    A = np.zeros((d, d), int)
    for a in range(d):
        for b in range(a):
            if r.rand() < density:
                A[perm[b], perm[a]] = 1                     # perm[b] -> perm[a]
    fns = [lambda x: 0.8 * np.tanh(1.5 * x),
           lambda x: 0.7 * np.sin(1.5 * x),
           lambda x: 0.3 * (x ** 2 - 1.0),
           lambda x: 0.8 * x / (1.0 + np.abs(x))]
    f = {}
    for c in range(d):
        for e in range(d):
            if A[c, e]:
                f[(c, e)] = fns[r.randint(len(fns))]
    X = np.zeros((n_samples, d))
    for e in perm:                                          # topological order
        v = r.uniform(-0.7, 0.7, n_samples)                 # non-Gaussian noise
        for c in range(d):
            if A[c, e]:
                v = v + f[(c, e)](X[:, c])
        X[:, e] = v
    return X.astype(np.float32), A


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
