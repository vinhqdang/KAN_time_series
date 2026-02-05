# CD-KAN: Causal Discovery Kolmogorov-Arnold Networks for Time Series

## The Fundamental Problem

**Current time-series models make a critical assumption**: relationships between variables are **static and immediate**. In reality:

- **Causality has delays**: Temperature affects humidity 2 hours later, not immediately
- **Relationships evolve**: X→Y might be strong in summer, weak in winter  
- **Causal structure is hidden**: We don't know which variables actually cause others

**Existing KAN limitations:**
- Standard KAN: φ(x) is timeless - same function always
- TKAN/TimeKAN: Add recurrence on top, but edges still static
- DeepOKAN/KANO: Operator learning, but no causal discovery

## Core Innovation: Time-Lagged Causal Edges

Instead of learning `y(t) = Σ φᵢⱼ(xⱼ(t))`, we learn:

**y(t) = Σᵢⱼ αᵢⱼ(t) · φᵢⱼ(xⱼ(t - τᵢⱼ)) · Iᵢⱼ(t)**

Where:
- **τᵢⱼ**: Learnable time delay (continuous, 0 to T_max)
- **αᵢⱼ(t)**: Time-varying causal strength (via temporal modulation network)
- **φᵢⱼ**: KAN univariate function (standard B-spline)
- **Iᵢⱼ(t)**: Causal indicator (0/1, whether edge is active at time t)

This is **fundamentally different** from all existing approaches:
1. **Learns causal delays** - not fixed at 1 time step
2. **Discovers sparse causal graphs** - which edges actually matter
3. **Handles non-stationary causality** - relationships that change over time
4. **Interpretable** - can extract "X causes Y with 3-hour delay"

---

## Architecture Design

### 1. Lagged Attention Mechanism

Instead of standard KAN edges, we use **Differentiable Time-Lag Attention**:

```
For edge (i ← j):
  1. Sample historical window: x_j(t-T:t)
  2. Compute attention weights over lags:
     w_τ = softmax(θᵢⱼ · [1, τ, τ², ..., τᵏ])  for τ ∈ [0, T]
  3. Extract time-lagged input:
     x̃_j = Σ_τ w_τ · x_j(t - τ)
  4. Apply KAN function:
     φᵢⱼ(x̃_j)
```

**Why this is novel:** Soft attention over continuous time lags enables gradient-based learning of causal delays. Previous work uses fixed discrete lags.

### 2. Temporal Causal Modulation

Each edge has a **time-varying strength** modeled by a lightweight network:

```python
class TemporalModulator(nn.Module):
    def __init__(self, context_dim=32):
        self.context_rnn = nn.GRU(input_size=1, hidden_size=context_dim)
        self.strength_mlp = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, time_index, history):
        # Extract temporal context
        context, _ = self.context_rnn(history)
        # Predict edge strength at current time
        strength = self.strength_mlp(context[time_index])
        return strength
```

**Why this is novel:** Existing models assume fixed relationship strength. This learns `P(edge is active | time, history)`.

### 3. Sparse Causal Graph Learner

Use **Gumbel-Softmax** for differentiable discrete causal graph structure learning:

```
For each potential edge (i ← j):
  1. Learn edge probability: pᵢⱼ = sigmoid(θᵢⱼ)
  2. Sample binary indicator via Gumbel-Softmax:
     Iᵢⱼ ~ Gumbel-Softmax(pᵢⱼ, temperature=T)
  3. During forward pass: use hard (0/1) assignments
  4. During backward pass: use soft gradients
  5. Apply sparsity penalty: L_sparse = λ · Σ pᵢⱼ
```

**Why this is novel:** First to combine neural structure learning with KAN. Discovers minimal causal DAG automatically.

---

## Complete CD-KAN Layer

```python
class CDKANLayer(nn.Module):
    """Causal Discovery KAN Layer"""
    
    def __init__(self, in_features, out_features, max_lag=10, 
                 spline_grid=5, learn_structure=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_lag = max_lag
        
        # For each edge (i ← j)
        self.edge_functions = nn.ModuleDict()
        self.lag_params = nn.ParameterDict()
        self.modulators = nn.ModuleDict()
        self.edge_probs = nn.ParameterDict()
        
        for i in range(out_features):
            for j in range(in_features):
                edge_id = f"{i}_{j}"
                
                # KAN univariate function
                self.edge_functions[edge_id] = BSplineFunction(
                    grid_size=spline_grid
                )
                
                # Learnable lag attention weights (polynomial coefficients)
                self.lag_params[edge_id] = nn.Parameter(
                    torch.randn(4)  # Polynomial order 3
                )
                
                # Temporal modulator
                self.modulators[edge_id] = TemporalModulator()
                
                # Edge existence probability (for structure learning)
                if learn_structure:
                    self.edge_probs[edge_id] = nn.Parameter(
                        torch.randn(1)  # Will be sigmoid'd
                    )
        
        # Gumbel temperature (annealed during training)
        self.register_buffer('temperature', torch.tensor(1.0))
        
    def compute_lag_weights(self, edge_id):
        """Compute attention weights over time lags"""
        # Polynomial attention: w(τ) = softmax(θ₀ + θ₁τ + θ₂τ² + θ₃τ³)
        lags = torch.arange(self.max_lag + 1, dtype=torch.float32)
        powers = torch.stack([lags**k for k in range(4)])
        
        logits = torch.matmul(self.lag_params[edge_id], powers)
        weights = F.softmax(logits, dim=0)
        return weights
    
    def forward(self, x_history, time_indices):
        """
        Args:
            x_history: (batch, time_window, in_features)
            time_indices: (batch,) - which time step to predict
        
        Returns:
            output: (batch, out_features)
        """
        batch_size = x_history.shape[0]
        T = x_history.shape[1]
        
        output = torch.zeros(batch_size, self.out_features)
        
        for i in range(self.out_features):
            edge_contributions = []
            
            for j in range(self.in_features):
                edge_id = f"{i}_{j}"
                
                # 1. Compute lag attention weights
                lag_weights = self.compute_lag_weights(edge_id)
                
                # 2. Extract time-lagged input via attention
                x_lagged = torch.zeros(batch_size)
                for b in range(batch_size):
                    t_current = time_indices[b]
                    
                    # Weighted sum over historical lags
                    for tau in range(min(t_current + 1, self.max_lag + 1)):
                        if t_current - tau >= 0:
                            x_lagged[b] += (
                                lag_weights[tau] * 
                                x_history[b, t_current - tau, j]
                            )
                
                # 3. Apply KAN function
                edge_output = self.edge_functions[edge_id](x_lagged)
                
                # 4. Temporal modulation (time-varying strength)
                modulation = self.modulators[edge_id](
                    time_indices, 
                    x_history[:, :, j]
                )
                
                # 5. Causal graph structure (Gumbel-Softmax)
                if hasattr(self, 'edge_probs'):
                    edge_prob = torch.sigmoid(self.edge_probs[edge_id])
                    
                    # Gumbel-Softmax trick for discrete sampling
                    logits = torch.stack([
                        torch.log(edge_prob + 1e-8),
                        torch.log(1 - edge_prob + 1e-8)
                    ])
                    
                    gumbel_noise = -torch.log(-torch.log(
                        torch.rand_like(logits) + 1e-8
                    ))
                    
                    indicator = F.softmax(
                        (logits + gumbel_noise) / self.temperature, 
                        dim=0
                    )[0]  # P(edge exists)
                    
                else:
                    indicator = 1.0
                
                # Combine: edge_output * modulation * indicator
                edge_contributions.append(
                    edge_output * modulation * indicator
                )
            
            # Sum contributions from all input features
            output[:, i] = sum(edge_contributions)
        
        return output
    
    def get_causal_graph(self, threshold=0.5):
        """Extract learned causal structure"""
        adjacency = np.zeros((self.out_features, self.in_features))
        lag_matrix = np.zeros((self.out_features, self.in_features))
        
        for i in range(self.out_features):
            for j in range(self.in_features):
                edge_id = f"{i}_{j}"
                
                # Edge existence probability
                if hasattr(self, 'edge_probs'):
                    prob = torch.sigmoid(self.edge_probs[edge_id]).item()
                    if prob > threshold:
                        adjacency[i, j] = prob
                        
                        # Expected lag
                        lag_weights = self.compute_lag_weights(edge_id)
                        lags = torch.arange(self.max_lag + 1)
                        expected_lag = (lag_weights * lags).sum().item()
                        lag_matrix[i, j] = expected_lag
        
        return adjacency, lag_matrix
```

---

## Training Strategy

### Phase 1: Warm-up (Epochs 1-20)
- Temperature = 5.0 (soft Gumbel-Softmax, continuous relaxation)
- All edges active (learn basic relationships)
- Loss = MSE only

### Phase 2: Structure Discovery (Epochs 21-60)
- Anneal temperature: 5.0 → 0.5 (sharpen to discrete)
- Add sparsity penalty: L = MSE + λ_sparse · Σ p_ij
- λ_sparse starts at 0.001, increases to 0.01
- Prune edges with p_ij < 0.1 every 10 epochs

### Phase 3: Fine-tuning (Epochs 61-100)
- Temperature = 0.1 (nearly discrete)
- Fix graph structure (top-K edges only)
- Optimize functions φ and modulators α
- Add causal consistency loss (see below)

---

## Novel Loss Functions

### 1. Causal Consistency Loss

Enforce that discovered causality respects temporal ordering:

```python
def causal_consistency_loss(adjacency, lag_matrix):
    """Penalize cycles and backward-in-time causality"""
    loss = 0
    
    # Penalty for cycles (should be DAG)
    for path_length in range(2, n_vars + 1):
        A_power = torch.linalg.matrix_power(adjacency, path_length)
        loss += torch.trace(A_power)  # Cycles return to diagonal
    
    # Penalty for negative lags (causes must precede effects)
    loss += torch.sum(F.relu(-lag_matrix) ** 2)
    
    return loss
```

### 2. Temporal Intervention Loss

Test causal hypotheses via simulated interventions:

```python
def intervention_loss(model, x, do_variable, do_value):
    """
    Predict counterfactual: what if we set x[do_variable] = do_value?
    
    Compare to actual data where x[do_variable] ≈ do_value naturally.
    """
    # Find samples where variable was naturally at do_value
    mask = torch.abs(x[:, :, do_variable] - do_value) < epsilon
    actual_outcomes = x[mask]
    
    # Predict under intervention
    x_intervened = x.clone()
    x_intervened[:, :, do_variable] = do_value
    predicted = model(x_intervened)
    
    # Should match actual outcomes
    return F.mse_loss(predicted[mask], actual_outcomes)
```

### 3. Granger Causality Regularization

Ensure learned structure aligns with statistical causality:

```python
def granger_regularization(adjacency, x, y):
    """
    For edge i←j with weight w_ij:
    If w_ij > 0, then x_j should Granger-cause y_i
    """
    loss = 0
    
    for i, j where adjacency[i,j] > threshold:
        # Fit AR model: y_i(t) ~ y_i(t-1:t-p)
        ar_error = fit_AR(y[:, i], lags=5)
        
        # Fit VAR model: y_i(t) ~ y_i(t-1:t-p) + x_j(t-1:t-p)
        var_error = fit_VAR(y[:, i], x[:, j], lags=5)
        
        # If edge exists, VAR should be better
        granger_score = ar_error - var_error
        loss += F.relu(-granger_score)  # Penalize if VAR not better
    
    return loss
```

---

## Why This Is Genuinely Novel

| Aspect | Existing Methods | CD-KAN |
|--------|-----------------|--------|
| **Time delays** | Fixed (1 step) or pre-specified grid | Continuous, learned via attention |
| **Causal structure** | Assumed or post-hoc analysis | Jointly learned with prediction |
| **Temporal variation** | Static relationships | Time-varying edge strengths α(t) |
| **Interpretability** | Black-box or post-hoc | Extracts causal DAG + lag estimates |
| **Integration** | Separate causal discovery + forecasting | End-to-end joint learning |

### Comparison to Related Work

**Neural Granger Causality (NGC):** Uses MLPs to test causality, doesn't discover structure  
**TCDF (Temporal Causal Discovery Framework):** Learns attention over lags but uses CNNs, not interpretable  
**DYNOTEARS:** Learns DAG structure but assumes linear relationships, no lag learning  
**CD-KAN:** Combines all three: DAG structure + nonlinear functions + continuous lags + interpretability

---

## Expected Performance

### Benchmark 1: Synthetic Lorenz System

**Setup:** 3 variables with known causal structure:
- dx/dt = σ(y - x)  
- dy/dt = x(ρ - z) - y
- dz/dt = xy - βz

**Expected Results:**
- **Graph recovery:** F1 > 0.95 for true edges
- **Lag estimation:** Mean error < 0.5 time steps
- **Forecasting MSE:** 30% better than TimeKAN (exploits true causal structure)

### Benchmark 2: Real-World Climate Data

**Setup:** 21 weather variables, hourly data, known physics:
- Temperature → Humidity (2-3 hour lag)
- Pressure → Wind speed (1-2 hour lag)
- Solar radiation → Temperature (4-6 hour lag)

**Expected Results:**
- **Discovered structure:** Matches domain knowledge (>80% precision)
- **Novel discoveries:** May find non-obvious causal links
- **Forecasting:** 10-15% MSE improvement over TimeKAN on long horizons (H=96)

### Benchmark 3: Financial Markets

**Setup:** Multi-asset time series, regime changes
- Stock indices, commodities, currencies
- Causality changes during crises vs normal times

**Expected Results:**
- **Dynamic causality detection:** Identifies when correlations → causations
- **Regime-specific graphs:** Different structures for bull/bear markets
- **Trading strategy:** Sharpe 1.9-2.1 (vs LSTM 1.85) by exploiting lead-lag relationships

---

## Interpretability: What You Can Extract

### 1. Causal Graph Visualization

```python
adjacency, lags = model.get_causal_graph(threshold=0.6)

# Plot directed graph with edge weights = causal strength
# Edge labels = time lags (e.g., "Temperature → Humidity: 2.3 hours")
```

### 2. Time-Varying Causal Strength

```python
# For edge i ← j, plot α_ij(t) over time
strengths = []
for t in range(T):
    strength = model.modulators[f"{i}_{j}"](t, x_history).item()
    strengths.append(strength)

plt.plot(strengths, label=f"Variable {j} → Variable {i}")
# Reveals when causality is active (e.g., only during business hours)
```

### 3. Counterfactual Predictions

```python
# "What if we had intervened on x_3 at time t=50?"
x_counterfactual = x.clone()
x_counterfactual[:, 50, 3] = intervention_value

y_factual = model(x)
y_counterfactual = model(x_counterfactual)

effect = (y_counterfactual - y_factual).mean(dim=0)
# Shows which downstream variables are affected
```

---

## Implementation Roadmap

**Week 1-2:** Core Components
- [ ] Differentiable lag attention mechanism
- [ ] Temporal modulator network
- [ ] Gumbel-Softmax structure learner
- [ ] Unit tests on toy data (2-variable system)

**Week 3-4:** Training & Losses
- [ ] Implement 3-phase training schedule
- [ ] Causal consistency loss
- [ ] Intervention loss framework
- [ ] Temperature annealing scheduler

**Week 5-6:** Benchmarks
- [ ] Lorenz system experiments (known ground truth)
- [ ] Compare vs NGC, TCDF, DYNOTEARS
- [ ] Graph recovery metrics (precision, recall, SHD)

**Week 7-8:** Real Data
- [ ] Weather/climate benchmarks
- [ ] Financial data with regime changes
- [ ] Ablation studies (remove each component)

**Week 9-10:** Analysis & Publication
- [ ] Interpretability case studies
- [ ] Theoretical analysis (when does it work?)
- [ ] Write paper for ICML/NeurIPS

---

## Potential Limitations & Mitigations

| Limitation | Mitigation |
|-----------|-----------|
| **Computational cost:** O(max_lag × edges) per forward pass | Use sparse graphs (prune aggressively); cache lag attention weights |
| **Identifiability:** Causal discovery fundamentally hard | Add domain constraints; use intervention data when available |
| **Long lags:** Hard to learn lag=100 with gradient descent | Hierarchical lag search (coarse→fine); add lag prior |
| **Non-stationarity:** Modulator may overfit | Regularize α(t) to be smooth; add trend decomposition |

---

## Conclusion: Why This Is Truly Novel

CD-KAN solves a problem **no existing method addresses**: jointly learning causal structure, temporal lags, and nonlinear dynamics in a unified interpretable framework.

**Key innovations:**
1. **Continuous lag learning** via differentiable attention (not discrete lag grid)
2. **Joint structure + function learning** (not separate stages)
3. **Time-varying causality** via temporal modulators (not static graphs)
4. **End-to-end interpretability** (extract DAG, lags, and symbolic functions)

**Expected impact:**
- Scientific discovery: Find true causal mechanisms in climate, biology, neuroscience
- Superior forecasting: Exploit causal structure for better long-horizon predictions
- Transparent AI: Users understand *why* the model makes predictions

This is not an incremental improvement - it's a fundamentally different paradigm for time-series modeling based on causal principles.