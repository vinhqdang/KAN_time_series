# CD-KAN v2: Algorithm Description

## Overview

**CD-KAN (Causal Discovery Kolmogorov-Arnold Networks) v2** is a deep learning architecture for simultaneous time series forecasting and causal structure discovery. It combines the expressive power of Kolmogorov-Arnold Networks (KAN) with rigorous differentiable DAG learning via the Augmented Lagrangian Method (ALM).

**Key Innovation**: First method to unify KAN-based function approximation with strict acyclicity enforcement for causal discovery, achieving SOTA performance (F1=0.8971) on benchmark datasets.

---

## Architecture Components

### 1. Input Normalization: RevIN (Reversible Instance Normalization)

**Purpose**: Handle non-stationary time series by normalizing distribution shifts.

**Mechanism**:
- **Forward**: Subtract mean, divide by std (per instance, per feature)
- **Backward**: Reverse transformation after prediction

**Mathematics**:
$$x_{\text{norm}} = \frac{x - \mu(x)}{\sigma(x)}$$

**Benefit**: Allows model to focus on temporal dynamics rather than distribution shifts, critical for financial/climate data.

### 2. Causal Discovery Layer (CDKANLayer)

**Purpose**: Learn sparse causal graph with temporal lag information.

#### 2.1 CausalStructure Module

Learnable adjacency matrix parameterized by logits:

$$A_{logits} \in \mathbb{R}^{d \times d}$$

**Gumbel-Sigmoid Sampling** for differentiable discrete decisions:
```python
def gumbel_sigmoid_sample(logits, temperature, hard=False):
    # Add logistic noise
    u = torch.rand_like(logits)
    noise = log(u) - log(1-u)
    y_soft = sigmoid((logits + noise) / temperature)
    
    if hard:  # Straight-through estimator
        y_hard = (y_soft > 0.5).float()
        return (y_hard - y_soft).detach() + y_soft
    return y_soft
```

**Properties**:
- **Differentiable**: Gradients flow through sigmoid
- **Discrete (in forward)**: Binary decisions when `hard=True`
- **Temperature annealing**: Start hot (soft distributions), cool down (sharper decisions)

#### 2.2 Edge Functions

For each potential edge $(i \rightarrow j)$:

**B-Spline KAN Function**:
$$f_{ij}(x) = \sum_{k=1}^{G} c_k B_k(x)$$

Where:
- $G$: Grid size (number of basis functions)
- $B_k$: B-spline basis functions (degree 3, C² continuity)
- $c_k$: Learnable coefficients

**Advantages over MLPs**:
- Smooth, interpretable non-linearity
- Local basis functions (better generalization)
- Differentiable with respect to grid placement

#### 2.3 Lag Attention

Learn which time lags are important for each edge:

$$w_{lag} = \text{softmax}(\theta_{lag})$$
$$x_{lagged} = \sum_{l=1}^{L} w_l \cdot x_{t-l}$$

Where $L$ is max lag (typically 5-10).

#### 2.4 Temporal Modulation

Context-dependent edge strength:

$$\alpha(x) = \sigma(W_x x + b)$$
$$y_{edge} = f_{ij}(x_{lagged}) \cdot \alpha(x)$$

**Benefit**: Captures time-varying causal relationships.

#### 2.5 Forward Pass

For output dimension $i$:
$$y_i = \sum_{j=1}^{d} A_{ij} \cdot f_{ij}(x_j^{lag}) \cdot \alpha_j(x)$$

Where $A_{ij}$ is the learned adjacency mask (0 or 1).

### 3. Deep KAN Backbone

**Residual KAN Blocks**:
```python
class ResidualKANBlock(nn.Module):
    def forward(self, x):
        return x + self.kan(self.dropout(x))
```

**Structure**:
- Layer 1: Input → Hidden (e.g., 5 → 128)
- Layers 2-N: Hidden → Hidden (e.g., 128 → 128)
- Each layer: KAN function + Dropout + Residual

**Depth**: Typically 3-4 layers for balance between capacity and trainability.

### 4. Output Projection

Final KAN layer: Hidden → Output dimensions

**Total Architecture**:
$$\text{Input} \xrightarrow{\text{RevIN}} \text{CD-Layer} \xrightarrow{\text{Backbone}} \text{Output} \xrightarrow{\text{RevIN}^{-1}} \text{Prediction}$$

---

## Training Algorithm: Augmented Lagrangian Method (ALM)

### Objective Function

Minimize:
$$\mathcal{L} = \text{MSE}(y, \hat{y}) + \lambda_{sparse} \|A\|_1 + \alpha \cdot h(A) + \frac{\rho}{2} h(A)^2$$

Where:
- **MSE**: Prediction loss
- **$\lambda_{sparse}$**: Sparsity penalty (L1 on adjacency)
- **$h(A)$**: DAG constraint (acyclicity measure)
- **$\alpha$**: Dual variable (Lagrange multiplier)
- **$\rho$**: Penalty coefficient (adaptive)

### DAG Constraint

$$h(A) = \text{tr}(e^{A \circ A}) - d$$

**Interpretation**:
- If $A$ is acyclic: $h(A) = 0$
- If $A$ has cycles: $h(A) > 0$
- Matrix exponential counts all paths (including cycles)

**Why this works**:
- Directed Acyclic Graph (DAG) ⟺ $(I - A)^{-1}$ exists
- Equivalent to: $\prod_{i=1}^{d} (1 - \lambda_i(A)) \neq 0$
- Approximated by: $\text{tr}(e^A) = d$ when acyclic

### Training Schedule

**Dual-Loop Optimization**:

```
Initialize: ρ = 1.0, α = 0.0
For outer_iter = 1 to max_outer:
    # Inner loop: minimize augmented Lagrangian
    For epoch = 1 to 10:
        Forward pass: compute L
        Backward pass: update model weights
    
    # Update dual variables
    h_A = compute_dag_constraint(A)
    
    If h_A > threshold (e.g., 1e-8):
        α ← α + ρ * h_A       # Dual ascent
        ρ ← min(2*ρ, 1e20)    # Increase penalty
    Else:
        break  # DAG constraint satisfied
```

**Key Parameters**:
- `lambda_sparse`: 0.005-0.01 (lower = denser graphs, higher = sparser)
- `rho_init`: 1.0
- `rho_max`: 1e20
- `threshold`: 1e-8

### Optimization

**Separate Parameter Groups**:
1. **Adjacency logits**: Lower learning rate (1e-3)
2. **KAN weights**: Standard learning rate (3e-4)

**Reason**: Structure learning requires careful gradients to avoid local minima.

---

## Hyperparameters

### Model Architecture

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `hidden_dim` | 128 | 64-256 | Hidden layer size |
| `n_layers` | 4 | 2-6 | Residual block depth |
| `max_lag` | 5 | 3-10 | Maximum temporal lag |
| `grid_size` | 10 | 5-20 | B-spline grid resolution |
| `dropout` | 0.1 | 0.0-0.3 | Regularization |

### Training

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epochs` | 100 | 50-200 | Total training epochs |
| `lr` | 3e-4 | 1e-4 to 1e-3 | Learning rate (KAN) |
| `lr_adj` | 1e-3 | 5e-4 to 5e-3 | Learning rate (adjacency) |
| `lambda_sparse` | 0.005 | 0.001-0.05 | Sparsity penalty |
| `temperature` | 1.0 → 0.1 | - | Gumbel temperature (annealed) |

### ALM-Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho_init` | 1.0 | Initial penalty coefficient |
| `rho_max` | 1e20 | Maximum penalty (prevents overflow) |
| `alpha_init` | 0.0 | Initial dual variable |
| `dag_threshold` | 1e-8 | Convergence criterion for h(A) |

---

## Inference

### Causal Graph Extraction

1. **Get adjacency probabilities**:
   ```python
   adj_logits = model.cd_layer.causal_structure.adj_logits
   adj_probs = torch.sigmoid(adj_logits)
   ```

2. **Thresholding**:
   ```python
   threshold = 0.1  # Adjust based on desired sparsity
   adj_binary = (adj_probs > threshold).float()
   ```

3. **Lag information** (per edge):
   ```python
   lag_weights = model.cd_layer.lag_attention[edge_id].get_weights()
   dominant_lag = torch.argmax(lag_weights)
   ```

### Forecasting

Standard forward pass with learned structure frozen:
```python
model.eval()
with torch.no_grad():
    predictions = model(x_history)
```

---

## Computational Complexity

**Training**:
- **CD-Layer**: $O(d^2 \cdot n \cdot L)$ where $d$ = features, $n$ = samples, $L$ = max lag
- **Backbone**: $O(n \cdot h^2 \cdot k)$ where $h$ = hidden dim, $k$ = layers
- **DAG Constraint**: $O(d^3)$ (matrix exponential)
- **Total per epoch**: $O(d^2 \cdot n \cdot L + d^3)$

**Inference**:
- $O(d^2 \cdot L)$ per sample (independent of training size)

**Scalability**: Linear in $n$, quadratic in $d$ (acceptable for $d < 50$).

---

## Advantages over Baselines

### vs. VAR-Lasso
- **Pro**: Learns non-linear relationships (sin, exp, etc.)
- **Con**: Slower training (neural network vs. closed-form)

### vs. PCMCI
- **Pro**: End-to-end differentiable, better precision-recall balance
- **Con**: Requires more hyperparameter tuning

### vs. NOTEARS/GOLEM
- **Pro**: KAN > MLP for smooth functions, ALM > continuous relaxation
- **Con**: Similar complexity, more complex implementation

### vs. Traditional KAN
- **Pro**: Explicit causal structure, interpretable adjacency
- **Con**: Requires DAG constraint (additional computation)

---

## Limitations and Future Work

### Current Limitations
1. **Scalability**: $O(d^3)$ DAG constraint limits to ~50 variables
2. **Identifiability**: Cannot distinguish between Markov equivalent graphs without interventions
3. **Stationarity**: Assumes time-invariant structure (extension to time-varying in progress)

### Future Directions
1. **Sparse DAG Constraint**: Exploit sparsity to reduce complexity to $O(d \cdot k^2)$ where $k$ = avg degree
2. **Interventional Data**: Incorporate do-calculus for causal effect estimation
3. **Bayesian Extension**: Uncertainty quantification for edge probabilities
4. **Multi-resolution**: Temporal hierarchies (hourly → daily → weekly)
5. **Transfer Learning**: Pre-train on large datasets, fine-tune on target domain

---

## Implementation Notes

### Critical Design Choices

1. **Why RevIN**: Financial/climate data has distribution shifts; RevIN is lightweight and effective
2. **Why B-Splines**: Smooth, local basis functions generalize better than global polynomials
3. **Why ALM**: Guarantees DAG convergence; continuous relaxations can fail
4. **Why Gumbel-Sigmoid**: Enables discrete sampling with gradient flow
5. **Why Residual Connections**: Deep KAN (>3 layers) unstable without skip connections

### Debugging Tips

1. **DAG not converging**: Increase `rho` growth rate or decrease `lr_adj`
2. **Too sparse**: Lower `lambda_sparse` or increase adjacency LR
3. **Too dense**: Raise `lambda_sparse` or lower temperature faster
4. **Unstable training**: Add gradient clipping, reduce learning rate
5. **Poor forecasting**: Check RevIN normalization, increase `hidden_dim`

---

## Reproducibility

**Benchmark Results** (F1=0.8971) achieved with:
- **Hardware**: NVIDIA GPU (optional, runs on CPU)
- **Seeds**: PyTorch seed=42, NumPy seed=42
- **Config**: `hidden_dim=128, n_layers=4, epochs=100, lambda_sparse=0.005`
- **Data**: 10 datasets (500-5000 samples, 5-15 nodes)

**Full results**: See `experimental_results/benchmark_comprehensive_sota.csv`

---

## References

1. **KAN**: Liu et al., "KAN: Kolmogorov-Arnold Networks", arXiv:2404.19756, 2024
2. **NOTEARS**: Zheng et al., "DAGs with NO TEARS", NeurIPS 2018
3. **ALM**: Nocedal & Wright, "Numerical Optimization", Chapter 17, 2006
4. **RevIN**: Kim et al., "Reversible Instance Normalization", ICLR 2022
5. **Gumbel-Softmax**: Jang et al., "Categorical Reparameterization", ICLR 2017

---

**Document Version**: 2.0  
**Last Updated**: February 2025  
**Author**: Dr. Dang Quang Vinh
