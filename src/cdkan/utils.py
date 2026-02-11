import torch
import torch.nn as nn
import torch.nn.functional as F

class LagAttention(nn.Module):
    def __init__(self, max_lag=10, order=3):
        super().__init__()
        self.max_lag = max_lag
        # Polynomial coefficients for attention weights over lags
        # w(tau) = softmax(poly(tau))
        self.coefs = nn.Parameter(torch.randn(order + 1))
        
        # Buffer for lag indices [0, 1, ..., max_lag]
        self.register_buffer('lags', torch.arange(max_lag + 1, dtype=torch.float32))
        self.register_buffer('powers', torch.stack([self.lags ** k for k in range(order + 1)]))

    def get_weights(self):
        # logits = coefs . powers
        logits = torch.matmul(self.coefs, self.powers)
        return F.softmax(logits, dim=0)

    def get_expected_lag(self):
        w = self.get_weights()
        return (w * self.lags).sum()

class TemporalModulator(nn.Module):
    def __init__(self, input_dim=1, context_dim=32):
        super().__init__()
        self.context_rnn = nn.GRU(input_size=input_dim, hidden_size=context_dim, batch_first=True)
        self.strength_mlp = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, history):
        # history: [batch, seq_len, input_dim]
        # We only need the context at the *end* of the history window to predict current edge strength
        # Or, if we modulate over time, we return sequence. 
        # For KAN edge strength at time T, we use hidden state at T.
        
        out, h_n = self.context_rnn(history)
        # h_n: [1, batch, context_dim]
        context = h_n.squeeze(0) 
        
        strength = self.strength_mlp(context)
        return strength

def gumbel_softmax_sample(logits, temperature):
    y = logits
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(y) + 1e-8) + 1e-8)
    y = y + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def gumbel_sigmoid_sample(logits, temperature, hard=False):
    # For binary case: P(1) = sigmoid(logits)
    # Gumbel-Softmax on [logits, 0] is equivalent to Gumbel-Sigmoid
    # Or simple logistic noise:
    # L = log(u) - log(1-u), y = sigmoid((logits + L) / temp)
    u = torch.rand_like(logits)
    logistic_noise = torch.log(u + 1e-8) - torch.log(1 - u + 1e-8)
    y = (logits + logistic_noise) / temperature
    y_soft = torch.sigmoid(y)
    
    if hard:
        # Straight-through estimator
        y_hard = (y_soft > 0.5).float()
        y = (y_hard - y_soft).detach() + y_soft
        return y
    
    return y_soft
