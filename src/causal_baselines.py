
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class CausalBaselines:
    @staticmethod
    def var_lasso(data, max_lag=3, alpha=0.1):
        """
        Infers causal graph using VAR with L1 penalty (Lasso).
        Simple proxy for linear Granger Causality.
        """
        n_samples, n_nodes = data.shape
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        # Prepare lagged features
        X_lagged = []
        Y_target = data[max_lag:]
        
        for t in range(max_lag, n_samples):
            lag_vec = []
            for l in range(1, max_lag + 1):
                lag_vec.extend(data[t-l])
            X_lagged.append(lag_vec)
            
        X_lagged = np.array(X_lagged)
        
        # Fit Lasso for each variable
        for j in range(n_nodes):
            model = Lasso(alpha=alpha)
            model.fit(X_lagged, Y_target[:, j])
            
            # Coefficients: [Node1_Lag1, Node2_Lag1, ... Node1_Lag2, ...]
            coeffs = model.coef_
            
            # Aggregate importance over lags
            for i in range(n_nodes):
                # Indices for node i across lags
                # 0, n, 2n...
                indices = [i + l*n_nodes for l in range(max_lag)]
                importance = np.sum(np.abs(coeffs[indices]))
                
                if importance > 0.01: # Threshold
                    adj_matrix[j, i] = 1
                    
        return adj_matrix

    @staticmethod
    def correlation_threshold(data, threshold=0.5):
        """
        Infers graph based on simple correlation threshold.
        Ignores time lags, just assumes contemporaneous correlation implies connection.
        Rough baseline.
        """
        df = pd.DataFrame(data)
        corr = df.corr().abs().values
        np.fill_diagonal(corr, 0)
        return (corr > threshold).astype(int)
