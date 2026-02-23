import torch
import torch.nn as nn
import time
import itertools
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .losses import (
    structural_sparsity_loss,
    causal_consistency_loss,
    compute_dag_residual,
)


# ---------------------------------------------------------------------------
# ALM / Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class CDKANTrainerConfig:
    """
    Centralised hyperparameter config for the CD-KAN trainer.

    ALM schedule
    ------------
    rho_init      : initial penalty coefficient rho
    rho_max       : maximum rho (clip to avoid numerical blow-up)
    alpha_init    : initial dual variable alpha
    h_tol         : convergence threshold on h(W) — training stops if h < h_tol
    update_freq   : epochs between ALM dual variable updates

    Regularisation
    --------------
    lambda_sparse : weight of L1 edge-probability sparsity penalty
    lambda_dag    : weight of NOTEARS h(W) penalty (multiplied by ALM terms)

    Temperature / Gumbel-Sigmoid annealing
    ---------------------------------------
    tau_init      : starting temperature for Gumbel-Sigmoid
    tau_final     : final (minimum) temperature
    tau_decay     : multiplicative decay applied every epoch

    Optimiser
    ---------
    lr            : learning rate for Adam

    Reproducibility
    ---------------
    seed          : random seed (set to None to skip seeding)
    """
    rho_init:      float = 1.0
    rho_max:       float = 1e16
    alpha_init:    float = 0.0
    h_tol:         float = 1e-8
    update_freq:   int   = 10

    lambda_sparse: float = 0.01
    lambda_dag:    float = 1.0

    tau_init:      float = 1.0
    tau_final:     float = 0.1
    tau_decay:     float = 0.99

    lr:            float = 1e-3
    seed:          Optional[int] = None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CDKANTrainer:
    """
    Trains CDKANForecaster (or any compatible model) with:
      - Augmented Lagrangian Method (ALM) for the NOTEARS DAG constraint
      - Temperature annealing for Gumbel-Sigmoid edge sampling
      - Per-epoch diagnostic logging (h(W) residual, sparse loss, DAG loss)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 config: Optional[CDKANTrainerConfig] = None):
        self.model  = model
        self.device = device
        self.cfg    = config or CDKANTrainerConfig()

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        self.criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Public training entry-point
    # ------------------------------------------------------------------

    def train(self, train_loader, test_loader, epochs: int = 100,
              patience: int = 10) -> Dict[str, Any]:
        """
        Train the model with ALM-controlled DAG enforcement.

        Returns
        -------
        history : dict with keys
            'train_loss', 'test_loss', 'h_residuals', 'sparse_losses',
            'dag_losses', 'rho_history', 'alpha_history', 'tau_history'
        """
        cfg = self.cfg
        rho   = cfg.rho_init
        alpha = cfg.alpha_init
        tau   = cfg.tau_init

        history: Dict[str, Any] = {
            'train_loss':   [],
            'test_loss':    [],
            'h_residuals':  [],
            'sparse_losses':[],
            'dag_losses':   [],
            'rho_history':  [],
            'alpha_history':[],
            'tau_history':  [],
        }

        best_loss        = float('inf')
        patience_counter = 0

        print(f"Starting ALM Training ({epochs} max epochs, device={self.device})")
        print(f"  tau: {tau:.3f}->{cfg.tau_final:.3f}  lam_sparse={cfg.lambda_sparse}  "
              f"lam_dag={cfg.lambda_dag}  rho_init={rho}  alpha_init={alpha}")

        for epoch in range(epochs):
            t0 = time.time()
            self.model.train()

            # Set current temperature
            self._set_temperature(tau)

            # Training step
            epoch_mse, epoch_sparse, epoch_dag, epoch_h = \
                self._run_epoch(train_loader, rho, alpha)

            # Validation
            val_mse = self.evaluate(test_loader)

            # Record history
            history['train_loss'].append(epoch_mse)
            history['test_loss'].append(val_mse)
            history['h_residuals'].append(epoch_h)
            history['sparse_losses'].append(epoch_sparse)
            history['dag_losses'].append(epoch_dag)
            history['rho_history'].append(rho)
            history['alpha_history'].append(alpha)
            history['tau_history'].append(tau)

            elapsed = time.time() - t0

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:04d}/{epochs} | tau={tau:.4f} | rho={rho:.2e} | "
                    f"alpha={alpha:.4f} | MSE={epoch_mse:.5f} | Val={val_mse:.5f} | "
                    f"h(W)={epoch_h:.4e} | Sparse={epoch_sparse:.5f} | "
                    f"DAG={epoch_dag:.5f} | t={elapsed:.1f}s"
                )

            # ALM dual update
            if (epoch + 1) % cfg.update_freq == 0:
                h_val = compute_dag_residual(self.model)
                if h_val > cfg.h_tol:
                    rho   = min(rho * 2.0, cfg.rho_max)
                    alpha = alpha + rho * h_val
                    print(f"  [ALM] Epoch {epoch+1}: h(W)={h_val:.4e} -> "
                          f"rho={rho:.2e}, alpha={alpha:.4f}")

            # Temperature annealing
            tau = max(cfg.tau_final, tau * cfg.tau_decay)

            # Early stopping (only after penalty is large enough)
            if val_mse < best_loss:
                best_loss        = val_mse
                patience_counter = 0
            else:
                patience_counter += 1

            converged = (
                patience_counter >= patience
                and rho > 100
                and epoch_h < max(cfg.h_tol * 100, 1e-4)
            )
            if converged:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(patience={patience}, h(W)={epoch_h:.4e})")
                break

        return history

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    @classmethod
    def sensitivity_run(cls, model_fn, train_loader, test_loader,
                        param_grid: Dict[str, list], epochs: int = 50,
                        device: str = 'cpu') -> list:
        """
        Sweep over a parameter grid and return a list of result dicts.

        Args:
            model_fn    : callable that returns a fresh model (no args)
            param_grid  : dict of {param_name: [values]}, e.g.
                          {'tau_init': [0.5, 1.0], 'lambda_sparse': [0.01, 0.1]}
            epochs      : epochs per run
            device      : compute device

        Returns:
            list of dicts, each containing param values, final val_mse, and h(W)
        """
        keys   = list(param_grid.keys())
        values = list(param_grid.values())
        results = []

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            cfg    = CDKANTrainerConfig(**{k: v for k, v in params.items()
                                          if hasattr(CDKANTrainerConfig, k) or
                                          k in CDKANTrainerConfig.__dataclass_fields__})
            model   = model_fn().to(device)
            trainer = cls(model, device=device, config=cfg)
            hist    = trainer.train(train_loader, test_loader,
                                    epochs=epochs, patience=epochs)  # no early stop

            row = {**params,
                   'final_val_mse': hist['test_loss'][-1],
                   'final_h':       hist['h_residuals'][-1]}
            results.append(row)
            print(f"  sensitivity: {params} => val_mse={row['final_val_mse']:.5f}, "
                  f"h(W)={row['final_h']:.4e}")

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(self, loader, rho: float, alpha: float):
        """Run one full training epoch; return (mse, sparse, dag, h)."""
        epoch_mse = epoch_sparse = epoch_dag = epoch_h = 0.0
        count = 0

        if isinstance(loader, dict):
            _, log = self._step(loader['X'], loader['y'], rho, alpha)
            return log['mse'], log['sparse'], log['dag'], log['h']

        for batch_X, batch_y in loader:
            _, log = self._step(batch_X, batch_y, rho, alpha)
            epoch_mse    += log['mse']
            epoch_sparse += log['sparse']
            epoch_dag    += log['dag']
            epoch_h      += log['h']
            count        += 1

        n = max(1, count)
        return epoch_mse / n, epoch_sparse / n, epoch_dag / n, epoch_h / n

    def _step(self, X: torch.Tensor, y: torch.Tensor,
              rho: float, alpha: float):
        self.optimizer.zero_grad()

        pred     = self.model(X)
        mse_loss = self.criterion(pred, y)

        # Sparsity
        sparse_loss = structural_sparsity_loss(self.model, self.cfg.lambda_sparse)

        # NOTEARS DAG constraint via ALM
        # causal_consistency_loss now returns (tensor, float_h)
        h_loss_tensor, h_val = causal_consistency_loss(self.model, lambda_dag=1.0)

        # ALM: alpha * h(W) + rho/2 * h(W)^2
        alm_loss = alpha * h_loss_tensor + (rho / 2.0) * h_loss_tensor * h_loss_tensor

        total_loss = mse_loss + sparse_loss + alm_loss
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), {
            'mse':    mse_loss.item(),
            'sparse': sparse_loss.item(),
            'dag':    alm_loss.item(),
            'h':      h_val,
        }

    def evaluate(self, loader) -> float:
        self.model.eval()
        total_mse = 0.0
        count     = 0
        with torch.no_grad():
            if isinstance(loader, dict):
                pred = self.model(loader['X'])
                return self.criterion(pred, loader['y']).item()
            for X, y in loader:
                pred = self.model(X)
                total_mse += self.criterion(pred, y).item()
                count     += 1
        return total_mse / max(1, count)

    def _set_temperature(self, tau: float):
        """Propagate current temperature to all CDKANLayer buffers."""
        for module in self.model.modules():
            if hasattr(module, 'temperature'):
                module.temperature.data.fill_(tau)
