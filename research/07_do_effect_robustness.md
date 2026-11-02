# 07 — Interventional-effect error and robustness (extends 06)

Committed BEFORE the code runs. Everything in 06 still applies (methods SSC = SPADE-scaled, SSR = SPADE-scaled + refit k=1.5 frozen from validation, BASILW = BASIL-W; same iSCM generator unless stated).

## Part 1 — interventional (do-effect) error
Quantity users act on: how E[x_e | do(x_c = v)] changes with v. For a learned model we obtain it by propagating the intervention through the learned SEM in the learned causal order: set x_c = v, recompute every node after c in the order with the learned edge functions plus its empirical training residual (each training row keeps its own residual), average over rows. Truth: simulate the true SEM with fresh noise (n=1000 draws, same noise family, same per-node standardization constants) under do(x_c = v).
- Learned order: source-peeling on edge strength ||coef_ec||; edges against that order are ignored in propagation. Same rule for all methods.
- Grid: 15 values of v between the 5% and 95% quantiles of x_c; curves centered over the grid.
- Pairs per dataset: up to 20 true DIRECT edges and 20 true INDIRECT ancestor pairs (path length >= 2), sampled with a fixed RNG; pairs with std(true curve) < 0.05 (effect on a unit-variance node) are skipped and their share reported.
- Metric: NRMSE_do = ||tau_hat - tau|| / ||tau||, a missing effect scores about 1. Reported separately for direct and indirect pairs.
Note: for direct pairs with an otherwise-unconnected parent this is close to the edge-function NRMSE; the indirect pairs are the new information.

## Part 2 — robustness cells (d=100, n=1000 unless stated, 5 test datasets each, seeds 1000-1004)
1. noise Gaussian; 2. Laplace; 3. heteroscedastic (ANM violated); 4. n=500; 5. n=2000; 6. different mechanism family (linear 0.9x, tanh(2x), 0.6(x^2-1), sigmoid threshold, 0.9 sin(2x), from G2) with uniform noise. Plus d=500, n=1000, uniform, 2 datasets (seeds 1000-1001) if time allows.

## Hypotheses and kill criteria
- **R1 (do-effect).** At d=100 and d=200 (uniform, n=1000), BASIL-W mean NRMSE_do <= 0.70 x SSC's, for direct pairs and for indirect pairs separately. Kill: ratio >= 0.90 in either. Also reported: BASIL-W minus SSR (tie band +-0.03).
- **R2 (robustness).** In each of the 6 d=100 cells BASIL-W edge-function NRMSE <= 0.70 x SSC's (kill: any cell > 0.90) and BASIL-W <= SSR + 0.03 in at least 4 of 6 cells.
- **R3 (scale).** At d=500, BASIL-W completes (total time <= 2.5 x SSC) with NRMSE <= 0.70 x SSC's on at least one dataset. Kill: not completed or ratio >= 0.90.
- **R4 (structure).** BASIL-W AUROC >= SSC - 0.03 in every cell (kill: any gap > 0.05).
- **R5 (uncertainty).** Band coverage >= 0.80 (kill < 0.60); ECE <= 0.10. Expected from 06: coverage ~0.78, i.e. a likely marginal miss.
## Stated in advance
5 datasets per cell; heteroscedastic noise and ANM violations may degrade all methods; indirect effects depend on the learned order, so an order error can inflate NRMSE_do even when edge functions are fine; truth is MC (finite noise ~ 1/sqrt(1000)); d=500 may not complete; mechanism family 2 includes a linear mechanism (easy for every method). Each failure will be reported.
