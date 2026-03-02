"""
extract_real_causal.py — Real-data causal discovery from CD-KAN on financial data.

Trains CD-KAN on the 2020-2025 financial dataset and extracts:
  1. Summary causal graph   — which asset historically drives which
  2. Lag attribution        — dominant lag (in days) per causal edge
  3. Functional forms       — learned spline shape per edge (linear/nonlinear)
  4. Stability validation   — Jaccard consistency across 3 rolling windows
  5. Network visualization  — directed graph with edge-strength as width

Output files (saved to experimental_results/):
  real_causal_adjacency.png      — heatmap of edge probabilities
  real_causal_network.png        — directed network graph
  real_causal_lag_heatmap.png    — expected lag (in days) per edge
  real_causal_functions.png      — learned f(x) curves for top-K edges
  real_causal_stability.png      — Jaccard stability across rolling windows
  real_causal_adjacency.csv      — raw edge probabilities table
  real_causal_expected_lags.csv  — expected lag table

Usage:
    python scripts/extract_real_causal.py
    python scripts/extract_real_causal.py --epochs 200 --fast
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cdkan.model import CDKANForecaster
from src.cdkan.trainer import CDKANTrainer, CDKANTrainerConfig
from src.evaluation import evaluate_causal_real, binarise_adjacency

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_DIR   = 'experimental_results'
DATA_PATH = 'data/financial_2020_2025.csv'
WINDOW    = 16
MAX_LAG   = 5   # 5 business-day lags

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_financial_data():
    """Load and clean the financial CSV, returning (data_np, asset_names, dates)."""
    df = pd.read_csv(DATA_PATH, index_col=0, header=[0, 1])
    # The CSV has two header rows (Price / Ticker); flatten to just asset columns
    df = pd.read_csv(DATA_PATH, index_col=0, skiprows=[1, 2])
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()].sort_index()
    df = df.apply(pd.to_numeric, errors='coerce').ffill().bfill().dropna()
    asset_names = df.columns.tolist()
    return df.values.astype(np.float32), asset_names, df


def normalise(data: np.ndarray):
    mean = data.mean(0, keepdims=True)
    std  = data.std(0, keepdims=True) + 1e-8
    return (data - mean) / std, mean, std


def make_windows(data: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------------------

def train_fold(X_tr, y_tr, X_te, y_te, n_assets, epochs, seed=42):
    dev = torch.device('cpu')
    model = CDKANForecaster(
        in_features=n_assets,
        hidden_dim=32,
        out_features=n_assets,
        max_lag=MAX_LAG,
        n_layers=2,
        grid_size=8,
        learn_structure=True,
    )
    cfg = CDKANTrainerConfig(
        tau_init=1.0, tau_final=0.1, tau_decay=0.98,
        lambda_sparse=0.02, rho_init=1.0, rho_max=1e12,
        h_tol=1e-6, update_freq=10, lr=5e-4, seed=seed,
    )
    trainer = CDKANTrainer(model, device='cpu', config=cfg)
    train_loader = {
        'X': torch.from_numpy(X_tr).to(dev),
        'y': torch.from_numpy(y_tr).to(dev),
    }
    test_loader = {
        'X': torch.from_numpy(X_te).to(dev),
        'y': torch.from_numpy(y_te).to(dev),
    }
    hist = trainer.train(train_loader, test_loader, epochs=epochs, patience=30)
    model.eval()
    return model, hist


# ---------------------------------------------------------------------------
# Spline function visualisation
# ---------------------------------------------------------------------------

def get_spline_curve(edge_fn, n_pts=200):
    """Evaluate a BSplineFunction over [-1, 1] and return (x, y) numpy arrays."""
    x_vals = torch.linspace(-1, 1, n_pts)
    with torch.no_grad():
        y_vals = edge_fn(x_vals).cpu().numpy()
    return x_vals.numpy(), y_vals


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_adjacency_heatmap(adj: np.ndarray, asset_names, filepath):
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.eye(len(asset_names), dtype=bool)
    sns.heatmap(
        adj, mask=mask,
        xticklabels=asset_names, yticklabels=asset_names,
        annot=True, fmt='.2f', cmap='YlOrRd',
        vmin=0, vmax=1, linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Edge Probability'}, ax=ax,
    )
    ax.set_title(
        'CD-KAN Learned Causal Graph\n(Financial Assets 2020-2025)\n'
        'Row = Effect, Column = Cause', fontweight='bold', fontsize=13,
    )
    ax.set_xlabel('Cause', fontweight='bold')
    ax.set_ylabel('Effect', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filepath}')


def plot_lag_heatmap(expected_lags: np.ndarray, adj: np.ndarray,
                     asset_names, threshold, filepath):
    """Show expected lag (days) only for edges above the probability threshold."""
    lags_masked = expected_lags.copy()
    lags_masked[adj < threshold] = np.nan

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.eye(len(asset_names), dtype=bool) | (adj < threshold)
    im = ax.imshow(lags_masked, cmap='Blues', vmin=1, vmax=MAX_LAG, aspect='auto')
    ax.set_xticks(range(len(asset_names)))
    ax.set_yticks(range(len(asset_names)))
    ax.set_xticklabels(asset_names, rotation=45, ha='right')
    ax.set_yticklabels(asset_names)
    for i in range(len(asset_names)):
        for j in range(len(asset_names)):
            if not np.isnan(lags_masked[i, j]):
                ax.text(j, i, f'{lags_masked[i, j]:.1f}d',
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        color='white' if lags_masked[i, j] > 3 else 'black')
    plt.colorbar(im, ax=ax, label='Expected Lag (trading days)')
    ax.set_title(
        'Dominant Lag per Causal Edge\n'
        f'(only edges with prob >= {threshold})',
        fontweight='bold', fontsize=13,
    )
    ax.set_xlabel('Cause', fontweight='bold')
    ax.set_ylabel('Effect', fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filepath}')


def plot_network_graph(adj: np.ndarray, asset_names, threshold, filepath):
    """Draw a directed causal network; edge width and colour = probability."""
    try:
        import networkx as nx
    except ImportError:
        print('  networkx not available, skipping network graph')
        return

    G = nx.DiGraph()
    G.add_nodes_from(asset_names)

    edges, weights = [], []
    for i, eff in enumerate(asset_names):
        for j, cau in enumerate(asset_names):
            if i != j and adj[i, j] >= threshold:
                G.add_edge(cau, eff, weight=adj[i, j])
                edges.append((cau, eff))
                weights.append(adj[i, j])

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.circular_layout(G)

    # Colour nodes by asset class
    class_colours = {
        'GLD': '#FFD700', 'SLV': '#C0C0C0',
        'WTI': '#8B4513', 'NG': '#FF8C00', 'COPPER': '#B87333',
        'SPX': '#1E90FF',
        'BTC': '#FF6347',
        'DXY': '#3CB371',
    }
    node_colours = [class_colours.get(n, '#AAAAAA') for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colours,
                           node_size=2200, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    if weights:
        edge_widths = [w * 6 for w in weights]
        edge_colours = [plt.cm.Reds(w) for w in weights]
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, width=edge_widths,
            edge_color=edge_colours, arrows=True,
            arrowsize=20, arrowstyle='->', connectionstyle='arc3,rad=0.15',
            ax=ax,
        )

    legend_elements = [
        mpatches.Patch(facecolor='#FFD700', label='Precious Metals'),
        mpatches.Patch(facecolor='#8B4513', label='Energy'),
        mpatches.Patch(facecolor='#1E90FF', label='Equity'),
        mpatches.Patch(facecolor='#FF6347', label='Crypto'),
        mpatches.Patch(facecolor='#3CB371', label='FX/DXY'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    ax.set_title(
        'CD-KAN Causal Network — Financial Assets 2020-2025\n'
        f'(Edges shown where prob >= {threshold}; width = probability)',
        fontweight='bold', fontsize=13,
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filepath}')


def plot_top_edge_functions(model, adj: np.ndarray, asset_names,
                            threshold, top_k, filepath):
    """Plot the learned spline function f(x) for the top-K strongest edges."""
    strong_edges = []
    for i, eff in enumerate(asset_names):
        for j, cau in enumerate(asset_names):
            if i != j and adj[i, j] >= threshold:
                strong_edges.append((adj[i, j], i, j, eff, cau))
    strong_edges.sort(reverse=True)
    strong_edges = strong_edges[:top_k]

    if not strong_edges:
        print('  No edges above threshold — skipping function plot')
        return

    ncols = min(3, len(strong_edges))
    nrows = (len(strong_edges) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                              squeeze=False)
    axes_flat = axes.flatten()

    for k, (prob, i, j, eff, cau) in enumerate(strong_edges):
        eid = f'{i}_{j}'
        if eid not in model.cd_layer.edge_functions:
            continue
        edge_fn = model.cd_layer.edge_functions[eid]
        xs, ys  = get_spline_curve(edge_fn)

        ax = axes_flat[k]
        ax.plot(xs, ys, linewidth=2.5, color='#2c7bb6')
        ax.axhline(0, color='grey', linewidth=0.7, linestyle='--')
        ax.axvline(0, color='grey', linewidth=0.7, linestyle='--')
        ax.fill_between(xs, ys, alpha=0.12, color='#2c7bb6')
        ax.set_title(f'{cau} -> {eff}\n(p={prob:.2f})', fontweight='bold', fontsize=11)
        ax.set_xlabel(f'{cau} (normalised)', fontsize=9)
        ax.set_ylabel(f'Effect on {eff}', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Annotate if curve appears nonlinear
        y_range = ys.max() - ys.min()
        if abs(ys[0] - ys[-1]) < 0.3 * y_range:
            ax.set_title(f'{cau} -> {eff}\n(p={prob:.2f}, nonlinear)',
                         fontweight='bold', fontsize=11, color='#d73027')

    for k in range(len(strong_edges), len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle(
        'Learned Functional Forms: Top Causal Edges\n'
        '(Nonlinear = red title; Linear-appearing = blue title)',
        fontweight='bold', fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filepath}')


def plot_stability(runs_adj: list, asset_names, threshold, filepath):
    """Bar chart of Jaccard stability across rolling windows per asset pair."""
    from src.evaluation import evaluate_causal_real
    stability = evaluate_causal_real(runs_adj, dataset_name='financial', threshold=threshold)

    # Per-edge Jaccard across runs
    n = len(runs_adj)
    D = len(asset_names)
    pairwise = np.zeros((D, D))
    for run_i in runs_adj:
        for run_j in runs_adj:
            bi = binarise_adjacency(run_i, threshold)
            bj = binarise_adjacency(run_j, threshold)
            inter = (bi & bj).astype(float)
            union = (bi | bj).astype(float)
            pairwise += inter / (union + 1e-8)
    pairwise /= n * n

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.eye(D, dtype=bool)
    sns.heatmap(
        pairwise, mask=mask,
        xticklabels=asset_names, yticklabels=asset_names,
        annot=True, fmt='.2f', cmap='Blues',
        vmin=0, vmax=1, linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Jaccard Stability'}, ax=ax,
    )
    ax.set_title(
        f'Causal Graph Stability Across {n} Rolling Windows\n'
        f'(Jaccard index; mean={stability["stability_jaccard"]:.3f})',
        fontweight='bold', fontsize=13,
    )
    ax.set_xlabel('Cause', fontweight='bold')
    ax.set_ylabel('Effect', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filepath}')
    return stability


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150,
                        help='Training epochs per fold (default=150)')
    parser.add_argument('--threshold', type=float, default=0.35,
                        help='Edge probability threshold for binary graph (default=0.35)')
    parser.add_argument('--top_k', type=int, default=9,
                        help='Number of top edges to plot functions for (default=9)')
    parser.add_argument('--fast', action='store_true',
                        help='Quick run: 30 epochs, 1 fold only')
    args = parser.parse_args()

    if args.fast:
        args.epochs = 30

    print('=' * 60)
    print('CD-KAN Real-Data Causal Discovery')
    print('Dataset: Financial Assets 2020-2025')
    print('=' * 60)

    # Load data
    raw_data, asset_names, df = load_financial_data()
    n_assets = len(asset_names)
    print(f'Assets ({n_assets}): {asset_names}')
    print(f'Time steps: {len(raw_data)}')

    # Rolling window folds for stability analysis
    T       = len(raw_data)
    n_train = int(T * 0.7)
    n_test  = int(T * 0.15)
    n_folds = 1 if args.fast else 3
    stride  = (T - n_train - n_test) // max(1, n_folds - 1) if n_folds > 1 else 0

    fold_adjs   = []
    fold_models = []
    fold_hists  = []

    for fold in range(n_folds):
        start = fold * stride
        end   = start + n_train + n_test
        if end > T:
            break

        # Normalize with training statistics only
        data_norm, mean, std = normalise(raw_data[start:end])
        train_scaled = data_norm[:n_train]
        test_scaled  = data_norm[n_train:]

        X_tr, y_tr = make_windows(train_scaled, WINDOW)
        X_te, y_te = make_windows(test_scaled, WINDOW)

        print(f'\n--- Fold {fold+1}/{n_folds} ---')
        print(f'  Train: {len(X_tr)} windows, Test: {len(X_te)} windows')

        model, hist = train_fold(X_tr, y_tr, X_te, y_te, n_assets,
                                 epochs=args.epochs, seed=42 + fold)

        adj    = model.get_summary_adjacency().detach().cpu().numpy()
        e_lags = model.get_expected_lags().detach().cpu().numpy()

        fold_adjs.append(adj)
        fold_models.append(model)
        fold_hists.append(hist)

        final_mse = hist['test_loss'][-1]
        final_h   = hist['h_residuals'][-1]
        print(f'  Final val MSE: {final_mse:.5f}')
        print(f'  Final h(W):    {final_h:.4e}')

        # ---- Save per-fold checkpoint immediately ----
        fold_prefix = f'fold{fold+1}'
        adj_df  = pd.DataFrame(adj,    index=asset_names, columns=asset_names)
        lags_df = pd.DataFrame(e_lags, index=asset_names, columns=asset_names)
        adj_df.to_csv( os.path.join(OUT_DIR, f'real_causal_adjacency_{fold_prefix}.csv'))
        lags_df.to_csv(os.path.join(OUT_DIR, f'real_causal_expected_lags_{fold_prefix}.csv'))

        plot_adjacency_heatmap(
            adj, asset_names,
            os.path.join(OUT_DIR, f'real_causal_adjacency_{fold_prefix}.png'),
        )
        plot_lag_heatmap(
            e_lags, adj, asset_names, args.threshold,
            os.path.join(OUT_DIR, f'real_causal_lag_heatmap_{fold_prefix}.png'),
        )
        plot_network_graph(
            adj, asset_names, args.threshold,
            os.path.join(OUT_DIR, f'real_causal_network_{fold_prefix}.png'),
        )
        plot_top_edge_functions(
            model, adj, asset_names,
            args.threshold, args.top_k,
            os.path.join(OUT_DIR, f'real_causal_functions_{fold_prefix}.png'),
        )

        # Summary log
        log_row = {
            'fold': fold + 1,
            'val_mse': final_mse,
            'h_W': final_h,
            'edge_density': float((adj >= args.threshold).mean()),
            'n_epochs': len(hist['test_loss']),
        }
        log_path = os.path.join(OUT_DIR, 'fold_summary.csv')
        log_df = pd.DataFrame([log_row])
        if os.path.exists(log_path):
            existing = pd.read_csv(log_path)
            # Replace row if fold already exists
            existing = existing[existing['fold'] != fold + 1]
            log_df = pd.concat([existing, log_df], ignore_index=True)
        log_df.to_csv(log_path, index=False)
        print(f'  Checkpoint saved for fold {fold+1} -> {OUT_DIR}/')

    # Use the first (primary) fold for main visualizations
    primary_model = fold_models[0]
    primary_adj   = fold_adjs[0]
    primary_lags  = primary_model.get_expected_lags().detach().cpu().numpy()

    print(f'\n{"="*60}')
    print('Generating Causal Insights...')
    print(f'  Threshold: {args.threshold}')

    # ---- 1. Adjacency heatmap ----
    plot_adjacency_heatmap(
        primary_adj, asset_names,
        os.path.join(OUT_DIR, 'real_causal_adjacency.png'),
    )

    # ---- 2. Lag heatmap ----
    plot_lag_heatmap(
        primary_lags, primary_adj, asset_names, args.threshold,
        os.path.join(OUT_DIR, 'real_causal_lag_heatmap.png'),
    )

    # ---- 3. Network graph ----
    plot_network_graph(
        primary_adj, asset_names, args.threshold,
        os.path.join(OUT_DIR, 'real_causal_network.png'),
    )

    # ---- 4. Learned functional forms ----
    plot_top_edge_functions(
        primary_model, primary_adj, asset_names,
        args.threshold, args.top_k,
        os.path.join(OUT_DIR, 'real_causal_functions.png'),
    )

    # ---- 5. Stability (only if multi-fold) ----
    if len(fold_adjs) > 1:
        stability = plot_stability(
            fold_adjs, asset_names, args.threshold,
            os.path.join(OUT_DIR, 'real_causal_stability.png'),
        )
        print(f'  Graph stability (Jaccard): {stability["stability_jaccard"]:.3f}')
        print(f'  Mean edge density:         {stability["mean_edge_density"]:.3f}')
    else:
        print('  (Stability plot skipped: single fold run)')

    # ---- 6. Save tables ----
    adj_df   = pd.DataFrame(primary_adj, index=asset_names, columns=asset_names)
    lags_df  = pd.DataFrame(primary_lags, index=asset_names, columns=asset_names)
    adj_df.to_csv(os.path.join(OUT_DIR, 'real_causal_adjacency.csv'))
    lags_df.to_csv(os.path.join(OUT_DIR, 'real_causal_expected_lags.csv'))
    print(f'  Saved: real_causal_adjacency.csv')
    print(f'  Saved: real_causal_expected_lags.csv')

    # ---- 7. Console insight summary ----
    print(f'\n{"="*60}')
    print('Top Causal Relationships (prob >= threshold):')
    print(f'{"Cause":<10} -> {"Effect":<10} | Prob   | Exp.Lag')
    print('-' * 44)
    rows = []
    for i, eff in enumerate(asset_names):
        for j, cau in enumerate(asset_names):
            if i != j and primary_adj[i, j] >= args.threshold:
                rows.append((primary_adj[i, j], cau, eff, primary_lags[i, j]))
    rows.sort(reverse=True)
    for prob, cau, eff, lag in rows:
        print(f'{cau:<10} -> {eff:<10} | {prob:.3f}  | {lag:.1f}d')

    print(f'\nAll outputs saved to: {OUT_DIR}/')


if __name__ == '__main__':
    main()
