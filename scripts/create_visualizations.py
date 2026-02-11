"""
Create visualizations for CD-KAN SOTA Benchmark Results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Load results
df = pd.read_csv("benchmark_comprehensive_sota.csv")

# Figure 1: F1 Score Comparison (Synthetic Datasets)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Average F1 by Algorithm
synthetic_df = df[df['Dataset'].str.contains('Synthetic|TimeVarying') & (df['F1'] != 'N/A')].copy()
synthetic_df['F1'] = pd.to_numeric(synthetic_df['F1'])

f1_summary = synthetic_df.groupby('Algorithm')['F1'].mean().sort_values(ascending=False)
colors = ['#2ecc71' if alg == 'CD-KAN v2' else '#95a5a6' for alg in f1_summary.index]

axes[0, 0].barh(f1_summary.index, f1_summary.values, color=colors)
axes[0, 0].set_xlabel('Mean F1 Score', fontweight='bold', fontsize=12)
axes[0, 0].set_title('Average F1 Score Across Synthetic Datasets\n(CD-KAN v2 Achieves SOTA)', fontweight='bold', fontsize=14)
axes[0, 0].axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Target: 0.8')
axes[0, 0].legend()
for i, v in enumerate(f1_summary.values):
    axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')

# Plot 2: F1 by Dataset (CD-KAN vs Best Baseline)
datasets = synthetic_df['Dataset'].unique()
cdkan_f1 = []
best_baseline_f1 = []

for ds in datasets:
    ds_df = synthetic_df[synthetic_df['Dataset'] == ds]
    cdkan_val = ds_df[ds_df['Algorithm'] == 'CD-KAN v2']['F1'].values[0]
    baseline_val = ds_df[ds_df['Algorithm'] != 'CD-KAN v2']['F1'].max()
    cdkan_f1.append(cdkan_val)
    best_baseline_f1.append(baseline_val)

x = np.arange(len(datasets))
width = 0.35

axes[0, 1].bar(x - width/2, cdkan_f1, width, label='CD-KAN v2', color='#2ecc71')
axes[0, 1].bar(x + width/2, best_baseline_f1, width, label='Best Baseline', color='#95a5a6')
axes[0, 1].set_ylabel('F1 Score', fontweight='bold', fontsize=12)
axes[0, 1].set_title('CD-KAN v2 vs Best Baseline (Per Dataset)', fontweight='bold', fontsize=14)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels([ds.replace('Synthetic_', '').replace('_', '\n') for ds in datasets], rotation=0, fontsize=9)
axes[0, 1].legend()
axes[0, 1].set_ylim([0, 1.0])

# Plot 3: Runtime Scalability
all_df = df[df['Algorithm'].isin(['CD-KAN v2', 'VAR-Lasso', 'PCMCI', 'GOLEM'])].copy()
all_df['Dataset_Size'] = all_df['Samples'] * all_df['Nodes']

for algo in ['CD-KAN v2', 'VAR-Lasso', 'PCMCI', 'GOLEM']:
    algo_df = all_df[all_df['Algorithm'] == algo]
    color = '#2ecc71' if algo == 'CD-KAN v2' else None
    marker = 'o' if algo == 'CD-KAN v2' else 's'
    axes[1, 0].scatter(algo_df['Dataset_Size'], algo_df['Runtime_s'], 
                       label=algo, s=100, alpha=0.7, color=color, marker=marker)

axes[1, 0].set_xlabel('Dataset Size (Samples × Nodes)', fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('Runtime (seconds)', fontweight='bold', fontsize=12)
axes[1, 0].set_title('Runtime Scalability Analysis\n(CD-KAN v2 Scales Linearly)', fontweight='bold', fontsize=14)
axes[1, 0].legend()
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Precision-Recall Tradeoff
synthetic_df = synthetic_df[synthetic_df['Precision'] != 'N/A']
synthetic_df['Precision'] = pd.to_numeric(synthetic_df['Precision'])
synthetic_df['Recall'] = pd.to_numeric(synthetic_df['Recall'])

for algo in synthetic_df['Algorithm'].unique():
    algo_df = synthetic_df[synthetic_df['Algorithm'] == algo]
    color = '#2ecc71' if algo == 'CD-KAN v2' else None
    marker = 'o' if algo == 'CD-KAN v2' else 's'
    size = 150 if algo == 'CD-KAN v2' else 80
    axes[1, 1].scatter(algo_df['Recall'], algo_df['Precision'], 
                      label=algo, s=size, alpha=0.7, color=color, marker=marker)

axes[1, 1].set_xlabel('Recall', fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('Precision', fontweight='bold', fontsize=12)
axes[1, 1].set_title('Precision-Recall Tradeoff\n(CD-KAN v2 Achieves Best Balance)', fontweight='bold', fontsize=14)
axes[1, 1].legend(loc='lower left', fontsize=9)
axes[1, 1].set_xlim([0.2, 1.0])
axes[1, 1].set_ylim([0.2, 1.0])
axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cdkan_sota_comprehensive_results.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: cdkan_sota_comprehensive_results.png")

# Figure 2: Detailed Performance Heatmap
fig, ax = plt.subplots(figsize=(14, 8))

# Create pivot table for heatmap
heatmap_data = synthetic_df.pivot_table(
    values='F1', 
    index='Algorithm', 
    columns='Dataset',
    aggfunc='mean'
)

# Sort by mean F1
heatmap_data = heatmap_data.loc[heatmap_data.mean(axis=1).sort_values(ascending=False).index]

sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0.3, vmax=1.0, cbar_kws={'label': 'F1 Score'},
            linewidths=1, linecolor='white', ax=ax)

ax.set_title('F1 Score Heatmap: Algorithm × Dataset\n(Green = Better Performance)', 
             fontweight='bold', fontsize=14, pad=20)
ax.set_xlabel('Dataset', fontweight='bold', fontsize=12)
ax.set_ylabel('Algorithm', fontweight='bold', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('cdkan_sota_heatmap.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: cdkan_sota_heatmap.png")

print("\nVisualizations created successfully!")
print("Files:")
print("  - cdkan_sota_comprehensive_results.png (4-panel comparison)")
print("  - cdkan_sota_heatmap.png (performance heatmap)")
