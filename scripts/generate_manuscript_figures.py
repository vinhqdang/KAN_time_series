import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Styles
    box_style = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1.5)
    highlight_style = dict(boxstyle="round,pad=0.3", fc="#e6f3ff", ec="#0066cc", lw=2)
    
    # 1. Input
    ax.text(1, 3, "Input\n$X_{t-L:t}$", ha="center", va="center", bbox=box_style, fontsize=12)
    
    # Arrow
    ax.annotate("", xy=(2, 3), xytext=(1.5, 3), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 2. RevIN
    ax.text(2.5, 3, "RevIN\nNorm", ha="center", va="center", bbox=box_style, fontsize=10)
    
    # Arrow
    ax.annotate("", xy=(3.5, 3), xytext=(3, 3), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 3. CD Layer (Big Box)
    rect = patches.FancyBboxPatch((3.5, 1), 3, 4, boxstyle="round,pad=0.1", fc="#f9f9f9", ec="gray", lw=1, linestyle="--")
    ax.add_patch(rect)
    ax.text(5, 5.3, "Causal Discovery Layer", ha="center", va="center", fontsize=11, fontweight="bold")

    # Inside CD Layer
    # Adjacency
    ax.text(5, 4, "Adjacency $A$\n(Learnable)", ha="center", va="center", bbox=highlight_style, fontsize=10)
    
    # KAN Functions
    ax.text(5, 2, "KAN Functions\n$\phi_{i \\to j}(x)$", ha="center", va="center", bbox=box_style, fontsize=10)
    
    # Arrows inside
    ax.annotate("", xy=(5, 3.4), xytext=(5, 2.6), arrowprops=dict(arrowstyle="<-", lw=1, linestyle="dashed"))
    ax.text(5.1, 3, "Masking", ha="left", va="center", fontsize=9, color="gray")

    # Arrow out of CD Layer
    ax.annotate("", xy=(7, 3), xytext=(6.5, 3), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 4. Residual KAN Backbone
    ax.text(8, 3, "Residual\nKAN Blocks\n(N Layers)", ha="center", va="center", bbox=box_style, fontsize=10)

    # Arrow
    ax.annotate("", xy=(9, 3), xytext=(8.5, 3), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 5. RevIN Denorm
    ax.text(9.5, 3, "RevIN\nDe-Norm", ha="center", va="center", bbox=box_style, fontsize=10)

    # Arrow
    ax.annotate("", xy=(10.5, 3), xytext=(10, 3), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 6. Output
    ax.text(11, 3, "Prediction\n$\hat{X}_{t+1}$", ha="center", va="center", bbox=box_style, fontsize=12)

    # ALM Loop (Bottom visualization)
    ax.annotate("", xy=(5, 0.8), xytext=(5, 1.0), arrowprops=dict(arrowstyle="-", lw=1, linestyle="dotted"))
    ax.text(5, 0.5, "DAG Constraint\n$Tr(e^A) - d = 0$", ha="center", va="center", fontsize=10, color="#cc0000", fontweight="bold")

    plt.tight_layout()
    plt.savefig('manuscript/figures/cdkan_architecture.png', dpi=300, bbox_inches='tight')
    print("Created cdkan_architecture.png")

if __name__ == "__main__":
    create_architecture_diagram()
