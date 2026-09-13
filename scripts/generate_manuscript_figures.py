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
    sum_style = dict(boxstyle="round,pad=0.3", fc="#fff4e6", ec="#cc7a00", lw=2)

    # 1. Input
    ax.text(1, 3, "Input\n$X_{t-L:t}$", ha="center", va="center", bbox=box_style, fontsize=12)

    # Arrow
    ax.annotate("", xy=(2, 3), xytext=(1.5, 3), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 2. RevIN
    ax.text(2.5, 3, "RevIN\nNorm", ha="center", va="center", bbox=box_style, fontsize=10)

    # Arrow
    ax.annotate("", xy=(3.5, 3), xytext=(3, 3), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 3. CD Layer (Big Box) -- the ONLY layer between input and output.
    # No downstream network re-mixes variables after this layer (Eq. 4/eq:predict):
    # each target's forecast is the additive sum of its own incoming edge functions,
    # and nothing else touches it before RevIN de-normalization.
    rect = patches.FancyBboxPatch((3.5, 0.9), 5, 4.2, boxstyle="round,pad=0.1", fc="#f9f9f9", ec="gray", lw=1, linestyle="--")
    ax.add_patch(rect)
    ax.text(6, 5.35, "Causal Discovery Layer (component-wise, no downstream mixing)", ha="center", va="center", fontsize=11, fontweight="bold")

    # Inside CD Layer
    # Adjacency
    ax.text(5, 4.1, "Adjacency\n$A_0$ (DAG) & $A_{1\\dots L}$", ha="center", va="center", bbox=highlight_style, fontsize=10)

    # KAN edge functions
    ax.text(5, 2.2, "KAN edge functions\n$\\phi^{(h)}_{ij}(x)$", ha="center", va="center", bbox=box_style, fontsize=10)

    # Arrows inside (masking gates which edges are active)
    ax.annotate("", xy=(5, 3.5), xytext=(5, 2.8), arrowprops=dict(arrowstyle="<-", lw=1, linestyle="dashed"))
    ax.text(5.15, 3.15, "Masking", ha="left", va="center", fontsize=9, color="gray")

    # Per-target additive combiner -- explicit: each target i is the SUM of its own
    # incoming edges only (Eq. eq:predict); no shared/residual network afterwards.
    ax.text(7.3, 2.2, "Per-target sum\n$\\hat{x}_{t,i}=b_i+\\sum_{j,h}\\phi^{(h)}_{ij}$", ha="center", va="center", bbox=sum_style, fontsize=9)
    ax.annotate("", xy=(6.6, 2.2), xytext=(5.9, 2.2), arrowprops=dict(arrowstyle="->", lw=1.5))

    # Arrow out of CD Layer directly to RevIN de-norm (no intervening backbone)
    ax.annotate("", xy=(9.15, 3), xytext=(8.5, 2.6), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 4. RevIN Denorm
    ax.text(9.7, 3, "RevIN\nDe-Norm", ha="center", va="center", bbox=box_style, fontsize=10)

    # Arrow
    ax.annotate("", xy=(10.7, 3), xytext=(10.2, 3), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 5. Output
    ax.text(11.2, 3, "Prediction\n$\\hat{X}_{t+1}$", ha="center", va="center", bbox=box_style, fontsize=12)

    # ALM Loop (Bottom visualization)
    ax.annotate("", xy=(5, 0.7), xytext=(5, 0.9), arrowprops=dict(arrowstyle="-", lw=1, linestyle="dotted"))
    ax.text(5, 0.4, "DAG Constraint\n$Tr(e^{A_0 \\circ A_0}) - d = 0$", ha="center", va="center", fontsize=10, color="#cc0000", fontweight="bold")

    plt.tight_layout()
    plt.savefig('manuscript/figures/cdkan_architecture.png', dpi=300, bbox_inches='tight')
    print("Created cdkan_architecture.png")

if __name__ == "__main__":
    create_architecture_diagram()
