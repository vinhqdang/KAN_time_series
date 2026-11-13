"""Shared matplotlib style for the data figures of the MLJ paper (TrueType fonts, Times-like STIX, 7 pt text)."""
import matplotlib as mpl

# categorical slots 1-5 of the validated reference palette (light mode): blue, orange, aqua, violet, magenta
BLUE, ORANGE, AQUA, VIOLET, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"
INK, INK2, GRID, SURFACE = "#1d1d1b", "#52514e", "#dcdad4", "#fcfcfb"
W_FULL = 5.15  # inches, = \textwidth of sn-jnl (372 pt)
W_HALF = 2.5


def setup():
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "axes.axisbelow": True,
        "lines.linewidth": 1.3, "lines.markersize": 4,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 150,
    })


def panel_label(ax, s, dx=-0.0, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")
