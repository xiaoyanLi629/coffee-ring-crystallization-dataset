"""Shared figure style for the manuscript (Nature / Scientific Data conventions).

Single column 89 mm, double column 183 mm; Arimo (metrically identical to Arial) 7 pt, panel letters 8 pt bold;
thin axes, outward ticks, no top/right spines; Okabe-Ito colour-blind-safe palette with fixed colours per element,
per salt and per SEM detector; figures saved as vector PDF (TrueType fonts embedded) and 600-dpi PNG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

MM = 1 / 25.4
COL1, COL2 = 89 * MM, 183 * MM

for f in font_manager.findSystemFonts():
    if "Arimo" in f:
        font_manager.fontManager.addfont(f)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arimo", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "legend.frameon": False, "legend.handlelength": 1.4, "legend.borderaxespad": 0.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out",
    "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0,
    "lines.markersize": 3, "patch.linewidth": 0.5, "axes.titlepad": 3, "axes.labelpad": 2,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular", "axes.unicode_minus": True,
})

OKABE = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
         "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000", "grey": "#8C8C8C"}
ELEMENT = {"C": "#4D4D4D", "O": "#56B4E9", "Na": "#E69F00", "Mg": "#009E73", "S": "#C9B200", "Cl": "#CC79A7", "Ca": "#D55E00"}
SALT = {"NaHCO3": "#E69F00", "CaCl2": "#D55E00", "MgSO4": "#009E73"}
DETECTOR = {"BEC": "#0072B2", "SEI": "#D55E00"}
TABLE = {1: "#FDE0C5", 2: "#FAC09A", 3: "#F59E72", 4: "#E8724B", 5: "#C2452E"}  # NaHCO3 low -> high
SALT_TEX = {"NaHCO3": "NaHCO$_3$", "CaCl2": "CaCl$_2$", "MgSO4": "MgSO$_4$", "MgCl2": "MgCl$_2$"}


def panel(ax, letter, x=-0.18, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".pdf")
    fig.savefig(path_no_ext + ".png")
    plt.close(fig)
