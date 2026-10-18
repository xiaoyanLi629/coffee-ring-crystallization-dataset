"""Shared figure style for the manuscript, matched to the companion dataset paper (CoffeeRing_Multimodal_Dataset).

Scientific Data sizes: 180 mm full width, 88 mm single column. Arial 7 pt for all text, 6 pt for tick labels,
bold 8 pt panel letters. Axes are placed in millimetres from the top-left corner (axes_mm), so panels line up across
figures and panel letters sit at the same offset everywhere. Figures are saved as vector PDF with TrueType fonts and as
a 300 dpi PNG preview.
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

MM = 1 / 25.4
FULL, HALF = 180, 88

for d in ["/usr/share/fonts/truetype/arial", "/System/Library/Fonts/Supplemental", "/Library/Fonts"]:
    for f in Path(d).glob("[Aa]rial*.ttf"):
        font_manager.fontManager.addfont(str(f))

INK = "#222222"
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial", "mathtext.cal": "Arial", "mathtext.default": "regular",
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3, "legend.columnspacing": 0.9,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.major.size": 2, "ytick.major.size": 2, "xtick.minor.size": 1.2,
    "ytick.minor.size": 1.2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "xtick.direction": "out",
    "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 0.9,
    "lines.markersize": 2.6, "patch.linewidth": 0.4, "axes.titlepad": 2.5, "axes.labelpad": 1.5,
    "hatch.linewidth": 0.4, "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.unicode_minus": True, "axes.axisbelow": True,
})

# element colours as in the companion paper's EDS tiles (Ca blue, Cl orange, O green), the rest chosen to stay distinct
ELEMENT = {"Ca": "#2C8FD6", "Cl": "#F08A1C", "O": "#3CB54A", "S": "#D9B310", "Mg": "#C04FC8", "Na": "#8C6BB1", "C": "#555555"}
# brighter versions for drawing signal on a black background
ELEMENT_ON_BLACK = {"Ca": "#3FB4FF", "Cl": "#FFA024", "O": "#43D65A", "S": "#FFE14D", "Mg": "#E46CFF", "Na": "#B39DFF", "C": "#DDDDDD"}
SALT = {"NaHCO3": "#7F7F7F", "CaCl2": "#2C8FD6", "MgSO4": "#C04FC8"}
DETECTOR = {"BEC": "#1F4E79", "SEI": "#C8553D"}
# NaHCO3 level (Table 1-5): one sequential ramp, light to dark
TABLE = {1: "#C6DBEF", 2: "#9ECAE1", 3: "#6BAED6", 4: "#3182BD", 5: "#08519C"}
GREY = "#8C8C8C"
LIGHT = "#D9D9D9"
SALT_TEX = {"NaHCO3": "NaHCO$_3$", "CaCl2": "CaCl$_2$", "MgSO4": "MgSO$_4$", "MgCl2": "MgCl$_2$"}


def figure(w_mm, h_mm):
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def axes_mm(fig, x, y, w, h, **kw):
    """axes at (x, y) mm from the top-left corner, w x h mm"""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def letter(fig, x, y, s):
    """bold panel letter with its top-left corner at (x, y) mm"""
    W, H = fig.get_size_inches() / MM
    fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight="bold", ha="left", va="top")


def save(fig, path_no_ext):
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    fig.savefig(path_no_ext + ".pdf")
    fig.savefig(path_no_ext + ".png", dpi=300)
    plt.close(fig)
