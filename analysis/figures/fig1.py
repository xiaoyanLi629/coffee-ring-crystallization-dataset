"""Fig. 1: experimental design and the image record of eight droplets.

python fig1.py   ->  manuscript/figures/fig1_overview.{pdf,png}

Display processing of the EDS maps (described in the caption): each exported pseudo-colour map is reduced to its
brightest channel, median filtered (3 px) and Gaussian smoothed (sigma 1.5 px); the substrate level, measured as the
median outside the dilated droplet mask, plus three robust standard deviations is subtracted; the result is scaled to
its 99.5th percentile inside the droplet (never below a floor, so that a map without signal stays dark) and drawn in
the element's colour on black. SEM and optical images are only cropped and contrast-stretched.
"""
import sys
import numpy as np, pandas as pd, cv2
from scipy import ndimage as ndi
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import to_rgb
sys.path.insert(0, "/root/autodl-fs/CoffeeRing_Crystallization_Dataset/analysis/figures")
from style import figure, axes_mm, letter, save, ELEMENT_ON_BLACK, TABLE, SALT_TEX, INK, GREY

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
OUT = ROOT + "/manuscript/figures/fig1_overview"

# ---- parameters
RECIPES = ["T1_S5", "T1_S21", "T2_S13", "T3_S25", "T4_S13", "T4_S23", "T5_S5", "T5_S25"]   # replicate chosen in pick()
ROWS = [("Optical", "optical"), ("SEM", "sem"), ("EDS Ca", "Ca"), ("EDS S", "S"), ("EDS Cl", "Cl"), ("EDS Mg", "Mg")]
CROP = 1.22                 # tile side / droplet diameter
EDS_MEDIAN, EDS_SIGMA, EDS_K = 3, 1.5, 3.0
EDS_FLOOR = 0.16            # minimum display range above background (fraction of the export's full scale)
EDS_GAMMA = 0.75
TILE, GAP, LABEL_W = 21.0, 0.6, 9.0   # mm
TOP = 47.0                  # mm reserved for panel a
LEVELS = {"NaHCO3": [0.25, 0.5, 0.75, 1, 2], "CaCl2": [0.5, 0.75, 1, 1.5, 2], "MgSO4": [0.25, 0.5, 0.75, 1, 2]}

meta = pd.read_csv(ROOT + "/metadata/samples.csv").set_index("droplet_id")
reg = pd.read_csv(ROOT + "/release/derived/registration.csv").set_index("droplet_id")
MK = np.load(ROOT + "/derived/masks_eds.npz")
MASKS = dict(zip(MK["paths"], MK["masks"]))


def pick(recipe):
    """replicate imaged at 50x with a standard EDS export, no quality flag and a converged SEM-EDS registration, best
    match first; if the recipe has none, the nearest CaCl2 level at the same MgSO4 level is used instead"""
    t, smp = map(int, recipe[1:].split("_S"))
    ok = meta[(meta.table == t) & (meta.sem_mag == 50) & (meta.eds_Ca_w == 1010) & (meta.n_flags == 0)].join(reg)
    ok = ok[~ok.fallback_to_typical]
    row = (smp - 1) // 5
    for s in sorted(range(5 * row + 1, 5 * row + 6), key=lambda x: abs(x - smp)):
        d = ok[ok["sample"] == s]
        if len(d):
            return d.ncc_registered.idxmax()
    raise ValueError(recipe)


def square(img, cx, cy, side, fill=0):
    """square crop of side `side` centred on (cx, cy); pads with `fill` outside the image"""
    h = int(round(side / 2)); cx, cy = int(round(cx)), int(round(cy))
    pad = [(h, h), (h, h)] + [(0, 0)] * (img.ndim - 2)
    p = np.pad(img, pad, constant_values=fill)
    return p[cy:cy + 2 * h, cx:cx + 2 * h]


def mask_geom(m):
    ys, xs = np.nonzero(m)
    return xs.mean(), ys.mean(), 2 * np.sqrt(m.sum() / np.pi)


def optical_mask(g):
    """same segmentation as analysis/g_derive.py (dark-field image: droplet = elevated brightness and texture)"""
    from skimage.filters import threshold_otsu
    t = ndi.gaussian_filter(np.abs(g - ndi.gaussian_filter(g, 2)), 6) + ndi.gaussian_filter(g, 6) * 0.5
    t = np.log(t + 1e-3)
    m = ndi.binary_closing(t > threshold_otsu(t), np.ones((5, 5)), iterations=4)
    lab, n = ndi.label(m)
    big = lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)
    return ndi.binary_fill_holes(big)


def eds_tile(path, e, m_full, cx, cy, side):
    a = cv2.imread(f"{ROOT}/experiment/data/{path}/{e}.jpeg").astype(np.float32).max(2) / 255
    a = ndi.gaussian_filter(ndi.median_filter(a, EDS_MEDIAN), EDS_SIGMA)
    out = ~ndi.binary_dilation(m_full, iterations=20)
    bg = np.median(a[out]); sd = 1.4826 * np.median(np.abs(a[out] - bg))
    v = np.clip(a - bg - EDS_K * sd, 0, None)
    top = max(np.percentile(v[m_full], 99.5), EDS_FLOOR)
    v = np.clip(v / top, 0, 1) ** EDS_GAMMA
    rgb = v[..., None] * np.array(to_rgb(ELEMENT_ON_BLACK[e]))
    return square(rgb, cx, cy, side)


def tiles(did):
    r = meta.loc[did]; path = r.path; d = f"{ROOT}/experiment/data/{path}"
    out = {}
    # EDS: mask on the 379 x 505 grid, maps at their export size
    a0 = cv2.imread(f"{d}/Ca.jpeg", 0); He, We = a0.shape
    m = cv2.resize(MASKS[path].astype(np.uint8), (We, He), interpolation=cv2.INTER_NEAREST).astype(bool)
    cx, cy, dia = mask_geom(m)
    for _, e in ROWS[2:]:
        out[e] = eds_tile(path, e, m, cx, cy, CROP * dia)
    # SEM: registration maps the EDS grid (1010 x 758) onto the SEM image resized to that grid
    sem = cv2.imread(f"{d}/sem.jpg", 0).astype(np.float32)
    g = reg.loc[did]; kx, ky = 1010 / We, 758 / He
    sx = sem.shape[1] / 1010
    scx = (g.scale * cx * kx + g.tx_px) * sx; scy = (g.scale * cy * ky + g.ty_px) * sx
    sem = square(sem, scx, scy, CROP * dia * kx * g.scale * sx, fill=np.median(sem))
    lo, hi = np.percentile(sem, [0.5, 99.7]); out["sem"] = np.clip((sem - lo) / (hi - lo), 0, 1)
    # optical: own segmentation on a 648 px wide copy
    o = cv2.imread(f"{d}/image.jpg")[:, :, ::-1].astype(np.float32)
    k = 648 / o.shape[1]
    om = optical_mask(cv2.resize(o.max(2), (648, int(round(o.shape[0] * k))), interpolation=cv2.INTER_AREA))
    ocx, ocy, odia = mask_geom(om)
    o = square(o, ocx / k, ocy / k, CROP * odia / k)
    out["optical"] = np.clip(o / np.percentile(o, 99.8), 0, 1)
    return out, r


def design(fig):
    """panel a: the 5 x 5 CaCl2 x MgSO4 grid numbered by Sample, repeated at five NaHCO3 levels (Tables)"""
    ax = axes_mm(fig, 4, 3, 62, 41); ax.set_xlim(-2.6, 5.2); ax.set_ylim(-1.9, 5.6); ax.set_aspect("equal"); ax.axis("off")
    c = 0.9
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((i, j), c, c, fc="#EEF3F8", ec="#9DB4CC", lw=0.4))
            ax.text(i + c / 2, j + c / 2, f"{5 * j + i + 1}", ha="center", va="center", fontsize=6)
    for i, v in enumerate(LEVELS["CaCl2"]):
        ax.text(i + c / 2, -0.15, f"{v:g}", ha="center", va="top", fontsize=6)
    ax.text(2.45, -0.95, f"{SALT_TEX['CaCl2']} (mM)", ha="center", va="top")
    for j, v in enumerate(LEVELS["MgSO4"]):
        ax.text(-0.15, j + c / 2, f"{v:g}", ha="right", va="center", fontsize=6)
    ax.text(-1.75, 2.45, f"{SALT_TEX['MgSO4']} (mM)", ha="center", va="center", rotation=90)
    ax.text(2.45, 5.15, "Sample 1–25", ha="center", va="bottom", fontweight="bold")
    # the five Tables
    ax2 = axes_mm(fig, 66, 3, 34, 41); ax2.set_xlim(0, 3.6); ax2.set_ylim(-1.9, 5.6); ax2.axis("off")
    ax2.text(0.0, 2.45, "×", ha="center", va="center", fontsize=11)
    for t in range(1, 6):
        y = 5 - t
        ax2.add_patch(Rectangle((0.5, y), 0.9, c, fc=TABLE[t], ec="none"))
        ax2.text(0.95, y + c / 2, f"{t}", ha="center", va="center", fontsize=6, color="white" if t >= 3 else INK)
        ax2.text(1.55, y + c / 2, f"{LEVELS['NaHCO3'][t - 1]:g} mM", ha="left", va="center", fontsize=6)
    ax2.text(0.95, 5.15, "Table", ha="center", va="bottom", fontweight="bold")
    ax2.text(1.55, 5.15, SALT_TEX["NaHCO3"], ha="left", va="bottom", fontweight="bold")
    # text block: what every water and droplet carries
    tx = axes_mm(fig, 112, 5, 66, 40); tx.axis("off")
    lines = [("Waters", f"5 × 5 × 5 full factorial, 125 compositions\nall with 0.25 mM {SALT_TEX['MgCl2']}"),
             ("Droplets", "5 replicates of 2 µL per water, 625 in total\n6061 aluminium, 23–26 °C, 45–50 % RH"),
             ("Files per droplet", "optical image (image.jpg)\nSEM image and record (sem.jpg, sem.txt)\nEDS maps of C, Ca, Cl, Mg, Na, O and S")]
    y = 1.0
    for head, body in lines:
        tx.text(0, y, head, fontweight="bold", va="top", transform=tx.transAxes)
        tx.text(0, y - 0.085, body, va="top", transform=tx.transAxes, linespacing=1.35)
        y -= 0.34


def main():
    DROPLETS = [pick(r) for r in RECIPES]
    print(DROPLETS)
    nrow, ncol = len(ROWS), len(DROPLETS)
    H = TOP + 9 + nrow * (TILE + GAP) + 1
    fig = figure(180, H)
    design(fig); letter(fig, 0, 1, "a")
    letter(fig, 0, TOP + 1, "b")
    for j, did in enumerate(DROPLETS):
        T, r = tiles(did)
        x = LABEL_W + j * (TILE + GAP)
        fig.text((x + TILE / 2) / 180, 1 - (TOP + 2.2) / H, did.replace("_", " "), ha="center", va="top", fontsize=6, fontweight="bold")
        fig.text((x + TILE / 2) / 180, 1 - (TOP + 5.0) / H, f"{r.NaHCO3_mM:g} · {r.CaCl2_mM:g} · {r.MgSO4_mM:g}",
                 ha="center", va="top", fontsize=6)
        for i, (lab, key) in enumerate(ROWS):
            ax = axes_mm(fig, x, TOP + 9 + i * (TILE + GAP), TILE, TILE)
            im = T[key]
            ax.imshow(im, cmap="gray" if key == "sem" else None, vmin=0, vmax=1, interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if key == "sem":
                ax.text(0.04, 0.05, r.sem_signal, transform=ax.transAxes, fontsize=5.5, color="white", va="bottom",
                        bbox=dict(fc="black", ec="none", alpha=0.45, pad=0.6))
            if j == 0:
                ax.text(-0.08, 0.5, lab, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6.5,
                        fontweight="bold")
    save(fig, OUT)


if __name__ == "__main__":
    main()
