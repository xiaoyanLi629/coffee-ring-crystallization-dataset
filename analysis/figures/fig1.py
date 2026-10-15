"""Fig. 1: optical, SEM and EDS images of eight droplets spanning the design.

python fig1.py   ->  manuscript/figures/fig1_overview.{pdf,png}

All processing is for display only and is stated in the caption; the released files are unchanged.
Every tile is a square crop centred on the droplet (side = CROP droplet diameters). The droplet mask comes from the
EDS maps (derived/masks_eds.npz), carried to the SEM image through the SEM-EDS registration, and from the optical
segmentation of analysis/g_derive.py for the optical image. The masks are used to equalise the tiles:
  optical  white balance on the brighter (colourless salt) pixels of the residue, saturation
           reduced to OPT_CHROMA; substrate set to the same dark level; a gamma curve brings
           the median brightness of the residue to OPT_MID in every tile
  SEM      contrast-limited adaptive histogram equalisation; a gamma curve sets the substrate to SEM_SUBSTRATE
  EDS      brightest channel, 3 px median and Gaussian (sigma 1.5 px) smoothing; substrate level plus three robust
           standard deviations subtracted; scaled to the 99.5th percentile in the droplet (not below EDS_FLOOR, so
           that a map without signal stays dark); a gamma curve brings the mean of maps with signal to EDS_MEAN
"""
import sys
import numpy as np, pandas as pd, cv2
from scipy import ndimage as ndi
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu
from matplotlib.colors import to_rgb
sys.path.insert(0, "/root/autodl-fs/CoffeeRing_Crystallization_Dataset/analysis/figures")
from style import figure, axes_mm, save, ELEMENT_ON_BLACK

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
OUT = ROOT + "/manuscript/figures/fig1_overview"

# ---- parameters
RECIPES = ["T1_S5", "T1_S21", "T2_S13", "T3_S25", "T4_S13", "T4_S23", "T5_S5", "T5_S25"]   # replicate chosen in pick()
ROWS = [("Optical", "optical"), ("SEM", "sem"), ("EDS Ca", "Ca"), ("EDS S", "S"), ("EDS Cl", "Cl"), ("EDS Mg", "Mg")]
CROP = 1.22                     # tile side / droplet diameter
PX = 600                        # tiles are resampled to PX x PX
OPT_MID, OPT_SUBSTRATE, OPT_CHROMA = 0.30, 0.04, 0.35
SEM_SUBSTRATE = 0.55
EDS_MEDIAN, EDS_SIGMA, EDS_K = 3, 1.5, 3.0
EDS_FLOOR, EDS_MEAN = 0.16, 0.16
TILE, GAP, LABEL_W, HEAD = 21.1, 0.6, 6.5, 8.5   # mm

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


def square(img, cx, cy, side, fill=0.0):
    """square crop of `side` px centred on (cx, cy), resampled to PX x PX; outside the image it is padded with `fill`,
    or by mirroring the image when fill is None"""
    h = int(round(side / 2)); cx, cy = int(round(cx)), int(round(cy))
    pad = [(h, h), (h, h)] + [(0, 0)] * (img.ndim - 2)
    a = img.astype(np.float32)
    c = (np.pad(a, pad, mode="symmetric") if fill is None else np.pad(a, pad, constant_values=fill))[cy:cy + 2 * h, cx:cx + 2 * h]
    return cv2.resize(c, (PX, PX), interpolation=cv2.INTER_AREA)


def geom(m):
    ys, xs = np.nonzero(m)
    return xs.mean(), ys.mean(), 2 * np.sqrt(m.sum() / np.pi)


def optical_mask(g):
    """segmentation of analysis/g_derive.py (dark-field image: droplet = elevated brightness and texture)"""
    t = np.log(ndi.gaussian_filter(np.abs(g - ndi.gaussian_filter(g, 2)), 6) + ndi.gaussian_filter(g, 6) * 0.5 + 1e-3)
    m = ndi.binary_closing(t > threshold_otsu(t), np.ones((5, 5)), iterations=4)
    lab, n = ndi.label(m)
    return ndi.binary_fill_holes(lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1))


def gamma_to(x, region, target):
    """x in [0, 1]; gamma chosen so that the median of x inside `region` becomes `target`"""
    med = np.clip(np.median(x[region]), 1e-3, 0.999)
    return x ** np.clip(np.log(target) / np.log(med), 0.4, 2.5)


def tone_optical(o, m):
    bg = ~ndi.binary_dilation(m, iterations=int(0.04 * PX))
    br = m & (o.mean(2) > np.percentile(o.mean(2)[m], 60))               # the salt residue is colourless:
    gain = np.clip(o[br].mean() / np.maximum(o[br].mean(0), 1e-3), 0.6, 1.6)  # balance on its brighter pixels
    o = o * gain
    L = o.mean(2, keepdims=True)                                          # the colour of dark-field illumination is
    o = L + OPT_CHROMA * (o - L)                                          # not a sample property: keep only a hint
    L = o.mean(2)
    lo, hi = np.median(L[bg]), np.percentile(L[m], 99.7)
    x = np.clip((o - lo) / (hi - lo), 0, 1)
    x = gamma_to(x, np.repeat(m[..., None], 3, 2), OPT_MID)
    return np.clip(OPT_SUBSTRATE + (1 - OPT_SUBSTRATE) * x, 0, 1)


def tone_sem(s, m):
    lo, hi = np.percentile(s, [0.5, 99.7])
    x = equalize_adapthist(np.clip((s - lo) / (hi - lo), 0, 1), kernel_size=PX // 6, clip_limit=0.012)
    bg = ~ndi.binary_dilation(m, iterations=int(0.04 * PX))
    return gamma_to(x, bg, SEM_SUBSTRATE)


def tone_eds(a, m_full, e):
    a = ndi.gaussian_filter(ndi.median_filter(a, EDS_MEDIAN), EDS_SIGMA)
    out = ~ndi.binary_dilation(m_full, iterations=20)
    bg = np.median(a[out]); sd = 1.4826 * np.median(np.abs(a[out] - bg))
    v = np.clip(a - bg - EDS_K * sd, 0, None)
    top = np.percentile(v[m_full], 99.5)
    v = np.clip(v / max(top, EDS_FLOOR), 0, 1)
    if top > EDS_FLOOR:                                                   # map with signal: equalise its mean
        mean = np.clip(v[m_full].mean(), 1e-3, 0.99)
        v = v ** np.clip(np.log(EDS_MEAN) / np.log(mean), 0.4, 1.0)
    return v[..., None] * np.array(to_rgb(ELEMENT_ON_BLACK[e]))


def tiles(did):
    r = meta.loc[did]; d = f"{ROOT}/experiment/data/{r.path}"
    out = {}
    He, We = cv2.imread(f"{d}/Ca.jpeg", 0).shape
    m = cv2.resize(MASKS[r.path].astype(np.uint8), (We, He), interpolation=cv2.INTER_NEAREST).astype(bool)
    cx, cy, dia = geom(m)
    m_tile = square(m, cx, cy, CROP * dia) > 0.5
    for _, e in ROWS[2:]:
        a = cv2.imread(f"{d}/{e}.jpeg").astype(np.float32).max(2) / 255
        rgb = tone_eds(a, m, e)
        out[e] = np.clip(np.dstack([square(rgb[..., k], cx, cy, CROP * dia) for k in range(3)]), 0, 1)
    # SEM: the registration maps the EDS grid (1010 x 758) onto the SEM image resized to that grid
    sem = cv2.imread(f"{d}/sem.jpg", 0).astype(np.float32)
    g = reg.loc[did]; kx, ky = 1010 / We, 758 / He; sx = sem.shape[1] / 1010
    sem = square(sem, (g.scale * cx * kx + g.tx_px) * sx, (g.scale * cy * ky + g.ty_px) * sx, CROP * dia * kx * g.scale * sx,
                 fill=None)
    out["sem"] = tone_sem(sem, m_tile)
    # optical: own segmentation on a 648 px wide copy
    o = cv2.imread(f"{d}/image.jpg")[:, :, ::-1].astype(np.float32) / 255
    k = 648 / o.shape[1]
    om = optical_mask(cv2.resize(o.max(2), (648, int(round(o.shape[0] * k))), interpolation=cv2.INTER_AREA))
    ocx, ocy, odia = geom(om)
    om_tile = square(om, ocx, ocy, CROP * odia) > 0.5
    o = np.dstack([square(o[..., c], ocx / k, ocy / k, CROP * odia / k, fill=None) for c in range(3)])
    out["optical"] = tone_optical(o, om_tile)
    return out, r


def main():
    droplets = [pick(x) for x in RECIPES]
    print(droplets)
    H = HEAD + len(ROWS) * (TILE + GAP)
    fig = figure(180, H)
    for j, did in enumerate(droplets):
        T, r = tiles(did)
        x = LABEL_W + j * (TILE + GAP)
        fig.text((x + TILE / 2) / 180, 1 - 0.4 / H, did.replace("_", " "), ha="center", va="top", fontsize=6, fontweight="bold")
        fig.text((x + TILE / 2) / 180, 1 - 3.6 / H, f"{r.NaHCO3_mM:g} · {r.CaCl2_mM:g} · {r.MgSO4_mM:g}", ha="center", va="top",
                 fontsize=6)
        for i, (lab, key) in enumerate(ROWS):
            ax = axes_mm(fig, x, HEAD + i * (TILE + GAP), TILE, TILE)
            ax.imshow(T[key], cmap="gray" if key == "sem" else None, vmin=0, vmax=1, interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if key == "sem":
                ax.text(0.04, 0.05, r.sem_signal, transform=ax.transAxes, fontsize=5.5, color="white", va="bottom",
                        bbox=dict(fc="black", ec="none", alpha=0.45, pad=0.6))
            if j == 0:
                ax.text(-0.07, 0.5, lab, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6.5,
                        fontweight="bold")
    save(fig, OUT)


if __name__ == "__main__":
    main()
