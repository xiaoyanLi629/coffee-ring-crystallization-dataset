"""Derived per-droplet quantities for modules D, E, F (stretch-insensitive EDS measures).

EDS decode: (R+G+B)/765 per element (monotone, no saturation), common half grid 379x505.
Droplet mask: smoothed sum of normalised O/Na/Cl/Ca/S, Otsu, closing; convex hull when the ring is open.
Per droplet and element: d = (in-out)/sd_out, coverage = frac(in > out+2sd), mean radius <r> (bg-subtracted,
intensity weighted, 0 centre 1 edge), shape-normalised radial profile (12 bins), Gini of in-droplet signal.
Pairwise co-localisation: Pearson r of smoothed maps inside the eroded droplet, observed vs shifted control.
SEM (raw, substrate-normalised, per detector mode used downstream): radial profile and texture inside the droplet.
Outputs: derived/masks_eds.npz, analysis/out_def/{features.csv, radial.npz, coloc.npz}, analysis/out_def/fig_mask_qc.png
"""
import os, json, itertools
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
OUT = ROOT + "/analysis/out_def"
os.makedirs(OUT, exist_ok=True); os.makedirs(ROOT + "/derived", exist_ok=True)
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
MASK_EL = ["O", "Na", "Cl", "Ca", "S"]
H, W = 379, 505
NB = 12
PAIRS = list(itertools.combinations(range(7), 2))
SHIFTS = [(12, 0), (-12, 0), (0, 12), (0, -12)]  # half-grid px (~24 px on the EDS export grid)


def decode(p):
    a = cv2.imread(p).astype(np.float32).sum(2) / 765.0
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)


def plausible(m):
    f = m.mean()
    if f < 0.08 or f > 0.75:
        return False
    ys, xs = np.nonzero(m)
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]))))
    return ev[1] / (ev[0] + 1e-9) < 2.6  # axis ratio < ~1.6


def ellipse_mask(pts_mask):
    ys, xs = np.nonzero(pts_mask)
    if len(xs) < 50:
        return None
    (cx, cy), (a, b), ang = cv2.fitEllipse(np.c_[xs, ys].astype(np.float32))
    out = np.zeros((H, W), np.uint8)
    cv2.ellipse(out, ((cx, cy), (a, b), ang), 1, -1)
    return out.astype(bool)


def make_mask(maps, sem_small):
    s = sum(ndi.gaussian_filter(maps[e], 4) / (maps[e].mean() + 1e-6) for e in MASK_EL)
    t = threshold_otsu(s)
    m = s > t
    m = ndi.binary_closing(m, structure=np.ones((5, 5)), iterations=3)
    lab, n = ndi.label(m)
    if n:
        big = lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)
        filled = ndi.binary_fill_holes(big)
        if filled.sum() >= 0.6 * convex_hull_image(big).sum() and plausible(filled):
            return filled, "fill"
        # open ring: fit an ellipse to the boundary of all sizeable components
        sizes = ndi.sum(m, lab, range(1, n + 1))
        comps = np.isin(lab, np.where(sizes > 0.002 * m.size)[0] + 1)
        edge = comps & ~ndi.binary_erosion(comps, iterations=2)
        el = ellipse_mask(edge)
        if el is not None and plausible(el):
            return el, "ellipse"
    # weak EDS: segment the droplet on the SEM image (deviation from the border substrate)
    z = ndi.gaussian_filter(sem_small, 6)
    border = np.r_[z[:15].ravel(), z[-15:].ravel(), z[:, :15].ravel(), z[:, -15:].ravel()]
    dev = np.abs(z - np.median(border))
    mm = dev > threshold_otsu(dev)
    mm = ndi.binary_opening(mm, iterations=3)
    lab, n = ndi.label(mm)
    if n:
        big = ndi.binary_fill_holes(lab == (np.argmax(ndi.sum(mm, lab, range(1, n + 1))) + 1))
        if plausible(big):
            return big, "sem"
        el = ellipse_mask(big & ~ndi.binary_erosion(big, iterations=2))
        if el is not None and plausible(el):
            return el, "sem_ellipse"
    return (m if m.any() else np.zeros((H, W), bool)), "failed"


def one(path):
    d = f"{ROOT}/experiment/data/{path}"
    maps = {e: decode(f"{d}/{e}.jpeg") for e in EL}
    sem = cv2.imread(f"{d}/sem.jpg", cv2.IMREAD_GRAYSCALE)
    sem_small = cv2.resize(sem, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    m, how = make_mask(maps, sem_small)
    row = {"path": path, "mask_method": how, "mask_frac": float(m.mean())}
    dist = ndi.distance_transform_edt(m)
    r = 1 - dist / (dist.max() + 1e-6)
    out = ~ndi.binary_dilation(m, iterations=8)
    inner = ndi.binary_erosion(m, iterations=4)
    radial = np.full((7, NB), np.nan, np.float32)
    sm = {}
    for k, e in enumerate(EL):
        a = maps[e]
        mo, so = a[out].mean(), a[out].std() + 1e-6
        ain = a[m]
        row[f"{e}_d"] = float((ain.mean() - mo) / so)
        row[f"{e}_cov"] = float((ain > mo + 2 * so).mean())
        row[f"{e}_contrast"] = float(ain.mean() - mo)
        w = np.clip(ain - mo, 0, None)
        row[f"{e}_rbar"] = float((w * r[m]).sum() / (w.sum() + 1e-9))
        ws = np.sort(w); n = len(ws)
        row[f"{e}_gini"] = float((2 * np.arange(1, n + 1) - n - 1).dot(ws) / (n * ws.sum() + 1e-9))
        for b in range(NB):
            sel = m & (r >= b / NB) & (r < (b + 1) / NB + (b == NB - 1) * 1e-3)
            radial[k, b] = a[sel].mean() - mo if sel.any() else np.nan
        sm[e] = ndi.gaussian_filter(a, 2)
    radial_shape = radial / (np.nanmean(np.abs(radial), 1, keepdims=True) + 1e-9)
    for k, e in enumerate(EL):
        row[f"{e}_rim_index"] = float(np.nanmean(radial[k, -2:]) / (np.nanmean(radial[k, :6]) + 1e-4)) if np.nanmean(radial[k, :6]) > 1e-4 else np.nan
    coloc = np.full((len(PAIRS), 2), np.nan, np.float32)
    for j, (a_, b_) in enumerate(PAIRS):
        A, B = sm[EL[a_]], sm[EL[b_]]
        coloc[j, 0] = np.corrcoef(A[inner], B[inner])[0, 1]
        ctrl = []
        for dy, dx in SHIFTS:
            Bs = np.roll(np.roll(B, dy, 0), dx, 1)
            mm = inner & np.roll(np.roll(inner, dy, 0), dx, 1)
            ctrl.append(np.corrcoef(A[mm], Bs[mm])[0, 1])
        coloc[j, 1] = np.mean(ctrl)
    # SEM inside droplet, substrate-normalised
    s = sem_small
    med = np.median(s[out]); iqr = np.subtract(*np.percentile(s[out], [75, 25])) + 1e-3
    z = (s - med) / iqr
    hp = z - ndi.gaussian_filter(z, 3)
    row.update(sem_z_in_mean=float(z[m].mean()), sem_z_in_sd=float(z[m].std()), sem_hf_in=float(hp[m].std()),
               sem_hf_rim=float(hp[m & (r > .85)].std()), sem_hf_centre=float(hp[m & (r < .5)].std()),
               sem_bright_frac=float((z[m] > 2).mean()), sem_dark_frac=float((z[m] < -2).mean()))
    sem_rad = np.array([z[m & (r >= b / NB) & (r < (b + 1) / NB + (b == NB - 1) * 1e-3)].mean() for b in range(NB)], np.float32)
    # rim width proxy: fraction of radius over which the SEM rim deviates > 1 IQR from the droplet interior median
    interior = np.median(z[m & (r < .5)])
    dev = np.abs(sem_rad - interior)
    row["sem_rim_width"] = float((dev > 1.0).sum() / NB)
    return row, radial, radial_shape, coloc, sem_rad, m


if __name__ == "__main__":
    meta = pd.read_csv(ROOT + "/metadata/samples.csv")
    paths = meta.path.tolist()
    with Pool(48) as p:
        res = p.map(one, paths, chunksize=4)
    f = pd.DataFrame([r[0] for r in res])
    f = meta.merge(f, on="path")
    f.to_csv(OUT + "/features.csv", index=False)
    np.savez_compressed(OUT + "/radial.npz", paths=np.array(paths), radial=np.stack([r[1] for r in res]),
                        radial_shape=np.stack([r[2] for r in res]), sem_radial=np.stack([r[4] for r in res]))
    np.savez_compressed(OUT + "/coloc.npz", paths=np.array(paths), pairs=np.array(PAIRS), coloc=np.stack([r[3] for r in res]))
    masks = np.stack([r[5] for r in res])
    np.savez_compressed(ROOT + "/derived/masks_eds.npz", paths=np.array(paths), masks=masks, grid=np.array([H, W]))
    # QC montage: all 625 masks, 25 x 25 tiles
    tiles = [cv2.resize(mk.astype(np.uint8) * 255, (80, 60), interpolation=cv2.INTER_NEAREST) for mk in masks]
    rows_ = [np.hstack(tiles[i:i + 25] + [np.zeros((60, 80), np.uint8)] * (25 - len(tiles[i:i + 25]))) for i in range(0, len(tiles), 25)]
    mont = np.vstack(rows_)
    for i in range(1, 25):
        mont[:, i * 80] = 128; mont[i * 60 if i * 60 < mont.shape[0] else -1, :] = 128
    cv2.imwrite(OUT + "/fig_mask_qc.png", mont)
    print("derive done", f.shape, f.mask_method.value_counts().to_dict())
