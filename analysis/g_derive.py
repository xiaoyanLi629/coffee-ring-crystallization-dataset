"""Module G (derive): cross-modal per-droplet quantities.

G0 file provenance: modification dates of image.jpg / sem.jpg / EDS maps, optical resolution; *_processed.jpeg vs raw.
G2 SEM grey level vs EDS composition (pixel level, inside the droplet, after the SEM-EDS registration of the modelling
   project): per droplet OLS of substrate-normalised SEM grey on the 7 standardised element maps, plus single-element
   correlations against a shifted control.
G5 optical image: droplet segmentation, geometry, and registration to the SEM image (scale from droplet size, rotation
   and mirror searched on gradient-magnitude images); control = the same search against another replicate of the recipe.
Outputs analysis/out_g/{derived.csv, fig_optical_qc.png}
"""
import os, json, datetime
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
OUT = ROOT + "/analysis/out_g"
os.makedirs(OUT, exist_ok=True)
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
H, W = 379, 505
OW = 648  # optical working width
CROP = 128
REG = json.load(open("/root/autodl-fs/coffee-ring-water-chemistry/data/derived/eds_registration.json"))
REG = {r["rep"]: r for r in REG["droplets"]}
MK = np.load(ROOT + "/derived/masks_eds.npz")
MASKS = dict(zip(MK["paths"], MK["masks"]))


def mdate(p):
    return datetime.datetime.fromtimestamp(os.stat(p).st_mtime).strftime("%Y-%m-%d")


def decode(p):
    a = cv2.imread(p).astype(np.float32).sum(2) / 765.0
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)


def plausible(m):
    if m[:3].any() or m[-3:].any() or m[:, :3].any() or m[:, -3:].any():  # droplets are never cut by the frame
        return False
    f = m.mean()
    if f < 0.03 or f > 0.75:
        return False
    ys, xs = np.nonzero(m)
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]))))
    return ev[1] / (ev[0] + 1e-9) < 2.6


def optical_mask(g):
    """dark-field optical image: droplet = region of elevated local brightness/texture"""
    t = ndi.gaussian_filter(np.abs(g - ndi.gaussian_filter(g, 2)), 6) + ndi.gaussian_filter(g, 6) * 0.5
    t = np.log(t + 1e-3)
    m = t > threshold_otsu(t)
    m = ndi.binary_closing(m, np.ones((5, 5)), iterations=4)
    lab, n = ndi.label(m)
    if not n:
        return None, "failed"
    big = lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)
    f = ndi.binary_fill_holes(big)
    if plausible(f):
        return f, "fill"
    edge = big & ~ndi.binary_erosion(big, iterations=2)
    edge[:6] = edge[-6:] = False; edge[:, :6] = edge[:, -6:] = False
    ys, xs = np.nonzero(edge)
    if len(xs) > 50:
        el = np.zeros(g.shape, np.uint8)
        cv2.ellipse(el, cv2.fitEllipse(np.c_[xs, ys].astype(np.float32)), 1, -1)
        if plausible(el.astype(bool)):
            return el.astype(bool), "ellipse"
    return f, "failed"


def geom(m):
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    return cx, cy, 2 * np.sqrt(m.sum() / np.pi)


def crop_norm(img, m, flip=False):
    """square crop centred on the droplet, side = 1.3 equivalent diameters, resized to CROP; gradient magnitude"""
    cx, cy, d = geom(m)
    half = 0.65 * d
    M = np.float32([[CROP / (2 * half), 0, CROP / 2 - cx * CROP / (2 * half)], [0, CROP / (2 * half), CROP / 2 - cy * CROP / (2 * half)]])
    c = cv2.warpAffine(img.astype(np.float32), M, (CROP, CROP), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
    if flip:
        c = c[:, ::-1]
    c = ndi.gaussian_filter(c, 1)
    gm = np.hypot(ndi.sobel(c, 0), ndi.sobel(c, 1))
    return np.log1p(gm / (np.median(gm) + 1e-6))


YY, XX = np.mgrid[:CROP, :CROP]
DISK = (XX - CROP / 2 + .5) ** 2 + (YY - CROP / 2 + .5) ** 2 < (CROP / 2 / 1.3 * 0.97) ** 2


def best_rot(a, b):
    """max NCC over rotations (3 deg) and mirror of a against b, inside the droplet disk"""
    bb = b[DISK]; bb = (bb - bb.mean()) / (bb.std() + 1e-6)
    best = (-1, 0, False)
    for fl in (False, True):
        aa0 = a[:, ::-1] if fl else a
        for ang in range(0, 360, 3):
            R = cv2.getRotationMatrix2D((CROP / 2 - .5, CROP / 2 - .5), ang, 1.0)
            r = cv2.warpAffine(aa0, R, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)[DISK]
            v = float(((r - r.mean()) / (r.std() + 1e-6) * bb).mean())
            if v > best[0]:
                best = (v, ang, fl)
    return best


def load_opt(path):
    o = cv2.imread(f"{ROOT}/experiment/data/{path}/image.jpg")
    h0, w0 = o.shape[:2]
    g = cv2.resize(o.max(2).astype(np.float32), (OW, int(round(OW * h0 / w0))), interpolation=cv2.INTER_AREA)
    return g, (w0, h0), o


def one(path):
    d = f"{ROOT}/experiment/data/{path}"
    row = {"path": path, "date_optical": mdate(d + "/image.jpg"), "date_sem": mdate(d + "/sem.jpg"),
           "date_eds": mdate(d + "/Ca.jpeg"), "date_processed": mdate(d + "/Ca_processed.jpeg") if os.path.exists(d + "/Ca_processed.jpeg") else ""}
    m = MASKS[path]
    # ---- G2 SEM grey vs composition
    sem = cv2.imread(d + "/sem.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_AREA)
    rg = REG.get(path)
    if rg is not None:
        u = rg["use"]
        M = np.float32([[u["s"], 0, u["tx"] / 2], [0, u["s"], u["ty"] / 2]])
        sem = cv2.warpAffine(sem, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        row["reg_ncc"] = rg["ncc_registered"]; row["reg_fallback"] = rg["fallback"]
    out = ~ndi.binary_dilation(m, iterations=8)
    inner = ndi.binary_erosion(m, iterations=4)
    med = np.median(sem[out]); iqr = np.subtract(*np.percentile(sem[out], [75, 25])) + 1e-3
    z = ndi.gaussian_filter((sem - med) / iqr, 2)
    maps = {e: ndi.gaussian_filter(decode(f"{d}/{e}.jpeg"), 2) for e in EL}
    if inner.sum() > 500:
        X = np.stack([(maps[e][inner] - maps[e][inner].mean()) / (maps[e][inner].std() + 1e-6) for e in EL], 1)
        y = z[inner]; y = (y - y.mean()) / (y.std() + 1e-6)
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(y)), X], y, rcond=None)
        row["sem_on_eds_R2"] = float(1 - ((y - np.c_[np.ones(len(y)), X] @ beta) ** 2).mean())
        for k, e in enumerate(EL):
            row[f"semb_{e}"] = float(beta[k + 1])
            r_obs = np.corrcoef(z[inner], maps[e][inner])[0, 1]
            ctrl = []
            for dy, dx in [(12, 0), (-12, 0), (0, 12), (0, -12)]:
                Bs = np.roll(np.roll(maps[e], dy, 0), dx, 1); mm = inner & np.roll(np.roll(inner, dy, 0), dx, 1)
                ctrl.append(np.corrcoef(z[mm], Bs[mm])[0, 1])
            row[f"semr_{e}"] = float(r_obs); row[f"semr_{e}_ctrl"] = float(np.mean(ctrl))
    # ---- G5 optical
    g, (w0, h0), o = load_opt(path)
    row.update(opt_w=w0, opt_h=h0, opt_mean_rgb=json.dumps(np.round(o.reshape(-1, 3).mean(0)[::-1], 2).tolist()))
    om, how = optical_mask(g)
    row["opt_mask_method"] = how
    if om is not None and how != "failed":
        cx, cy, dd = geom(om)
        row.update(opt_frac=float(om.mean()), opt_diam_px=float(dd * w0 / OW))
        oin = g[om]; oout = g[~ndi.binary_dilation(om, iterations=6)]
        dist = ndi.distance_transform_edt(om); r = 1 - dist / (dist.max() + 1e-6)
        so = oout.std() + 1e-6
        row.update(opt_in_d=float((oin.mean() - oout.mean()) / so), opt_rim_d=float((g[om & (r > .85)].mean() - oout.mean()) / so),
                   opt_centre_d=float((g[om & (r < .5)].mean() - oout.mean()) / so), opt_bright_frac=float((oin > oout.mean() + 4 * so).mean()),
                   opt_hf=float((g - ndi.gaussian_filter(g, 2))[om].std() / so))
        semfull = cv2.resize(cv2.imread(d + "/sem.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        a = crop_norm(g, om); b = crop_norm(semfull, m)
        v, ang, fl = best_rot(a, b)
        row.update(optreg_ncc=v, optreg_angle=ang, optreg_flip=fl, eds_diam_px_half=float(geom(m)[2]))
        t, s_, rr = path.split("/")
        alt = f"{t}/{s_}/{int(rr.split('_')[0]) % 5 + 1}_rep"
        if alt in MASKS:
            sa = cv2.resize(cv2.imread(f"{ROOT}/experiment/data/{alt}/sem.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
            row["optreg_ncc_ctrl"] = best_rot(a, crop_norm(sa, MASKS[alt]))[0]
    return row, (cv2.resize((om if om is not None else np.zeros_like(g, bool)).astype(np.uint8) * 255, (80, 60), interpolation=cv2.INTER_NEAREST),
                 cv2.resize(np.clip(g / (np.percentile(g, 99.5) + 1e-6) * 255, 0, 255).astype(np.uint8), (80, 60), interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    meta = pd.read_csv(ROOT + "/metadata/samples.csv")
    with Pool(48) as p:
        res = p.map(one, meta.path.tolist(), chunksize=2)
    f = meta.merge(pd.DataFrame([r[0] for r in res]), on="path")
    f.to_csv(OUT + "/derived.csv", index=False)
    tiles = []
    for mk, im in [r[1] for r in res]:
        t = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cnt, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cnt, -1, (0, 200, 255), 1)
        tiles.append(t)
    rows_ = [np.hstack(tiles[i:i + 25]) for i in range(0, len(tiles), 25)]
    cv2.imwrite(OUT + "/fig_optical_qc.png", np.vstack(rows_))
    print("G derive done", f.shape, f.opt_mask_method.value_counts().to_dict())
