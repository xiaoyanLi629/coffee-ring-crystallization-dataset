"""Pilot per-droplet features from the raw files (experiment/data).

EDS pseudo-colour JPEG -> scalar intensity (max over RGB) -> common 252x189 grid.
Droplet mask from the smoothed sum of O, Na, Cl, Ca, S (Mg and C excluded: substrate/background).
Outputs pilot/features.csv and pilot/radial.npz under the dataset project.
"""
import os, re, glob, json
import numpy as np, pandas as pd, cv2
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from multiprocessing import Pool

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
RAW = ROOT + "/experiment/data"
OUT = ROOT + "/pilot"
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
MASK_EL = ["O", "Na", "Cl", "Ca", "S"]
W, H = 252, 189
NB = 10  # radial bins


def read_semtxt(p):
    kv = {}
    for line in open(p, errors="ignore"):
        m = re.match(r"\$+(\S+)\s*(.*)", line.strip())
        if m:
            kv[m.group(1)] = m.group(2)
    return kv


def eds_intensity(p):
    a = cv2.imread(p, cv2.IMREAD_COLOR)
    if a is None:
        return None, None
    shape = a.shape[:2]
    a = a.max(2).astype(np.float32) / 255.0
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA), shape


def droplet_mask(maps):
    s = sum(ndi.gaussian_filter(maps[e], 3) / (maps[e].mean() + 1e-6) for e in MASK_EL)
    t = threshold_otsu(s)
    m = s > t
    m = ndi.binary_closing(m, iterations=3)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (np.argmax(sizes) + 1)
    return ndi.binary_fill_holes(m)


def one(rep):
    d = os.path.join(RAW, rep)
    t, s, r = rep.split("/")
    row = {"rep": rep, "table": int(t.split("_")[1]), "sample": int(s.split("_")[1]), "replicate": int(r.split("_")[0])}
    kv = read_semtxt(os.path.join(d, "sem.txt"))
    row.update(sem_signal=kv.get("CM_SIGNAL"), sem_mag=float(kv.get("CM_MAG", "nan")), sem_kv=float(kv.get("CM_ACCEL_VOLT", "nan")),
               sem_date=kv.get("CM_DATE"), sem_time=kv.get("CM_TIME"))
    maps, shapes = {}, {}
    for e in EL:
        maps[e], shapes[e] = eds_intensity(os.path.join(d, e + ".jpeg"))
    row["eds_w"], row["eds_h"] = shapes["Ca"][1], shapes["Ca"][0]
    m = droplet_mask(maps)
    row["mask_frac"] = float(m.mean())
    dist = ndi.distance_transform_edt(m)
    rnorm = np.where(m, 1 - dist / (dist.max() + 1e-6), np.nan)  # 0 centre, 1 edge
    ys, xs = np.nonzero(m)
    row["drop_cx"], row["drop_cy"] = xs.mean() / W, ys.mean() / H
    # equivalent diameter as fraction of field width, normalised to 50x
    row["drop_d_frac"] = float(np.sqrt(4 * m.sum() / np.pi) / W)
    row["drop_d_norm50"] = row["drop_d_frac"] * 50.0 / row["sem_mag"]
    if len(xs) > 10:
        cov = np.cov(np.vstack([xs, ys]))
        ev = np.sort(np.linalg.eigvalsh(cov))
        row["drop_ecc"] = float(np.sqrt(1 - ev[0] / ev[1]))
    out = ~ndi.binary_dilation(m, iterations=6)
    radial = np.full((len(EL), NB), np.nan, np.float32)
    for k, e in enumerate(EL):
        a = maps[e]
        mi, mo, so = a[m].mean(), a[out].mean(), a[out].std() + 1e-6
        row[f"{e}_in"], row[f"{e}_out"], row[f"{e}_outsd"] = float(mi), float(mo), float(so)
        row[f"{e}_contrast"] = float(mi - mo)
        row[f"{e}_d"] = float((mi - mo) / so)
        row[f"{e}_total"] = float((a[m] - mo).sum())  # background-subtracted integrated signal (px units)
        for b in range(NB):
            sel = m & (rnorm >= b / NB) & (rnorm < (b + 1) / NB + (b == NB - 1) * 1e-3)
            if sel.any():
                radial[k, b] = a[sel].mean() - mo
        inner = np.nanmean(radial[k, :5]); rim = np.nanmean(radial[k, 8:])
        row[f"{e}_rim_ratio"] = float(rim / (abs(inner) + 1e-4))
    # SEM raw style + geometry
    sem = cv2.imread(os.path.join(d, "sem.jpg"), cv2.IMREAD_GRAYSCALE)
    row["sem_raw_shape"] = f"{sem.shape[1]}x{sem.shape[0]}"
    sm = cv2.resize(sem, (640, 480), interpolation=cv2.INTER_AREA).astype(np.float32)
    mm = cv2.resize(m.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST).astype(bool)
    oo = cv2.resize(out.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST).astype(bool)
    for q in [1, 5, 25, 50, 75, 95, 99]:
        row[f"semstyle_p{q}"] = float(np.percentile(sm, q))
    row["semstyle_mean"], row["semstyle_std"] = float(sm.mean()), float(sm.std())
    row["semstyle_out_mean"], row["semstyle_out_std"] = float(sm[oo].mean()), float(sm[oo].std())
    hp = sm - cv2.GaussianBlur(sm, (0, 0), 3)
    row["semstyle_out_hf"] = float(hp[oo].std())
    row["semstyle_lapvar"] = float(cv2.Laplacian(sm, cv2.CV_32F).var())
    row["semstyle_rowjump"] = float(np.abs(np.diff(sm.mean(1))).max())
    row["sem_in_mean"], row["sem_in_std"] = float(sm[mm].mean()), float(sm[mm].std())
    row["sem_contrast"] = row["sem_in_mean"] - row["semstyle_out_mean"]
    # optical
    im = np.ascontiguousarray(np.rot90(cv2.imread(os.path.join(d, "image.jpg"), cv2.IMREAD_COLOR), -1))  # back to the acquisition orientation
    row["opt_shape"] = f"{im.shape[1]}x{im.shape[0]}"
    io = cv2.resize(im, (648, 486), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = io.mean(2)
    row["opt_mean"], row["opt_std"] = float(g.mean()), float(g.std())
    row["opt_b"], row["opt_g"], row["opt_r"] = [float(x) for x in io.reshape(-1, 3).mean(0)]
    row["opt_p99"] = float(np.percentile(g, 99))
    return row, radial, m


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    reps = sorted(os.path.relpath(p, RAW) for p in glob.glob(RAW + "/Table_*/Sample_*/*_rep"))
    with Pool(48) as pool:
        res = pool.map(one, reps, chunksize=4)
    df = pd.DataFrame([r for r, _, _ in res])
    conc = pd.read_csv(ROOT + "/data/element_concentrations.csv").rename(columns={"Table": "table", "Sample_ID": "sample"})
    df = df.merge(conc, on=["table", "sample"], how="left")
    df.to_csv(OUT + "/features.csv", index=False)
    np.savez_compressed(OUT + "/radial.npz", radial=np.stack([r for _, r, _ in res]), reps=np.array(reps),
                        masks=np.stack([m for _, _, m in res]))
    print("done", df.shape)
