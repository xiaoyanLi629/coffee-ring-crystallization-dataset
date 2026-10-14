"""Module P: crystal (grain) inventory from the EDS maps, with phases assigned by stoichiometric templates.

Single EDS pixels are too noisy to show which elements occur together, so composition is read at the scale of grains.
Each map is decoded ((R+G+B)/765, common 379x505 grid), smoothed (sigma 1.5 px) and expressed relative to the substrate
of the same map, z = (I - mean_out) / sd_out (removes the per-map stretch); each element is then divided by its 99th
percentile over in-droplet pixels of all droplets so that elements span comparable ranges. Grains are connected regions
inside the droplet where the summed enrichment exceeds a fixed threshold (>= 4 px). The mean element vector of each
grain is decomposed by non-negative least squares onto unit-norm templates of the expected phases:
  halite Na+Cl; gypsum Ca+S+O; Mg sulfate Mg+S+O; Ca carbonate Ca+C+O; Na/Mg carbonate C+O; chloride brine Cl+Mg+Ca.
The grain is assigned to the template with the largest coefficient; its purity is that coefficient's share and the fit
quality is the cosine similarity between the grain vector and its reconstruction.
Outputs analysis/out_p/{grains.csv, phase_droplet.csv, element_scale.npy, phase_maps.npz}
"""
import os
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from scipy import ndimage as ndi
from scipy.optimize import nnls

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
OUT = ROOT + "/analysis/out_p"
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
H, W = 379, 505
MK = np.load(ROOT + "/derived/masks_eds.npz")
MASKS = dict(zip(MK["paths"], MK["masks"]))
PHASES = {"halite": ["Na", "Cl"], "gypsum": ["Ca", "S", "O"], "Mg sulfate": ["Mg", "S", "O"],
          "Ca carbonate": ["Ca", "C", "O"], "Na/Mg carbonate": ["C", "O"], "chloride brine": ["Cl", "Mg", "Ca"]}
NAMES = list(PHASES)
T = np.array([[1.0 if e in v else 0.0 for e in EL] for v in PHASES.values()])
T = T / np.linalg.norm(T, axis=1, keepdims=True)
THR = 2.0  # summed scaled enrichment that defines a grain


def zmaps(path):
    m = MASKS[path]
    out = ~ndi.binary_dilation(m, iterations=8)
    Z = []
    for e in EL:
        a = cv2.imread(f"{ROOT}/experiment/data/{path}/{e}.jpeg").astype(np.float32).sum(2) / 765
        a = ndi.gaussian_filter(cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA), 1.5)
        Z.append((a - a[out].mean()) / (a[out].std() + 1e-6))
    return np.clip(np.stack(Z, -1), 0, None), m


def inside_sample(path, n=3000):
    Z, m = zmaps(path)
    v = Z[ndi.binary_erosion(m, iterations=2)]
    rng = np.random.default_rng(abs(hash(path)) % 2 ** 32)
    return v[rng.choice(len(v), min(n, len(v)), replace=False)]


def grains(args):
    path, scale = args
    Z, m = zmaps(path)
    Zs = Z / scale
    inner = ndi.binary_erosion(m, iterations=2)
    S = Zs.sum(-1)
    cc, n = ndi.label(inner & (S > THR))
    dist = ndi.distance_transform_edt(m); r = 1 - dist / (dist.max() + 1e-6)
    lab = np.full((H, W), -1, np.int8)
    rows = []
    for k in range(1, n + 1):
        sel = cc == k
        size = int(sel.sum())
        if size < 4:
            continue
        v = Zs[sel].mean(0)
        coef, _ = nnls(T.T, v)
        j = int(np.argmax(coef))
        rec = T.T @ coef
        lab[sel] = j
        rows.append({"path": path, "grain": k, "size_px": size, "r": float(r[sel].mean()), "phase": NAMES[j],
                     "purity": float(coef[j] / (coef.sum() + 1e-9)), "fit_cos": float(v @ rec / (np.linalg.norm(v) * np.linalg.norm(rec) + 1e-9)),
                     **{f"z_{e}": float(v[i]) for i, e in enumerate(EL)}})
    return rows, lab, float(m.sum())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    meta = pd.read_csv(ROOT + "/metadata/samples.csv")
    f = pd.read_csv(ROOT + "/analysis/out_def/features.csv")
    keep = f[f.usable_eds & (f.mask_method != "failed")].path.tolist()
    with Pool(48) as p:
        X = np.concatenate(p.map(inside_sample, keep, chunksize=4))
    scale = np.percentile(X, 99, axis=0) + 1e-6
    np.save(OUT + "/element_scale.npy", scale)
    with Pool(48) as p:
        res = p.map(grains, [(pth, scale) for pth in keep], chunksize=4)
    g = pd.DataFrame([row for r in res for row in r[0]])
    g.to_csv(OUT + "/grains.csv", index=False)
    area = {pth: r[2] for pth, r in zip(keep, res)}
    rows = []
    for pth in keep:
        d = g[g.path == pth]
        row = {"path": pth, "n_grains": len(d)}
        for ph in NAMES:
            dd = d[d.phase == ph]
            row[f"n_{ph}"] = len(dd)
            row[f"area_{ph}"] = float(dd.size_px.sum() / area[pth])
            row[f"r_{ph}"] = float(np.average(dd.r, weights=dd.size_px)) if len(dd) else np.nan
            row[f"size_{ph}"] = float(dd.size_px.median()) if len(dd) else np.nan
        rows.append(row)
    meta.merge(pd.DataFrame(rows), on="path").to_csv(OUT + "/phase_droplet.csv", index=False)
    np.savez_compressed(OUT + "/phase_maps.npz", paths=np.array(keep), labels=np.stack([r[1] for r in res]), names=np.array(NAMES))
    print("grains done", len(g), "grains;", g.phase.value_counts().to_dict())
    print("median purity", g.purity.median().round(3), "median fit cos", g.fit_cos.median().round(3))
