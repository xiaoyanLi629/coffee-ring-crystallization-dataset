"""Module C: what the EDS pseudo-colour JPEGs can be used for.

C1 colour map: is each element map a single hue scaled from black (chroma constant with intensity)?
C2 quantisation comb: period of the 8-bit histogram -> number of count levels per map (proxy of max counts)
C3 per-map stretching: do the brightest pixels stay fixed while concentration changes?
C4 background / SNR per element and per session
C5 Table 5 EDS export sizes vs field of view (registration scale) and intensity
Outputs analysis/out_c/.
"""
import os, json, glob
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from scipy import stats, signal

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
OUT = ROOT + "/analysis/out_c"
os.makedirs(OUT, exist_ok=True)
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
OWN = {"C": "NaHCO3_mM", "Na": "NaHCO3_mM", "Ca": "CaCl2_mM", "Cl": "CaCl2_mM", "Mg": "MgSO4_mM", "S": "MgSO4_mM", "O": "MgSO4_mM"}
meta = pd.read_csv(ROOT + "/metadata/samples.csv")
pil = pd.read_csv(ROOT + "/pilot/features.csv")
pil["droplet_id"] = [f"T{a}_S{b}_R{c}" for a, b, c in zip(pil.table, pil["sample"], pil.replicate)]
df = meta.merge(pil[["droplet_id"] + [c for c in pil.columns if c.split("_")[0] in EL and c.split("_")[-1] in ("contrast", "total", "out", "outsd", "d", "in")] + ["mask_frac", "drop_d_frac"]], on="droplet_id")
df["recipe"] = df.table.astype(str) + "_" + df["sample"].astype(str)


def comb_period(h):
    """dominant period (in grey levels) of the fine structure of a histogram over levels 3..150"""
    lo, hi = 3, 150
    x = np.log(h[lo:hi].astype(float) + 1)
    trend = signal.savgol_filter(x, 21, 2)
    r = x - trend
    if r.std() < 1e-6:
        return np.nan, 0.0
    f, p = signal.periodogram(r, detrend="linear")
    sel = (f > 1 / 40) & (f < 1 / 2.05)
    if not sel.any():
        return np.nan, 0.0
    k = np.argmax(p[sel])
    peak = p[sel][k]
    return float(1 / f[sel][k]), float(peak / (np.median(p[sel]) + 1e-12))


def per_map(path):
    rows = []
    for e in EL:
        a = cv2.imread(f"{ROOT}/experiment/data/{path}/{e}.jpeg")[:, :, ::-1].astype(np.float32)  # RGB
        mx = a.max(2)
        sel = mx > 40
        px = a[sel]
        chroma = (px / px.max(1, keepdims=True)).mean(0) if len(px) else np.full(3, np.nan)
        # chroma at low vs high intensity: constant -> single hue ramp; drifting to white -> saturating LUT
        lo = a[(mx > 40) & (mx < 100)]; hi = a[mx > 180]
        clo = (lo / lo.max(1, keepdims=True)).mean(0) if len(lo) > 50 else np.full(3, np.nan)
        chi = (hi / hi.max(1, keepdims=True)).mean(0) if len(hi) > 50 else np.full(3, np.nan)
        h = np.bincount(mx.astype(np.uint8).ravel(), minlength=256)
        per, strength = comb_period(h)
        rows.append(dict(path=path, element=e, max=float(mx.max()), p999=float(np.percentile(mx, 99.9)),
                         p99=float(np.percentile(mx, 99)), mean=float(mx.mean()), frac_255=float((mx >= 254).mean()),
                         chroma_r=chroma[0], chroma_g=chroma[1], chroma_b=chroma[2],
                         chroma_lo=json.dumps(np.round(clo, 3).tolist()), chroma_hi=json.dumps(np.round(chi, 3).tolist()),
                         comb_period=per, comb_strength=strength, n_levels=int((h > 0).sum())))
    return rows


with Pool(48) as p:
    rows = sum(p.map(per_map, df.path.tolist(), chunksize=4), [])
mp = pd.DataFrame(rows).merge(df[["path", "droplet_id", "table", "session", "sem_signal", "recipe", "usable_eds", "eds_export",
                                   "NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"]], on="path")
mp.to_csv(OUT + "/per_map.csv", index=False)
mp = mp[mp.usable_eds]
res = {}

# C1
c1 = {}
for e, g in mp.groupby("element"):
    lo = np.array([json.loads(x) for x in g.chroma_lo]); hi = np.array([json.loads(x) for x in g.chroma_hi])
    c1[e] = {"chroma_mean": np.round(g[["chroma_r", "chroma_g", "chroma_b"]].mean().values, 3).tolist(),
             "chroma_low_intensity": np.round(np.nanmean(lo, 0), 3).tolist(), "chroma_high_intensity": np.round(np.nanmean(hi, 0), 3).tolist()}
res["C1_colour"] = c1

# C2 comb period -> levels
mp["comb_levels"] = 255.0 / mp.comb_period
c2 = {}
for e, g in mp.groupby("element"):
    strong = g.comb_strength > 8
    gg = g[strong]
    c2[e] = {"frac_maps_with_clear_comb": round(float(strong.mean()), 3),
             "period_median_IQR": np.round([gg.comb_period.median(), gg.comb_period.quantile(.25), gg.comb_period.quantile(.75)], 2).tolist() if len(gg) else None,
             "spearman_levels_vs_own_salt": round(float(stats.spearmanr(gg.comb_levels, gg[OWN[e]]).correlation), 3) if len(gg) > 20 else None,
             "spearman_levels_vs_mean_intensity": round(float(stats.spearmanr(gg.comb_levels, gg["mean"]).correlation), 3) if len(gg) > 20 else None}
res["C2_comb"] = c2

# C3 stretching: spearman of brightest pixels with own salt vs of mean intensity with own salt; CV of p99.9
c3 = {}
for e, g in mp.groupby("element"):
    c3[e] = {"max_median": float(g["max"].median()), "frac_maps_max>=250": round(float((g["max"] >= 250).mean()), 3),
             "p999_cv": round(float(g.p999.std() / g.p999.mean()), 3), "mean_cv": round(float(g["mean"].std() / g["mean"].mean()), 3),
             "rho_p999_own": round(float(stats.spearmanr(g.p999, g[OWN[e]]).correlation), 3),
             "rho_mean_own": round(float(stats.spearmanr(g["mean"], g[OWN[e]]).correlation), 3)}
res["C3_stretch"] = c3

# C4 background / SNR by element and session
d = df[df.usable_eds]
c4 = {}
for e in EL:
    c4[e] = {"detectable_frac(d>1)": round(float((d[f"{e}_d"] > 1).mean()), 3),
             "d_median": round(float(d[f"{e}_d"].median()), 2),
             "background_by_session": d.groupby("session")[f"{e}_out"].median().round(4).to_dict(),
             "background_CV_across_sessions": round(float(d.groupby("session")[f"{e}_out"].median().std() / d.groupby("session")[f"{e}_out"].median().mean()), 3)}
res["C4_background"] = c4

# C5 Table 5 export sizes
reg = json.load(open("/root/autodl-fs/coffee-ring-water-chemistry/data/derived/eds_registration.json"))
rg = pd.DataFrame([{"path": r["rep"], "reg_s": r["s"], "reg_ncc": r["ncc_registered"], "reg_fallback": r["fallback"]} for r in reg["droplets"]])
t5 = df[df.table == 5].merge(rg, on="path")
c5 = {}
for ex, g in t5.groupby("eds_export"):
    c5[ex] = {"n": int(len(g)), "sessions": g.session.value_counts().to_dict(), "reg_scale_median": round(float(g.reg_s.median()), 4),
              "eds_mask_frac_median": round(float(g.mask_frac.median()), 3), "drop_d_frac_median": round(float(g.drop_d_frac.median()), 3),
              **{f"{e}_out_median": round(float(g[f"{e}_out"].median()), 4) for e in EL},
              **{f"{e}_contrast_median": round(float(g[f"{e}_contrast"].median()), 4) for e in EL}}
res["C5_table5_exports"] = c5
allreg = df.merge(rg, on="path")
res["C5_reg_scale_by_export"] = allreg.groupby("eds_export").reg_s.median().round(4).to_dict()

json.dump(res, open(OUT + "/summary.json", "w"), indent=1, default=float)
print("C done")
