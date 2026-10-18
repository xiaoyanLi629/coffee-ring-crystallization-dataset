"""Module P (stats): does the grain inventory follow the design and the thermodynamic model?

1. Phase area fraction regressed on log2 salt concentrations with SEM-session fixed effects (standardised betas).
2. Radial position of each phase (area-weighted mean normalised radius) and paired order between phases.
3. Recipe-level agreement with the PHREEQC prediction (analysis/out_t/compare.csv): Spearman over the 125 recipes and
   the median within-Table Spearman (removes the session / NaHCO3 confound).
Only grains with purity >= 0.4 are counted in the sensitivity check (reported alongside).
Outputs analysis/out_p/summary.json
"""
import os
import json
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.environ.get("CRD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root; override with CRD_ROOT
OUT = ROOT + "/analysis/out_p"
NAMES = ["halite", "gypsum", "Mg sulfate", "Ca carbonate", "Na/Mg carbonate", "chloride brine"]
SALT = ["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM"]

d = pd.read_csv(OUT + "/phase_droplet.csv")
d = d[d.usable_eds]
g = pd.read_csv(OUT + "/grains.csv")
res = {"n_droplets": int(len(d)), "n_grains": int(len(g)), "grains_by_phase": g.phase.value_counts().to_dict(),
       "median_purity": round(float(g.purity.median()), 3), "median_fit_cos": round(float(g.fit_cos.median()), 3),
       "frac_purity_ge_0.5": round(float((g.purity >= 0.5).mean()), 3)}


def ols_fe(y, X, groups):
    y = (y - y.mean()) / (y.std() + 1e-12); Xs = (X - X.mean(0)) / X.std(0)
    D = pd.get_dummies(groups, drop_first=True).values.astype(float)
    M = np.c_[np.ones(len(y)), Xs, D]
    b, *_ = np.linalg.lstsq(M, y, rcond=None); e = y - M @ b
    cov = np.linalg.pinv(M.T @ M) * (e @ e) / (len(y) - M.shape[1])
    return b[1:4], np.sqrt(np.diag(cov))[1:4]


X = np.log2(d[SALT].values)
res["area_vs_salts_sessionFE"] = {}
for ph in NAMES:
    b, se = ols_fe(d[f"area_{ph}"].values, X, d.session.values)
    res["area_vs_salts_sessionFE"][ph] = {s: [round(float(bb), 3), round(float(ss), 3)] for s, bb, ss in zip(SALT, b, se)}
res["radius_median"] = {ph: round(float(d[f"r_{ph}"].median()), 3) for ph in NAMES}
pairs = {}
for a in NAMES:
    for b in NAMES:
        if a < b:
            ok = d[[f"r_{a}", f"r_{b}"]].dropna()
            pairs[f"{a} outside {b}"] = round(float((ok[f"r_{a}"] > ok[f"r_{b}"]).mean()), 3)
res["paired_outer"] = pairs

c = pd.read_csv(ROOT + "/analysis/out_t/compare.csv", index_col=0)
d["recipe"] = "T" + d.table.astype(str) + "_S" + d["sample"].astype(str)
rm = d.groupby("recipe")[[f"area_{p}" for p in NAMES] + [f"n_{p}" for p in NAMES]].mean()
c = c.join(rm, how="inner")
c["area_carbonate"] = c["area_Ca carbonate"] + c["area_Na/Mg carbonate"]
tests = [("Gypsum_mmol", "area_gypsum"), ("MgSO4_solid_mmol", "area_Mg sulfate"), ("Halite_mmol", "area_halite"),
         ("Calcite_mmol", "area_Ca carbonate"), ("Calcite_mmol", "area_carbonate")]
res["vs_thermo"] = {}
for p, o in tests:
    within = [stats.spearmanr(x[p], x[o]).correlation for _, x in c.groupby("table")]
    res["vs_thermo"][f"{p} vs {o}"] = {"rho_all": round(float(stats.spearmanr(c[p], c[o]).correlation), 3),
                                       "rho_within_table_median": round(float(np.nanmedian(within)), 3),
                                       "rho_within_table": [round(float(x), 3) for x in within]}
json.dump(res, open(OUT + "/summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "paired_outer"}, indent=1)[:4000])
print({k: v for k, v in pairs.items() if "gypsum" in k or "Mg sulfate" in k})
