"""Compare the PHREEQC evaporation model with what the EDS maps show, recipe by recipe.

Predicted (analysis/out_t/onset.csv, at the drying-humidity endpoint a_w = 0.475 or the end of the converged path):
  moles of each solid phase (primary tests), fraction of Ca as calcite / gypsum, fraction of Mg in solid Mg salts, gypsum onset concentration factor.
Observed (analysis/out_def, droplet values averaged over replicates, unusable droplets excluded):
  co-localisation excess of Ca-S, C-Ca, C-O, Na-Cl, Mg-S; mean radius of Ca, S, Mg; contrast-to-noise d.
Agreement is reported as Spearman correlation over the 125 recipes and, to remove the session/NaHCO3 confound, as the
median Spearman correlation within each Table (25 recipes imaged in one or two sessions).
Outputs analysis/out_t/{compare.csv, compare_summary.json}
"""
import json, itertools
import numpy as np, pandas as pd
from scipy import stats

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
OUT = ROOT + "/analysis/out_t"
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
PAIRS = list(itertools.combinations(range(7), 2))

o = pd.read_csv(OUT + "/onset.csv")
f = pd.read_csv(ROOT + "/analysis/out_def/features.csv")
f = f[f.usable_eds & (f.mask_method != "failed")]
C = np.load(ROOT + "/analysis/out_def/coloc.npz")
ci = {p: i for i, p in enumerate(C["paths"])}
for j, (a, b) in enumerate(PAIRS):
    f[f"coloc_{EL[a]}-{EL[b]}"] = [C["coloc"][ci[p]][j, 0] - C["coloc"][ci[p]][j, 1] for p in f.path]
f["recipe"] = "T" + f.table.astype(str) + "_S" + f["sample"].astype(str)
obs_cols = ["coloc_Ca-S", "coloc_C-Ca", "coloc_C-O", "coloc_Cl-Na", "coloc_Mg-S", "Ca_rbar", "S_rbar", "Mg_rbar",
            "C_rbar", "Na_rbar", "Ca_rim_index", "S_rim_index", "Mg_d", "S_d", "Ca_d", "C_d"]
obs = f.groupby("recipe").agg({**{c: "mean" for c in obs_cols}, "table": "first", "NaHCO3_mM": "first", "CaCl2_mM": "first", "MgSO4_mM": "first", "Ca_mM": "first", "Mg_mM": "first"})

o = o.set_index("recipe")
mol = lambda m: o.get(f"{m}_mol_at_aw0475", pd.Series(0.0, index=o.index)).fillna(0)
pred = pd.DataFrame(index=o.index)
pred["f_Ca_calcite"] = mol("Calcite") / (obs.Ca_mM / 1000)
pred["f_Ca_gypsum"] = (mol("Gypsum") + mol("Glauberite")) / (obs.Ca_mM / 1000)
pred["f_Mg_solid"] = (mol("Epsomite") + mol("Hexahydrite") + mol("Kieserite") + mol("Pentahydrite") + mol("Bloedite")
                      + mol("Nesquehonite") + mol("Bischofite")) / (obs.Mg_mM / 1000)
pred["gypsum_onset_log10cf"] = np.log10(o["Gypsum_onset_cf"])
pred["calcite_to_gypsum"] = np.log10((mol("Calcite") + 1e-9) / (mol("Gypsum") + 1e-9))
for m_ in ("Calcite", "Gypsum", "Halite"):
    pred[f"{m_}_mmol"] = mol(m_) * 1000
pred["MgSO4_solid_mmol"] = (mol("Epsomite") + mol("Hexahydrite") + mol("Kieserite") + mol("Pentahydrite") + mol("Bloedite")) * 1000
pred["reached_aw0475"] = o["aw_end"] < 0.475
d = obs.join(pred, how="inner")
d.to_csv(OUT + "/compare.csv")

TESTS = [("Gypsum_mmol", "coloc_Ca-S", +1, "more gypsum -> stronger Ca-S co-localisation"),
         ("Calcite_mmol", "coloc_C-Ca", +1, "more calcite -> stronger C-Ca co-localisation"),
         ("Halite_mmol", "coloc_Cl-Na", +1, "more halite -> stronger Na-Cl co-localisation"),
         ("MgSO4_solid_mmol", "coloc_Mg-S", +1, "more solid Mg sulfate -> Mg-S co-localisation"),
         ("MgSO4_solid_mmol", "Mg_d", +1, "more solid Mg sulfate -> stronger Mg signal"),("f_Ca_gypsum", "coloc_Ca-S", +1, "more Ca as gypsum -> stronger Ca-S co-localisation"),
         ("f_Ca_calcite", "coloc_C-Ca", +1, "more Ca as calcite -> stronger C-Ca co-localisation"),
         ("f_Ca_calcite", "coloc_C-O", +1, "more carbonate solid -> stronger C-O co-localisation"),
         ("calcite_to_gypsum", "coloc_Ca-S", -1, "calcite:gypsum ratio up -> Ca-S co-localisation down"),
         ("gypsum_onset_log10cf", "S_rbar", -1, "earlier gypsum saturation -> S closer to the rim"),
         ("gypsum_onset_log10cf", "S_rim_index", -1, "earlier gypsum saturation -> stronger S rim enrichment"),
         ("f_Mg_solid", "coloc_Mg-S", +1, "more Mg sulfate solid -> Mg-S co-localisation"),
         ("f_Mg_solid", "Mg_rbar", +1, "more solid Mg salt -> Mg less confined to the centre")]
res = {"n_recipes": int(len(d)), "frac_paths_reaching_aw0475": round(float(d.reached_aw0475.mean()), 3), "tests": []}
for p, ob, sign, text in TESTS:
    ok = d[[p, ob]].replace([np.inf, -np.inf], np.nan).dropna()
    rho_all = stats.spearmanr(ok[p], ok[ob])
    within = [stats.spearmanr(g[p], g[ob]).correlation for _, g in d.loc[ok.index].groupby("table") if g[p].nunique() > 2]
    res["tests"].append({"predicted": p, "observed": ob, "expected_sign": sign, "hypothesis": text,
                         "rho_all": round(float(rho_all.correlation), 3), "p_all": float(rho_all.pvalue),
                         "rho_within_table_median": round(float(np.nanmedian(within)), 3),
                         "rho_within_table": [round(float(x), 3) for x in within],
                         "agrees": bool(np.sign(np.nanmedian(within)) == sign)})
res["pred_by_NaHCO3"] = d.groupby("NaHCO3_mM")[["f_Ca_calcite", "f_Ca_gypsum", "f_Mg_solid"]].mean().round(3).to_dict()
res["pred_by_MgSO4"] = d.groupby("MgSO4_mM")[["f_Ca_calcite", "f_Ca_gypsum", "f_Mg_solid", "gypsum_onset_log10cf"]].mean().round(3).to_dict()
json.dump(res, open(OUT + "/compare_summary.json", "w"), indent=1)
for t in res["tests"]:
    print(f"{t['predicted']:>22} vs {t['observed']:<14} all {t['rho_all']:+.2f}  within-table median {t['rho_within_table_median']:+.2f} {t['rho_within_table']}  agrees={t['agrees']}")
