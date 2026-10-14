"""Thermodynamic model of droplet evaporation (PHREEQC, Pitzer database, 25 C, open to atmospheric CO2).

For each of the 125 recipes, water is removed stepwise down to 1e-4 of its initial mass while the solution stays in
equilibrium with CO2(g) (log pCO2 = -3.4) and with any mineral that becomes supersaturated (kinetically inhibited
dolomite, magnesite, aragonite and anhydrite are excluded). For every mineral we record the concentration factor at which it first
precipitates and the moles precipitated at the end. Because the solution at the contact line reaches a given
concentration factor first, earlier-precipitating phases are expected closer to the rim.
At the drying relative humidity (45-50 %) a residue stops losing water once the water activity of the remaining brine
falls to about 0.475; phases that would only form below that activity stay dissolved in a deliquescent film. Onsets are
therefore reported both for the full path and for the path truncated at a_w = 0.475.
Outputs analysis/out_t/{evap_paths.csv, onset.csv}
"""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
OUT = ROOT + "/analysis/out_t"
DB = "/root/autodl-tmp/phreeqc_db/pitzer.dat"
LIB = "/root/autodl-tmp/coffee-env/lib/python3.10/site-packages/phreeqpy/iphreeqc/phreeqc3/libiphreeqc-3.7.3.so"
MINERALS = ["Calcite", "Gypsum", "Halite", "Epsomite", "Hexahydrite", "Kieserite", "Bischofite",
            "Nahcolite", "Natron", "Trona", "Thenardite", "Mirabilite", "Glauberite", "Bloedite", "Nesquehonite",
            "Gaylussite", "Pirssonite", "Brucite", "Antarcticite", "Tachyhydrite", "Carnallite", "Leonhardtite", "Pentahydrite"]


def phreeqc():
    from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc
    p = IPhreeqc(LIB)
    p.load_database(DB)
    return p


def available(p):
    txt = open(DB, encoding="latin-1").read().split("PHASES")[1].split("\nEND")[0]
    return [m for m in MINERALS if f"\n{m}\n" in txt]


def run(rec):
    """full path to W_LEFT[-1]; if PHREEQC fails to converge near dryness, retry with a shorter path"""
    for end in (-3.9, -3.8, -3.7, -3.5, -3.2):
        r = run_once(rec, np.r_[1.0, np.logspace(-0.02, end, 200)])
        if r[1] is not None:
            r[1]["path_end_log10_w"] = end
            return r
    return r


def run_once(rec, w_left):
    p = phreeqc()
    mins = available(p)
    na, ca, mg, cl, s = rec["Na"], rec["Ca"], rec["Mg"], rec["Cl"], rec["S"]
    removed = " ".join(f"{55.508 * (1 - w):.6f}" for w in w_left)
    eq = "\n".join(f"  {m} 0 0" for m in mins)
    sel = " ".join(mins)
    inp = f"""
SOLUTION 1
  temp 25
  units mmol/kgw
  pH 8.3 charge
  Na {na}
  Ca {ca}
  Mg {mg}
  Cl {cl}
  S(6) {s}
  C(4) {rec['C']}
EQUILIBRIUM_PHASES 1
  CO2(g) -3.4 10
{eq}
REACTION 1
  H2O -1
  {removed} moles
INCREMENTAL_REACTIONS false
KNOBS
  -step_size 10
  -pe_step_size 5
SELECTED_OUTPUT 1
  -reset false
  -step true
  -ph true
  -ionic_strength true
  -water true
  -activities H2O
  -equilibrium_phases {sel}
  -saturation_indices {sel}
END
"""
    try:
        p.run_string(inp)
    except Exception as e:  # report and keep going
        return rec["recipe"], None, str(e)[:300]
    arr = p.get_selected_output_array()
    df = pd.DataFrame(arr[1:], columns=arr[0])
    df = df[df["step"] != 0] if "step" in df else df
    df.insert(0, "recipe", rec["recipe"])
    return rec["recipe"], df, ""


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_csv(ROOT + "/metadata/samples.csv").drop_duplicates(["table", "sample"])
    recs = [{"recipe": f"T{r.table}_S{r['sample']}", "Na": r.Na_mM, "Ca": r.Ca_mM, "Mg": r.Mg_mM, "Cl": r.Cl_mM,
             "S": r.S_mM, "C": r.C_mM} for _, r in m.iterrows()]
    if len(sys.argv) > 1:
        recs = recs[:int(sys.argv[1])]
    # one process per recipe with a hard time limit: a few PHREEQC runs stall near dryness instead of failing
    import multiprocessing as mp
    res = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(24, maxtasksperchild=1) as pool:
        jobs = [(rec, pool.apply_async(run, (rec,))) for rec in recs]
        for rec, j in jobs:
            try:
                res.append(j.get(timeout=60))
            except mp.TimeoutError:
                res.append((rec["recipe"], None, "timeout"))
            print(rec["recipe"], res[-1][2][:60] or "ok", flush=True)
    errs = [(r, e) for r, d, e in res if d is None]
    paths = pd.concat([d for r, d, e in res if d is not None], ignore_index=True)
    paths.to_csv(OUT + "/evap_paths.csv", index=False)
    # onset concentration factor and final moles per mineral
    rows = []
    mins = [c[2:] for c in paths.columns if c.startswith("d_")]
    for rec, d in paths.groupby("recipe"):
        cf = 1.0 / d["mass_H2O"].values
        aw = 10 ** d["la_H2O"].values
        dry = aw >= 0.475
        row = {"recipe": rec, "cf_at_aw0475": float(cf[dry][-1]), "path_end_cf": float(cf[-1]), "aw_end": float(aw[-1])}
        for mn in mins:
            amt = d[mn].values
            idx = np.nonzero(amt > 1e-12)[0]
            row[f"{mn}_onset_cf"] = float(cf[idx[0]]) if len(idx) else np.nan
            row[f"{mn}_final_mol"] = float(amt[-1])
            row[f"{mn}_mol_at_aw0475"] = float(amt[dry][-1])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT + "/onset.csv", index=False)
    print("thermo done", len(rows), "recipes; errors:", errs[:3], "minerals:", mins)
