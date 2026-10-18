"""Module A: master metadata table and quality flags, built from the raw copy (experiment/data).

Outputs (under the dataset project):
  metadata/samples.csv        one row per droplet
  metadata/quality_flags.csv  one row per (droplet, flag)
  metadata/README.md          column dictionary
"""
import os, re, glob, json
import numpy as np, pandas as pd
from PIL import Image

ROOT = os.environ.get("CRD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root; override with CRD_ROOT
RAW = ROOT + "/experiment/data"
SPLIT = "/root/autodl-fs/coffee-ring-water-chemistry/data/splits/cv5_seed2026.json"
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
MW = {"C": 12.011, "Ca": 40.078, "Cl": 35.45, "Mg": 24.305, "Na": 22.990, "O": 15.999, "S": 32.06}
os.makedirs(ROOT + "/metadata", exist_ok=True)


def semtxt(p):
    kv = {}
    for line in open(p, errors="ignore"):
        m = re.match(r"\$+(\S+)\s*(.*)", line.strip())
        if m:
            kv[m.group(1)] = m.group(2).strip()
    return kv


rows = []
for d in sorted(glob.glob(RAW + "/Table_*/Sample_*/*_rep")):
    rep = os.path.relpath(d, RAW)
    t, s, r = rep.split("/")
    kv = semtxt(d + "/sem.txt")
    row = dict(droplet_id=f"T{t[6:]}_S{s[7:]}_R{r[0]}", path=rep, table=int(t[6:]), sample=int(s[7:]), replicate=int(r[0]),
               sem_signal=kv.get("CM_SIGNAL"), sem_mag=int(kv["CM_MAG"]), sem_kv=float(kv["CM_ACCEL_VOLT"]),
               sem_wd_mm=float(kv["SM_WD"]), sem_spot=int(kv["SM_SPOT_SIZE"]), sem_date=kv["CM_DATE"], sem_time=kv["CM_TIME"],
               sem_stage_x=float(kv["CM_STAGE_X"]), sem_stage_y=float(kv["CM_STAGE_Y"]))
    with Image.open(d + "/image.jpg") as im:
        row["optical_w"], row["optical_h"] = im.size
    with Image.open(d + "/sem.jpg") as im:
        row["sem_w"], row["sem_h"] = im.size
    sizes = {}
    for e in EL:
        with Image.open(f"{d}/{e}.jpeg") as im:
            sizes[e] = im.size
            row[f"eds_{e}_w"], row[f"eds_{e}_h"] = im.size
    row["eds_size_consistent"] = len(set(sizes.values())) == 1
    rows.append(row)
df = pd.DataFrame(rows)

# recipe and nominal composition
conc = pd.read_csv(ROOT + "/data/element_concentrations.csv").rename(columns={"Table": "table", "Sample_ID": "sample"})
df = df.merge(conc, on=["table", "sample"], how="left")
df["NaHCO3_mM"], df["CaCl2_mM"], df["MgSO4_mM"], df["MgCl2_mM"] = df.C_mM, df.Ca_mM, df.S_mM, 0.25
assert np.allclose(df.Mg_mM, df.MgSO4_mM + 0.25) and np.allclose(df.Cl_mM, 2 * df.CaCl2_mM + 0.5)
mol = df[[f"{e}_mM" for e in EL]].values
mass = mol * np.array([MW[e] for e in EL])
for i, e in enumerate(EL):
    df[f"{e}_molfrac"] = mol[:, i] / mol.sum(1)
    df[f"{e}_massfrac"] = mass[:, i] / mass.sum(1)
df["hardness_mgL_CaCO3"] = (df.Ca_mM + df.Mg_mM) * 100.09
df["ionic_strength_mM"] = 0.5 * (df.Na_mM * 1 + df.Ca_mM * 4 + df.Mg_mM * 4 + df.Cl_mM * 1
                                 + df.S_mM * 4 + df.C_mM * 1)  # HCO3- as monovalent, SO4 2-
df["tds_mg_L"] = (df.NaHCO3_mM * 84.007 + df.CaCl2_mM * 110.98 + df.MgSO4_mM * 120.37 + df.MgCl2_mM * 95.21)

# acquisition: datetime, session (calendar day), relative pixel size
df["sem_datetime"] = pd.to_datetime(df.sem_date + " " + df.sem_time)
df["session"] = df.sem_date
df["sem_px_rel"] = 50.0 / df.sem_mag  # pixel size relative to a 50x image (absolute scale needs the instrument's field width)
df["eds_export"] = df.eds_Ca_w.astype(str) + "x" + df.eds_Ca_h.astype(str)

# CV folds (recipe-grouped)
sp = json.load(open(SPLIT))
fold = {}
for k, f in enumerate(sp["folds"]):
    for t, s in f["test"]:
        fold[(t, s)] = k
df["cv_test_fold"] = [fold[(t, s)] for t, s in zip(df.table, df["sample"])]

# quality flags
flags = []
def flag(ids, name, note):
    for i in ids:
        flags.append(dict(droplet_id=i, flag=name, note=note))
flag(["T1_S3_R5", "T1_S4_R1"], "eds_set_duplicated", "all 7 EDS maps byte-identical between these two droplets; true maps of one droplet missing")
flag(["T2_S16_R2"], "eds_element_duplicated", "C.jpeg identical to Ca.jpeg")
flag(["T5_S22_R2"], "eds_element_duplicated", "Mg.jpeg identical to O.jpeg")
flag(["T4_S20_R3", "T4_S20_R4"], "sem_duplicated", "sem.jpg and sem.txt byte-identical between replicates 3 and 4")
flag(["T2_S18_R2", "T2_S18_R3", "T2_S18_R4", "T2_S18_R5", "T2_S19_R5", "T2_S22_R5", "T2_S23_R1", "T3_S1_R1"],
     "semtxt_wrong_in_released_copy", "data/ (enhanced) sem.txt overwritten by a copy of T2_S18_R1; raw copy is correct and used here")
flag(df.loc[~df.eds_size_consistent, "droplet_id"], "eds_mixed_sizes", "EDS maps of this droplet differ in pixel size")
flag(df.loc[df.optical_w != 2592, "droplet_id"], "optical_low_resolution", "optical image 1280x1024 instead of 2592x1944")
flag(df.loc[df.sem_kv != 15, "droplet_id"], "sem_10kV", "SEM at 10 kV instead of 15 kV")
pil = pd.read_csv(ROOT + "/pilot/features.csv")
pil["droplet_id"] = [f"T{a}_S{b}_R{c}" for a, b, c in zip(pil.table, pil["sample"], pil.replicate)]
flag(pil.loc[pil.mask_frac < 0.05, "droplet_id"], "pilot_mask_suspect", "pilot EDS droplet mask covers <5% of the field")
fl = pd.DataFrame(flags)
df["n_flags"] = df.droplet_id.map(fl.droplet_id.value_counts()).fillna(0).astype(int)
df["usable_eds"] = ~df.droplet_id.isin(fl[fl.flag.isin(["eds_set_duplicated"])].droplet_id)
df["usable_sem"] = ~df.droplet_id.isin(fl[fl.flag == "sem_duplicated"].droplet_id)

df.to_csv(ROOT + "/metadata/samples.csv", index=False)
fl.to_csv(ROOT + "/metadata/quality_flags.csv", index=False)
print(df.shape, "flags", len(fl), fl.flag.value_counts().to_dict())
print(df.groupby("session").agg(n=("droplet_id", "size"), tables=("table", lambda x: sorted(set(x))),
                                modes=("sem_signal", lambda x: dict(x.value_counts()))).to_string())
