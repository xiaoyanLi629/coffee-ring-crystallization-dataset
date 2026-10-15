"""Assemble the public data record in release/ and the Zenodo upload archives in release_zenodo/.

release/
  README.md                      description of the record (copied from analysis/release_README.md)
  raw/Table_t/Sample_s/r_rep/    sem.jpg, sem.txt, {C,Ca,Cl,Mg,Na,O,S}.jpeg, image.jpg as written by the instruments
  metadata/                      samples.csv, quality_flags.csv, recipes.csv
  derived/                       masks_eds.npz, eds_features.csv, registration.csv, optical.csv, grains.csv,
                                 phase_droplet.csv, phase_maps.npz, thermodynamics.csv
  splits.csv                     fold or group of every droplet under the benchmark protocols
  MD5SUMS                        checksum of every file above
release_zenodo/
  raw_Table_{t}_Samples_{a}-{b}.zip (25 archives of five Samples each, ~100 MB, small enough to survive unstable uploads),
  tables.zip (metadata, derived, splits), README.md, MD5SUMS, MD5SUMS_archives

Raw files come from experiment/data/, excluding the derived *_processed.jpeg images. They are hard-linked when the file
system allows it and copied otherwise; nothing in experiment/ is modified.
"""
import hashlib, json, os, shutil, zipfile
import numpy as np, pandas as pd

ROOT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset"
SRC = ROOT + "/experiment/data"
REL = ROOT + "/release"
ZEN = ROOT + "/release_zenodo"
REG = "/root/autodl-fs/CNN-Vision-Transformer_Model_for_Elements_Concentration_Estimation_by_Coffee-Ring_Effect_Residue_Patterns/data/derived/eds_registration.json"
EL = ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]
RAW = ["sem.jpg", "sem.txt", "image.jpg"] + [f"{e}.jpeg" for e in EL]


def link(a, b):
    os.makedirs(os.path.dirname(b), exist_ok=True)
    if os.path.exists(b):
        os.remove(b)
    try:
        os.link(a, b)
    except OSError:
        shutil.copy2(a, b)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


m = pd.read_csv(ROOT + "/metadata/samples.csv")
ID = dict(zip(m.path, m.droplet_id))
os.makedirs(REL + "/metadata", exist_ok=True); os.makedirs(REL + "/derived", exist_ok=True)

# ---- raw files
n_raw = 0
for p in m.path:
    for f in RAW:
        link(f"{SRC}/{p}/{f}", f"{REL}/raw/{p}/{f}"); n_raw += 1

# ---- metadata
shutil.copy2(ROOT + "/metadata/samples.csv", REL + "/metadata/samples.csv")
shutil.copy2(ROOT + "/metadata/quality_flags.csv", REL + "/metadata/quality_flags.csv")
rec = m.groupby(["table", "sample"])[["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM", "MgCl2_mM", "Na_mM", "Ca_mM", "Mg_mM", "Cl_mM",
                                      "S_mM", "C_mM", "O_mM", "hardness_mgL_CaCO3", "ionic_strength_mM", "tds_mg_L"]].first().reset_index()
rec.insert(0, "recipe", "T" + rec.table.astype(str) + "_S" + rec["sample"].astype(str))
rec.to_csv(REL + "/metadata/recipes.csv", index=False)

# ---- derived
mk = np.load(ROOT + "/derived/masks_eds.npz")
np.savez_compressed(REL + "/derived/masks_eds.npz", droplet_id=np.array([ID[p] for p in mk["paths"]]), masks=mk["masks"], grid=mk["grid"])

f = pd.read_csv(ROOT + "/analysis/out_def/features.csv")
keep = ["mask_method", "mask_frac"] + [c for c in f.columns if c.split("_")[0] in EL and c.split("_", 1)[1] in
                                       ("d", "cov", "contrast", "rbar", "gini", "rim_index")] + [c for c in f.columns if c.startswith("sem_") and c not in m.columns]
f[["droplet_id"] + keep].to_csv(REL + "/derived/eds_features.csv", index=False)

reg = json.load(open(REG))
rg = pd.DataFrame([{"droplet_id": ID[r["rep"]], "scale": r["use"]["s"], "tx_px": r["use"]["tx"], "ty_px": r["use"]["ty"],
                    "ncc_identity": r["ncc_identity"], "ncc_registered": r["ncc_registered"], "ncc_second_peak": r["ncc_second_peak"],
                    "fallback_to_typical": r["fallback"]} for r in reg["droplets"]])
rg.sort_values("droplet_id").to_csv(REL + "/derived/registration.csv", index=False)

g = pd.read_csv(ROOT + "/analysis/out_g/derived.csv")
ocols = ["date_optical", "opt_w", "opt_h", "opt_mean_rgb", "opt_mask_method", "opt_frac", "opt_diam_px", "opt_in_d", "opt_rim_d",
         "opt_centre_d", "opt_bright_frac", "opt_hf", "optreg_ncc", "optreg_angle", "optreg_flip", "optreg_ncc_ctrl"]
g[["droplet_id"] + ocols].to_csv(REL + "/derived/optical.csv", index=False)

gr = pd.read_csv(ROOT + "/analysis/out_p/grains.csv")
gr.insert(0, "droplet_id", gr.path.map(ID)); gr = gr.drop(columns="path").rename(columns={"r": "r_norm"})
gr.to_csv(REL + "/derived/grains.csv", index=False)
pd_ = pd.read_csv(ROOT + "/analysis/out_p/phase_droplet.csv")
pcols = [c for c in pd_.columns if c not in m.columns]
pd_[["droplet_id"] + pcols].to_csv(REL + "/derived/phase_droplet.csv", index=False)
pm = np.load(ROOT + "/analysis/out_p/phase_maps.npz")
np.savez_compressed(REL + "/derived/phase_maps.npz", droplet_id=np.array([ID[p] for p in pm["paths"]]), labels=pm["labels"],
                    names=pm["names"], element_scale=np.load(ROOT + "/analysis/out_p/element_scale.npy"))

th = pd.read_csv(ROOT + "/analysis/out_t/onset.csv")
th.to_csv(REL + "/derived/thermodynamics.csv", index=False)

# ---- splits
sp = m[["droplet_id", "table", "sample", "replicate", "cv_test_fold", "session", "sem_signal"]].merge(g[["droplet_id", "date_optical"]], on="droplet_id")
sp = sp.rename(columns={"cv_test_fold": "P1_recipe_fold", "session": "P2_sem_session", "date_optical": "P2o_optical_date", "sem_signal": "P3_detector"})
sp.to_csv(REL + "/splits.csv", index=False)

# ---- README, checksums (README.md travels next to the archives, so it is not listed in MD5SUMS)
shutil.copy2(ROOT + "/analysis/release_README.md", REL + "/README.md")
files = sorted(os.path.relpath(os.path.join(d, x), REL) for d, _, xs in os.walk(REL) for x in xs if x not in ("MD5SUMS", "README.md"))
with open(REL + "/MD5SUMS", "w") as fo:
    for x in files:
        fo.write(f"{md5(os.path.join(REL, x))}  {x}\n")

# ---- Zenodo archives (JPEG is already compressed, so archives are stored without compression)
os.makedirs(ZEN, exist_ok=True)
for x in os.listdir(ZEN):
    if x.startswith("raw_Table_") and x.endswith(".zip"):
        os.remove(os.path.join(ZEN, x))
for t in range(1, 6):
    for k in range(5):
        lo, hi = 5 * k + 1, 5 * k + 5
        want = tuple(f"raw/Table_{t}/Sample_{s}/" for s in range(lo, hi + 1))
        with zipfile.ZipFile(f"{ZEN}/raw_Table_{t}_Samples_{lo:02d}-{hi:02d}.zip", "w", zipfile.ZIP_STORED) as z:
            for x in files:
                if x.startswith(want):
                    z.write(os.path.join(REL, x), x)
with zipfile.ZipFile(f"{ZEN}/tables.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for x in files:
        if not x.startswith("raw/"):
            z.write(os.path.join(REL, x), x)
for x in ("README.md", "MD5SUMS"):
    if os.path.exists(f"{REL}/{x}"):
        shutil.copy2(f"{REL}/{x}", f"{ZEN}/{x}")
with open(ZEN + "/MD5SUMS_archives", "w") as fo:
    for x in sorted(os.listdir(ZEN)):
        if x.endswith(".zip"):
            fo.write(f"{md5(os.path.join(ZEN, x))}  {x}\n")

print(json.dumps({"raw_files": n_raw, "files": len(files), "grains": len(gr), "recipes": len(rec), "phase_droplets": len(pd_),
                  "zenodo": {x: round(os.path.getsize(f"{ZEN}/{x}") / 2 ** 20, 1) for x in sorted(os.listdir(ZEN))}}, indent=1))
