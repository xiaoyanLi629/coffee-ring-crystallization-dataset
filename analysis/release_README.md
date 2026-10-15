# SEM, EDS and optical images of coffee-ring salt residues from 125 synthetic tap-water compositions

**Xiaoyan Li**¹˒² · **Cuicui Jiang**³ · **Rebecca Lahr**⁴ (corresponding author, rlahr@a2gov.org)

¹ Department of Automation, Tsinghua University, Beijing, China
² Department of Computational Mathematics, Science and Engineering, Michigan State University, East Lansing, MI, USA
³ College of Computer Science, Inner Mongolia University, Hohhot, China
⁴ Department of Civil and Environmental Engineering, Michigan State University, East Lansing, MI, USA

| | |
|---|---|
| **DOI** | [10.6084/m9.figshare.34516434](https://doi.org/10.6084/m9.figshare.34516434) |
| **Licence** | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| **Code** | <https://github.com/xiaoyanLi629/coffee-ring-crystallization-dataset> |
| **Size** | 29 files, 2.4 GB (6,262 files after extraction) |

---

## Contents

1. [Overview](#overview)
2. [Experimental design](#experimental-design)
3. [Acquisition](#acquisition)
4. [Files in this record](#files-in-this-record)
5. [Getting started](#getting-started)
6. [Directory layout after extraction](#directory-layout-after-extraction)
7. [Tables and derived data](#tables-and-derived-data)
8. [Known limitations](#known-limitations)
9. [Recommended evaluation protocols](#recommended-evaluation-protocols)
10. [How to cite](#how-to-cite)
11. [Licence](#licence)

---

## Overview

When a drop of tap water dries, its dissolved salts crystallise into a residue. The ring, film and crystals of that residue depend on the water's composition. This record contains **625 dried droplet residues**: five replicate 2 µL droplets of each of **125 synthetic waters**, dried on mirror-polished 6061 aluminium at 23–26 °C and 45–50 % relative humidity.

Each residue was observed in three ways:

| Modality | Files per droplet | Format |
|---|---|---|
| Optical microscopy | `image.jpg` | RGB, 2592 × 1944 px (586 droplets) or 1280 × 1024 px (39 droplets) |
| Scanning electron microscopy (SEM) | `sem.jpg` and the instrument record `sem.txt` | 8-bit greyscale, 2560 × 1920 px |
| Energy-dispersive X-ray spectroscopy (EDS) | `C.jpeg`, `Ca.jpeg`, `Cl.jpeg`, `Mg.jpeg`, `Na.jpeg`, `O.jpeg`, `S.jpeg` | Pseudo-colour maps, mostly 1010 × 757/758 px |

Alongside the raw files, the record provides a metadata table, quality flags, droplet masks, SEM–EDS registration, stretch-invariant EDS descriptors, an inventory of 84,245 crystal grains with phase assignments, a PHREEQC equilibrium-evaporation prediction for every water, and benchmark splits.

## Experimental design

The waters form a full 5 × 5 × 5 factorial design of three salts on a constant background of **0.25 mM MgCl₂**.

| Factor | Encoded by | Levels (mM) |
|---|---|---|
| NaHCO₃ | Table `t` = 1–5 | 0.25, 0.50, 0.75, 1.00, 2.00 |
| CaCl₂ | Sample `s` (within each row of five) | 0.50, 0.75, 1.00, 1.50, 2.00 |
| MgSO₄ | Sample `s` (rows of five) | 0.25, 0.50, 0.75, 1.00, 2.00 |

Within each Table, Sample `s` = 1–25 runs over CaCl₂ fastest and MgSO₄ slowest:

```
s = 5 × (MgSO₄ level index) + (CaCl₂ level index) + 1      (indices 0–4)

               CaCl₂ (mM)   0.50  0.75  1.00  1.50  2.00
MgSO₄ 0.25 mM                  1     2     3     4     5
MgSO₄ 0.50 mM                  6     7     8     9    10
MgSO₄ 0.75 mM                 11    12    13    14    15
MgSO₄ 1.00 mM                 16    17    18    19    20
MgSO₄ 2.00 mM                 21    22    23    24    25
```

Across the design, hardness ranges from 100 to 425 mg/L as CaCO₃ and total dissolved solids from 130 to 655 mg/L. The complete mapping, including element concentrations, hardness, ionic strength and dissolved solids, is in `metadata/recipes.csv`.

## Acquisition

| Step | Details |
|---|---|
| Substrate | 6061 aluminium, mirror-like (#8) finish, 1 × 3 inch slides, ten droplets per slide |
| Deposition | 2 µL droplets from a programmable four-axis autosampler; drying took about 20 min in a chamber at 23–26 °C and 45–50 % RH |
| Optical | Celestron Digital Microscope Pro (5 MP) at 100×; photographs taken July–August 2021 |
| SEM | JEOL JSM-6610LV, high vacuum, 15 kV (10 kV for two droplets), 10 mm working distance, 40–55× magnification; twelve sessions, 14–28 August 2021 |
| EDS | Oxford Instruments system on the same microscope; seven elemental maps per droplet, exported as contrast-stretched JPEG |

Samples were not coated before electron microscopy. Full methods are given in the accompanying Data Descriptor.

## Files in this record

| File | Size | Contents |
|---|---|---|
| `raw_Table_{t}_Samples_{a}-{b}.zip` × 25 | 80–125 MB each, 2.4 GB in total | Raw instrument files for one Table and five Samples (25 droplets, 250 files) |
| `tables.zip` | 11 MB | `metadata/`, `derived/` and `splits.csv` |
| `README.md` | — | This document |
| `MD5SUMS` | 0.4 MB | MD5 checksum of each of the 6,262 files after extraction |
| `MD5SUMS_archives` | 2 KB | MD5 checksum of each of the 26 archives |

The raw data are divided into 25 archives so that each one is small enough to download reliably. The archives are stored without compression, because JPEG images are already compressed.

## Getting started

**1. Download** all files into one folder.

**2. Verify the downloads** (optional):

```bash
md5sum -c MD5SUMS_archives          # macOS: brew install coreutils, then gmd5sum -c
```

**3. Extract** every archive into the same folder. This rebuilds the `raw/` tree and adds `metadata/`, `derived/` and `splits.csv`:

```bash
for f in *.zip; do unzip -q "$f"; done
md5sum -c --quiet MD5SUMS           # prints nothing if all 6,262 files are intact
```

**4. Load a droplet** in Python:

```python
import pandas as pd
from PIL import Image

meta = pd.read_csv("metadata/samples.csv")
d = meta.set_index("droplet_id").loc["T2_S13_R1"]
print(d[["NaHCO3_mM", "CaCl2_mM", "MgSO4_mM", "sem_signal", "usable_sem", "usable_eds"]])

folder = f"raw/{d.path}"                       # e.g. raw/Table_2/Sample_13/1_rep
sem = Image.open(f"{folder}/sem.jpg")
eds = {e: Image.open(f"{folder}/{e}.jpeg") for e in ["C", "Ca", "Cl", "Mg", "Na", "O", "S"]}
optical = Image.open(f"{folder}/image.jpg")
```

## Directory layout after extraction

```
raw/
└── Table_{t}/Sample_{s}/{r}_rep/     t = 1–5 (NaHCO₃), s = 1–25 (CaCl₂ × MgSO₄), r = 1–5 (replicate)
    ├── sem.jpg                       SEM image of the whole residue
    ├── sem.txt                       SEM acquisition record written by the instrument
    ├── C.jpeg … S.jpeg               seven EDS maps, pseudo-colour, contrast-stretched per map
    └── image.jpg                     optical photograph
metadata/
├── samples.csv                       one row per droplet (69 columns)
├── quality_flags.csv                 one row per flag (80 flags), with an explanation
└── recipes.csv                       Table/Sample → concentrations for the 125 waters
derived/
├── masks_eds.npz                     droplet masks on the 379 × 505 EDS grid
├── eds_features.csv                  stretch-invariant EDS and SEM descriptors
├── registration.csv                  SEM–EDS scale and translation per droplet
├── optical.csv                       optical summary and optical-to-SEM registration
├── grains.csv                        84,245 crystal grains
├── phase_droplet.csv                 grain counts, area fractions and positions per phase
├── phase_maps.npz                    grain phase label maps on the EDS grid
└── thermodynamics.csv                PHREEQC prediction per water
splits.csv                            benchmark groups for every droplet
```

Every droplet has the identifier **`T{t}_S{s}_R{r}`** (for example `T2_S13_R1`). It is the key that joins every table, and the `path` column in `samples.csv` gives its folder.

## Tables and derived data

| File | Rows | Description |
|---|---|---|
| `metadata/samples.csv` | 625 | Design factors; salt and element concentrations; molar and mass fractions; hardness, ionic strength and TDS; SEM parameters parsed from `sem.txt` (detector `sem_signal`, voltage, magnification, working distance, spot size, date, stage position); image sizes; SEM `session`; recommended fold `cv_test_fold`; `n_flags`, `usable_sem` and `usable_eds` |
| `metadata/quality_flags.csv` | 80 | `droplet_id`, `flag` and `note`, covering, for example, low-resolution optical images, mixed EDS map sizes, duplicated images and the two 10 kV acquisitions |
| `metadata/recipes.csv` | 125 | Concentrations, hardness, ionic strength and TDS for each recipe `T{t}_S{s}` |
| `derived/masks_eds.npz` | 625 | Arrays `droplet_id`, `masks` (625 × 379 × 505, boolean) and `grid` |
| `derived/eds_features.csv` | 625 | For each element, `{E}_d`, `_cov`, `_contrast`, `_rbar`, `_gini` and `_rim_index`, all expressed relative to the substrate of the same map; SEM descriptors prefixed `sem_` |
| `derived/registration.csv` | 625 | `scale`, `tx_px` and `ty_px` with x_SEM = s · x_EDS + t on the 758 × 1010 EDS grid; matching scores before and after registration; fallback indicator |
| `derived/optical.csv` | 625 | Photograph date, image size, droplet mask and diameter, ring and centre descriptors, and optical-to-SEM registration (`optreg_*`) |
| `derived/grains.csv` | 84,245 | One row per grain: size, normalised radius `r_norm`, assigned phase, purity, fit quality and per-element enrichment `z_{E}` |
| `derived/phase_droplet.csv` | 619 | Number, area fraction, mean radius and mean size of grains per phase |
| `derived/phase_maps.npz` | 619 | Label maps `labels` (619 × 379 × 505), phase `names` (halite, gypsum, Mg sulfate, Ca carbonate, Na/Mg carbonate, chloride brine) and `element_scale` |
| `derived/thermodynamics.csv` | 125 | Onset concentration factor and amount of each mineral (calcite, gypsum, halite, epsomite and others) along equilibrium evaporation |
| `splits.csv` | 625 | `P1_recipe_fold`, `P2_sem_session`, `P2o_optical_date` and `P3_detector` |

Six droplets (T1_S1_R4, T1_S1_R5, T1_S3_R5, T1_S4_R1, T1_S21_R5 and T3_S8_R2) have no grain inventory and are absent from `grains.csv` and the two phase files. Four of them carry quality flags for duplicated EDS maps or a suspect droplet mask.

## Known limitations

> **⚠ The SEM detector is confounded with NaHCO₃.**
> Table 1 was imaged with the backscattered-electron detector (BEC), Tables 4 and 5 with the secondary-electron detector (SEI), and Tables 2 and 3 with both (105/20 and 85/40 droplets BEC/SEI). The detector label alone explains about 58 % of the variance of log NaHCO₃. The column `sem_signal` records the detector for every image.

- **EDS maps are contrast-stretched individually.** Grey levels cannot be compared between maps or between droplets. Use the substrate-referenced descriptors in `derived/eds_features.csv`. The original spectra and count maps were not retained.
- **The Mg map includes a substrate signal.** The Al Kα line of the aluminium substrate lies close to Mg Kα, so at low MgSO₄ the Mg map is dominated by the substrate.
- **No physical pixel size.** `sem.txt` does not record the scale-bar length. A relative scale (50 / magnification) is given in `sem_px_rel`.
- **Optical images carry no capture metadata.** Photograph dates come from the file system and are listed in `derived/optical.csv`.
- **Quality flags.** Two droplets are unusable for SEM and two for EDS (see `usable_sem` and `usable_eds`). All 80 flags are explained in `metadata/quality_flags.csv`.

## Recommended evaluation protocols

| Protocol | Column in `splits.csv` | When to use |
|---|---|---|
| P2: leave one SEM session out | `P2_sem_session` | **Primary protocol** for any SEM- or EDS-based model |
| P2o: leave one optical date out | `P2o_optical_date` | Primary protocol for optical models |
| P1: recipe-grouped 5-fold | `P1_recipe_fold` | Secondary reference; replicates of a recipe never span train and test |
| P3: cross-detector transfer | `P3_detector` | Stress test for domain shift |

Report the detector-only baseline alongside any NaHCO₃ result. The code that reproduces every derived file and benchmark is at <https://github.com/xiaoyanLi629/coffee-ring-crystallization-dataset>.

## How to cite

If you use these data, please cite the dataset and the accompanying Data Descriptor:

> Li, X., Jiang, C. & Lahr, R. SEM, EDS and optical images of coffee-ring salt residues from 125 synthetic tap-water compositions. *figshare* https://doi.org/10.6084/m9.figshare.34516434 (2026).

```bibtex
@misc{li2026crystal,
  author    = {Li, Xiaoyan and Jiang, Cuicui and Lahr, Rebecca},
  title     = {{SEM, EDS and optical images of coffee-ring salt residues from 125 synthetic tap-water compositions}},
  publisher = {figshare},
  year      = {2026},
  doi       = {10.6084/m9.figshare.34516434}
}
```

## Licence

The data are released under the [Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You may share and adapt the data for any purpose, provided that you give appropriate credit.
