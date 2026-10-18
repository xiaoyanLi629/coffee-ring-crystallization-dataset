# Coffee-ring salt residues from 125 synthetic tap waters: analysis code

This repository holds the code behind the data descriptor *SEM, EDS and optical images of coffee-ring salt residues from 125 synthetic tap-water compositions* by Xiaoyan Li, Cuicui Jiang, Rumei Yang and Rebecca Lahr. The scripts build the metadata table, quality flags, droplet masks, stretch-invariant EDS descriptors, the crystal-grain inventory, the thermodynamic evaporation model, the reference benchmarks and the figures that describe the dataset.

The data themselves are not stored here. They are deposited in figshare under a CC BY 4.0 licence:

| | |
|---|---|
| **Dataset** | [doi:10.6084/m9.figshare.34516434](https://doi.org/10.6084/m9.figshare.34516434) |
| **Content** | 625 dried droplets (125 waters × 5 replicates), each with an optical photograph, an SEM image and seven EDS element maps (C, Ca, Cl, Mg, Na, O, S) |
| **Design** | Full 5 × 5 × 5 factorial of NaHCO₃, CaCl₂ and MgSO₄ on a constant 0.25 mM MgCl₂ background |

The record's own README describes every file, the directory layout, the known limitations of the images and the recommended evaluation protocols. This page covers only the code.

## Repository layout

`analysis/` contains the processing and analysis pipeline, one script per module, together with the small result tables and summaries each module writes to `analysis/out_*/`. `analysis/figures/` builds the figures from those outputs, using a shared millimetre-based style in `style.py`. `metadata/` holds the per-droplet metadata table and the quality flags that ship with the dataset. `experiment/recipies/` keeps the original recipe spreadsheet and element-concentration tables used to prepare the waters, and `data/` holds the recipe summaries derived from them. `docs/` has the pilot feature-extraction scripts that some modules read from, and `pilot/` holds their small outputs.

## Pipeline

Each script is self-contained and documents its inputs and outputs in its header. They are meant to be run in the order below, because later modules read the tables written by earlier ones.

1. `a_metadata.py` builds `metadata/samples.csv` and `metadata/quality_flags.csv` from the raw droplet folders.
2. `def_derive.py` decodes the EDS maps, segments each droplet and computes the stretch-invariant descriptors and radial profiles; `def_stats.py` tests them against the design.
3. `c_eds.py` characterises what the pseudo-colour EDS exports can and cannot be used for (colour map, quantisation, per-map contrast stretching).
4. `b_batch.py` decomposes variance into chemistry and acquisition session and runs the shortcut tests.
5. `g_derive.py` and `g_stats.py` cover cross-modal consistency: file provenance, SEM–EDS registration, SEM grey level against composition for the two detectors, Mg leakage from the aluminium substrate, and the optical modality.
6. `p_phase.py` builds the crystal-grain inventory and assigns phases by stoichiometric templates; `p_stats.py` relates it to the design.
7. `thermo_evap.py` simulates equilibrium evaporation of every water in PHREEQC with the Pitzer database; `thermo_compare.py` compares the predictions with the grain inventory.
8. `h_embed.py` extracts frozen ResNet-18 embeddings, `h_bench.py` runs the ridge-regression benchmarks and `m_deep.py` fine-tunes the CNN baselines under the recipe-grouped, session-held-out and cross-detector protocols.
9. `build_release.py` assembles the public data record, and `figshare_upload.py` uploads it. The upload script reads an API token from a local file and never publishes the item on its own.

The figures are rebuilt with `python analysis/figures/fig1.py` and `python analysis/figures/figs.py [fig2 fig3 ...]`.

## Running the code

By default the scripts treat the repository root as the project directory. Download and extract the figshare record so that the raw droplet folders sit under `experiment/data/Table_t/Sample_s/r_rep/`, or set the environment variable `CRD_ROOT` to a directory with that layout. Several modules (`a_metadata.py`, `c_eds.py`, `def_stats.py`, `g_derive.py`, `g_stats.py` and `build_release.py`) read the SEM–EDS registration and the fixed cross-validation folds produced by the companion modelling project; the same registration is distributed with the dataset as `derived/registration.csv`, and the folds as `splits.csv`. `thermo_evap.py` needs a local copy of the PHREEQC Pitzer database and of the IPhreeqc library, whose paths are set at the top of the script.

The analysis was run with Python 3.10 and NumPy 2.0, pandas 2.3, SciPy 1.15, scikit-learn 1.7, scikit-image 0.25, statsmodels 0.15, OpenCV 5.0, Matplotlib 3.10, PyTorch 2.14 with torchvision 0.29, and phreeqpy 0.6 (IPhreeqc 3.7.3).

## Citation

If you use the data, please cite the dataset:

> Li, X., Jiang, C., Yang, R. & Lahr, R. SEM, EDS and optical images of coffee-ring salt residues from 125 synthetic tap-water compositions. figshare (2026). https://doi.org/10.6084/m9.figshare.34516434

The companion dataset, which established the drying condition used here by comparing nine temperature and humidity settings, is Li, X., Jiang, C., Chen, J. & Lahr, R. *A physically grounded multi-modal imaging dataset of evaporative coffee ring deposits*. figshare (2026). https://doi.org/10.6084/m9.figshare.34516902

## Contact

Rebecca Lahr (corresponding author), rlahr@a2gov.org. For questions about the code, please open an issue.
