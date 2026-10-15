"""Create the Zenodo draft for the 625-droplet dataset (metadata + reserved DOI). Writes release_zenodo/deposition.json."""
import json, os, requests
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(k, None)
TOK = open("/root/.zenodo_token").read().strip()
H = {"Authorization": f"Bearer {TOK}"}
OUT = "/root/autodl-fs/CoffeeRing_Crystallization_Dataset/release_zenodo/deposition.json"
desc = ("<p>When a drop of tap water dries, its dissolved salts crystallise into a residue whose ring, film and crystals depend on the "
        "water's composition. This record contains 625 such residues: five replicate 2 &micro;L droplets of each of 125 synthetic waters, "
        "dried on mirror-polished 6061 aluminium at 23&ndash;26 &deg;C and 45&ndash;50 % relative humidity. The waters form a full factorial "
        "design of NaHCO<sub>3</sub>, CaCl<sub>2</sub> and MgSO<sub>4</sub> (five levels each) on a constant 0.25 mM MgCl<sub>2</sub> background. "
        "Every residue was imaged by digital optical microscopy and by scanning electron microscopy (SEM), and mapped for seven elements "
        "(C, Ca, Cl, Mg, Na, O, S) by energy-dispersive X-ray spectroscopy (EDS).</p>"
        "<p>The raw instrument files are distributed in archives by NaHCO<sub>3</sub> level (Table) and CaCl<sub>2</sub> &times; MgSO<sub>4</sub> "
        "combination (Sample). <code>tables.zip</code> holds the metadata table, quality flags, recipes, droplet masks, stretch-invariant EDS "
        "descriptors, SEM&ndash;EDS and optical&ndash;SEM registration, an inventory of 84,245 crystal grains, a PHREEQC equilibrium-evaporation "
        "prediction for every water and benchmark splits. See <code>README.md</code> for the layout and for points to know before use, in particular "
        "that the SEM detector (backscattered vs secondary electrons) changed between sessions and is confounded with the NaHCO<sub>3</sub> level.</p>")
meta = {"metadata": {
    "upload_type": "dataset",
    "title": "SEM, EDS and optical images of coffee-ring salt residues from 125 synthetic tap-water compositions",
    "creators": [
        {"name": "Li, Xiaoyan", "affiliation": "Tsinghua University; Michigan State University", "orcid": "0000-0002-2649-7575"},
        {"name": "Jiang, Cuicui", "affiliation": "Inner Mongolia University"},
        {"name": "Lahr, Rebecca", "affiliation": "Michigan State University", "orcid": "0000-0002-7988-7649"}],
    "description": desc,
    "access_right": "open",
    "license": "cc-by-4.0",
    "keywords": ["coffee-ring effect", "evaporative crystallisation", "scanning electron microscopy", "energy-dispersive X-ray spectroscopy",
                 "water quality", "multi-modal dataset", "distribution shift"],
    "prereserve_doi": True}}
if os.path.exists(OUT):
    raise SystemExit("deposition.json exists; refusing to create a second draft")
r = requests.post("https://zenodo.org/api/deposit/depositions", headers=H, json=meta, timeout=120)
print(r.status_code)
d = r.json()
if r.status_code != 201:
    raise SystemExit(json.dumps(d)[:1500])
json.dump(d, open(OUT, "w"), indent=1)
print(d["id"], d["metadata"]["prereserve_doi"]["doi"], d["links"]["bucket"], d["links"]["html"])
