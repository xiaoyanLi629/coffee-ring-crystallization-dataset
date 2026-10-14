# SEM, EDS and optical images of coffee-ring salt residues from 125 synthetic tap-water compositions

Xiaoyan Li, Cuicui Jiang, Rebecca Lahr (corresponding author, rlahr@a2gov.org)

This record contains 625 dried droplet residues: five replicate 2 µL droplets of each of 125 synthetic waters, dried on mirror-polished 6061 aluminium at 23–26 °C and 45–50 % relative humidity. The waters form a full factorial design of NaHCO3, CaCl2 and MgSO4 at five levels each on a constant 0.25 mM MgCl2 background. Every residue was photographed with a digital optical microscope, imaged by scanning electron microscopy (SEM) and mapped for C, Ca, Cl, Mg, Na, O and S by energy-dispersive X-ray spectroscopy (EDS). The record is described in the accompanying Data Descriptor; the code that builds every derived file is at https://github.com/xiaoyanLi629/coffee-ring-crystallization-dataset.

## Files

The five archives `raw_Table_1.zip` to `raw_Table_5.zip` hold the raw files of one NaHCO3 level each (125 droplets, 1,250 files per archive). `tables.zip` holds the metadata, the derived data and the benchmark splits. `MD5SUMS` lists the checksum of every file after extraction, and `MD5SUMS_archives` the checksums of the archives themselves.

After extraction the record has this layout:

```
raw/Table_{t}/Sample_{s}/{r}_rep/
    sem.jpg            SEM image of the whole residue (2560 x 1920)
    sem.txt            SEM acquisition record written by the instrument
    {C,Ca,Cl,Mg,Na,O,S}.jpeg   EDS maps, pseudo-colour, contrast stretched per map
    image.jpg          optical photograph (2592 x 1944, or 1280 x 1024 for 39 droplets)
metadata/samples.csv        one row per droplet: design, concentrations, acquisition, quality flags
metadata/quality_flags.csv  one row per flag with an explanation
metadata/recipes.csv        Table/Sample to concentration mapping for the 125 waters
derived/masks_eds.npz       droplet masks on the 379 x 505 EDS grid
derived/eds_features.csv    stretch-invariant EDS and SEM descriptors per droplet
derived/registration.csv    SEM-EDS scale and translation per droplet (x_sem = s * x_eds + t, 758 x 1010 EDS grid)
derived/optical.csv         optical image date, droplet mask summary, diameter and registration to SEM
derived/grains.csv          84,245 crystal grains: size, normalised radius, phase, purity, element enrichment
derived/phase_droplet.csv   grain counts, area fractions and positions per phase and droplet
derived/phase_maps.npz      grain phase label maps on the EDS grid
derived/thermodynamics.csv  PHREEQC prediction of precipitation onset and amounts per water
splits.csv                  fold or group of every droplet under protocols P1, P2, P2o and P3
```

Here t = 1–5 is the NaHCO3 level, s = 1–25 the CaCl2 × MgSO4 combination and r = 1–5 the replicate. Droplet identifiers have the form `T{t}_S{s}_R{r}` and join every table.

## Points to know before use

The SEM detector is not constant: Table 1 was imaged with backscattered electrons (BEC), Tables 4 and 5 with secondary electrons (SEI), and Tables 2 and 3 with both. Because Table encodes the NaHCO3 level, detector and NaHCO3 are confounded; the column `sem_signal` records the detector of every image. EDS maps are stretched independently, so their grey levels are not comparable between maps; use the descriptors in `derived/eds_features.csv`, which are expressed relative to the substrate of the same map. The aluminium substrate contributes to the Mg and O maps. Quality flags are listed in `metadata/quality_flags.csv`; the columns `usable_eds` and `usable_sem` in `samples.csv` summarise them.

## License

The data are released under the Creative Commons Attribution 4.0 International license (CC BY 4.0).
