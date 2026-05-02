# semianalytic_fesc

Code accompanying the paper:

> **Physically motivated modeling of LyC escape fraction during reionization**
> Ivan Kostyuk, Benedetta Ciardi, Andrea Ferrara
> *A&A 695, A32 (2025)* — [DOI: 10.1051/0004-6361/202449997](https://doi.org/10.1051/0004-6361/202449997) · [arXiv:2308.01476](https://arxiv.org/abs/2308.01476)

## Science context

Understanding how ionizing (Lyman-continuum, LyC) photons escape from early galaxies into the intergalactic medium is a central problem in cosmic reionization: without a sufficiently high escape fraction (*f*<sub>esc</sub>), galaxies cannot supply enough photons to reionize the universe by z ~ 6. Directly measuring *f*<sub>esc</sub> at high redshift is observationally extremely difficult, which makes physically motivated models that connect *f*<sub>esc</sub> to simulated galaxy properties essential.

This code implements a semi-analytic model that computes *f*<sub>esc</sub> as a post-processing step on IllustrisTNG galaxies. Each galaxy is projected onto a 2D grid, and for every cell the model self-consistently tracks radiation pressure from young stars, gravitational pressure from the gas disc, dust attenuation, and radiation-driven outflows to derive the fraction of ionizing photons that escape to the IGM.

Applied to ~600 000 galaxies from [TNG50](https://www.tng-project.org/) over z = 5.2–20, the model reveals a **bimodal nature of LyC escape**:

- **ext-mode** — high-metallicity (10<sup>−3.5</sup> < Z < 10<sup>−2</sup>), low-mass (M<sub>★</sub> < 10<sup>7</sup> M<sub>☉</sub>) galaxies with extended star formation, where photons escape from the outer regions of the galactic plane. This mode becomes prominent at later cosmic times once sufficient metal enrichment has occurred.
- **loc-mode** — low-metallicity (Z < 10<sup>−3</sup>), moderately massive (M<sub>★</sub> < 10<sup>8</sup> M<sub>☉</sub>) galaxies where star formation is confined to small high-density regions, producing localised LyC escape. Present at all redshifts studied.

Building on these results, the paper derives an **analytical fitting formula** that predicts *f*<sub>esc</sub> from stellar mass, gas mass, and redshift — a lightweight subgrid tool for use in large-scale reionization models.

## Physical model

For each grid cell the pipeline computes, in order:

1. Surface densities of gas, stars, and SFR
2. Ionizing and bolometric flux from the local SFR
3. Dust opacity from gas metallicity (normalised to Z<sub>☉</sub>)
4. Gas fraction and gravitational pressure
5. Radiation pressure from absorbed photons
6. Where radiation pressure exceeds gravity: radiation-driven outflow velocity and evacuated gas fraction (on the ~2 Myr lifetime of OB stars)
7. Reduced column density through the evacuated medium
8. LyC escape fraction accounting for both dust absorption and recombination (Strömgren-sphere correction)
9. A breakout condition ([Ferrara 2023](https://doi.org/10.1093/mnras/stad2431)): cells where the gas fraction falls below a critical threshold are set to *f*<sub>esc</sub> = 0

The galaxy-integrated *f*<sub>esc</sub> is the ionizing-flux-weighted mean over all cells.

## Pipeline

```bash
# 0. Build the initial galaxy catalogue from TNG
python build_initial_df_0.py

# 1. Project particles onto 2D grids
python gridded_maps_1.py

# 2. Compute f_esc for each grid cell
python calculate_gridded_fesc_2.py

# 3. Aggregate to galaxy-integrated values
python maps_to_df_3.py

# 4. Merge snapshots and fit the polynomial model
python build_full_df.py
python fitting_fesc.py
```

For cluster submission across a snapshot range, configure `config_parameters.yml` and use:

```bash
python submit_runs.py
```

## Configuration

All runtime parameters live in `config_parameters.yml`:

| Parameter | Description |
|---|---|
| `base_path` | Root directory for output dataframes and maps |
| `tng_datapath` | Path to IllustrisTNG data |
| `merger_history_path` | Path to the SubLink merger tree HDF5 file |
| `grid_size` | Number of cells per side for the projected grid |
| `with_breakout` | Enable/disable the Ferrara 2023 breakout condition |
| `snap_min` / `snap_max` | Snapshot range to process |

## Repository layout

```
semianalytic_fesc/
├── config.py                    # Points to base_path, df names, and hdf names
├── config_parameters.yml        # Snapshot range, grid size, breakout flag, I/O paths
├── run.py                       # Entry point: orchestrates the full pipeline
├── submit_runs.py               # Submits batch jobs per snapshot on a cluster
│
├── build_initial_df_0.py        # Step 0: build initial galaxy catalogue from TNG
├── gridded_maps_1.py            # Step 1: project galaxy particles onto 2D grids (HDF5)
├── add_sfr_height.py            # Adds SFR-weighted scale height to the catalogue
├── calculate_gridded_fesc_2.py  # Step 2: compute f_esc for each grid cell
├── maps_to_df_3.py              # Step 3: aggregate cell-level results to galaxy level
├── add_analytic_fesc_to_numeric.py  # Appends analytic f_esc estimates to the dataframe
├── build_full_df.py             # Merges per-snapshot dataframes into one
├── create_full_database.py      # Assembles the final HDF5 database
│
├── fitting_fesc.py              # Polynomial regression model for f_esc
├── nearest_neighbors.py         # k-NN interpolation utilities
├── merger_history.py            # Retrieves merger trees from SubLink
├── offset_ang_mom.py            # Angular momentum offset calculations
├── gini.py                      # Gini coefficient for morphology
├── utils.py                     # Shared helper functions
├── plotting_routines.py         # All plotting functions
│
├── deprecated/                  # Old scripts retained for reference
└── misc/                        # Miscellaneous one-off scripts
```

## Requirements

Python ≥ 3.8, NumPy, pandas, h5py, astropy, scikit-learn, matplotlib. Access to IllustrisTNG simulation data.

## Citation

If you use this code, please cite:

```bibtex
@article{Kostyuk2025fesc,
  author  = {Kostyuk, Ivan and Ciardi, Benedetta and Ferrara, Andrea},
  title   = {Physically motivated modeling of {LyC} escape fraction during reionization},
  journal = {A\&A},
  year    = {2025},
  volume  = {695},
  pages   = {A32},
  doi     = {10.1051/0004-6361/202449997},
  eprint  = {2308.01476},
  archivePrefix = {arXiv}
}
```
