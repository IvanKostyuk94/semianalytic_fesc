# semianalytic_fesc

A semi-analytic pipeline for computing the Lyman-continuum (LyC) escape fraction (*f*<sub>esc</sub>) of galaxies from IllustrisTNG simulation snapshots, together with a polynomial fitting model that predicts *f*<sub>esc</sub> from integrated galaxy properties.

## Overview

The physical model works cell by cell on a 2D projected grid for each galaxy. For every grid cell it:

1. Computes surface densities of gas, stars, and star-formation rate (SFR).
2. Derives the local ionizing and bolometric flux from the SFR.
3. Estimates dust opacity from the gas metallicity.
4. Compares radiation pressure to gravitational pressure; where radiation dominates, gas is partially evacuated on the lifetime of OB stars (~2 Myr).
5. Calculates the escape fraction through the reduced column density, accounting for both dust and recombination absorption (Strömgren-sphere correction).
6. Applies a breakout condition ([Ferrara 2023](https://doi.org/10.1093/mnras/stad2431)): cells where the gas fraction is below a critical threshold contribute zero escaped flux.

The galaxy-integrated *f*<sub>esc</sub> is then the ionizing-flux-weighted mean over all cells. A second module fits a polynomial model (stellar mass, gas mass, metallicity, redshift) to the gridded results, allowing fast *f*<sub>esc</sub> estimates for large galaxy samples.

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
├── plotting.ipynb               # Notebook for producing paper figures
│
├── deprecated/                  # Old scripts retained for reference
└── misc/                        # Miscellaneous one-off scripts
```

## Pipeline

Run the steps in order for each set of snapshots:

```bash
# 0. Build the initial galaxy catalogue
python build_initial_df_0.py

# 1. Project particles onto 2D grids
python gridded_maps_1.py

# 2. Compute the escape fraction on the grids
python calculate_gridded_fesc_2.py

# 3. Aggregate to galaxy-integrated values
python maps_to_df_3.py

# 4. (Optional) Merge snapshots and fit the polynomial model
python build_full_df.py
python fitting_fesc.py
```

For cluster submission, edit `config_parameters.yml` and use:

```bash
python submit_runs.py
```

## Configuration

All runtime parameters live in `config_parameters.yml`:

| Parameter | Description |
|---|---|
| `base_path` | Root directory for output dataframes and maps |
| `tng_datapath` | Path to the IllustrisTNG data |
| `merger_history_path` | Path to the SubLink merger tree HDF5 file |
| `grid_size` | Number of cells per side for the projected grid |
| `with_breakout` | Enable/disable the Ferrara 2023 breakout condition |
| `snap_min` / `snap_max` | Snapshot range to process |

## Requirements

- Python ≥ 3.8
- NumPy, pandas, h5py, astropy, scikit-learn, matplotlib
- Access to IllustrisTNG simulation data

## Reference

The physical model follows [Ferrara (2023)](https://doi.org/10.1093/mnras/stad2431), adapting the radiation-pressure-driven outflow and breakout framework to a cell-by-cell grid approach on IllustrisTNG galaxies.
