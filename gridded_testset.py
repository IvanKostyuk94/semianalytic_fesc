from gridded_maps import grid_halos
from utils import get_sim, get_snap
import numpy as np
from calculate_gridded_fesc_2 import update_maps
from maps_to_df_3 import update_map_df

df_name = "new_convergence/column_test"
hdf_name = "new_convergence/maps_column_test"
base_path = "/ptmp/mpa/ivkos/semianalytic_fesc"
snap_num = 13
sim, sim_path = get_sim()
snap = get_snap(snap_num)
grid_sizes = np.arange(2, 160, 20).astype(np.int64)

print("Working on gridded maps")
for grid_scale in grid_sizes:
    print(f"Working on grid size: {grid_scale}")
    grid_halos(df_name, snap_num, grid_scale, hdf_name, testing=True)

print("Working on creating gridded maps")
for grid_scale in grid_sizes:
    print(f"Working on grid size: {grid_scale}")
    update_maps(snap_num, grid_scale, hdf_name, df_name, testing=True)

print("Summarizing maps into df")
for grid_scale in grid_sizes:
    print(f"Working on grid size: {grid_scale}")
    update_map_df(snap_num, grid_scale, hdf_name, df_name, testing=True)
