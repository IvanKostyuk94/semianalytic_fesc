from get_height_and_maps_1 import update_df_height_make_maps
from calculate_gridded_fesc_2 import update_maps
from maps_to_df_3 import update_map_df

snap_num = 13
grids_to_test = [
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    90,
    100
]

df_name = "grid_tests/test_df_ad"
hdf_name = "grid_tests/maps_adaptive_full"

print("Working on scale height and base maps")
for grid in grids_to_test:
    print(f"Working on mfp dist: {grid}")
    update_df_height_make_maps(
        snap_num,
        df_name,
        hdf_name,
        to_hdf=True,
        adaptive=True,
        base_path="/ptmp/mpa/ivkos/semianalytic_fesc",
        avg_dist_weighting=grid,
        approx_grid_size=None,
    )

print("Working on creating gridded maps")
for grid in grids_to_test:
    print(f"Working on mfp dist: {grid}")
    update_maps(snap_num, grid, hdf_name, df_name)

print("Summarizing maps into df")
for grid in grids_to_test:
    print(f"Working on mfp dist: {grid}")
    update_map_df(snap_num, grid, hdf_name, df_name)
