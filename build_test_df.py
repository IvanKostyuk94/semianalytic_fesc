from get_height_and_maps_1 import update_df_height_make_maps
from calculate_gridded_fesc_2 import update_maps
from maps_to_df_3 import update_map_df

snap_num = 13
grids_to_test = [
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
    1.5,
    10.0,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.5,
    9.0,
    9.5,
]

df_name = "grid_tests/test_df2_adaptive"
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
