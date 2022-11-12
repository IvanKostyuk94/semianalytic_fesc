from get_height_and_maps_1 import update_df_height_make_maps
from calculate_gridded_fesc_2 import update_maps
from maps_to_df_3 import update_map_df

snap_num = 13
grids_to_test = [
    0.1,
    0.11513954,
    0.13257114,
    0.1526418,
    0.17575106,
    0.20235896,
    0.23299518,
    0.26826958,
    0.30888436,
    0.35564803,
    0.40949151,
    0.47148664,
    0.54286754,
    0.62505519,
    0.71968567,
    0.82864277,
    0.95409548,
    1.09854114,
    1.26485522,
    1.45634848,
    1.67683294,
    1.93069773,
    2.22299648,
    2.55954792,
    2.9470517,
    3.39322177,
    3.90693994,
    4.49843267,
    5.17947468,
    5.96362332,
    6.86648845,
    7.90604321,
    9.10298178,
    10.48113134,
    12.06792641,
    13.89495494,
    15.9985872,
    18.42069969,
    21.20950888,
    24.42053095,
    28.11768698,
    32.37457543,
    37.2759372,
    42.9193426,
    49.41713361,
    56.89866029,
    65.51285569,
    75.43120063,
    86.85113738,
    100.0,
]

df_name = "new_convergence/test_adaptive"
hdf_name = "new_convergence/maps_adaptive"
adaptive = True
print("Working on scale height and base maps")
for grid in grids_to_test:
    print(f"Working on mfp dist: {grid:.2f}")
    update_df_height_make_maps(
        snap_num,
        df_name,
        hdf_name,
        to_hdf=True,
        adaptive=adaptive,
        base_path="/ptmp/mpa/ivkos/semianalytic_fesc",
        avg_dist_weighting=grid,
        approx_grid_size=None,
    )

print("Working on creating gridded maps")
for grid in grids_to_test:
    print(f"Working on mfp dist: {grid:.2f}")
    update_maps(snap_num, grid, hdf_name, df_name)

print("Summarizing maps into df")
for grid in grids_to_test:
    print(f"Working on mfp dist: {grid:.2f}")
    update_map_df(snap_num, grid, hdf_name, df_name)
