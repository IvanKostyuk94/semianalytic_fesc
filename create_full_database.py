from build_initial_df_0 import build_new_df
from get_height_and_maps_1 import update_df_height_make_maps
from calculate_gridded_fesc_2 import update_maps
from maps_to_df_3 import update_map_df


def create_database(
    snap_nums,
    df_name="df",
    maps_name="maps",
    scale=1,
    base="/ptmp/mpa/ivkos/semianalytic_fesc",
):
    for num in snap_nums:
        df_name_full = df_name + "_" + str(num)
        maps_name_full = maps_name + "_" + str(num)
        build_new_df(snap_num=num, base=base)
        update_df_height_make_maps(
            snap_num=num,
            df_name=df_name_full,
            hdf_name=maps_name_full,
            to_hdf=True,
            adaptive=True,
            avg_dist_weighting=1,
        )
        update_maps(
            snap_num=num,
            grid_size=scale,
            hdf_name=maps_name_full,
            df_name=df_name_full,
        )
        update_map_df(
            snap_num=num,
            scale=scale,
            hdf_name=maps_name_full,
            df_name=df_name_full,
        )
    return


if __name__ == "__main__":
    snap_nums = [0]
    create_database(snap_nums)
