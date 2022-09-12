from utils import get_sim, get_dataset_df, reduce_df, get_snap
import os


def build_new_df(snap_num, save_name="df", base="/ptmp/mpa/ivkos/semianalytic_fesc"):
    sim, _ = get_sim()
    full_df = get_dataset_df(sim, snap_num=snap_num)
    save_df = reduce_df(full_df)
    snap = get_snap(snap_num)
    snap_path = os.path.join(base, snap)
    if not os.path.exists(snap_path):
        print(f"Creating directory '{snap_path}' as it does not exist yet.")
        os.system(f"mkdir {snap_path}")
    save_path = os.path.join(snap_path, "0_" + save_name + ".pickle")
    save_df.to_pickle(save_path)
    return


if __name__ == "__main__":
    snap_num = 13
    build_new_df(snap_num)
