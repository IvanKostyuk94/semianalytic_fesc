import os
import pandas as pd
import pyTNG.utils as utils
from utils import get_sim
from utils import get_snap


def get_velocities_df(snap_num):
    sim, _ = get_sim()
    dataset = next(sim.group_cat[snap_num].chunk_generator("subhalo"))
    keys_needed = ["SubhaloVMax", "SubhaloVelDisp"]
    sub_dict = {key: dataset[key] for key in keys_needed}
    dataset_df = utils.dfFromArrDict(sub_dict)

    new_df = pd.DataFrame().assign(
        V_rot=dataset_df[("SubhaloVel", 0)],
        V_disp=dataset_df[("SubhaloVelDisp", 0)],
    )
    return new_df


def update_velocities(
    df_prefix,
    snap_range=(1, 17),
    base_path="/ptmp/mpa/ivkos/semianalytic_fesc",
):
    for i in range(*snap_range):
        snap = get_snap(i)
        dir_path = os.path.join(base_path, snap)
        df_name = df_prefix + str(i) + ".pickle"
        df_path = os.path.join(dir_path, df_name)
        df = pd.read_pickle(df_path)
        velocities_df = get_velocities_df(i)
        RotationVelocity = []
        VelocityDispersion = []
        for galaxy in df.index:
            velocities = velocities_df.loc[galaxy]
            RotationVelocity.append(velocities.V_rot)
            VelocityDispersion.append(velocities.V_disp)
        df["RotationVelocity"] = RotationVelocity
        df["VelocityDispersion"] = VelocityDispersion
        df.to_pickle(df_path)
    return


if __name__ == "__main__":
    update_velocities("gridded_df_")
