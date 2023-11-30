import os
import numpy as np
import pandas as pd
from config import config


def get_snap_num(z):
    z_to_snap = {6: 13, 8: 8, 10: 4}
    return z_to_snap[z]


def add_analytic_fesc_numerical_df(
    numerical_df_name,
    analytic_df_name=config["full_df_name"],
    base_path=config["base_path"],
):
    numerical_df_path = os.path.join(base_path, numerical_df_name)
    numerical_df = pd.read_pickle(numerical_df_path)

    analytic_df_path = os.path.join(base_path, analytic_df_name + ".pickle")
    analytic_df = pd.read_pickle(analytic_df_path)

    numerical_df["analytic_fesc"] = np.nan
    numerical_df["analytic_Q0"] = np.nan

    redshifts = [6, 8, 10]
    counter = 0
    for z in redshifts:
        numerical_sub_df = numerical_df[numerical_df.z == z]
        snap_num = get_snap_num(z)
        analytic_sub_df = analytic_df[
            analytic_df.z == analytic_df.z.unique()[snap_num]
        ]

        for _, element in numerical_sub_df.iterrows():
            galaxy_id = element.GalaxyID
            analytic_fesc = analytic_sub_df.loc[
                analytic_sub_df["idx"] == galaxy_id
            ].f_esc
            analytic_Q0 = analytic_sub_df.loc[
                analytic_sub_df["idx"] == galaxy_id
            ]["Ion_em"]
            try:
                numerical_df.loc[
                    (numerical_df.z == z) & (numerical_df.ID == element.ID),
                    "analytic_fesc",
                ] = np.float64(analytic_fesc)
                numerical_df.loc[
                    (numerical_df.z == z) & (numerical_df.ID == element.ID),
                    "analytic_Q0",
                ] = np.float64(analytic_Q0)
            except:
                print(analytic_fesc)

            counter += 1
            if counter % 100 == 0:
                print(counter)

    numerical_df.to_pickle(numerical_df_path)


if __name__ == "__main__":
    add_analytic_fesc_numerical_df("numerical_runs.pickle")
