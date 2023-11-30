import numpy as np
import pandas as pd
import h5py
import os
import pickle
from utils import get_snap
from multiprocessing import Pool
from functools import partial

np.seterr(divide="ignore", invalid="ignore")


def get_average_N_d(maps):
    return 1 / np.sum(1 / np.array(maps["N_d"]))


def get_df_quantity(idces, prop, prop_maps, df, scale, testing=False):

    results = []

    flux_quantities = ["Ion_flux", "Bol_flux"]
    summed_quantities = [
        "M_gas",
        "M_star",
        "SFR",
    ]
    average_quantities = [
        "Dust_norm",
        "Column_dens",
        "Column_dens_stroemgren",
        "N_d",
        "N_ratio",
        "N_red",
        "U",
        "U1",
        "f_g",
        "f_g_crit",
        "p_r",
        "sigma_d_H",
        "n_gas",
    ]

    for index in idces:
        prop_dict = {}
        prop_dict["idx"] = index
        if prop in summed_quantities:
            quant = np.sum(prop_maps[prop][index])
        elif prop in flux_quantities:
            if testing:
                grid_column = f"Grid_cell_size_{scale}"
            else:
                grid_column = "Grid_cell_size"
            grid_size = df.loc[index, grid_column]
            quant = np.sum(prop_maps[prop][index]) * grid_size**2
        elif prop in average_quantities:
            quant = np.mean(np.ma.masked_invalid(prop_maps[prop][index]))
        elif prop == "Metallicity":
            quant = np.sum(
                np.array(prop_maps["M_gas"][index])
                * np.ma.masked_invalid(
                    np.array(prop_maps["Metallicity"][index])
                )
            ) / np.sum(prop_maps["M_gas"][index])
        elif prop == "f_esc":
            quant = np.sum(
                np.array(prop_maps["f_esc"][index])
                * np.array(prop_maps["Ion_flux"][index])
            ) / np.sum(prop_maps["Ion_flux"][index])
        else:
            continue
        # The scale here is only for testing
        if testing:
            if prop in flux_quantities:
                column_name = prop[:-4] + "em_" + str(scale)
            else:
                column_name = prop + "_" + str(scale)
        else:
            if prop in flux_quantities:
                column_name = prop[:-4] + "em"
            else:
                column_name = prop

        prop_dict[prop] = quant
        results.append(prop_dict)

    return results


def convert_to_dict(hdf_file, scale, df):
    full_dict = {}
    idces = np.array(df.index, dtype="int")
    for prop in hdf_file[scale][str(df.index[0])].keys():
        prop_dict = {}
        for idx in idces:
            prop_dict[idx] = np.copy(hdf_file[scale][str(idx)][prop])
        full_dict[prop] = prop_dict
    return full_dict


def add_map_quantities(df, prop_dict, scale, Nproc, testing=False):
    for prop in prop_dict.keys():
        if testing:
            column_name = prop + "_" + scale
        else:
            column_name = prop

        if not (column_name in df.columns):
            df[column_name] = np.nan

        idces = np.array(df.index, dtype="int")

        np.random.shuffle(idces)
        chunks = np.array_split(idces, Nproc)

        get_df_quant_batch = partial(
            get_df_quantity,
            prop=prop,
            prop_maps=prop_dict,
            df=df,
            scale=scale,
            testing=testing,
        )

        with Pool(Nproc) as executor:
            pool_results = executor.map(get_df_quant_batch, chunks)

        for result in pool_results:
            # print(result)
            for element in result:
                df.loc[element["idx"], prop] = element[prop]

        # for idx in df.index:
        #     get_df_quantity(prop, hdf_file, df, idx, scale, testing=testing)
    return


def update_map_df(
    snap_num,
    scale,
    hdf_name,
    df_name,
    base,
    output_name=None,
    testing=False,
    Nproc=40,
):
    if output_name is None:
        output_name = df_name
    snap = get_snap(snap_num)
    hdf_filename = hdf_name + ".hdf5"
    hdf_path = os.path.join(base, snap, hdf_filename)

    df_filename = df_name + ".pickle"
    output_filename = output_name + ".pickle"

    dict_filename = hdf_name + ".pickle"
    dict_path = os.path.join(base, snap, dict_filename)
    origin_path = os.path.join(base, snap, df_filename)
    destination_path = os.path.join(base, snap, output_filename)

    hdf_file = h5py.File(hdf_path, "a")
    df = pd.read_pickle(origin_path)

    if not os.path.isfile(dict_path):
        print("Creating dictionary file")
        convert_hdf_to_pickle(
            snap_num, str(scale), hdf_name, df_name, base, output_name=None
        )
        print("Finished creating dictionary file")

    with open(dict_path, "rb") as handle:
        prop_dict = pickle.load(handle)

    add_map_quantities(df, prop_dict, str(scale), Nproc, testing=testing)

    df.to_pickle(destination_path)
    hdf_file.close()
    return


def convert_hdf_to_pickle(
    snap_num, scale, hdf_name, df_name, base, output_name=None
):
    if output_name is None:
        output_name = hdf_name

    snap = get_snap(snap_num)
    hdf_filename = hdf_name + ".hdf5"
    hdf_path = os.path.join(base, snap, hdf_filename)

    df_filename = df_name + ".pickle"
    output_filename = output_name + ".pickle"

    origin_path = os.path.join(base, snap, df_filename)
    destination_path = os.path.join(base, snap, output_filename)

    hdf_file = h5py.File(hdf_path, "a")
    df = pd.read_pickle(origin_path)

    full_dict = convert_to_dict(hdf_file, scale, df)

    with open(destination_path, "wb") as handle:
        pickle.dump(full_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return
