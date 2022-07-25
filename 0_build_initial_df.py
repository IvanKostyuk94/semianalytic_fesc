from utils import *
import os


def build_new_df(snap_num, save_name='df'):
    sim, simpath = get_sim()
    full_df = get_dataset_df(sim, snap_num=snap_num)
    save_df = reduce_df(full_df)
    base = '/ptmp/mpa/ivkos/semianalytic_fesc/sn013'
    save_path = os.path.join(base, '0_' + save_name + '.pickle')
    save_df.to_pickle(save_path)
    return


if __name__ == '__main__':
    snap_num = 13
    build_new_df(snap_num)
