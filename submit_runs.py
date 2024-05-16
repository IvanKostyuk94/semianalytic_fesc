import os
from utils import get_snap
import pandas as pd
from config import config


def write_job(snap_num, jobname):
    job = f"""#!/bin/bash

#SBATCH -D ./
#SBATCH -o run-{snap_num}.out
#SBATCH -e run-{snap_num}.err
#SBATCH --mail-user=ivkos@mpa-garching.mpg.de
#SBATCH --mail-type=ALL

#SBATCH -J run-{snap_num}
#SBATCH -t 24:00:00

python run.py {snap_num}
"""

    filename = jobname

    with open(filename, "w") as job_file:
        job_file.write(job)

    os.system("chmod +x " + filename)
    return


def is_df_done(df_name, snap_num, base_path=config["base_path"]):
    df_name_full = df_name + "_" + str(snap_num) + ".pickle"
    snap = get_snap(snap_num)
    df_path = os.path.join(base_path, snap, df_name_full)
    try:
        df = pd.read_pickle(df_path)
        contains_fesc = "f_esc" in df.columns
        return all(df["processed"]) and contains_fesc and all(df["map_to_df_done"])
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    df_name = config["df_name"]
    snap_min = config["snap_min"]
    snap_max = config["snap_max"]
    for snap in range(snap_min, snap_max + 1):
        jobname = f"{config['jobname']}_{snap}.sh"
        if is_df_done(df_name, snap):
            print(f"snap {snap} is already finished")
            os.system(f"rm {jobname}")
        else:
            write_job(snap, jobname)
            os.system(f"sbatch {jobname}")
