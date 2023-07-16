# The base path where the maps and dataframes will be stored
# For each snapshot a subfolder named sn*snap_num* will be
# created if it does not exist
base_path = "/ptmp/mpa/ivkos/semianalytic_fesc"

# Prefix of the dataframe and maps file
# The full name will be name_snapshot.hdf5/pickle
df_name = "df"
maps_name = "maps"

# What is the size of the grid for each galaxy
# Make sure the star-forming regions are sufficiently resolved
grid_size = 100

# Use breakout condition (Ferrara 2023 eq. 14)
with_breakout = True

# Range of snapshots to be processed
snap_min = 0
snap_max = 16

# Name of the batchjob the snapnumber will be added to it
jobname = "gridding"
