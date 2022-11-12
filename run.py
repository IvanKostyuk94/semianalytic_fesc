import sys
from create_full_database import create_database


df_name = "gridded_df"
maps_name = "gridded_maps"
create_database(int(sys.argv[1]), df_name=df_name, maps_name=maps_name)
