"""
Steps before running this script:
0. make sure you have the necessary packages installed to use the cstdn module
1. change the append path to your path for vasp-job-automation
2. change the ion_list to the ions you want to plot
3. decide if you want to save the data to a json file
4. decide if you want to save the figure to a png file
5. specify the name of the OUTCAR file you want to extract data from
6. specify the path of the 'config_x' dir where x is the config number
7. specify the name of the OSZICAR file you want to extract data from
e.g. 'path/and/stuff/config_1' (alternatively, leave the path as os.getcwd(),
copy this script to the 'config_x' dir, and run it from there)

Note: The 'config_x' dir must contain the 'vol_x' dirs which contain
the OUTCAR files.

"""


import sys
import os
import pandas as pd

# 1. 
sys.path.append('/storage/home/lam7027/bin/vasp-job-automation') #Change this to your path
import cstdn
import os

# 2.
ion_list = [i for i in range(1, 9)] #recall that range(n, m) = [n, ..., m-1]

# directory that contains the all the 'config_x' dirs
configurations_dir = '/storage/home/lam7027/work/FeSe/cmme_2x2x1_vol_series_r2scan_rvv10_fixed_poscar/configurations'

#make a list of all the 'config_x' dirs
config_dirs = [os.path.join(configurations_dir, d) for d in os.listdir(configurations_dir) if os.path.isdir(os.path.join(configurations_dir, d))]

df_list = []
for config_dir in config_dirs:
    config_df = cstdn.extract_config_data(config_dir, ion_list, outcar_name='OUTCAR.2relax', oszicar_name='OSZICAR.2relax') # get the data 
    ev_fig = cstdn.plot_ev(config_df, show_fig=False)
    mv_fig = cstdn.plot_mv(config_df, show_fig=False)
    ev_fig.write_image(f'{os.path.basename(config_dir)}_ev_fig.png')
    mv_fig.write_image(f'{os.path.basename(config_dir)}_mv_fig.png')
    df_list.append(config_df)

df = pd.concat(df_list, ignore_index=True)
cstdn.plot_ev(df, show_fig=True)

# plot the data with plotly. opens a browser window, if show_fig=True
fig = cstdn.plot_ev(df, show_fig=True) 

# 3.
#optionally save the data to a json file. Change the name of the json file as needed
#df.to_json('mv_data.json') 

# 4.
#Optionally, save the figure as a png file. Change the name of the png file as needed
#fig.write_image('mv_fig.png')
