"""
Steps before running this script:
0. make sure you have the necessary packages installed to use the cstdn module
1. change the append path to your path for vasp-job-automation
2. change the ion_list to the ions you want to plot
3. decide if you want to save the data to a json file
4. decide if you want to save the figure to a png file
5. specify the name of the OUTCAR file you want to extract data from
6. specify the path of the 'config_x' dir where x is the config number
e.g. 'path/and/stuff/config_1' (alternatively, leave the path as os.getcwd(),
copy this script to the 'config_x' dir, and run it from there)

Note: The 'config_x' dir must contain the 'vol_x' dirs which contain
the OUTCAR files.

"""


import sys
import os

# 1. 
sys.path.append('/storage/home/lam7027/bin/vasp-job-automation') #Change this to your path
import cstdn

# 2.
ion_list = [i for i in range(1, 9)] #recall that range(n, m) = [n, ..., m-1]

# 5. and 6.
df = cstdn.extract_config_data(os.getcwd(), ion_list, outcar_name='OUTCAR', oszicar_name='OSZICAR') # get the data 

fig = cstdn.plot_mv(df, show_fig=True) # plot the data with plotly. opens a browser window, if show_fig=True

# 3.
#optionally save the data to a json file. Change the name of the json file as needed
#df.to_json('mv_data.json') 

# 4.
#Optionally, save the figure as a png file. Change the name of the png file as needed
#fig.write_image('mv_fig.png')
