
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import leastsq
import os
import glob
import shutil
import pandas as pd
import numpy as np
import plotly.express as px
import json
import sys
import plotly.graph_objects as go

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler
from custodian.vasp.jobs import VaspJob
from pymatgen.core import structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun

"""
eos fuctions
Shun-Li Shang wrote the original MATLAB code for EOS fitting.
Hui Sun converted the MATLAB code to python code.
Nigel Hew modified the python code to make it more user-friendly and added more functions.
Luke Myers performed some additional modifications to the python code.
"""

"""
This EOS fitting code is based on the following paper:
Shun-Li Shang et al., Computational Materials Science, 47, 4, (2010).
https://doi.org/10.1016/j.commatsci.2009.12.006

Equations of State:
1:  4-parameter (Teter-Shang) mBM4   1
2:  5-parameter (Teter-Shang) mBM5   2
3:  4-parameter               BM4    3
4:  5-parameter               BM5    4
5:  4-parameter Natural       Log4   5
6:  5-parameter Natural       Log5   6
7:  4-parameter Murnaghan     Mur    7
8:  4-parameter Vinet         Vinet  8
9:  4-parameter Morse         Morse  9
"""
"""
end eos functions
"""

# Function to extract the last occurrence of volume from OUTCAR files
def extract_volume(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'volume' in line:
                volume = float(line.split()[-1])
                break  # Stop searching after finding the last occurrence
    return volume


# Function to extract the last occurrence of pressure from OUTCAR files
def extract_pressure(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'pressure' in line:
                pressure = float(line.split()[3])
                break  # Stop searching after finding the last occurrence
    return pressure


# Function to extract energy from OSZICAR files
def extract_energy(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'F=' in line:
                energy = float(line.split()[4])
                break  # Stop searching after finding the last occurrence
    return energy


def extract_mag_data(outcar_path='OUTCAR'):
    if not os.path.isfile(outcar_path):
        print(f"Warning: File {outcar_path} does not exist. Skipping.")
        return None
    with open(outcar_path, 'r') as file:
        data = []
        step = 0
        found_mag_data = False
        data_start = False
        lines = file.readlines()
        for line in lines:
            if 'magnetization (x)' in line:
                found_mag_data = True
                step += 1
            elif found_mag_data and not data_start and '----' in line:
                data_start = True
            elif data_start and '----' not in line:
                ion = int(line.split()[0])
                s = float(line.split()[1])
                p = float(line.split()[2])
                d = float(line.split()[3])
                tot = float(line.split()[4])
                data.append((step, ion, s, p, d, tot))
            elif data_start and '----' in line:
                data_start = False
                found_mag_data = False
        df = pd.DataFrame(data, columns=['step', '# of ion', 's', 'p', 'd', 'tot'])
        return df


"""
Returns only the 'tot' magnetization of the last step for each specified ion

ion_list should be a list of integers ex: [1, 2, 3, 4]
"""
def extract_simple_mag_data(ion_list, outcar_path='OUTCAR'):
    all_mag_data = extract_mag_data(outcar_path)
    last_step_data = all_mag_data[all_mag_data['step'] == all_mag_data['step'].max()]
    simple_data = last_step_data[last_step_data['# of ion'].isin(ion_list)][['# of ion', 'tot']]
    simple_data.reset_index(drop=True, inplace=True)
    return simple_data

"""
df is a data frame with columns ['config', '# of ion', 'vol', 'tot']
breaks if missing these columns
"""
def plot_mv(df, show_fig=True):
    fig = px.line(df,
                    x='vol',
                    y='tot',
                    color='# of ion',symbol='# of ion',
                    hover_data=['config', '# of ion', 'vol', 'tot'],
                    template='plotly_white')
    fig.update_layout(title='Mag-V',
                        xaxis_title='Volume [A^3]',
                        yaxis_title='Magnetic Moment [mu_B]')
    
    fig.update_yaxes(nticks=10)
    fig.update_xaxes(nticks=10)
    
    # Loop over each trace and update dash length
    for i, trace in enumerate(fig.data):
        dash_length = f"{2+(i+1)}px,{2+2*(i+1)}px"  # Dash length changes with each iteration
        fig.data[-i-1].update(mode='markers+lines',
                            marker=dict(size=8, line=dict(width=1), opacity=0.5),
                            line=dict(width=3, dash=dash_length))


    if show_fig:
        fig.show()
    return fig

"""
~~~WARNING~~~ The currect intent is to replace this function with extract_config_data()
This function grabs the necessary magnetic and volume data from the OUTCAR
for each volume and returns a data frame.

Within the path, there should be folders named vol_0, vol_1, etc.

There should be no other files or directories in the path with 
names starting with 'vol_'.

outcar_name and oszicar_name must be the same in each volume folder.

Consider adding config_name column to the data frame
"""
def extract_config_mv_data(path, ion_list, outcar_name='OUTCAR'):
    dfs_list = []
    start = path.find('config_') + len('config_') # Find the index where "config_" starts and add its length
    config = path[start:] #get the string following "config_"
    for vol_dir in glob.glob(os.path.join(path, 'vol_*')):
        outcar_path = os.path.join(vol_dir, outcar_name)
        if not os.path.isfile(outcar_path):
            print(f"Warning: File {outcar_path} does not exist. Skipping.")
            continue
        vol = extract_volume(outcar_path)
        mag_data = extract_simple_mag_data(ion_list, outcar_path)
        mag_data['vol'] = vol
        mag_data['config'] = config
        dfs_list.append(mag_data)
    df = pd.concat(dfs_list, ignore_index=True).sort_values(by=['vol', '# of ion']).reset_index(drop=True)
    return df

"""
This function grabs all necessary data from the OUTCAR
for each volume and returns a data frame in the tidy data format.

Todo: extract the pressure data,
Todo: extract any other data that might be useful

Within the path, there should be folders named vol_0, vol_1, etc.

There should be no other files or directories in the path with 
names starting with 'vol_'.

outcar_name and oszicar_name must be the same in each volume folder.

Consider adding config_name column to the data frame
"""

def extract_config_data(path, ion_list, outcar_name='OUTCAR', oszicar_name='OSZICAR'):
    dfs_list = []
    start = path.find('config_') + len('config_') # Find the index where "config_" starts and add its length
    config = path[start:] #get the string following "config_"
    for vol_dir in glob.glob(os.path.join(path, 'vol_*')):
        
        outcar_path = os.path.join(vol_dir, outcar_name)
        if not os.path.isfile(outcar_path):
            print(f"Warning: File {outcar_path} does not exist. Skipping.")
            continue

        oszicar_path = os.path.join(vol_dir, oszicar_name)
        if not os.path.isfile(oszicar_path):
            print(f"Warning: File {oszicar_path} does not exist. Skipping.")
            continue

        vol = extract_volume(outcar_path)
        energy = extract_energy(oszicar_path)
        data_collection = extract_simple_mag_data(ion_list, outcar_path)
        data_collection['vol'] = vol
        data_collection['config'] = config
        data_collection['energy'] = energy
        dfs_list.append(data_collection)
    df = pd.concat(dfs_list, ignore_index=True).sort_values(by=['vol', '# of ion']).reset_index(drop=True)
    return df

def three_step_relaxation(path, vasp_cmd, handlers, backup=True):  # Path should contain necessary VASP config files
    original_dir = os.getcwd()
    os.chdir(path)
    step1 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=True,
        final=False,
        suffix='.1relax',
        backup=backup,
    )

    step2 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=True,
        final=False,
        suffix='.2relax',
        backup=backup,
        settings_override=[
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
        ]
    )

    step3 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=True,
        final=True,
        suffix='.3static',
        backup=backup,
        settings_override=[
            {"dict": "INCAR", "action": {"_set": {
                "IBRION": -1,
                "NSW": 0,
                "ISMEAR": -5
            }}},
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
        ]
    )

    jobs = [step1, step2, step3]
    c = Custodian(handlers, jobs, max_errors=3)
    c.run()
    os.chdir(original_dir)


"""
!!!WARNING!!! You probably want to have volumes in decreasing order eg:
volumes = []
for vol in range(300, 370, 10):
    volumes.append(vol)
volumes.reverse()

or
volumes = list(np.linspace(340, 270, 11))

Path should contain starting POSCAR, POTCAR, INCAR, KPOINTS

When restarting, the last volume folder will be deleted and
the second to last volume folder will be used as the starting point.

"""
def vol_series(path, volumes, vasp_cmd, handlers, restarting=False):  
    #write a params.json file to keep track of the parameters used
    #unfortunately, handlers is not json serializable, so the value is replace by a useless string
    params = {'path': path,
              'volumes': volumes,
              'vasp_cmd': vasp_cmd,
              'handlers': 'handlers is not json serializable',
              'restarting': restarting}
    params_json_path = os.path.join(path, 'params.json')
    n = 0
    while os.path.isfile(params_json_path):
        n += 1
        params_json_path = os.path.join(path, 'params_' + str(n) + '.json')
    with open(params_json_path, 'w') as file:
        json.dump(params, file)
    
    # If restarting, find the last volume folder and delete it
    if restarting:
        for j in range(len(volumes)):
            vol_folder_name = 'vol_' + str(j)
            vol_folder_path = os.path.join(path, vol_folder_name)
            if not os.path.exists(vol_folder_path):
                last_vol_folder_name = 'vol_' + str(j - 1)
                last_vol_folder_path = os.path.join(path, last_vol_folder_name)
                break
        if j == 0:
            print("No volumes to restart from. You might want to set restarting=False (which is the default) or check if 'vol_0' exists inside the path")
            return
        
        # Delete the last volume folder
        last_vol_index = j-1
        shutil.rmtree(last_vol_folder_path)

    for i, vol in enumerate(volumes):
        # if restarting, skip volumes that have already been run
        if restarting and i < last_vol_index:
            continue

        # Create vol folder
        vol_folder_name = 'vol_' + str(i)
        vol_folder_path = os.path.join(path, vol_folder_name)
        os.makedirs(vol_folder_path)

        if i == 0:  # Copy from path
            files_to_copy = ['INCAR', 'KPOINTS', 'POSCAR', 'POTCAR']
            for file_name in files_to_copy:
                if os.path.isfile(os.path.join(path, file_name)):
                    shutil.copy2(os.path.join(path, file_name), os.path.join(vol_folder_path, file_name))
        else:  # Copy from previous folder and delete WAVECARs, CHGCARs, CHGs, PROCARs from previous volume folder
            previous_vol_folder_path = os.path.join(path, 'vol_' + str(i - 1))
            source_name_dest_name = [('CONTCAR.3static', 'POSCAR'),
                                    ('INCAR.2relax', 'INCAR'),
                                    ('KPOINTS.1relax', 'KPOINTS'),
                                    ('POTCAR', 'POTCAR'),
                                    ('WAVECAR.3static', 'WAVECAR'),
                                    ('CHGCAR.3static', 'CHGCAR')]
            for file_name in source_name_dest_name:
                file_source = os.path.join(previous_vol_folder_path, file_name[0])
                file_dest = os.path.join(vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)
            # After copying, it is safe to delete some of the WAVECARS, CHGCARS, CHG and PROCARS from the previous volume folder to save space
            # Keeps WAVECAR.3static and CHGCAR.3static
            files_to_delete = ['WAVECAR.1relax', 'WAVECAR.2relax',
                            'CHGCAR.1relax', 'CHGCAR.2relax',
                            'CHG.1relax', 'CHG.2relax', 'CHG.3static',
                            'PROCAR.1relax', 'PROCAR.2relax', 'PROCAR.3static']
            paths_to_delete = []
            for file_name in files_to_delete:
                file_path = os.path.join(previous_vol_folder_path, file_name)
                paths_to_delete.append(file_path)

            for file_path in paths_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                elif restarting and i == last_vol_index:
                    pass
                else:
                    print(f"The file {file_path} does not exist.")

        # Change the volume of the POSCAR
        poscar = os.path.join(vol_folder_path, 'POSCAR')
        struct = structure.Structure.from_file(poscar)
        struct.scale_lattice(vol)
        struct.to_file(poscar, "POSCAR")

        # Run VASP
        print('Running three step relaxation for volume ' + str(vol))
        three_step_relaxation(vol_folder_path, vasp_cmd, handlers, backup=False)


"""
kpoints_list should be a list of strings ex:
    ['1 1 1', '2 2 2', '3 3 3']
incar_tags should be a dictionary ex:
    {'encut' 'ISMEAR': -5, 'IBRION': 2}
only edits the forth line of the KPOINTS file

Todo: use pymatgen to edit the KPOINTS file, not koints_list
"""
def kpoints_conv_test(path, kpoints_list, vasp_cmd, handlers,
                      backup=False):  # Path should contain starting POSCAR, POTCAR, INCAR, KPOINTS
    original_dir = os.getcwd()
    kpoints_conv_dir = os.path.join(path, 'kpoints_conv')
    os.makedirs(kpoints_conv_dir)
    shutil.copy2(os.path.join(path, 'POSCAR'), os.path.join(kpoints_conv_dir, 'POSCAR'))
    shutil.copy2(os.path.join(path, 'POTCAR'), os.path.join(kpoints_conv_dir, 'POTCAR'))
    shutil.copy2(os.path.join(path, 'INCAR'), os.path.join(kpoints_conv_dir, 'INCAR'))
    shutil.copy2(os.path.join(path, 'KPOINTS'), os.path.join(kpoints_conv_dir, 'KPOINTS'))
    os.chdir(kpoints_conv_dir)
    for i, el in enumerate(kpoints_list):
        # Change the kpoints file
        with open('KPOINTS', 'r') as file:
            lines = file.readlines()
            lines[3] = el + '\n'

        with open('KPOINTS', 'w') as file:
            file.writelines(lines)

        # Run the VASP job
        if i == len(kpoints_list) - 1:
            final = True
        else:
            final = False

        job = VaspJob(
            vasp_cmd=vasp_cmd,
            final=False,
            suffix=f'.{i}',
            backup=backup
        )
        c = Custodian(handlers, [job], max_errors=3)
        c.run()
        if os.path.isfile(f'WAVECAR.[i-1]'):
            os.remove(f'WAVECAR.[i-1]')
        if os.path.isfile(f'CHGCAR.[i-1]'):
            os.remove(f'CHGCAR.[i-1]')
        if os.path.isfile(f'CHG.[i-1]'):
            os.remove(f'CHG.[i-1]')
        if os.path.isfile(f'PROCAR.[i-1]'):
            os.remove(f'PROCAR.[i-1]')
    os.chdir(original_dir)


# TODO: Good idea for the below. Maybe we can combine the convergence and plot in the above functions?
def calculate_kpoint_convergence():
    pass


def plot_kpoint_convergence():
    pass


def calculate_encut_convergence():
    pass


def encut_convergence_test():
    pass


def plot_encut_convergence():
    pass


if __name__ == "__main__":
    print("This is a module for importing. It is not meant to be run directly.")

    # # Specify custodian handlers
    # subset = list(VaspErrorHandler.error_msgs.keys())
    # subset.remove("algo_tet")
    # handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]

    # # Specify VASP command
    # vasp_cmd = ["srun", "vasp_std"]

    # # three_step_relaxation('', vasp_cmd, handlers)
    
    # volumes = list(np.linspace(370, 270, 15))

    # vol_series(os.getcwd(), volumes, vasp_cmd, handlers, restarting=True)

    # # kpoints_list = ['4 4 5', '5 5 6', '6 6 7', '7 7 8', '7 7 9', '8 8 10', '12 12 15']
    # # kpoints_conv_test(os.getcwd(), kpoints_list, vasp_cmd, handlers, backup=False)
