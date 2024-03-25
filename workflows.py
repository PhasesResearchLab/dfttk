import os
import sys
import glob
import json
import shutil
import numpy as np
import pandas as pd

from custodian.custodian import Custodian
from custodian.vasp.jobs import VaspJob
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints


def extract_volume(path):
    # Function to extract the last occurrence of volume from an OUTCAR file
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'volume' in line:
                volume = float(line.split()[-1])
                break
    return volume


def extract_pressure(path):
    # Function to extract the last occurrence of pressure from an OUTCAR file
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'pressure' in line:
                pressure = float(line.split()[3])
                break
    return pressure


def extract_energy(path):
    # Function to extract the final energy from an OSZICAR file
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'F=' in line:
                energy = float(line.split()[4])
                break
    return energy


def write_ev(path):
    # Function to write the volumes and energies to a text file
    os.chdir(path)
    folders = glob.glob(os.path.join(path, 'vol_*'))
    data = []
    for folder in folders:
        os.chdir(folder)
        volume = extract_volume('OUTCAR.3static')
        energy = extract_energy('OSZICAR.3static')
        data.append([volume, energy])
        os.chdir(path)
    data = np.array(data)
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]
    np.savetxt('volume_energy.txt', sorted_data, fmt='%f')
    return


def write_pv(path):
    # Function to write the volumes and pressures to a text file
    os.chdir(path)
    folders = glob.glob(os.path.join(path, 'vol_*'))
    data = []
    for folder in folders:
        os.chdir(folder)
        volume = extract_volume('OUTCAR.3static')
        pressure = extract_pressure('OUTCAR.3static')
        data.append([volume, pressure])
        os.chdir(path)
    data = np.array(data)
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]
    np.savetxt('volume_pressure.txt', sorted_data, fmt='%f')
    return


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
        df = pd.DataFrame(
            data, columns=['step', '# of ion', 's', 'p', 'd', 'tot'])
        return df


def extract_simple_mag_data(ion_list, outcar_path='OUTCAR'):
    """
    Returns only the 'tot' magnetization of the last step for each specified ion.
    The ion_list should be a list of integers ex: [1, 2, 3, 4].
    """

    all_mag_data = extract_mag_data(outcar_path)
    last_step_data = all_mag_data[all_mag_data['step']
                                  == all_mag_data['step'].max()]
    simple_data = last_step_data[last_step_data['# of ion'].isin(ion_list)][[
        '# of ion', 'tot']]
    simple_data.reset_index(drop=True, inplace=True)
    return simple_data


def append_energy_per_atom(df):
    """
    I think this function could be replaced by adding this line the the extract_config_data() function
    or something like that. I'm not sure yet.
    """

    df['energy_per_atom'] = df['energy'] / df['number_of_atoms']
    return df


def remove_magmom_data(df):
    """
    This function exists because I should not have combined the magmom data for each ion into the main dataframe.
    Future plans include a redefinition of the extract_config_data() function to return a dataframe with a pointer
    to another dataframe containing the magmom data for each ion for that volume of that config.
    """

    try:
        new_df = df.drop('# of ion', axis=1)
        new_df = new_df.drop('tot', axis=1)
        new_df = new_df.drop_duplicates()
    except Exception as e:
        print("There was an error removing the magmom data. Are you sure there was magnetic data in the df? e: ", e)
        new_df = df
    return new_df


def get_lowest_atomic_energy_configs(df, number_of_lowest=1):
    # This function takes a dataframe and returns the rows with the lowest energy per atom

    lowest_energy_configs = df.nsmallest(number_of_lowest, 'energy_per_atom')
    return lowest_energy_configs


def extract_config_mv_data(path, ion_list, outcar_name='OUTCAR'):
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

    dfs_list = []
    # Find the index where "config_" starts and add its length
    start = path.find('config_') + len('config_')
    config = path[start:]  # Get the string following "config_"
    for vol_dir in glob.glob(os.path.join(path, 'vol_*')):
        outcar_path = os.path.join(vol_dir, outcar_name)
        if not os.path.isfile(outcar_path):
            print(f"Warning: File {outcar_path} does not exist. Skipping.")
            continue
        vol = extract_volume(outcar_path)
        mag_data = extract_simple_mag_data(ion_list, outcar_path)
        mag_data['volume'] = vol
        mag_data['config'] = config
        dfs_list.append(mag_data)
    df = pd.concat(dfs_list, ignore_index=True).sort_values(
        by=['volume', '# of ion']).reset_index(drop=True)
    return df


def extract_config_data(path, ion_list, outcar_name='OUTCAR', oszicar_name='OSZICAR', contcar_name='CONTCAR'):
    """
    !!!Warning!!! this function will soon be deprecated. Use extract_configuration_data() instead if possible.

    This function grabs all necessary data from the OUTCAR
    for each volume and returns a data frame in the tidy data format.

    TODO: extract the pressure data
    TODO: extract any other data that might be useful

    Within the path, there should be folders named vol_0, vol_1, etc.

    There should be no other files or directories in the path with 
    names starting with 'vol_'.

    outcar_name and oszicar_name must be the same in each volume folder.

    Consider adding config_name column to the data frame
    """

    dfs_list = []
    # Find the index where "config_" starts and add its length
    start = path.find('config_') + len('config_')
    config = path[start:]  # Get the string following "config_"
    for vol_dir in glob.glob(os.path.join(path, 'vol_*')):

        outcar_path = os.path.join(vol_dir, outcar_name)
        if not os.path.isfile(outcar_path):
            print(f"Warning: File {outcar_path} does not exist. Skipping.")
            continue

        oszicar_path = os.path.join(vol_dir, oszicar_name)
        if not os.path.isfile(oszicar_path):
            print(f"Warning: File {oszicar_path} does not exist. Skipping.")
            continue

        contcar_path = os.path.join(vol_dir, contcar_name)
        if not os.path.isfile(contcar_path):
            print(f"Warning: File {contcar_path} does not exist. Skipping.")
            continue

        struct = Structure.from_file(contcar_path)
        number_of_atoms = len(struct.sites)
        vol = extract_volume(outcar_path)
        energy = extract_energy(oszicar_path)
        data_collection = extract_simple_mag_data(ion_list, outcar_path)
        data_collection['volume'] = vol
        data_collection['config'] = config
        data_collection['energy'] = energy
        data_collection['number_of_atoms'] = number_of_atoms
        dfs_list.append(data_collection)
    df = pd.concat(dfs_list, ignore_index=True).sort_values(
        by=['volume', '# of ion']).reset_index(drop=True)
    return df


def extract_configuration_data(path, ion_list=[1], outcar_name='OUTCAR', oszicar_name='OSZICAR',
                               contcar_name='CONTCAR', collect_mag_data='False'):
    row_list = []
    # Find the index where "config_" starts and add its length
    start = path.find('config_') + len('config_')
    config = path[start:]  # get the string following "config_"
    for vol_dir in glob.glob(os.path.join(path, 'vol_*')):

        outcar_path = os.path.join(vol_dir, outcar_name)
        if not os.path.isfile(outcar_path):
            print(f"Warning: File {outcar_path} does not exist. Skipping.")
            continue

        oszicar_path = os.path.join(vol_dir, oszicar_name)
        if not os.path.isfile(oszicar_path):
            print(f"Warning: File {oszicar_path} does not exist. Skipping.")
            continue

        contcar_path = os.path.join(vol_dir, contcar_name)
        if not os.path.isfile(contcar_path):
            print(f"Warning: File {contcar_path} does not exist. Skipping.")
            continue

        struct = Structure.from_file(contcar_path)
        number_of_atoms = len(struct.sites)
        vol = extract_volume(outcar_path)
        energy = extract_energy(oszicar_path)
        if collect_mag_data == True:
            mag_data = extract_simple_mag_data(ion_list, outcar_path)
            row = {'volume': vol,
                   'config': config,
                   'energy': energy,
                   'number_of_atoms': number_of_atoms,
                   'mag_data': mag_data
                   }
        else:
            row = {'volume': vol,
                   'config': config,
                   'energy': energy,
                   'number_of_atoms': number_of_atoms
                   }
        row_list.append(row)
    df = pd.DataFrame(row_list)
    return df


def three_step_relaxation(path, vasp_cmd, handlers, copy_magmom=False, backup=False):

    # Path should contain necessary VASP config files
    original_dir = os.getcwd()
    os.chdir(path)
    step1 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=False,
        suffix='.1relax',
        backup=backup,
    )

    step2 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=False,
        suffix='.2relax',
        backup=backup,
        settings_override=[
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
        ]
    )

    step3 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=True,
        suffix='.3static',
        backup=backup,
        settings_override=[
            {"dict": "INCAR", "action": {"_set": {
                "ALGO": "Normal",
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


def ev_curve_series(path, volumes, vasp_cmd, handlers, restarting=False, keep_wavecar=False, keep_chgcar=False, copy_magmom=False):
    """
    For spin-polarized calculations (ISPIN=2), you probably want to have volumes in decreasing order, e.g.:
    volumes = []
    for vol in range(300, 370, 10):
        volumes.append(vol)
    volumes.reverse()

    or
    volumes = list(np.linspace(340, 270, 11))

    Path should contain starting POSCAR, POTCAR, INCAR, and KPOINTS files

    When restarting, the last volume folder will be deleted and
    the second last volume folder will be used as the starting point.
    
    #To Do: fix restarting so that it looks at files relative to the input path.
    """

    # Write a params.json file to keep track of the parameters used
    errors_subset_list = [
        handler.errors_subset_to_catch for handler in handlers]
    params = {'path': path,
              'volumes': volumes,
              'vasp_cmd': vasp_cmd,
              'handlers': errors_subset_list[0],
              'restarting': restarting}
    params_json_path = os.path.join(path, 'params.json')

    n = 0
    params_json_path = os.path.join(path, 'params_' + str(n) + '.json')
    while os.path.isfile(params_json_path):
        n += 1
        params_json_path = os.path.join(path, 'params_' + str(n) + '.json')

    with open(params_json_path, 'w') as file:
        json.dump(params, file)

    # If restarting, the volumes in the vol folders should match the volumes list in order
    # You must supply a volumes list greater than or equal to the number of vol folders
    if restarting:
        vol_folders = [f for f in os.listdir(path) if os.path.isdir(f) and f.startswith('vol')]
        print(vol_folders)
        
        # read volumes completed/started
        volumes_started = []
        for vol_folder in vol_folders:
            try:
                struct = Structure.from_file(os.path.join(path, vol_folder, 'POSCAR.1relax'))
            except Exception as e:
                print(f'possible error: {e}, trying POSCAR')
                try:
                    struct = Structure.from_file(os.path.join(path, vol_folder, 'POSCAR'))
                except Exception as e:
                    print(f"Error: {e}. Could not extract volumes from POSCAR files. Do the files POSCAR.1relax or POSCAR exist in each volume folder?")
                    sys.exit(1)
            vol_started = struct.volume
            volumes_started.append(round(vol_started, 6)) # round to 6 decimal places to avoid floating point errors
            rounded_volumes = [round(vol, 6) for vol in volumes]
            
        # compare volumes started to the begining of the inputed volumes. if they don't match exit.
        if not volumes_started == rounded_volumes[:len(volumes_started)]:
            print(f"Error: The volumes completed/started do not match the start of the inputed volumes list. \n rounded_input_volumes: {rounded_volumes} \n volumes_started (rounded): {volumes_started} Exiting.")
            sys.exit(1)
        else:
            print("The volumes completed/started match the start of the inputed volumes list. continuing restart")

        j = len(vol_folders) - 1
        last_vol_folder_name = 'vol_' + str(j)
        last_vol_folder_path = os.path.join(path, last_vol_folder_name)

        # Failed at the third step
        if all(os.path.isfile(os.path.join(last_vol_folder_path, file)) for file in ['INCAR.2relax', 'POSCAR.2relax', 'KPOINTS.2relax']):
            files = ['INCAR.2relax', 'POSCAR.2relax', 'KPOINTS.2relax',
                     'POTCAR', 'CHGCAR.2relax', 'WAVECAR.2relax']
            source_name_dest_name = [('INCAR.2relax', 'INCAR'),
                                     ('CONTCAR.2relax', 'POSCAR'),
                                     ('KPOINTS.2relax', 'KPOINTS'),
                                     ('CHGCAR.2relax', 'CHGCAR'),
                                     ('WAVECAR.2relax', 'WAVECAR')]
            for file_name in source_name_dest_name:
                file_source = os.path.join(last_vol_folder_path, file_name[0])
                file_dest = os.path.join(last_vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)

            keep_files = [name[1]
                          for name in source_name_dest_name] + ['POTCAR']
            for filename in os.listdir(last_vol_folder_path):
                file_path = os.path.join(last_vol_folder_path, filename)
                if filename not in keep_files:
                    os.remove(file_path)

            # Run VASP
            print('Running three step relaxation for volume ' +
                  str(volumes[j]))
            three_step_relaxation(last_vol_folder_path, vasp_cmd,
                                  handlers, backup=False, copy_magmom=copy_magmom)
            last_vol_index = j + 1

        # Failed at the second step
        elif all(os.path.isfile(os.path.join(last_vol_folder_path, file)) for file in ['INCAR.1relax', 'POSCAR.1relax', 'KPOINTS.1relax']):
            files = ['INCAR.1relax', 'POSCAR.1relax', 'KPOINTS.1relax',
                     'POTCAR', 'CHGCAR.1relax', 'WAVECAR.1relax']
            source_name_dest_name = [('INCAR.1relax', 'INCAR'),
                                     ('CONTCAR.1relax', 'POSCAR'),
                                     ('KPOINTS.1relax', 'KPOINTS'),
                                     ('CHGCAR.1relax', 'CHGCAR'),
                                     ('WAVECAR.1relax', 'WAVECAR')]
            for file_name in source_name_dest_name:
                file_source = os.path.join(last_vol_folder_path, file_name[0])
                file_dest = os.path.join(last_vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)

            keep_files = [name[1]
                          for name in source_name_dest_name] + ['POTCAR']
            for filename in os.listdir(last_vol_folder_path):
                file_path = os.path.join(last_vol_folder_path, filename)
                if filename not in keep_files:
                    os.remove(file_path)

            # Run VASP
            print('Running three step relaxation for volume ' +
                  str(volumes[j]))
            three_step_relaxation(last_vol_folder_path, vasp_cmd,
                                  handlers, backup=False, copy_magmom=copy_magmom)
            last_vol_index = j + 1

        # Failed at the first step
        else:
            # Delete the last volume folder
            shutil.rmtree(last_vol_folder_path)
            last_vol_index = j

    files_to_delete = ['WAVECAR.1relax', 'WAVECAR.2relax',
                       'WAVECAR.3static', 'CHGCAR.3static',
                       'CHGCAR.1relax', 'CHGCAR.2relax',
                       'CHG.1relax', 'CHG.2relax', 'CHG.3static',
                       'PROCAR.1relax', 'PROCAR.2relax', 'PROCAR.3static']
    
    for i, vol in enumerate(volumes):
        # If restarting, skip volumes that have already been run
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
                    shutil.copy2(os.path.join(path, file_name),
                                 os.path.join(vol_folder_path, file_name))
        else:  # Copy from previous folder and delete WAVECARs, CHGCARs, CHGs, PROCARs from previous volume folder
            previous_vol_folder_path = os.path.join(path, 'vol_' + str(i - 1))
            source_name_dest_name = [('CONTCAR.3static', 'POSCAR'),
                                     ('INCAR.2relax', 'INCAR'),
                                     ('KPOINTS.1relax', 'KPOINTS'),
                                     ('POTCAR', 'POTCAR'),
                                     ('WAVECAR.3static', 'WAVECAR'),
                                     ('CHGCAR.3static', 'CHGCAR')]
            for file_name in source_name_dest_name:
                file_source = os.path.join(
                    previous_vol_folder_path, file_name[0])
                file_dest = os.path.join(vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)

            # After copying, it is safe to delete the WAVECAR, CHGCAR, CHG, and PROCAR files from the previous volume folder to save space
            if keep_wavecar:
                files_to_delete.remove('WAVECAR.3static')
            if keep_chgcar:
                files_to_delete.remove('CHGCAR.3static')
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
        struct = Structure.from_file(poscar)
        struct.scale_lattice(vol)
        struct.to_file(poscar, "POSCAR")

        # Run VASP
        print('Running three step relaxation for volume ' + str(vol))
        three_step_relaxation(vol_folder_path, vasp_cmd,
                              handlers, backup=False, copy_magmom=copy_magmom)

    # Delete some files in the last volume folder to save space
    previous_vol_folder_path = os.path.join(path, 'vol_' + str(i))
    paths_to_delete = []
    for file_name in files_to_delete:
        file_path = os.path.join(previous_vol_folder_path, file_name)
        paths_to_delete.append(file_path)

    for file_path in paths_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(f"The file {file_path} does not exist.")


def kpoints_conv_test(path, kppa_list, vasp_cmd, handlers, backup=False):
    original_dir = os.getcwd()
    kpoints_conv_dir = os.path.join(path, 'kpoints_conv')
    os.makedirs(kpoints_conv_dir)

    # Copy VASP input files except KPOINTS
    shutil.copy2(os.path.join(path, 'POSCAR'),
                 os.path.join(kpoints_conv_dir, 'POSCAR'))
    shutil.copy2(os.path.join(path, 'POTCAR'),
                 os.path.join(kpoints_conv_dir, 'POTCAR'))
    shutil.copy2(os.path.join(path, 'INCAR'),
                 os.path.join(kpoints_conv_dir, 'INCAR'))

    # Create KPOINTS file and run VASP
    os.chdir(kpoints_conv_dir)
    struct = Structure.from_file('POSCAR')
    for i, kppa in enumerate(kppa_list):
        kpoints = Kpoints.automatic_density(struct, kppa)
        kpoints.write_file('KPOINTS')

        if i == len(kppa_list) - 1:
            final = True
        else:
            final = False

        # Run the VASP job
        job = VaspJob(
            vasp_cmd=vasp_cmd,
            final=final,
            backup=backup,
            suffix=f'.{kppa}',
            settings_override=[
                {"dict": "INCAR", "action": {"_set": {
                    "ISIF": 2, "NSW": 0
                }}}]
        )
        c = Custodian(handlers, [job], max_errors=3)
        c.run()

        # Remove these files incase you didn't set up the incar correctly.
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
