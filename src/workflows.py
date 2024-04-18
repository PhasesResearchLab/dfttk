import os
import sys
import glob
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from custodian.custodian import Custodian
from custodian.vasp.jobs import VaspJob
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Kpoints


def extract_volume(path):
    """Function to extract the last occurrence of volume from an OUTCAR file

    Args:
        path (str): the path to an OUTCAR file

    Returns:
        float: the volume from an OUTCAR file
    """

    with open(path, "r") as file:
        file_name = os.path.basename(path)
        assert file_name.startswith("OUTCAR"), "File name does not start with 'OUTCAR'"

        lines = file.readlines()
        for line in reversed(lines):
            if "volume" in line:
                volume = float(line.split()[-1])
                break
    return volume


def extract_pressure(path):
    """Function to extract the last occurrence of pressure from an OUTCAR file

    Args:
        path (str): the path to an OUTCAR file

    Returns:
        float: the pressure from an OUTCAR file
    """

    with open(path, "r") as file:
        file_name = os.path.basename(path)
        assert file_name.startswith("OUTCAR"), "File name does not start with 'OUTCAR'"

        lines = file.readlines()
        for line in reversed(lines):
            if "pressure" in line:
                pressure = float(line.split()[3])
                break
    return pressure


def extract_energy(path):
    """Function to extract the final energy from an OSZICAR file

    Args:
        path (str): the path to an OSZICAR file

    Returns:
        float: the final energy from an OSZICAR file
    """

    with open(path, "r") as file:
        file_name = os.path.basename(path)
        assert file_name.startswith(
            "OSZICAR"
        ), "File name does not start with 'OSZICAR'"

        lines = file.readlines()
        for line in reversed(lines):
            if "F=" in line:
                energy = float(line.split()[4])
                break
    return energy


def write_ev(path):
    """Function to write the volumes and energies obtained from ev_curve_series to a text file.
    The data will be obtained from vol_* folders.

    Args:
        path (str): the path to the directory containing the vol_* folders
    """

    original_dir = os.getcwd()
    os.chdir(path)

    folders = [
        name
        for name in os.listdir(os.getcwd())
        if os.path.isdir(name) and name.startswith("vol")
    ]

    data = []
    for folder in folders:
        os.chdir(folder)
        volume = extract_volume("OUTCAR.3static")
        energy = extract_energy("OSZICAR.3static")
        data.append([volume, energy])
        os.chdir("../")

    data = np.array(data)
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]
    np.savetxt("volume_energy.txt", sorted_data, fmt="%f")
    os.chdir(original_dir)


def write_pv(path):
    """Function to write the volumes and pressures obtained from ev_curve_series to a text file.
    The data will be obtained from vol_* folders.

    Args:
        path (str): the path to the directory containing the vol_* folders
    """

    original_dir = os.getcwd()
    os.chdir(path)

    folders = [
        name
        for name in os.listdir(os.getcwd())
        if os.path.isdir(name) and name.startswith("vol")
    ]

    data = []
    for folder in folders:
        os.chdir(folder)
        volume = extract_volume("OUTCAR.3static")
        pressure = extract_pressure("OUTCAR.3static")
        data.append([volume, pressure])
        os.chdir("../")

    data = np.array(data)
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]
    np.savetxt("volume_pressure.txt", sorted_data, fmt="%f")
    os.chdir(original_dir)


def extract_mag_data(outcar_path="OUTCAR"):
    """Extracts the magnetization data from an OUTCAR file and returns a pandas DataFrame.

    Args:
        outcar_path (str, optional): Path to an OUTCAR file. Defaults to "OUTCAR".

    Returns:
        <class 'pandas.core.frame.DataFrame'>: a pandas DataFrame containing the magnetization data
    """    
    
    if not os.path.isfile(outcar_path):
        print(f"Warning: File {outcar_path} does not exist. Skipping.")
        return None
    
    with open(outcar_path, "r") as file:
        data = []
        step = 0
        found_mag_data = False
        data_start = False
        lines = file.readlines()
        for line in lines:
            if "magnetization (x)" in line:
                found_mag_data = True
                step += 1
            elif found_mag_data and not data_start and "----" in line:
                data_start = True
            elif data_start and "----" not in line:
                ion = int(line.split()[0])
                s = float(line.split()[1])
                p = float(line.split()[2])
                d = float(line.split()[3])
                tot = float(line.split()[4])
                data.append((step, ion, s, p, d, tot))
            elif data_start and "----" in line:
                data_start = False
                found_mag_data = False
        df = pd.DataFrame(data, columns=["step", "# of ion", "s", "p", "d", "tot"])
        return df


def extract_tot_mag_data(ion_list, outcar_path="OUTCAR"):
    """Returns only the 'tot' magnetization of the last step for each specified ion.

    Args:
        ion_list (list): The ion_list should be a list of integers ex: [1, 2, 3, 4].
        outcar_path (str, optional): Path to an OUTCAR file. Defaults to "OUTCAR".

    Returns:
        <class 'pandas.core.frame.DataFrame'>: a pandas DataFrame containing the 'tot' magnetization data
    """    

    all_mag_data = extract_mag_data(outcar_path)
    last_step_data = all_mag_data[all_mag_data["step"] == all_mag_data["step"].max()]
    tot_data = last_step_data[last_step_data["# of ion"].isin(ion_list)][
        ["# of ion", "tot"]
    ]
    tot_data.reset_index(drop=True, inplace=True)
    return tot_data


def append_energy_per_atom(df):
    """
    I think this function could be replaced by adding this line the the extract_config_data() function
    or something like that. I'm not sure yet.
    """

    df["energy_per_atom"] = df["energy"] / df["number_of_atoms"]
    return df


def remove_magmom_data(df):
    """
    This function exists because I should not have combined the magmom data for each ion into the main dataframe.
    Future plans include a redefinition of the extract_config_data() function to return a dataframe with a pointer
    to another dataframe containing the magmom data for each ion for that volume of that config.
    """

    try:
        new_df = df.drop("# of ion", axis=1)
        new_df = new_df.drop("tot", axis=1)
        new_df = new_df.drop_duplicates()
    except Exception as e:
        print(
            "There was an error removing the magmom data. Are you sure there was magnetic data in the df? e: ",
            e,
        )
        new_df = df
    return new_df


def get_lowest_atomic_energy_configs(df, number_of_lowest=1):
    # This function takes a dataframe and returns the rows with the lowest energy per atom

    lowest_energy_configs = df.nsmallest(number_of_lowest, "energy_per_atom")
    return lowest_energy_configs


def extract_config_mv_data(path, ion_list, outcar_name="OUTCAR"):
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
    start = path.find("config_") + len("config_")
    config = path[start:]  # Get the string following "config_"
    for vol_dir in glob.glob(os.path.join(path, "vol_*")):
        outcar_path = os.path.join(vol_dir, outcar_name)
        if not os.path.isfile(outcar_path):
            print(f"Warning: File {outcar_path} does not exist. Skipping.")
            continue
        vol = extract_volume(outcar_path)
        mag_data = extract_simple_mag_data(ion_list, outcar_path)
        mag_data["volume"] = vol
        mag_data["config"] = config
        dfs_list.append(mag_data)
    df = (
        pd.concat(dfs_list, ignore_index=True)
        .sort_values(by=["volume", "# of ion"])
        .reset_index(drop=True)
    )
    return df


def extract_config_data(
    path, ion_list, outcar_name="OUTCAR", oszicar_name="OSZICAR", contcar_name="CONTCAR"
):
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
    start = path.find("config_") + len("config_")
    config = path[start:]  # Get the string following "config_"
    for vol_dir in glob.glob(os.path.join(path, "vol_*")):

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
        data_collection["volume"] = vol
        data_collection["config"] = config
        data_collection["energy"] = energy
        data_collection["number_of_atoms"] = number_of_atoms
        dfs_list.append(data_collection)
    df = (
        pd.concat(dfs_list, ignore_index=True)
        .sort_values(by=["volume", "# of ion"])
        .reset_index(drop=True)
    )
    return df


def extract_configuration_data(
    path,
    ion_list=[1],
    outcar_name="OUTCAR",
    oszicar_name="OSZICAR",
    contcar_name="CONTCAR",
    collect_mag_data="False",
):
    row_list = []
    # Find the index where "config_" starts and add its length
    start = path.find("config_") + len("config_")
    config = path[start:]  # get the string following "config_"
    for vol_dir in glob.glob(os.path.join(path, "vol_*")):

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
            row = {
                "volume": vol,
                "config": config,
                "energy": energy,
                "number_of_atoms": number_of_atoms,
                "mag_data": mag_data,
            }
        else:
            row = {
                "volume": vol,
                "config": config,
                "energy": energy,
                "number_of_atoms": number_of_atoms,
            }
        row_list.append(row)
    df = pd.DataFrame(row_list)
    return df


def three_step_relaxation(
    path,
    vasp_cmd,
    handlers,
    copy_magmom=False,
    backup=False,
    default_settings=True,
    settings_override_2relax=None,
    settings_override_3static=None,
):
    """This function runs a three-step relaxation (two consecutive relaxations followed by
       one static) for a given path using VASP. The path should contain the necessary VASP
       input files: POSCAR, POTCAR, INCAR, and KPOINTS.

    Args:
        path (str): the path to the folder containing the VASP input files
        vasp_cmd (list): the VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers (class 'list'): custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        copy_magmom (bool, optional): If True, copies the magmom from an OUTCAR file of one run to the INCAR
        file of the next run. Defaults to False.
        backup (bool, optional): If True, appends the original POSCAR, POTCAR, INCAR, and KPOINTS files with
        .orig. Defaults to False.
    """

    if default_settings:
        settings_override_2relax = [
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
        ]
        settings_override_3static = [
            {
                "dict": "INCAR",
                "action": {
                    "_set": {"ALGO": "Normal", "IBRION": -1, "NSW": 0, "ISMEAR": -5}
                },
            },
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
        ]

    original_dir = os.getcwd()
    os.chdir(path)
    step1 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=False,
        suffix=".1relax",
        backup=backup,
    )

    step2 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=False,
        suffix=".2relax",
        backup=backup,
        settings_override=settings_override_2relax,
    )

    step3 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=True,
        suffix=".3static",
        backup=backup,
        settings_override=settings_override_3static,
    )

    jobs = [step1, step2, step3]
    c = Custodian(handlers, jobs, max_errors=3)
    c.run()
    os.chdir(original_dir)


# TODO: write tests for this function
# TODO: write something to tell you when NELM is reached and in which folder
def ev_curve_series(
    path,
    volumes,
    vasp_cmd,
    handlers,
    restarting=False,
    keep_wavecar=False,
    keep_chgcar=False,
    copy_magmom=False,
    default_settings=True,
    settings_override_2relax=None,
    settings_override_3static=None,
):
    """This function runs a series of three_step_relaxation calculations for a list of volumes. It starts with the first volume, then
       copies the relevant files to the next volume folder, scales the volume of the POSCAR accordingly, and so on.

    Args:
        path (str): the path to the folder containing the VASP input files
        volumes (list): the list of volumes to run the calculations for
        vasp_cmd (list): the VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers (class 'list'): custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        restarting (bool, optional): for restarting failed jobs. Defaults to False.
        keep_wavecar (bool, optional): if True, does not delete WAVECAR.3static. Defaults to False.
        keep_chgcar (bool, optional): if True, does not delete CHGCAR.3static. Defaults to False.
        copy_magmom (bool, optional): If True, copies the magmom from an OUTCAR file of one run to the INCAR
        file of the next run. Defaults to False.
    """

    # Writes a params.json file to keep track of the parameters used
    errors_subset_list = [handler.errors_subset_to_catch for handler in handlers]
    params = {
        "path": path,
        "volumes": volumes,
        "vasp_cmd": vasp_cmd,
        "handlers": errors_subset_list[0],
        "restarting": restarting,
    }

    n = 0
    params_json_path = os.path.join(path, "params_" + str(n) + ".json")
    while os.path.isfile(params_json_path):
        n += 1
        params_json_path = os.path.join(path, "params_" + str(n) + ".json")

    with open(params_json_path, "w") as file:
        json.dump(params, file)

    # Currently, restarting only supports:
    # 1) Volumes list greater than or equal to the number of vol folders
    # 2) The volumes in the vol folders should match the volumes list in order
    if restarting:

        vol_folders = [
            folder
            for folder in os.listdir(path)
            if os.path.isdir(folder) and folder.startswith("vol")
        ]

        # TODO: This is assuming all the previous volumes succeeded. Do we need these try and except blocks?
        volumes_started = []
        for vol_folder in vol_folders:
            try:
                struct = Structure.from_file(
                    os.path.join(path, vol_folder, "POSCAR.1relax")
                )
            except Exception as e:
                print(f"possible error: {e}, trying POSCAR")
                try:
                    struct = Structure.from_file(
                        os.path.join(path, vol_folder, "POSCAR")
                    )
                except Exception as e:
                    print(
                        f"Error: {e}. Could not extract volumes from POSCAR files. Do the files POSCAR.1relax or POSCAR exist in each volume folder?"
                    )
                    sys.exit(1)
            volume_started = struct.volume
            volumes_started.append(round(volume_started, 6))
            rounded_volumes_supplied = [round(volume, 6) for volume in volumes]

        if not volumes_started == rounded_volumes_supplied[: len(volumes_started)]:
            print(
                f"Error: The volumes completed do not match the inputed volumes list of the same number starting from the beginning. \n rounded_input_volumes: {rounded_volumes_supplied} \n volumes_started (rounded): {volumes_started} \n Exiting."
            )
            sys.exit(1)
        else:
            print(
                "The volumes completed match the inputed volumes list of the same number starting from the beginning. Continuing restart"
            )

        j = len(vol_folders) - 1
        last_vol_folder_name = "vol_" + str(j)
        last_vol_folder_path = os.path.join(path, last_vol_folder_name)

        # If the job failed at the third step of three_step_relaxation, restart using the files from the second step
        if all(
            os.path.isfile(os.path.join(last_vol_folder_path, file))
            for file in ["INCAR.2relax", "POSCAR.2relax", "KPOINTS.2relax"]
        ):
            source_name_dest_name = [
                ("INCAR.2relax", "INCAR"),
                ("CONTCAR.2relax", "POSCAR"),
                ("KPOINTS.2relax", "KPOINTS"),
                ("CHGCAR.2relax", "CHGCAR"),
                ("WAVECAR.2relax", "WAVECAR"),
            ]
            for file_name in source_name_dest_name:
                file_source = os.path.join(last_vol_folder_path, file_name[0])
                file_dest = os.path.join(last_vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)

            keep_files = [name[1] for name in source_name_dest_name] + ["POTCAR"]
            for filename in os.listdir(last_vol_folder_path):
                file_path = os.path.join(last_vol_folder_path, filename)
                if filename not in keep_files:
                    os.remove(file_path)

            print("Running three step relaxation for volume " + str(volumes[j]))
            three_step_relaxation(
                last_vol_folder_path,
                vasp_cmd,
                handlers,
                backup=False,
                copy_magmom=copy_magmom,
            )
            last_vol_index = j + 1

        # If the job failed at the second step of three_step_relaxation, restart using the files from the first step
        elif all(
            os.path.isfile(os.path.join(last_vol_folder_path, file))
            for file in ["INCAR.1relax", "POSCAR.1relax", "KPOINTS.1relax"]
        ):
            source_name_dest_name = [
                ("INCAR.1relax", "INCAR"),
                ("CONTCAR.1relax", "POSCAR"),
                ("KPOINTS.1relax", "KPOINTS"),
                ("CHGCAR.1relax", "CHGCAR"),
                ("WAVECAR.1relax", "WAVECAR"),
            ]
            for file_name in source_name_dest_name:
                file_source = os.path.join(last_vol_folder_path, file_name[0])
                file_dest = os.path.join(last_vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)

            keep_files = [name[1] for name in source_name_dest_name] + ["POTCAR"]
            for filename in os.listdir(last_vol_folder_path):
                file_path = os.path.join(last_vol_folder_path, filename)
                if filename not in keep_files:
                    os.remove(file_path)

            print("Running three step relaxation for volume " + str(volumes[j]))
            three_step_relaxation(
                last_vol_folder_path,
                vasp_cmd,
                handlers,
                backup=False,
                copy_magmom=copy_magmom,
            )
            last_vol_index = j + 1

        # If the job failed at the first step of three_step_relaxation, delete the folder
        else:
            shutil.rmtree(last_vol_folder_path)
            last_vol_index = j

    files_to_delete = [
        "WAVECAR.1relax",
        "WAVECAR.2relax",
        "WAVECAR.3static",
        "CHGCAR.3static",
        "CHGCAR.1relax",
        "CHGCAR.2relax",
        "CHG.1relax",
        "CHG.2relax",
        "CHG.3static",
        "PROCAR.1relax",
        "PROCAR.2relax",
        "PROCAR.3static",
    ]

    # This starts the EV curve calculations in series. If restarting, skip the volumes that have already been completed.
    for i, vol in enumerate(volumes):
        if restarting and i < last_vol_index:
            continue

        vol_folder_name = "vol_" + str(i)
        vol_folder_path = os.path.join(path, vol_folder_name)
        os.makedirs(vol_folder_path)

        if i == 0:
            files_to_copy = ["INCAR", "KPOINTS", "POSCAR", "POTCAR"]
            for file_name in files_to_copy:
                if os.path.isfile(os.path.join(path, file_name)):
                    shutil.copy2(
                        os.path.join(path, file_name),
                        os.path.join(vol_folder_path, file_name),
                    )
        else:
            previous_vol_folder_path = os.path.join(path, "vol_" + str(i - 1))
            source_name_dest_name = [
                ("CONTCAR.3static", "POSCAR"),
                ("INCAR.2relax", "INCAR"),
                ("KPOINTS.1relax", "KPOINTS"),
                ("POTCAR", "POTCAR"),
                ("WAVECAR.3static", "WAVECAR"),
                ("CHGCAR.3static", "CHGCAR"),
            ]
            for file_name in source_name_dest_name:
                file_source = os.path.join(previous_vol_folder_path, file_name[0])
                file_dest = os.path.join(vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)

            if keep_wavecar:
                files_to_delete.remove("WAVECAR.3static")
            if keep_chgcar:
                files_to_delete.remove("CHGCAR.3static")
            paths_to_delete = []

            for file_name in files_to_delete:
                file_path = os.path.join(previous_vol_folder_path, file_name)
                paths_to_delete.append(file_path)

            for file_path in paths_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                elif restarting and i == last_vol_index:
                    pass
                
        poscar = os.path.join(vol_folder_path, "POSCAR")
        struct = Structure.from_file(poscar)
        struct.scale_lattice(vol)
        struct.to_file(poscar, "POSCAR")

        print("Running three step relaxation for volume " + str(vol))
        three_step_relaxation(
            vol_folder_path,
            vasp_cmd,
            handlers,
            backup=False,
            copy_magmom=copy_magmom,
            default_settings=default_settings,
            settings_override_2relax=settings_override_2relax,
            settings_override_3static=settings_override_3static,
        )

    previous_vol_folder_path = os.path.join(path, "vol_" + str(i))
    paths_to_delete = []
    for file_name in files_to_delete:
        file_path = os.path.join(previous_vol_folder_path, file_name)
        paths_to_delete.append(file_path)

    for file_path in paths_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)

    vol_folders = [d for d in os.listdir(path) if d.startswith('vol')]
    for vol_folder in vol_folders:
        error_folders = [f for f in os.listdir(os.path.join(path, vol_folder)) if f.startswith('error')]
        if len(error_folders) > 0:
            print(f'In {vol_folder} there are error folders: {error_folders}')
        

def run_phonons(vasp_cmd, handlers, copy_magmom=False, backup=False):
    # TODO: add a way to override the default settings

    step1 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=False,
        suffix=".1relax",
        backup=backup,
    )

    step2 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=copy_magmom,
        final=True,
        suffix=".2phonons",
        backup=backup,
        settings_override=[
            {
                "dict": "INCAR",
                "action": {
                    "_set": {
                        "EDIFF": "1E-6",
                        "IBRION": 6,
                        "NSW": 1,
                        "ISIF": 0,
                        "POTIM": 0.015,
                    }
                },
            },
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
        ],
    )

    jobs = [step1, step2]
    c = Custodian(handlers, jobs, max_errors=3)
    c.run()


def phonons_parallel(path, phonon_volumes, supercell_size, kppa, sbatch_command):

    # Copy files to phonon folders
    vol_folders = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder)) and folder.startswith("vol")
    ]

    ev_volumes_finished = []
    ev_folder_names = []
    for vol_folder in vol_folders:
        structure = Structure.from_file(
            os.path.join(path, vol_folder, "CONTCAR.3static")
        )
        ev_volumes_finished.append(round(structure.volume, 6))
        ev_folder_names.append(vol_folder)

    ev_volumes_and_folders_finished = [
        [a, b] for a, b in zip(ev_volumes_finished, ev_folder_names)
    ]

    for i in range(len(ev_volumes_and_folders_finished)):
        ev_volumes_and_folders_finished[i][1] = ev_volumes_and_folders_finished[i][
            1
        ].replace("vol_", "")

    phonon_volumes_and_folders = []
    for ev_volume_finished, folder in ev_volumes_and_folders_finished:
        if ev_volume_finished in phonon_volumes:
            phonon_volumes_and_folders.append([ev_volume_finished, folder])

    for phonon_volume, phonon_folder in phonon_volumes_and_folders:
        os.makedirs(os.path.join(path, f"phonon_{phonon_folder}"), exist_ok=True)

    source_name_dest_name = [
        ("CONTCAR.3static", "POSCAR"),
        ("INCAR.2relax", "INCAR"),
        ("POTCAR", "POTCAR"),
    ]

    for phonon_volume, phonon_folder in phonon_volumes_and_folders:
        for source_name, dest_name in source_name_dest_name:
            file_source = os.path.join(path, f"vol_{phonon_folder}", source_name)
            file_dest = os.path.join(path, f"phonon_{phonon_folder}", dest_name)
            if os.path.isfile(file_source):
                shutil.copy2(file_source, file_dest)
            shutil.copy2(
                os.path.join(path, sbatch_command),
                os.path.join(path, f"phonon_{phonon_folder}", sbatch_command),
            )

    # Create a supercell and write the KPOINTS file
    for phonon_volume, phonon_folder in phonon_volumes_and_folders:
        structure = Structure.from_file(
            os.path.join(path, f"phonon_{phonon_folder}", "POSCAR")
        )
        structure.make_supercell(supercell_size)
        structure.to_file(
            os.path.join(path, f"phonon_{phonon_folder}", "POSCAR"), "POSCAR"
        )
        kpoints = Kpoints.automatic_density(structure, kppa, force_gamma=True)
        kpoints.write_file(os.path.join(path, f"phonon_{phonon_folder}", "KPOINTS"))

    # Run the phonon calculations in parallel
    for phonon_volume, phonon_folder in phonon_volumes_and_folders:
        os.chdir(os.path.join(path, f"phonon_{phonon_folder}"))
        os.system(sbatch_command)
        os.chdir(path)


def kpoints_conv_test(
    path, kppa_list, vasp_cmd, handlers, force_gamma=True, backup=False
):
    """This function runs a series of VASP calculations with different k-point densities for convergence testing.

    Args:
        path (str): the path to the folder containing the VASP input files
        kppa_list (list): the list of k-point densities to run the calculations for
        vasp_cmd (list): the VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers (class 'list'): custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        force_gamma (bool, optional): If True, forces a gamma-centered mesh. Defaults to True.
        backup (bool, optional): If True, appends the original POSCAR, POTCAR, INCAR, and KPOINTS files with
        .orig. Defaults to False.
    """

    original_dir = os.getcwd()
    kpoints_conv_dir = os.path.join(path, "kpoints_conv")
    os.makedirs(kpoints_conv_dir)

    shutil.copy2(os.path.join(path, "POSCAR"), os.path.join(kpoints_conv_dir, "POSCAR"))
    shutil.copy2(os.path.join(path, "POTCAR"), os.path.join(kpoints_conv_dir, "POTCAR"))
    shutil.copy2(os.path.join(path, "INCAR"), os.path.join(kpoints_conv_dir, "INCAR"))

    os.chdir(kpoints_conv_dir)
    struct = Structure.from_file("POSCAR")
    for i, kppa in enumerate(kppa_list):
        kpoints = Kpoints.automatic_density(struct, kppa, force_gamma=force_gamma)
        kpoints.write_file("KPOINTS")

        if i == len(kppa_list) - 1:
            final = True
        else:
            final = False

        job = VaspJob(
            vasp_cmd=vasp_cmd,
            final=final,
            backup=backup,
            suffix=f".{kppa}",
            settings_override=[
                {"dict": "INCAR", "action": {"_set": {"IBRION": -1, "NSW": 0}}}
            ],
        )
        c = Custodian(handlers, [job], max_errors=3)
        c.run()

        if os.path.isfile(f"WAVECAR.{kppa}"):
            os.remove(f"WAVECAR.{kppa}")
        if os.path.isfile(f"CHGCAR.{kppa}"):
            os.remove(f"CHGCAR.{kppa}")
        if os.path.isfile(f"CHG.{kppa}"):
            os.remove(f"CHG.{kppa}")
        if os.path.isfile(f"PROCAR.{kppa}"):
            os.remove(f"PROCAR.{kppa}")
    os.chdir(original_dir)
    return


def calculate_kpoint_conv(path, kppa_list, plot=True):
    """This function calculates the energy convergence with respect to k-point density and plots the results.

    Args:
        path (str): the path to the folder containing the VASP input files
        kppa_list (list): the list of k-point densities to run the calculations for
        plot (bool, optional): If True, plots the results. Defaults to True.
    """
    original_dir = os.getcwd()
    kpoints_conv_dir = os.path.join(path, "kpoints_conv")

    os.chdir(kpoints_conv_dir)
    data = []
    for kppa in kppa_list:
        energy = extract_energy(f"OSZICAR.{kppa}")
        data.append([kppa, energy])
    data = np.array(data)
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]
    num_atoms = len(Structure.from_file(f"POSCAR.{kppa}").sites)
    sorted_data = np.column_stack((sorted_data, np.zeros(len(sorted_data))))
    sorted_data[1:, 2] = (sorted_data[1:, 1] - sorted_data[:-1, 1]) / num_atoms * 1000
    os.chdir(path)
    np.savetxt("kppa_energy.txt", sorted_data, fmt="%f")

    if plot:
        fig, axis = plt.subplots(1, 2, figsize=(12, 6))
        axis[0].plot(sorted_data[:, 0], sorted_data[:, 1], marker="o")
        axis[0].set_xlabel("k-point density")
        axis[0].set_ylabel("Energy (eV)")
        axis[1].plot(sorted_data[:, 0], sorted_data[:, 2], marker="o")
        axis[1].axhline(y=1, color="black", linestyle="--")
        axis[1].axhline(y=-1, color="black", linestyle="--")
        axis[1].set_xlabel("k-point density")
        axis[1].set_ylabel("ΔEnergy (meV/atom)")
        plt.tight_layout()
        plt.savefig("kpoint_conv.png", dpi=300)
    os.chdir(original_dir)


def encut_conv_test(path, encut_list, vasp_cmd, handlers, backup=False):
    """This function runs a series of VASP calculations with different ENCUT values for convergence testing.

    Args:
        path (str): the path to the folder containing the VASP input files
        encut_list (list): the list of ENCUT values to run the calculations for
        vasp_cmd (list): the VASP commands to run VASP specific to your system. E.g. ["srun", "vasp_std"].
        handlers (class 'list'): custodian handlers to catch errors. See class 'custodian.vasp.handlers.VaspErrorHandler'.
        backup (bool, optional): If True, appends the original POSCAR, POTCAR, INCAR, and KPOINTS files with
        .orig. Defaults to False.
    """
    original_dir = os.getcwd()
    encut_conv_dir = os.path.join(path, "encut_conv")
    os.makedirs(encut_conv_dir)

    shutil.copy2(os.path.join(path, "POSCAR"), os.path.join(encut_conv_dir, "POSCAR"))
    shutil.copy2(os.path.join(path, "KPOINTS"), os.path.join(encut_conv_dir, "KPOINTS"))
    shutil.copy2(os.path.join(path, "POTCAR"), os.path.join(encut_conv_dir, "POTCAR"))
    shutil.copy2(os.path.join(path, "INCAR"), os.path.join(encut_conv_dir, "INCAR"))

    os.chdir(encut_conv_dir)
    for i, encut in enumerate(encut_list):
        if i == len(encut_list) - 1:
            final = True
        else:
            final = False

        job = VaspJob(
            vasp_cmd=vasp_cmd,
            final=final,
            backup=backup,
            suffix=f".{encut}",
            settings_override=[
                {
                    "dict": "INCAR",
                    "action": {"_set": {"IBRION": -1, "NSW": 0, "ENCUT": encut}},
                }
            ],
        )
        c = Custodian(handlers, [job], max_errors=3)
        c.run()

        if os.path.isfile(f"WAVECAR.{encut}"):
            os.remove(f"WAVECAR.{encut}")
        if os.path.isfile(f"CHGCAR.{encut}"):
            os.remove(f"CHGCAR.{encut}")
        if os.path.isfile(f"CHG.{encut}"):
            os.remove(f"CHG.{encut}")
        if os.path.isfile(f"PROCAR.{encut}"):
            os.remove(f"PROCAR.{encut}")
    os.chdir(original_dir)


def calculate_encut_conv(path, encut_list, plot=True):
    """This function calculates the energy convergence with respect to ENCUT and plots the results.

    Args:
        path (str): the path to the folder containing the VASP input files
        encut_list (list): the list of ENCUT values to run the calculations for
        plot (bool, optional): If True, plots the results. Defaults to True.
    """
    original_dir = os.getcwd()
    encut_conv_dir = os.path.join(path, "encut_conv")

    os.chdir(encut_conv_dir)
    data = []
    for encut in encut_list:
        energy = extract_energy(f"OSZICAR.{encut}")
        data.append([encut, energy])
    data = np.array(data)
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]
    num_atoms = len(Structure.from_file(f"POSCAR.{encut}").sites)
    sorted_data = np.column_stack((sorted_data, np.zeros(len(sorted_data))))
    sorted_data[1:, 2] = (sorted_data[1:, 1] - sorted_data[:-1, 1]) / num_atoms * 1000
    os.chdir(path)
    np.savetxt("encut_energy.txt", sorted_data, fmt="%f")

    if plot:
        fig, axis = plt.subplots(1, 2, figsize=(12, 6))
        axis[0].plot(sorted_data[:, 0], sorted_data[:, 1], marker="o")
        axis[0].set_xlabel("ENCUT (eV)")
        axis[0].set_ylabel("Energy (eV)")
        axis[1].plot(sorted_data[:, 0], sorted_data[:, 2], marker="o")
        axis[1].axhline(y=1, color="black", linestyle="--")
        axis[1].axhline(y=-1, color="black", linestyle="--")
        axis[1].set_xlabel("ENCUT")
        axis[1].set_ylabel("ΔEnergy (meV/atom)")
        plt.tight_layout()
        plt.savefig("encut_conv.png", dpi=300)
    os.chdir(original_dir)


if __name__ == "__main__":
    print("This is a module for importing. It is not meant to be run directly.")
    print("But anyway, here are some tests!")

    # TODO: Change test_data to something more appropriate
    # TODO: Is there a better way to specify these paths?
    # At the moment, have to run the tests from the src directory
    OUTCAR_path = "../test_data/FeSe/configurations/config_18/vol_1/OUTCAR.3static"
    OSZICAR_path = "../test_data/FeSe/configurations/config_18/vol_1/OSZICAR.3static"

    volume = extract_volume(OUTCAR_path)
    pressure = extract_pressure(OUTCAR_path)
    energy = extract_energy(OSZICAR_path)

    assert extract_volume(OUTCAR_path) == 333.0
    assert extract_pressure(OUTCAR_path) == -19.74
    assert extract_energy(OSZICAR_path) == -101.28406

    path = "../test_data/FeSe/configurations/config_18"
    write_ev(path)
    data = np.loadtxt(os.path.join(path, "volume_energy.txt"))
    expected_data = np.array(
        [
            [298.0, -101.64358],
            [305.0, -101.58832],
            [312.0, -101.52038],
            [319.0, -101.44049],
            [326.0, -101.36327],
            [333.0, -101.28406],
        ]
    )

    assert np.array_equal(data, expected_data), "Data does not match expected values"

    write_pv(path)
    data = np.loadtxt(os.path.join(path, "volume_pressure.txt"))
    expected_data = np.array(
        [
            [298.0, -10.74],
            [305.0, -18.71],
            [312.0, -14.49],
            [319.0, -19.19],
            [326.0, -29.42],
            [333.0, -19.74],
        ]
    )

    assert np.array_equal(data, expected_data), "Data does not match expected values"

    print("Tests passed")
