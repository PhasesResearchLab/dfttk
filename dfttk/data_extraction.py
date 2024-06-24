# Standard library imports
import os
import glob

# Related third party imports
import numpy as np
import pandas as pd

# Local application/library specific imports
from pymatgen.core.structure import Structure

# DFTTK imports
from dfttk.magnetism import determine_magnetic_ordering


def extract_volume(path: str) -> float:
    """Extract the volume of a structure from a POSCAR/CONTCAR file

    Args:
        path: the path to a POSCAR/CONTCAR file

    Returns:
        The the volume of the structure
    """

    structure = Structure.from_file(path)
    volume = round(structure.volume, 6)

    return volume


def extract_pressure(path: str) -> float:
    """Extract the last occurrence of pressure from an OUTCAR file

    Args:
        path: the path to an OUTCAR file

    Returns:
        the pressure from an OUTCAR file
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


def extract_energy(path: str) -> float:
    """Extract the final energy from an OSZICAR file

    Args:
        path: the path to an OSZICAR file

    Returns:
        The final energy from an OSZICAR file
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


def write_ev(path: str) -> None:
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
        volume = extract_volume("CONTCAR.3static")
        energy = extract_energy("OSZICAR.3static")
        data.append([volume, energy])
        os.chdir("../")

    data = np.array(data)
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]
    np.savetxt("volume_energy.txt", sorted_data, fmt="%f")
    os.chdir(original_dir)


def write_pv(path: str) -> None:
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
        volume = extract_volume("CONTCAR.3static")
        pressure = extract_pressure("OUTCAR.3static")
        data.append([volume, pressure])
        os.chdir("../")

    data = np.array(data)
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]
    np.savetxt("volume_pressure.txt", sorted_data, fmt="%f")
    os.chdir(original_dir)


def extract_mag_data(outcar_path: str = "OUTCAR") -> pd.DataFrame:
    """Extracts the magnetization data from an OUTCAR file and returns the data as a pandas DataFrame in the same format and headings as seen in the OUTCAR.

    Args:
        outcar_path: Path to an OUTCAR file. Defaults to "OUTCAR".

    Returns:
        Pandas DataFrame containing the magnetization data
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
            elif found_mag_data and not data_start and "# of ion" in line:
                headers = line.split()
                headers.pop(0)  # '#'
                headers.pop(0)  # 'of'
                headers.pop(0)  # 'ion'
                headers.insert(0, '#_of_ion')
            elif found_mag_data and not data_start and "----" in line:
                data_start = True
            elif data_start and "----" not in line:
                ion = int(line.split()[0])
                data_line = line.split()[1:]
                data_line = [float(data) for data in data_line]
                data.append((step, ion, *data_line))
            elif data_start and "----" in line:
                data_start = False
                found_mag_data = False
        columns = ["step"] + headers
        df = pd.DataFrame(data, columns=columns)
        return df


# TODO just get mag data for all the ions
def extract_tot_mag_data(outcar_path: str = "OUTCAR") -> pd.DataFrame:
    """Returns only the 'tot' magnetization of the last step for each specified ion.

    Args:
        outcar_path: Path to an OUTCAR file. Defaults to "OUTCAR".

    Returns:
        a pandas DataFrame containing the 'tot' magnetization data
    """

    all_mag_data = extract_mag_data(outcar_path)
    last_step_data = all_mag_data[all_mag_data["step"] == all_mag_data["step"].max()]
    tot_data = last_step_data[["#_of_ion", "tot"]]
    tot_data.reset_index(drop=True, inplace=True)
    return tot_data

def parse_magmom_line(line: str) -> pd.DataFrame:
    """reads vasp formatted MAGMOM line from an INCAR or OUTCAR

    Args:
        line: string containing vasp formatted MAGMOM line

    Returns:
        pd.DataFrame: with columns '#_of_ion' and 'tot' containing the input magnetic moments for each atom.
    """    
    # Remove "MAGMOM = " from the line and split it into parts
    parts = line.replace("MAGMOM = ", "").split()
    magmoms = []
    for part in parts:
        if '*' in part:
            count, value = part.split('*')
            magmoms.extend([float(value)] * int(count))
        else:
            magmoms.append(float(part))

    number_of_ion = list(range(1, len(magmoms) + 1))
    df = pd.DataFrame({'#_of_ion': number_of_ion, 'tot': magmoms})
    return df

def extract_input_mag_data(outcar_path: str = "OUTCAR") -> pd.DataFrame:
    """reads the first line of the OUTCAR that contains "MAGMOM", which should be the input magnetic moments for each atom.
    Also works for INCARs

    Args:
        outcar_path: path to the OUTCAR. Defaults to "OUTCAR".

    Raises:
        ValueError: if there is no line that contains MAGMOM. (non magnetic calculation)

    Returns:
        pd.DataFrame: with columns '#_of_ion' and 'tot' containing the input magnetic moments for each atom.
    """    
    if not os.path.isfile(outcar_path):
        print(f"Warning: File {outcar_path} does not exist. Skipping.")
        return None

    with open(outcar_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            caps_line = line.upper()
            if "MAGMOM" in caps_line:
                return parse_magmom_line(line)
        raise ValueError("No MAGMOM line found in OUTCAR")
                
def extract_configuration_data(
    path: list[str],
    outcar_name: str = "OUTCAR.3static",
    oszicar_name: str = "OSZICAR.3static",
    contcar_name: str = "CONTCAR.3static",
    collect_mag_data: bool = False,
    magmom_tolerance: float = 1e-12,
    total_magnetic_moment_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Extracts the volume, configuration, energy, number of atoms, and magnetization data (if specified) from calculations
    run by ev_curve_series and returns a pandas DataFrame.

    Args:
        path: the path containing a config_* folder which contain vol_* folders
        outcar_name: name of the OUTCAR file. Defaults to "OUTCAR".
        oszicar_name: name of the OSZICAR file. Defaults to "OSZICAR".
        contcar_name: name of the CONTCAR file. Defaults to "CONTCAR".
        collect_mag_data: if True, collect the magnetization data using extract_tot_mag_data. Defaults to
        False.
        magmom_tolerance: the tolerance for the total magnetic moment to be considered zero. Defaults to 0.

    Returns:
        pandas DataFrame: a pandas DataFrame containing the volume, configuration, energy, number of atoms, and
        magnetization data (if specified)
    """

    # Find the index where "config_" starts and add its length
    start = path.find("config_") + len("config_")
    config = path[start:]  # get the string following "config_"

    row_list = []
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
        vol = extract_volume(contcar_path)
        energy = extract_energy(oszicar_path)
        energy_per_atom = energy / number_of_atoms
        vol_per_atom = vol / number_of_atoms
        if collect_mag_data == True:
            mag_data = extract_tot_mag_data(outcar_path)
            total_magnetic_moment = mag_data["tot"].sum()
            magnetic_ordering = determine_magnetic_ordering(
                mag_data,
                magmom_tolerance=magmom_tolerance,
                total_magnetic_moment_tolerance=total_magnetic_moment_tolerance,
            )

            row = {
                "config": config,
                "volume": vol,
                "volume_per_atom": vol_per_atom,
                "energy": energy,
                "energy_per_atom": energy_per_atom,
                "number_of_atoms": number_of_atoms,
                "total_magnetic_moment": total_magnetic_moment,
                "magnetic_ordering": magnetic_ordering,
                "mag_data": mag_data,
            }
        else:
            row = {
                "config": config,
                "volume": vol,
                "volume_per_atom": vol_per_atom,
                "energy": energy,
                "energy_per_atom": energy_per_atom,
                "number_of_atoms": number_of_atoms,
            }
        row_list.append(row)
    df = pd.DataFrame(row_list)
    return df

def recursive_extract_configuration_data(
    config_dirs: list[str],
    outcar_name: str = "OUTCAR",
    oszicar_name: str = "OSZICAR",
    contcar_name: str = "CONTCAR",
    collect_mag_data: bool = False,
    magmom_tolerance: float = 0,
    total_magnetic_moment_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """convenience function to extract configuration data from multiple config directories.
    Runs extract_configuration_data for each config directory in a list.

    Args:
        config_dirs: list of paths to config directories that will be passed to extract_configuration_data()
        outcar_name: name of the OUTCAR file. Defaults to "OUTCAR".
        oszicar_name: name of the OSZICAR file. Defaults to "OSZICAR".
        contcar_name: name of the CONTCAR file. Defaults to "CONTCAR".
        collect_mag_data: if True, collect the magnetization data using extract_tot_mag_data. Defaults to
        False.
        OUTCAR. Defaults to [1].
        magmom_tolerance: the tolerance for the total magnetic moment to be considered zero. Defaults to 0.

    """
    df_list = []
    for config_dir in config_dirs:
        try:
            config_df = extract_configuration_data(
                config_dir,
                outcar_name=outcar_name,
                oszicar_name=oszicar_name,
                contcar_name=contcar_name,
                collect_mag_data=collect_mag_data,
                magmom_tolerance=magmom_tolerance,
                total_magnetic_moment_tolerance=total_magnetic_moment_tolerance,
            )
            df_list.append(config_df)
        except Exception as e:
            print(f"Error in {config_dir}: {e}")
    df = pd.concat(df_list, ignore_index=True)
    return df


