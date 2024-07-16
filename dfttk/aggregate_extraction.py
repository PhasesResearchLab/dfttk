"""
Extract relevant data from VASP output files.
"""

# Standard library imports
import os
import glob

# Related third party imports
import pandas as pd

# Local application/library specific imports
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.inputs import Incar

# DFTTK imports
from dfttk.data_extraction import (
    extract_volume,
    extract_energy,
    extract_tot_mag_data,
    extract_kpoints,
)
from dfttk.magnetism import determine_magnetic_ordering


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
        space_group = SpacegroupAnalyzer(struct).get_space_group_symbol()
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
                "number_of_atoms": number_of_atoms,
                "volume": vol,
                "volume_per_atom": vol_per_atom,
                "energy": energy,
                "energy_per_atom": energy_per_atom,
                "total_magnetic_moment": total_magnetic_moment,
                "magnetic_ordering": magnetic_ordering,
                "mag_data": mag_data,
            }
        else:
            row = {
                "config": config,
                "number_of_atoms": number_of_atoms,
                "volume": vol,
                "volume_per_atom": vol_per_atom,
                "energy": energy,
                "energy_per_atom": energy_per_atom,
                "space_group": space_group,
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

def extract_convergence_data(path: str) -> pd.DataFrame:
    """Calculates the energy convergence with respect to ENCUT and plots the results.

    Args:
        path (str): path to the folder containing the VASP input files
        plot (bool, optional): If True, plots the energy per atom vs. ENCUT. Defaults to True.

    Returns:
        pd.DataFrame: a pandas dataframe containing the ENCUT, energy, number of atoms, energy per atom, and difference in energy per atom.
        go.Figure: a plotly figure of the energy per atom vs. ENCUT.
    """

    OSZICAR_files = [
        file
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file))
        and file.startswith("OSZICAR")
    ]
    
    conv_items = sorted([int(file.split(".")[1]) for file in OSZICAR_files])
    
    encut_list = []
    energy_list = []
    number_of_atoms_list = []
    kpoints_list = []
    for item in conv_items:
        oszicar_path = os.path.join(path, f"OSZICAR.{item}")
        outcar_path = os.path.join(path, f"OUTCAR.{item}")
        incar_path = os.path.join(path, f"INCAR.{item}")
        poscar_path = os.path.join(path, f"POSCAR.{item}")
        incar = Incar.from_file(incar_path)
        struct = Structure.from_file(poscar_path)
        
        encut_list.append(incar.get("ENCUT", None))
        energy_list.append(extract_energy(oszicar_path))
        number_of_atoms_list.append(len(struct.sites))
        kpoints_list.append(extract_kpoints(outcar_path))

    energy_per_atom_list = [
        energy / num_atoms 
        for energy, num_atoms in zip(energy_list, number_of_atoms_list)
    ]
    
    difference_meV_per_atom_list = [
        (energy_per_atom_list[i] - energy_per_atom_list[i - 1]) * 1000
        for i in range(1, len(energy_per_atom_list))
    ]
    difference_meV_per_atom_list.insert(0, float("nan"))
    df = pd.DataFrame(
        {
            "ENCUT": encut_list,
            "energy": energy_list,
            "number_of_atoms": number_of_atoms_list,
            "energy_per_atom": energy_per_atom_list,
            "difference_meV_per_atom": difference_meV_per_atom_list,
        }
    )
    
    return df
