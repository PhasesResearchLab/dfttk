# Standard library imports
import os
import glob

# Related third party imports
import pandas as pd

# Local application/library specific imports
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# DFTTK imports
from dfttk.data_extraction import (
    extract_volume,
    extract_energy,
    extract_tot_mag_data,
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
