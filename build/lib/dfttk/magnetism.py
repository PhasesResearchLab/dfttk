# Standard library imports
import os
import itertools
import numbers


# Related third party imports
import numpy as np
import pandas as pd

# Local application/library specific imports
from pymatgen.core.structure import Structure
from pymatgen.analysis.magnetism.analyzer import (
    CollinearMagneticStructureAnalyzer as CMSA,
)
from pymatgen.io.vasp.outputs import Poscar


# DFTTK imports
from dfttk.data_extraction import extract_tot_mag_data, extract_input_mag_data


def determine_magnetic_ordering(
    df: pd.DataFrame,
    magmom_tolerance: float = 1e-12,
    total_magnetic_moment_tolerance: float = 1e-12,
) -> str:
    """Determines the magnetic ordering of a structure from the magnetization
    data in a pandas DataFrame. e.g. 'FM', 'AFM', 'FiM', 'NM', 'SF'

    Args:
        df (pandas DataFrame): a pandas DataFrame containing the magnetization data
        magmom_tolerance (float, optional): the tolerance for the total magnetic moment on each atom to be considered zero.
        Total_magmom_tolerance (float, optional) the tolerance for the sum of the total magnentic moments for each atom.
        Defaults to 1e-12 to handle floating point errors.

    Returns:
        The magnetic ordering of the structure
    """

    if (np.isclose(df["tot"], 0, atol=magmom_tolerance)).all():
        return "NM"
    elif np.isclose(df["tot"].sum(), 0, atol=total_magnetic_moment_tolerance) == True:
        return "AFM"
    elif (df["tot"] >= 0 + magmom_tolerance).all() or (
        df["tot"] <= 0 - magmom_tolerance
    ).all():
        return "FM"
    elif (df["tot"] > 0 + magmom_tolerance).sum() == (
        df["tot"] < 0 - magmom_tolerance
    ).sum():
        return "FiM"
    else:
        return "SF"


def get_magnetic_structure(poscar: str, outcar: str) -> Structure:
    """Combines the magmom data from the outcar with the structure from the poscar
    to return a pymatgen magnetic Structures object (e.g. Structures with
    associated magmom tags).

    Args:
        poscar (str): name of the POSCAR file
        outcar (str): name of the OUTCAR file

    Returns:
        Structure: pymatgen Structure object with magmom tags
    """
    structure = Structure.from_file(poscar)
    mag_data = extract_tot_mag_data(outcar)
    structure.add_site_property("magmom", mag_data["tot"])
    return structure


# TODO: make this magnetic/non-magnetic agnostic
def equivalent_orderings(
    path: str, contcar_name: str = "CONTCAR", outcar_name: str = "OUTCAR"
) -> bool:
    """finds equivalent magnetic orderings for a set of configurations in a path
    Works rather slow. Needs to be optimized. 350 configurations takes about 10 minutes.

    Args:
        path: Path to "configurations" folder
        contcar_name: name of the CONTCAR file. Defaults to 'CONTCAR'.
        outcar_name: name of the OUTCAR file. Defaults to 'OUTCAR'.

    Raises:
        FileNotFoundError: if the contcar/outcar files are not found for a config

    Returns:
        a dictionary where the keys are the configurations and the values are lists of configurations with matching magnetic ordering
    """
    struct_dict = {}
    for config_dir in os.listdir(path):
        config_dir_path = os.path.join(path, config_dir)
        if os.path.isdir(config_dir_path) and config_dir.startswith("config_"):
            for subdir in os.listdir(config_dir_path):
                subdir_path = os.path.join(config_dir_path, subdir)
                if os.path.isdir(subdir_path) and subdir.startswith("vol_"):
                    try:
                        magnetic_structure = get_magnetic_structure(
                            os.path.join(subdir_path, contcar_name),
                            os.path.join(subdir_path, outcar_name),
                        )
                        config = config_dir.split("config_")[1]
                        struct_dict[config] = magnetic_structure
                        structure_found = True
                        break
                    except FileNotFoundError as e:
                        print(
                            f"missing CONTCAR/OUTCAR in {subdir_path}: {e}. Did you use the correct CONTCAR/OUTCAR name?"
                        )
            if not structure_found:
                raise FileNotFoundError(
                    f"Could not make magnetic structure for config in {config_dir_path}"
                )
    equivalence_dict = {config: [] for config in struct_dict.keys()}
    items = struct_dict.items()
    for i, (config, magnetic_structure) in enumerate(items):
        analyzer = CMSA(magnetic_structure)
        for remaining_config, remaining_magnetic_structure in itertools.islice(
            items, i + 1, len(struct_dict)
        ):
            if analyzer.matches_ordering(remaining_magnetic_structure):
                equivalence_dict[config].append(remaining_config)
                equivalence_dict[remaining_config].append(config)
    return equivalence_dict


def remove_equivalent_orderings(
    df: pd.DataFrame, equivalence_dict: dict
) -> pd.DataFrame:
    remove_list = []
    sorted_df = df.sort_values(by="energy_per_atom")
    for index, row in sorted_df.iterrows():
        if row["config"] in remove_list:
            continue
        elif equivalence_dict[row["config"]] == []:
            continue
        else:
            remove_list.extend(equivalence_dict[row["config"]])

    # keep rows that are not in the remove_list
    return df[~df["config"].isin(remove_list)]


# TODO: support specify min and max for each ion (dict) and min/max (tuple) for
# magmom_tol. it may be beneficial to have a range of acceptable values instead
# a tolerance.
def significant_magmom_change(
    outcar_path: str = "OUTCAR", magmom_tol: float = 0.5
) -> bool:
    """determines if the resulting magnetic moment is significantly different from the input magnetic moment for any of the atoms.

    Args:
        outcar_path: Path to the OUTCAR. Defaults to "OUTCAR".
        magmom_tol: tolerance for change in magnetic moment for each atom. Defaults to 0.5.

    Raises:
        ValueError: if the magmom_tol is not a real number (float, int, etc).

    Returns:
        bool: True if at least one of the atoms in the struct has a resulting magnetic moment that is significantly different from the input.
    """
    input_magmoms = extract_input_mag_data(outcar_path)
    output_magmoms = extract_tot_mag_data(outcar_path)

    if isinstance(magmom_tol, numbers.Real):
        magmom_tol = abs(magmom_tol)
        min_df = input_magmoms.copy()
        max_df = input_magmoms.copy()
        min_df["tot"] = min_df["tot"] - magmom_tol
        max_df["tot"] = max_df["tot"] + magmom_tol
    # elif isinstance(magmom_tol, dict):
    #     pass
    # elif isinstance(magmom_tol, tuple):
    #     pass
    else:
        raise ValueError(
            "magmom_tol must be a real number (float, int, etc) or a dictionary"
        )
    for index, row in output_magmoms.iterrows():
        if row["tot"] < min_df["tot"][index] or row["tot"] > max_df["tot"][index]:
            return True
    return False


# TODO: make magmoms written in NIONS*magmom format
def rearrange_sites_and_magmoms(config_dir):
    """
    this function is a patch to rearrange the sites and magmoms in the POSCAR and
    INCAR files. If the sites are not grouped by specie, VASP will look for more
    potentials than supplied/necessary.
    """
    incar_file = os.path.join(config_dir, "INCAR")
    poscar_file = os.path.join(config_dir, "POSCAR")
    struct = Structure.from_file(poscar_file)  # read poscar
    orig_magmom_df = extract_input_mag_data(incar_file)  # read magmom from incar
    orig_magmoms = orig_magmom_df["tot"].tolist()  # get the magmoms
    struct.add_site_property("magmom", orig_magmoms)  # add magmom to structure
    struct = struct.get_sorted_structure()  # sort structure with the magmoms
    rearranged_magmoms = struct.site_properties["magmom"]  # get the rearranged magmoms
    numeric_strings = [
        str(value) for value in rearranged_magmoms
    ]  # convert values to strings
    result_string = " ".join(numeric_strings)  # join the strings
    result_string = "MAGMOM = " + result_string  # add the MAGMOM = part

    # Write the result_string to the INCAR file
    with open(incar_file, "r") as file:
        lines = file.readlines()
    with open(incar_file, "w") as file:
        for line in lines:
            if line.startswith("MAGMOM ="):
                file.write(result_string + "\n")
            else:
                file.write(line)

    # Write the rearranged structure to the POSCAR file
    poscar = Poscar(struct)
    poscar.write_file(poscar_file)
    return None
