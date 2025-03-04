"""
Module for magnetic analysis of VASP calculations. 
"""

# Related third party imports
import numpy as np
import pandas as pd

# Local application/library specific imports
from pymatgen.core.structure import Structure

# DFTTK imports
from dfttk.data_extraction import extract_tot_mag_data


def determine_magnetic_ordering(
    df: pd.DataFrame,
    magmom_tolerance: float = 1e-12,
    total_magnetic_moment_tolerance: float = 1e-12,
) -> str:
    """Determines the magnetic ordering of a structure from the magnetization
    data in a pandas DataFrame. e.g. 'FM', 'AFM', 'FiM', 'NM', 'SF'.

    Args:
        df (pandas DataFrame): a pandas DataFrame containing the magnetization data.
        magmom_tolerance (float, optional): the tolerance for the total magnetic moment on each atom to be considered zero.
        Total_magmom_tolerance (float, optional) the tolerance for the sum of the total magnentic moments for each atom.
        Defaults to 1e-12 to handle floating point errors.

    Returns:
        str: the magnetic ordering of the structure.
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
    """Combines the magmom data from the OUTCAR file with the structure from the POSCAR file
    to return a pymatgen magnetic structures object (e.g. structures with
    associated magmom tags).

    Args:
        poscar (str): path to the POSCAR file.
        outcar (str): path to the OUTCAR file.

    Returns:
        Structure: pymatgen Structure object with magmom tags.
    """

    structure = Structure.from_file(poscar)
    mag_data = extract_tot_mag_data(outcar, poscar)
    structure.add_site_property("magmom", mag_data["tot"])

    return structure
