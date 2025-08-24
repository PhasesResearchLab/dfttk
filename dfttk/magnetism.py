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

# TODO: write tests for this
def determine_magnetic_ordering(
    df: pd.DataFrame,
    magmom_tolerance: float = 0.05,
    total_magnetic_moment_tolerance: float = 0.1,
) -> str:
    """Determines the magnetic ordering of a structure from the magnetization
    data in a pandas DataFrame. e.g. 'FM', 'AFM', 'FiM', 'NM', 'SF'.

    Args:
        df (pandas DataFrame): a pandas DataFrame containing the magnetization data from extract_tot_mag_data.
        magmom_tolerance (float, optional): the tolerance for the total magnetic moment on each atom to be considered zero. Defaults to 0.05 to handle floating point errors.
        total_magnetic_moment_tolerance (float, optional): the tolerance for the sum of the total magnetic moments for each atom. Defaults to 0.1 to handle floating point errors.

    Returns:
        str: the magnetic ordering of the structure.
    """
    
    moments = df["tot"].to_numpy()
    
    # NM
    if np.all(np.isclose(moments, 0, atol=magmom_tolerance)):
        return "NM"

    # Filter out near-zero moments
    significant = moments[np.abs(moments) > magmom_tolerance]
    pos_count = np.sum(significant > 0)
    neg_count = np.sum(significant < 0)
    net_moment = np.sum(significant)

    # FM: all significant moments have the same sign
    if significant.size > 0 and (pos_count == 0 or neg_count == 0):
        return "FM"

    # AFM: net moment near zero and balanced signs
    if np.isclose(net_moment, 0, atol=total_magnetic_moment_tolerance) and pos_count == neg_count:
        return "AFM"

    # FiM: net moment not zero and balanced signs
    if not np.isclose(net_moment, 0, atol=total_magnetic_moment_tolerance) and pos_count == neg_count:
        return "FiM"

    # SF: net moment not zero and unbalanced signs
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
