"""
Extract relevant data from VASP output files.
"""

# Standard library imports
import os

# Related third party imports
import numpy as np
import pandas as pd

# Local application/library specific imports
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar


def extract_kpoints(path: str) -> list[str]:
    """Extract kpoints from an OUTCAR file

    Args:
        path (str): the path to an OUTCAR file

    Returns:
        list[int]: kpoints in the format [9, 9, 9]
    """

    with open(path, "r") as file:
        file_name = os.path.basename(path)
        assert file_name.startswith("OUTCAR"), "File name does not start with 'OUTCAR'"

        lines = file.readlines()
        for line in lines:
            if "generate k-points for" in line:
                kpoints = line.split()[3:6]
                kpoints = [int(x) for x in kpoints]
                break
    return kpoints


def extract_atomic_masses(outcar_path: str) -> float:
    """
    Extract the mass of each atom (POMASS values) from an OUTCAR file as a dictionary.
    """

    atoms = []
    masses = []
    with open(outcar_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "TITEL" in line:
                atoms.append(line.split()[-2])
            elif "POMASS" and "mass and valenz" in line:
                mass = line.split()[2].replace(";", "")
                masses.append(float(mass))

    # Clean atoms so that only the species is contained. e.g. 'Fe_pv' -> 'Fe'
    atoms = [atom.split("_")[0] for atom in atoms]

    atomic_masses = dict(zip(atoms, masses))
    return atomic_masses


def extract_average_mass(
    contcar_path: str, outcar_path: str, average: str = "arithmetic"
) -> float:
    atomic_masses = extract_atomic_masses(outcar_path)
    structure = Structure.from_file(contcar_path)
    masses = [atomic_masses[site.specie.symbol] for site in structure]
    if average == "arithmetic":
        average_mass = sum(masses) / len(masses)
    elif average == "geometric":
        average_mass = np.prod(masses) ** (1 / len(masses))
    elif average == "harmonic":
        average_mass = len(masses) / sum([1 / mass for mass in masses])
    else:
        raise ValueError(
            f"Average type {average} not recognized. Must be 'arithmetic', 'geometric', or 'harmonic'."
        )
    return average_mass


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
                headers.insert(0, "#_of_ion")
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
def extract_tot_mag_data(
    outcar_path: str = "OUTCAR", contcar_path: str = "CONTCAR"
) -> pd.DataFrame:
    """Returns only the 'tot' magnetization of the last step for each specified ion.

    Args:
        outcar_path: Path to an OUTCAR file. Defaults to "OUTCAR".
        contcar_path: Path to a CONTCAR file. Defaults to "CONTCAR".

    Returns:
        a pandas DataFrame containing the 'tot' magnetization data
    """

    all_mag_data = extract_mag_data(outcar_path)
    last_step_data = all_mag_data[all_mag_data["step"] == all_mag_data["step"].max()]
    tot_data = last_step_data[["#_of_ion", "tot"]]
    tot_data.reset_index(drop=True, inplace=True)

    contcar = Poscar.from_file(contcar_path)
    species = [site.specie.symbol for site in contcar.structure]

    tot_data = tot_data.copy()
    tot_data.loc[:, "species"] = species

    return tot_data
