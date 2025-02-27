"""
Extract relevant data from VASP output files.
"""

# Standard library imports
import os

# Related third-party imports
import numpy as np
import pandas as pd

# Local application/library-specific imports
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.vasp.outputs import Vasprun


def extract_atomic_masses(outcar_path: str) -> float:
    """Extracts the atomic masses from an OUTCAR file and returns them as a dictionary with the species as the key and the mass as the value.

    Args:
        outcar_path (str): path to an OUTCAR file.

    Returns:
        dict: dictionary containing the atomic masses.
    """

    # Initialize lists to store the atomic species and masses
    atoms = []
    masses = []

    # Open the OUTCAR file and read its contents
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

    # Create a dictionary of atomic masses
    atomic_masses = dict(zip(atoms, masses))

    return atomic_masses


def extract_average_mass(
    contcar_path: str, outcar_path: str, average: str = "arithmetic"
) -> float:
    """Calculates the average atomic mass of a structure from a CONTCAR file and an OUTCAR file.

    Args:
        contcar_path (str): path to a CONTCAR file.
        outcar_path (str): path to an OUTCAR file.
        average (str, optional): Type of average to calculate. Must be 'arithmetic', 'geometric', or 'harmonic'. Defaults to "arithmetic".

    Raises:
        ValueError: If the average type is not recognized.

    Returns:
        float: average atomic mass of the structure.
    """

    # Extract the atomic masses from the OUTCAR file
    atomic_masses = extract_atomic_masses(outcar_path)

    # Read the structure from the CONTCAR file
    structure = Structure.from_file(contcar_path)

    # Create a list of atomic masses for each site in the structure
    masses = [atomic_masses[site.specie.symbol] for site in structure]

    # Calculate the average mass based on the specified type
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


def extract_mag_data(outcar_path: str) -> pd.DataFrame:
    """Extracts the magnetization data from an OUTCAR file and returns the data as a pandas DataFrame in the same format and headings as seen in the OUTCAR.

    Args:
        outcar_path (str): path to an OUTCAR file.

    Returns:
        pd.DataFrame: DataFrame containing the magnetization data.
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
                try:
                    ion = int(line.split()[0])
                    data_line = line.split()[1:]
                    data_line = [float(data) for data in data_line]
                    data.append((step, ion, *data_line))
                except:
                    continue
            elif data_start and "----" in line:
                try:
                    data_start = False
                    found_mag_data = False
                except:
                    continue
        columns = ["step"] + headers
        df = pd.DataFrame(data, columns=columns)
        return df


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


def parse_doscar(vasprun_path, doscar_path):

    vasprun = Vasprun(vasprun_path)
    nedos = vasprun.parameters["NEDOS"]
    fermi_energy = vasprun.efermi

    with open(doscar_path, "r") as file:
        lines = file.readlines()

        header_lines = 6
        data_lines = lines[header_lines : header_lines + nedos]

        energies = []
        dos_up = []
        dos_down = []
        integrated_dos_up = []
        integrated_dos_down = []

        for line in data_lines:
            if line.strip():
                parts = line.split()

                if len(parts) == 3:
                    energies.append(float(parts[0]))
                    dos_up.append(float(parts[1]))
                    integrated_dos_up.append(float(parts[2]))

                if len(parts) == 5:
                    energies.append(float(parts[0]))
                    dos_up.append(float(parts[1]))
                    dos_down.append(float(parts[2]))
                    integrated_dos_up.append(float(parts[3]))
                    integrated_dos_down.append(float(parts[4]))

        energies = np.array(energies)
        dos_up = np.array(dos_up)
        dos_down = np.array(dos_down)
        integrated_dos_up = np.array(integrated_dos_up)
        integrated_dos_down = np.array(integrated_dos_down)

    return (
        energies,
        dos_up,
        dos_down,
        integrated_dos_up,
        integrated_dos_down,
        fermi_energy,
    )
