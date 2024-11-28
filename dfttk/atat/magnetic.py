"""
Generates magnetic spin configurations using the icamag tool from the ATAT package.
Please follow the installation instructions in atat/install/README.md.
"""

# Standard library imports
import os
import re
import subprocess

# Third-party imports
import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

# DFTTK imports
from dfttk import vasp_input


def poscar2lat(
    path: str,
    poscar_file: str,
    magnetic_sites: dict = {},
    scaling_matrix: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    lat_file: str = "lat.in",
):
    """Converts a POSCAR file to a lat.in file for use with ATAT's icamag tool.

    Args:
        path (str): path to the folder containing the POSCAR file
        poscar_file (str): name of the POSCAR file
        magnetic_sites (dict, optional): sites with magnetic moments. E.g., {"Fe": ["Fe+5", "Fe-5"]}. Defaults to {}.
        scaling_matrix (np.ndarray, optional): scaling matrix for the lattice. Defaults to np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).
        lat_file (str, optional): name of the lat.in file. Defaults to "lat.in".
    """

    poscar = Poscar.from_file(os.path.join(path, poscar_file))

    lattice = poscar.structure.lattice
    direct_coords = poscar.structure.frac_coords
    species = poscar.structure.species

    with open(os.path.join(path, lat_file), "w") as f:
        for i in range(3):
            f.write(" ".join(f"{x:.10f}" for x in lattice.matrix[i]) + "\n")

        for i in range(3):
            f.write(" ".join(str(x) for x in scaling_matrix[i]) + "\n")

        for i, specie in enumerate(species):
            coord_str = " ".join(f"{x:.10f}" for x in direct_coords[i])

            if magnetic_sites:
                site = magnetic_sites.get(specie.symbol, [specie.symbol])
                site_str = ", ".join(site)

            else:
                site_str = specie.symbol

            f.write(f"{coord_str} {site_str}\n")


def call_icamag(path: str, output_file: str = "spin_configs"):
    """Calls the icamag tool from the ATAT package to generate spin configurations.

    Args:
        path (str): path to the folder containing the lat.in file
        output_file (str, optional): name of output file. Defaults to "spin_configs".

    Raises:
        FileNotFoundError: if the specified path does not exist
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path {path} does not exist.")

    with open(os.path.join(path, output_file), "w") as output:
        subprocess.run(["icamag", "-d"], cwd=path, stdout=output)


def determine_magnetic_ordering(
    magmom: np.ndarray,
    magmom_tolerance: float = 0,
    total_magnetic_moment_tolerance: float = 0,
) -> str:
    """
    Determines the magnetic ordering based on the magnetic moments.

    Args:
        magmom (np.ndarray): Array of magnetic moments.
        magmom_tolerance (float, optional): Tolerance for individual magnetic moments. Defaults to 0.
        total_magnetic_moment_tolerance (float, optional): Tolerance for the total magnetic moment. Defaults to 0.

    Returns:
        str: The magnetic ordering, which can be "NM", "AFM", "FM", "FiM", or "SF".
    """
    if (np.isclose(magmom, 0, atol=magmom_tolerance)).all():
        magnetic_ordering = "NM"
    elif np.isclose(sum(magmom), 0, atol=total_magnetic_moment_tolerance) == True:
        magnetic_ordering = "AFM"
    elif (magmom >= 0 + magmom_tolerance).all() or (
        magmom <= 0 - magmom_tolerance
    ).all():
        magnetic_ordering = "FM"
    elif (magmom > 0 + magmom_tolerance).sum() == (magmom < 0 - magmom_tolerance).sum():
        magnetic_ordering = "FiM"
    else:
        magnetic_ordering = "SF"

    return magnetic_ordering


def create_poscar_object(
    lattice_vectors: np.ndarray,
    scaling_matrix: np.ndarray,
    species_elements: list[str],
    coords: np.ndarray,
    magmom: np.ndarray
) -> Poscar:
    """
    Creates a POSCAR object from the given lattice vectors, scaling matrix, species elements, coordinates, and magnetic moments.

    Args:
        lattice_vectors (np.ndarray): Array of lattice vectors.
        scaling_matrix (np.ndarray): Scaling matrix to apply to the lattice vectors.
        species_elements (list[str]): List of species elements.
        coords (np.ndarray): Array of atomic coordinates.
        magmom (np.ndarray): Array of magnetic moments.

    Returns:
        Poscar: A POSCAR object representing the structure.
    """
    scaled_lattice_vectors = np.dot(lattice_vectors, scaling_matrix)
    lattice = Lattice(scaled_lattice_vectors)
    structure = Structure(
        lattice, species_elements, coords, site_properties={"MAGMOM": magmom}
    )
    poscar_object = Poscar(structure)

    return poscar_object


def parse_spin_config(
    path: str,
    spin_config_file: str = "spin_configs",
    magmom_tolerance: float = 0,
    total_magnetic_moment_tolerance: float = 0,
) -> pd.DataFrame:
    """parse the spin configurations generated by icamag.

    Args:
        path (str): path to the folder containing the output file from icamag.
        spin_config_file (str, optional): name of the output file. Defaults to "spin_configs".
        magmom_tolerance (float, optional): individual magmom tolerance. Defaults to 0.
        total_magnetic_moment_tolerance (float, optional): total magmom tolerance. Defaults to 0.

    Returns:
        pd.DataFrame: dataframe containing information about the spin configurations
    """

    with open(os.path.join(path, spin_config_file), "r") as f:
        lines = f.readlines()

    number_of_atoms_list = []
    volume_list = []
    space_group_list = []
    multiplicity_list = []
    coords_list = []
    species_list = []
    poscar_object_list = []
    magmom_list = []
    magnetic_ordering_list = []
    mag_data_list = []

    count = 0
    for line in lines:
        line = line.strip().split()

        if count == 0:
            multiplicity = int(line[0])
            multiplicity_list.append(multiplicity)

        elif count == 1:
            lattice_vector_1 = np.array([float(x) for x in line])

        elif count == 2:
            lattice_vector_2 = np.array([float(x) for x in line])

        elif count == 3:
            lattice_vector_3 = np.array([float(x) for x in line])
            lattice_vectors = np.vstack(
                (lattice_vector_1, lattice_vector_2, lattice_vector_3)
            )

        elif count == 4:
            scaling_vector_1 = np.array([float(x) for x in line])

        elif count == 5:
            scaling_vector_2 = np.array([float(x) for x in line])

        elif count == 6:
            scaling_vector_3 = np.array([float(x) for x in line])
            scaling_matrix = np.vstack(
                (scaling_vector_1, scaling_vector_2, scaling_vector_3)
            )

        elif count > 6 and "end" not in line and len(line) > 1:
            coord = np.array([float(x) for x in line[:3]])
            species = line[3]

            coords_list.append(coord)
            species_list.append(species)

        count += 1

        if not line:
            # Collect the non-letters (magmom) from species_list
            magmom = ["".join(re.findall(r"[^a-zA-Z]", x)) for x in species_list]

            # Set magmom = 0 for non-magnetic sites
            magmom = [float(x) if x else 0 for x in magmom]
            magmom = np.array(magmom)
            magmom_list.append(magmom)

            magnetic_ordering = determine_magnetic_ordering(
                magmom,
                magmom_tolerance=magmom_tolerance,
                total_magnetic_moment_tolerance=total_magnetic_moment_tolerance,
            )
            magnetic_ordering_list.append(magnetic_ordering)

            # Remove any non-letter characters (magmom) from species
            species_elements = ["".join(filter(str.isalpha, x)) for x in species_list]
            coords = np.vstack(coords_list)
            poscar_object = create_poscar_object(
                lattice_vectors, scaling_matrix, species_elements, coords, magmom
            )
            poscar_object_list.append(poscar_object)

            number_of_atoms = sum(poscar_object.natoms)
            number_of_atoms_list.append(number_of_atoms)
            volume = poscar_object.structure.volume
            volume_list.append(volume)
            space_group = poscar_object.structure.get_space_group_info()[0]
            space_group_list.append(space_group)
            species = [site.specie.symbol for site in poscar_object.structure]

            mag_data = pd.DataFrame()
            mag_data["#_of_ion"] = range(len(magmom))
            mag_data["tot"] = magmom
            mag_data = mag_data.copy()
            mag_data.loc[:, "species"] = species
            mag_data_list.append(mag_data)

            # Reset count and lists
            count = 0
            species_list = []
            coords_list = []

    # Create a dataframe called spin_configs
    spin_configs = pd.DataFrame()
    spin_configs["config"] = [str(i) for i in range(len(poscar_object_list))]
    spin_configs["multiplicity"] = multiplicity_list
    spin_configs["number_of_atoms"] = number_of_atoms_list
    spin_configs["volume"] = volume_list
    spin_configs["volume_per_atom"] = (
        spin_configs["volume"] / spin_configs["number_of_atoms"]
    )
    spin_configs["space_group"] = space_group_list
    spin_configs["magnetic_ordering"] = magnetic_ordering_list
    spin_configs["mag_data"] = mag_data_list

    return spin_configs, poscar_object_list


def write_spin_config(
    path: str,
    spin_configs: pd.DataFrame,
    poscar_object_list: list,
    material_type: str,
    encut: int = 520,
    kppa: int = 4000,
    other_settings: dict = {},
):
    """Writes the spin configurations to separate folders and generates VASP input files.

    Args:
        path (str): path to the folder to write the spin configurations
        spin_configs (pd.DataFrame): dataframe containing information about the spin configurations
        material_type (str): metal or non_metal
        encut (int, optional): ENCUt value. Defaults to 520.
        kppa (int, optional): k-point mesh per reciprocal atom. Defaults to 4000.
        other_settings (dict, optional): include other INCAR tags. Defaults to {}.
    """

    spin_configs["poscar_object"] = poscar_object_list
    config_values = spin_configs["config"].values
    for config_value in config_values:
        poscar_object = spin_configs[spin_configs["config"] == config_value][
            "poscar_object"
        ].values[0]

        config_dir = os.path.join(path, f"config_{config_value}")
        os.makedirs(config_dir, exist_ok=True)

        poscar_object.write_file(os.path.join(config_dir, "POSCAR"))

        magmom_settings = poscar_object.structure.site_properties
        other_settings.update(magmom_settings)
        vasp_input.ev_curve_set(
            config_dir,
            material_type=material_type,
            encut=encut,
            kppa=kppa,
            other_settings=other_settings,
        )


def gen_spin_configs(
    path: str,
    magnetic_sites: dict,
    material_type: str,
    poscar_file: str = "POSCAR",
    encut: int = 520,
    kppa: int = 4000,
    other_settings: dict = {},
):
    """Convenience function to generate spin configurations using the POSCAR file in the specified path.

    Args:
        path (str): path to the folder to write the spin configurations
        magnetic_sites (dict): sites with magnetic moments
        material_type (str): metal or non_metal
        poscar_file (str, optional): name of POSCAR file. Defaults to "POSCAR".
        encut (int, optional): ENCUT value. Defaults to 520.
        kppa (int, optional): k-point mesh per reciprocal atom. Defaults to 4000.
        other_settings (dict, optional): include other INCAR tags. Defaults to {}.
    """

    poscar2lat(path, poscar_file=poscar_file, magnetic_sites=magnetic_sites)
    call_icamag(path)

    spin_configs, poscar_object_list = parse_spin_config(path)
    write_spin_config(
        path,
        spin_configs,
        poscar_object_list,
        material_type=material_type,
        encut=encut,
        kppa=kppa,
        other_settings=other_settings,
    )
