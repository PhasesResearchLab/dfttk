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


def call_icamag(path, output_file: str = "spin_configs"):

    # Ensure the path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path {path} does not exist.")

    # Run the subprocess in the specified path
    with open(os.path.join(path, output_file), "w") as output:
        subprocess.run(["icamag", "-d"], cwd=path, stdout=output)


def parse_spin_config(
    path: str,
    spin_config_file: str = "spin_configs",
    magmom_tolerance: float = 0,
    total_magnetic_moment_tolerance: float = 0,
):

    with open(os.path.join(path, spin_config_file), "r") as f:
        lines = f.readlines()

    multiplicity_list = []
    coords_list = []
    species_list = []
    poscar_object_list = []
    magmom_list = []
    magnetic_ordering_list = []

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
            coords = np.vstack(coords_list)

            # Remove any non-letter characters (magmom) from species
            species_elements = ["".join(filter(str.isalpha, x)) for x in species_list]

            # Collect the non-letters (magmom) from species_list
            magmom = ["".join(re.findall(r"[^a-zA-Z]", x)) for x in species_list]

            # Set magmom = 0 for non-magnetic sites
            magmom = [float(x) if x else 0 for x in magmom]
            magmom = np.array(magmom)
            magmom_list.append(magmom)

            # Determine the magnetic ordering, ignoring non-magnetic sites
            if (np.isclose(magmom, 0, atol=magmom_tolerance)).all():
                magnetic_ordering = "NM"
            elif (
                np.isclose(sum(magmom), 0, atol=total_magnetic_moment_tolerance) == True
            ):
                magnetic_ordering = "AFM"
            elif (magmom >= 0 + magmom_tolerance).all() or (
                magmom <= 0 - magmom_tolerance
            ).all():
                magnetic_ordering = "FM"
            elif (magmom > 0 + magmom_tolerance).sum() == (
                magmom < 0 - magmom_tolerance
            ).sum():
                magnetic_ordering = "FiM"
            else:
                magnetic_ordering = "SF"

            magnetic_ordering_list.append(magnetic_ordering)

            lattice_vectors = np.dot(lattice_vectors, scaling_matrix)
            lattice = Lattice(lattice_vectors)

            structure = Structure(
                lattice, species_elements, coords, site_properties={"MAGMOM": magmom}
            )

            poscar_object = Poscar(structure)
            poscar_object_list.append(poscar_object)

            # reset count and lists
            count = 0
            species_list = []
            coords_list = []

    # create a dataframe called spin_configs
    spin_configs = pd.DataFrame()
    spin_configs["config"] = range(len(poscar_object_list))
    spin_configs["multiplicity"] = multiplicity_list
    spin_configs["poscar_object"] = poscar_object_list
    spin_configs["magnetic_ordering"] = magnetic_ordering_list

    return spin_configs


def write_spin_config(
    path, spin_configs, material_type, encut=520, kppa=4000, other_settings={}
):

    config_values = spin_configs["config"].values
    for config_value in config_values:
        poscar_object = spin_configs[spin_configs["config"] == config_value][
            "poscar_object"
        ].values[0]

        config_dir = os.path.join(path, f"config_{config_value}")
        os.makedirs(config_dir, exist_ok=True)

        poscar_object.write_file(os.path.join(config_dir, "POSCAR"))

        other_settings = poscar_object.structure.site_properties
        vasp_input.ev_curve_set(
            config_dir,
            material_type=material_type,
            encut=encut,
            kppa=kppa,
            other_settings=other_settings,
        )


def gen_spin_configs(
    path,
    magnetic_sites,
    material_type,
    poscar_file="POSCAR",
    encut=520,
    kppa=4000,
    other_settings={},
):

    poscar2lat(path, poscar_file=poscar_file, magnetic_sites=magnetic_sites)
    call_icamag(path)

    spin_configs = parse_spin_config(path)
    write_spin_config(
        path,
        spin_configs,
        material_type=material_type,
        encut=encut,
        kppa=kppa,
        other_settings=other_settings,
    )
