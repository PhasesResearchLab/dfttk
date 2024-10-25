"""
Prepare VASP input files for a given POSCAR file.
Follow the POTCAR setup for pymatgen: https://pymatgen.org/installation.html

For more information on available functionals,
see https://github.com/materialsproject/pymatgen/blob/master/src/pymatgen/io/vasp/inputs.py#L2581
"""

# Standard library imports
import os

# Third-party imports
import numpy as np
import yaml
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar

with open("potcar_dict.yaml", "r") as file:
    POTCAR_DICT = yaml.safe_load(file)

with open("magmom_dict.yaml", "r") as file:
    MAGMOM_DICT = yaml.safe_load(file)

def base_set(
    path: str,
    material_type: str,
    encut: int = 520,
    kppa: int = 4000,
    magnetic: bool = False,
    potcar_functional: str = "PBE_54",
    incar_functional: str = "PBE",
) -> None:

    poscar_path = os.path.join(path, "POSCAR")
    struct = Structure.from_file(poscar_path)

    kpoints = Kpoints.automatic_density(struct, kppa, force_gamma=True)
    kpoints.write_file(os.path.join(path, "KPOINTS"))

    potcar_symbols = [site.specie.symbol for site in struct.sites]
    unique_potcar_symbols = list(dict.fromkeys(potcar_symbols))
    unique_potcar_symbols = [POTCAR_DICT[symbol] for symbol in unique_potcar_symbols]
    potcar = Potcar(symbols=unique_potcar_symbols, functional=potcar_functional)
    potcar.write_file(os.path.join(path, "POTCAR"))

    incar_settings = {
        "ENCUT": encut,
        "PREC": "Accurate",
        "NELM": 200,
        "EDIFF": 1e-6,
        "EDIFFG": -0.01,
        "LREAL": False,
        "LWAVE": False,
        "LCHARG": False,
    }

    # TODO: Include all possible functionals
    # For more details, see: https://www.vasp.at/wiki/index.php/GGA and https://www.vasp.at/wiki/index.php/METAGGA
    incar_functional_settings = {
        "PBE": {"GGA": "PE"},
        "PBEsol": {"GGA": "PS"},
        "r2SCAN": {"METAGGA": "R2SCAN", "LASPH": True},
    }

    incar_settings.update(incar_functional_settings.get(incar_functional, {}))

    material_settings = {
        "metal": {"ISMEAR": 1, "SIGMA": 0.2},
        "non_metal": {"ISMEAR": 0, "SIGMA": 0.05},
    }

    incar_settings.update(material_settings.get(material_type, {}))
    
    if magnetic:
        magmom = [MAGMOM_DICT[site.specie.symbol] for site in struct.sites]
        incar_settings.update({"MAGMOM": magmom, "ISPIN": 2, "LORBIT": 11})

    return incar_settings


def volume_relax_set(
    path: str,
    material_type: str,
    encut: int = 520,
    kppa: int = 4000,
    magnetic: bool = False,
    potcar_functional: str = "PBE_54",
    incar_functional: str = "PBE",
    other_settings: dict = {},
) -> None:

    incar_settings = base_set(
        path, material_type, encut, kppa, magnetic, potcar_functional, incar_functional
    )
    incar_settings.update(
        {
            "ENCUT": encut,
            "IBRION": 2,
            "ISIF": 3,
            "NSW": 200,
        }
    )
    incar_settings.update(other_settings)
    incar = Incar(incar_settings)
    
    incar.write_file(os.path.join(path, "INCAR"))


def conv_set(
    path: str,
    encut: int = 520,
    kppa: int = 4000,
    magnetic: bool = False,
    potcar_functional: str = "PBE_54",
    incar_functional: str = "PBE",
    other_settings: dict = {},
) -> None:

    material_type = "metal"  # Will be overwridden by ISMEAR = -5
    incar_settings = base_set(
        path, material_type, encut, kppa, magnetic, potcar_functional, incar_functional
    )
    incar_settings.update(
        {
            "ENCUT": encut,
            "ISMEAR": -5,
            "IBRION": -1,
            "ISIF": 2,
            "NSW": 0,
        }
    )
    incar_settings.update(other_settings)

    incar = Incar(incar_settings)
    incar.write_file(os.path.join(path, "INCAR"))


def ev_curve_set(
    path: str,
    material_type: str,
    encut: int = 520,
    kppa: int = 4000,
    potcar_functional: str = "PBE_54",
    incar_functional: str = "PBE",
    other_settings: dict = {},
) -> None:

    incar_settings = base_set(
        path, material_type, encut, kppa, potcar_functional, incar_functional
    )
    incar_settings.update(
        {
            "ENCUT": encut,
            "IBRION": 2,
            "ISIF": 4,
            "NSW": 100,
        }
    )
    incar_settings.update(other_settings)

    incar = Incar(incar_settings)
    incar.write_file(os.path.join(path, "INCAR"))


def perturb_structure(
    path, displacement_magnitude, atoms_to_perturb, number_of_perturbations
):
    parent_dir = os.path.dirname(path)

    structure = Poscar.from_file(path).structure

    for i in range(number_of_perturbations):
        structure_copy = structure.copy()

        # Same magnitude, but different vector components for each perturbation on each atom
        for atom in atoms_to_perturb:
            random_vector = np.random.rand(3) - 0.5
            random_vector_magnitude = np.linalg.norm(random_vector)
            normalized_random_vector = random_vector / random_vector_magnitude
            displacement_vector = normalized_random_vector * displacement_magnitude
            structure_copy[atom].coords = (
                structure_copy[atom].coords + displacement_vector
            )

        perturbed_structure = Poscar(structure_copy)
        os.makedirs(os.path.join(parent_dir, f"perturb_{i}"))
        perturbed_structure.write_file(
            os.path.join(parent_dir, f"perturb_{i}", "POSCAR")
        )
