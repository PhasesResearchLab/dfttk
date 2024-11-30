"""
Module for magnetic analysis of VASP calculations. 
"""

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


# TODO: While this does work, we also want the option to determine the magnetic ordering based on only certain elements.
# For example, for Fe3Pt, we may want to determine the magnetic ordering based only on the Fe atoms.
# In principle though, it would be wrong to label the whole structure based on only some of the atoms. So we have to be clear in the comments this meaning of "AFM", "SFC", etc.
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


def filter_mag_data_by_species(row: pd.Series, species: list[str]) -> pd.DataFrame:
    """
    Filters the mag_data to include only the specified species.

    Args:
        row (pd.Series): A row from a DataFrame containing mag_data.
        species (list): A list of species to filter by.

    Returns:
        pd.DataFrame: Filtered mag_data containing only the specified species.
    """
    mag_data = row["mag_data"]
    return mag_data[mag_data["species"].isin(species)]


def assign_tot_sign(row: pd.Series) -> pd.DataFrame:
    """
    Assigns a tot_sign column to the mag_data based on the sign of the tot column.

    Args:
        row (pd.Series): A row from a DataFrame containing mag_data.

    Returns:
        pd.DataFrame: mag_data with an additional tot_sign column.
    """
    mag_data = row["mag_data"].copy()
    mag_data["tot_sign"] = np.sign(mag_data["tot"])
    return mag_data


def check_discontinuities(
    config_df: pd.DataFrame, ref_tot_sign: pd.Series
) -> tuple[list[bool], list[int]]:
    """Checks for discontinuities in the magnetic moment for an ev-curve.

    Args:
        config_df (pd.DataFrame): DataFrame containing the magnetic moment data for a configuration.
        ref_tot_sign (pd.Series): Series containing the reference tot_sign values.

    Returns:
        tuple: A list of booleans indicating whether there is a discontinuity at each volume, and a list of indices.
    """
    match_list = []
    index_list = []
    count = 0
    for index, row in config_df.iterrows():
        current_tot_sign = row["mag_data"]["tot_sign"]
        if count == 0:
            match = current_tot_sign.equals(ref_tot_sign)
        else:
            previous_tot_sign = config_df.iloc[count - 1]["mag_data"]["tot_sign"]
            match = current_tot_sign.equals(previous_tot_sign)

        match_list.append(match)
        index_list.append(index)
        count += 1

    return match_list, index_list


def handle_no_discontinuity(
    match_list: list[bool],
    index_list: list[int],
    config_df: pd.DataFrame,
    ref: pd.DataFrame,
    filtered_df: pd.DataFrame,
) -> pd.DataFrame:
    """Handles the case where there are no discontinuities in the magnetic moment in an ev-curve.

    Args:
        match_list (list[bool]): list of booleans indicating whether there is a discontinuity at each volume.
        index_list (list[int]): list of indices.
        config_df (pd.DataFrame): DataFrame containing the magnetic moment data for a configuration.
        ref (pd.DataFrame): DataFrame containing the reference tot_sign values.
        filtered_df (pd.DataFrame): DataFrame containing the filtered configurations.

    Returns:
        pd.DataFrame: DataFrame containing the filtered configurations.
    """
    if all(match_list):
        multiplicity = ref["multiplicity"].values[0]
        config_df.insert(1, "multiplicity", multiplicity)
        filtered_df = pd.concat([filtered_df, config_df.loc[index_list]])

    return filtered_df


def handle_initial_jump(
    config: str,
    config_df: pd.DataFrame,
    unstable_initial_states: list[tuple[str, int, int]],
) -> list[tuple[str, int, int]]:
    """Handles the case where there is a jump in the magnetic moment at the initial volume in an ev-curve.

    Args:
        config (str): name of the configuration.
        config_df (pd.DataFrame): DataFrame containing the magnetic moment data for a configuration.
        unstable_initial_states (list[tuple[str, int, int]]): List of unstable initial states.

    Returns:
        list[tuple[str, int, int]]: List of unstable initial states.
    """
    unstable_initial_states = []
    tot_sign = config_df.iloc[0]["mag_data"]["tot_sign"]
    spin_up = (tot_sign[tot_sign == 1]).count()
    spin_down = (tot_sign[tot_sign == -1]).count()
    unstable_initial_states.append((config, spin_up, spin_down))

    return unstable_initial_states


def handle_later_jump(
    match_list: list[bool],
    index_list: list[int],
    config_df: pd.DataFrame,
    ref: pd.DataFrame,
    filtered_df: pd.DataFrame,
) -> pd.DataFrame:
    """Handles the case where there is a jump in the magnetic moment at a later volume in an ev-curve.

    Args:
        match_list (list[bool]): list of booleans indicating whether there is a discontinuity at each volume.
        index_list (list[int]): list of indices.
        config_df (pd.DataFrame): DataFrame containing the magnetic moment data for a configuration.
        ref (pd.DataFrame): DataFrame containing the reference tot_sign values.
        filtered_df (pd.DataFrame): DataFrame containing the filtered configurations.

    Returns:
        pd.DataFrame: DataFrame containing the filtered configurations.
    """
    # Gets the maximum number of True values in a row.
    max_true = 0
    count = 0
    for match in match_list:
        if match == True:
            count += 1
            if count > max_true:
                max_true = count
        else:
            break

    # Only keep the configuration if it has at least 8 True values in a row.
    if max_true > 8:
        first_false = index_list[match_list.index(False)]
        config_df = config_df.loc[: first_false - 1]
        multiplicity = ref["multiplicity"].values[0]
        config_df.insert(1, "multiplicity", multiplicity)
        new_index_list = index_list[: match_list.index(False)]
        filtered_df = pd.concat([filtered_df, config_df.loc[new_index_list]])

    return filtered_df


def identify_relaxed_config(
    unstable_initial_states: list[tuple[str, int, int]],
    df_copy: pd.DataFrame,
    filtered_df: pd.DataFrame,
) -> None:
    """Prints the configurations that the unstable states relaxed to.

    Args:
        unstable_initial_states (list[tuple[str, int, int]]): list of unstable initial states.
        df_copy (pd.DataFrame): DataFrame containing the relaxed energy-volume data.
        filtered_df (pd.DataFrame): DataFrame containing the filtered configurations.
    """
    for (
        unstable_config,
        unstable_spin_up,
        unstable_spin_down,
    ) in unstable_initial_states:
        config_df = df_copy[df_copy["config"] == unstable_config]
        unstable_config_volume = unstable_config["volume_per_atom"].reset_index(
            drop=True
        )
        unstable_config_energy = unstable_config["energy_per_atom"].reset_index(
            drop=True
        )

        unique_filtered_configs = filtered_df["config"].unique()
        for filtered_config in unique_filtered_configs:
            filtered_config_df = filtered_df[filtered_df["config"] == filtered_config]
            filtered_tot_sign = filtered_config_df["mag_data"].values[0]["tot_sign"]
            filtered_spin_up = (filtered_tot_sign[filtered_tot_sign == 1]).count()
            filtered_spin_down = (filtered_tot_sign[filtered_tot_sign == -1]).count()
            filtered_config_volume = filtered_config_df["volume_per_atom"].reset_index(
                drop=True
            )
            filtered_config_energy = filtered_config_df["energy_per_atom"].reset_index(
                drop=True
            )

            if unstable_config_volume.equals(filtered_config_volume):
                energy_difference = abs(unstable_config_energy - filtered_config_energy)

                if all(energy_difference < 0.001):
                    if (
                        unstable_spin_up == filtered_spin_up
                        or unstable_spin_up == filtered_spin_down
                    ):
                        print(
                            f"config {unstable_config} relaxed to config {filtered_config} for all volumes."
                        )


# TODO: What other cases to cover?
# TODO: For all unstable states, report which configurations they relaxed to.
def filter_magmom_configs(
    df: pd.DataFrame, spin_configs: pd.DataFrame, species: list[str]
) -> pd.DataFrame:
    """
    Filters the magnetic moment configurations by filtering species of interest,
    assigning total sign columns, and checking for discontinuities in magmom.

    Args:
        df (pd.DataFrame): DataFrame containing the relaxed energy-volume data.
        spin_configs (pd.DataFrame): DataFrame containing the original spin configurations from ATAT icamag.
        species (list): List of species to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame with consistent magnetic moment configurations.
    """
    # Keep only species of interest in df_copy and spin_configs_copy.
    df_copy = df.copy()
    df_copy["mag_data"] = df_copy.apply(
        filter_mag_data_by_species, species=species, axis=1
    )
    spin_configs_copy = spin_configs.copy()
    spin_configs_copy["mag_data"] = spin_configs_copy.apply(
        filter_mag_data_by_species, species=species, axis=1
    )

    # Assign a tot_sign column to each mag_data.
    df_copy["mag_data"] = df_copy.apply(assign_tot_sign, axis=1)
    spin_configs_copy["mag_data"] = spin_configs_copy.apply(assign_tot_sign, axis=1)

    unique_configs = spin_configs_copy["config"].unique()
    unstable_initial_states = []
    filtered_df = pd.DataFrame()
    for config in unique_configs:
        config_df = df_copy[df_copy["config"] == config]
        ref = spin_configs_copy[spin_configs_copy["config"] == config]
        ref_tot_sign = ref["mag_data"].values[0]["tot_sign"]

        # Check for any discontinuities.
        match_list, index_list = check_discontinuities(config_df, ref_tot_sign)

        if all(match_list):
            filtered_df = handle_no_discontinuity(
                match_list, index_list, config_df, ref, filtered_df
            )

        elif match_list[0] == False and all(match_list[1:]):
            unstable_intial_states = handle_initial_jump(
                config, config_df, unstable_initial_states
            )

        else:
            print(
                f"There is a jump in config {config} at volume {config_df.loc[index_list[match_list.index(False)]]['volume']}."
            )

            filtered_df = handle_later_jump(
                match_list, index_list, config_df, ref, filtered_df
            )

    # For the unstable states, report which configurations they relaxed to.
    identify_relaxed_config(unstable_initial_states, df_copy, filtered_df)

    return filtered_df
