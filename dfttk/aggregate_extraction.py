"""
Extract relevant data from VASP output files.
"""

# Standard library imports
import os
import glob

# Related third party imports
from natsort import natsorted
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Local application/library specific imports
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Poscar

# DFTTK imports
from dfttk.data_extraction import (
    extract_volume,
    extract_energy,
    extract_tot_mag_data,
    extract_kpoints,
    extract_atomic_masses,
    extract_average_mass,
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
    mass_average: str = "geometric",
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
        mass_average: the method used to calculate the average atomic mass. Options are "geometric" and "arithmetic".
        
    Returns:
        pandas DataFrame: a pandas DataFrame containing the volume, configuration, energy, number of atoms, and
        magnetization data (if specified)
    """

    # Find the index where "config_" starts and add its length
    start = path.find("config_") + len("config_")
    config = path[start:]  # get the string following "config_"

    row_list = []
    vol_dirs = glob.glob(os.path.join(path, "vol_*"))
    vol_dirs = natsorted(vol_dirs)
    for vol_dir in vol_dirs:
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
        atomic_masses = extract_atomic_masses(outcar_path)
        average_mass = extract_average_mass(contcar_path, outcar_path, mass_average)
        if collect_mag_data == True:
            mag_data = extract_tot_mag_data(outcar_path, contcar_path)
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
                "space_group": space_group,
                "atomic_masses": atomic_masses,
                "average_mass": average_mass,
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
                "atomic_masses": atomic_masses,
                "average_mass": average_mass,
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
    """Convenience function to extract configuration data from multiple config directories.
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


def extract_convergence_data(path: str) -> pd.DataFrame:
    """extracts and calculates energy convergence data for a series of VASP convergence calculations

    Args:
        path: path to the folder containing the VASP calculation outputs for convergence calculations.

    Returns:
        pd.DataFrame: a pandas dataframe containing the ENCUT, kpoint grid, kppa, energy, number of atoms, energy per atom, and difference in energy per atom
    """

    OSZICAR_files = [
        file
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file)) and file.startswith("OSZICAR")
    ]

    conv_items = sorted([int(file.split(".")[1]) for file in OSZICAR_files])

    encut_list = []
    energy_list = []
    number_of_atoms_list = []
    kpoint_grid_list = []
    for item in conv_items:
        oszicar_path = os.path.join(path, f"OSZICAR.{item}")
        outcar_path = os.path.join(path, f"OUTCAR.{item}")
        incar_path = os.path.join(path, f"INCAR.{item}")
        poscar_path = os.path.join(path, f"POSCAR.{item}")
        incar = Incar.from_file(incar_path)
        struct = Structure.from_file(poscar_path)

        encut_list.append(incar.get("ENCUT", None))
        energy_list.append(extract_energy(oszicar_path))
        number_of_atoms_list.append(len(struct.sites))
        kpoint_grid_list.append(extract_kpoints(outcar_path))
    energy_per_atom_list = [
        energy / num_atoms
        for energy, num_atoms in zip(energy_list, number_of_atoms_list)
    ]

    difference_meV_per_atom_list = [
        (energy_per_atom_list[i] - energy_per_atom_list[i - 1]) * 1000
        for i in range(1, len(energy_per_atom_list))
    ]
    difference_meV_per_atom_list.insert(0, float("nan"))

    kppa_list = []
    for i, kpoint_grid in enumerate(kpoint_grid_list):
        kppa = np.prod(kpoint_grid) * number_of_atoms_list[i]
        kppa_list.append(kppa)

    df = pd.DataFrame(
        {
            "encut": encut_list,
            "kpoint_grid": kpoint_grid_list,
            "kppa": kppa_list,
            "energy": energy_list,
            "number_of_atoms": number_of_atoms_list,
            "energy_per_atom": energy_per_atom_list,
            "difference_mev_per_atom": difference_meV_per_atom_list,
        }
    )

    return df


def plot_format(fig: go.Figure, x_title: str, y_title: str):
    """Updates an x-y plotly figure to the basic format used in DFTTK.
    Args:
        fig: A figure with x-y data.
        x_title (str): title of the x-axis.
        y_title (str): title of the y-axis
    """
    fig.update_layout(
        font=dict(family="Devaju Sans"),
        plot_bgcolor="white",
        width=840,
        height=600,
        legend=dict(font=dict(size=20, color="black")),
        xaxis=dict(
            title=x_title,
            titlefont=dict(size=22, color="rgb(0,0,0)"),
            showline=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            mirror="allticks",
            tickwidth=1,
            tickcolor="black",
            showgrid=False,
            tickfont=dict(color="rgb(0,0,0)", size=20),
        ),
        yaxis=dict(
            title=y_title,
            titlefont=dict(size=22, color="rgb(0,0,0)"),
            showline=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            mirror="allticks",
            tickwidth=1,
            tickcolor="black",
            showgrid=False,
            tickfont=dict(color="rgb(0,0,0)", size=20),
        ),
    )


def plot_encut_conv(df: pd.DataFrame, show_fig=True) -> go.Figure:
    """makes a plot for Encut convergence using plotly.

    Args:
        df: a pandas dataframe containing the ENCUT, kpoint grid, kppa, energy, number of atoms, energy per atom, and difference in energy per atom (as structured by the return of `extract_convergence_data()`).
        show_fig: wheather or not to call the fig.show() method. Defaults to True.

    Returns:
        go.Figure: a plotly figure of the energy per atom vs. ENCUT.
    """
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["encut"],
                y=df["energy_per_atom"],
                mode="lines+markers",
            )
        ]
    )
    plot_format(fig, "encut", "Energy (eV/atom)")
    kpoints = df["kpoint_grid"].iloc[0]
    fig.update_layout(
        title=dict(
            text=f"k-points: {kpoints[0]} x {kpoints[1]} x {kpoints[2]}",
            font=dict(size=24, color="rgb(0,0,0)"),
        )
    )
    if show_fig == True:
        fig.show()
    return fig


def calculate_encut_conv(
    path: str, plot: bool = True
) -> tuple[pd.DataFrame, go.Figure]:
    """Convenience fuction to calculate the energy convergence with respect to ENCUT and plots the results.

    Args:
        path: path to the folder containing the ENCUT convergence calculation results
        plot: If True, plots the energy per atom vs. ENCUT. Defaults to True.

    Returns:
        pd.DataFrame: a pandas dataframe containing the ENCUT, kpoint_grid, kppa, energy, number of atoms, energy per atom, and difference in energy per atom.
        go.Figure: a plotly figure of the energy per atom vs. ENCUT.
    """
    df = extract_convergence_data(path)

    if plot:
        fig = plot_encut_conv(df, show_fig=plot)
    else:
        fig = None

    return df, fig


def plot_kpoint_conv(df: pd.DataFrame, show_fig=True) -> go.Figure:
    """makes a plot for k-point convergence using plotly.

    Args:
        df: a pandas dataframe containing the ENCUT, kpoint grid, kppa, energy, number of atoms, energy per atom, and difference in energy per atom (as structured by the return of `extract_convergence_data()`).
        show_fig: wheather or not to call the fig.show() method. Defaults to True.

    Returns:
        go.Figure: a plotly figure of the energy per atom vs. k-point density (kppa).
    """
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["kppa"],
                y=df["energy_per_atom"],
                mode="lines+markers",
            )
        ]
    )
    plot_format(fig, "KPPA", "Energy (eV/atom)")
    encut = df["encut"].iloc[0]
    fig.update_layout(
        title=dict(
            text=f"ENCUT: {encut} eV",
            font=dict(size=24, color="rgb(0,0,0)"),
        )
    )
    if show_fig == True:
        fig.show()
    return fig


# TODO: Incorporate other convergence criteria
# See https://github.com/kavanase/vaspup2.0
def calculate_kpoint_conv(
    path: str, plot: bool = True
) -> tuple[pd.DataFrame, go.Figure]:
    """Convenience fuction to calculate the energy convergence with respect to k-point density and plots the results.

    Args:
        path: path to the folder containing the kpoint convergence calculation results
        plot: If True, plots the energy per atom vs. k-point density (kppa). Defaults to True.

    Returns:
        pd.DataFrame: a pandas dataframe containing the ENCUT, kpoint_grid, kppa, energy, number of atoms, energy per atom, and difference in energy per atom.
        go.Figure: a plotly figure of the energy per atom vs. ENCUT.
    """

    df = extract_convergence_data(path)
    df = df.drop_duplicates(subset=["kpoint_grid"])

    if plot:
        fig = plot_kpoint_conv(df, show_fig=plot)
    else:
        fig = None

    return df, fig
