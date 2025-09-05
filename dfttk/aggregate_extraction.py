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
import plotly.graph_objects as go


# Local application/library specific imports
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Oszicar, Outcar

# DFTTK imports
from dfttk.data_extraction import (
    extract_tot_mag_data,
    extract_atomic_masses,
    extract_average_mass,
)
from dfttk.magnetism import determine_magnetic_ordering


def extract_configuration_data(
    path: str,
    outcar_name: str = "OUTCAR.3static",
    oszicar_name: str = "OSZICAR.3static",
    contcar_name: str = "CONTCAR.3static",
    collect_mag_data: bool = False,
    magmom_tolerance: float = 0.01,
    total_magnetic_moment_tolerance: float = 0.01,
    mass_average: str = "geometric",
) -> pd.DataFrame:
    """Extracts the volume, configuration, energy, number of atoms, and magnetization data (if specified) from calculations
    run by ev_curve_series and returns a pandas DataFrame.

    Args:
        path: the path containing a config_* folder which contain vol_* folders.
        outcar_name: name of the OUTCAR file. Defaults to "OUTCAR".
        oszicar_name: name of the OSZICAR file. Defaults to "OSZICAR".
        contcar_name: name of the CONTCAR file. Defaults to "CONTCAR".
        collect_mag_data: if True, collect the magnetization data using extract_tot_mag_data. Defaults to
        False.
        magmom_tolerance: the tolerance for the total magnetic moment to be considered zero. Defaults to 0.01 to handle floating point errors.
        total_magnetic_moment_tolerance: the tolerance for the sum of the total magnetic moments for each atom.
        Defaults to 0.01 to handle floating point errors.
        mass_average: the method used to calculate the average atomic mass. Options are "geometric" and "arithmetic".

    Returns:
        pandas DataFrame: a pandas DataFrame containing the volume, configuration, energy, number of atoms, and
        magnetization data (if specified).
    """

    # Initialize lists to store data
    volumes = []
    energies = []
    mag_data_list = []
    total_magnetic_moments = []
    magnetic_orderings = []

    # Get list of volume directories and sort them naturally
    vol_dirs = glob.glob(os.path.join(path, "vol_*"))
    vol_dirs = natsorted(vol_dirs)

    # Loop through each volume directory and extract data
    for vol_dir in vol_dirs:
        outcar_path = os.path.join(vol_dir, outcar_name)
        oszicar_path = os.path.join(vol_dir, oszicar_name)
        contcar_path = os.path.join(vol_dir, contcar_name)

        # Check if all required files exist
        if not all(os.path.isfile(p) for p in [outcar_path, oszicar_path, contcar_path]):
            print(f"Warning: Required files do not exist in {vol_dir}. Skipping.")
            continue

        # Extract volume from CONTCAR file
        struct = Structure.from_file(contcar_path)
        number_of_atoms = len(struct.sites)
        vol = round(struct.volume, 6)

        # Extract energy from OSZICAR file
        oszicar = Oszicar(oszicar_path)
        energy = oszicar.final_energy

        # Extract atomic masses and average mass from OUTCAR and CONTCAR files
        atomic_masses = extract_atomic_masses(outcar_path)
        average_mass = extract_average_mass(contcar_path, outcar_path, mass_average)

        # Append volume and energy to lists
        volumes = np.append(volumes, vol)
        energies = np.append(energies, energy)

        # Collect magnetization data if specified
        if collect_mag_data == True:
            try:
                mag_data = extract_tot_mag_data(outcar_path, contcar_path)
                total_magnetic_moment = mag_data["tot"].sum()
            except:
                mag_data = []
                outcar = Outcar(outcar_path)
                total_magnetic_moment = outcar.total_mag

            try:
                magnetic_ordering = determine_magnetic_ordering(
                    mag_data,
                    magmom_tolerance=magmom_tolerance,
                    total_magnetic_moment_tolerance=total_magnetic_moment_tolerance,
                )
            except:
                magnetic_ordering = []

            if isinstance(mag_data, pd.DataFrame):
                mag_data_list.append(mag_data.to_numpy())
            else:
                pass

            total_magnetic_moments = np.append(total_magnetic_moments, total_magnetic_moment)
            magnetic_orderings = np.append(magnetic_orderings, magnetic_ordering)

    # After the loop, add these checks:
    if isinstance(total_magnetic_moments, np.ndarray) and np.all(total_magnetic_moments == None):
        total_magnetic_moments = np.array([])

    # Convert mag_data_list to numpy array
    mag_data_array = np.array(mag_data_list)

    return (
        number_of_atoms,
        volumes,
        energies,
        atomic_masses,
        average_mass,
        mag_data_array,
        total_magnetic_moments,
        magnetic_orderings,
    )


def extract_convergence_data(path: str) -> pd.DataFrame:
    """Extracts and calculates energy convergence data for a series of VASP convergence calculations.

    Args:
        path: path to the folder containing the VASP calculation outputs for convergence calculations.

    Returns:
        pd.DataFrame: a pandas dataframe containing the ENCUT, kpoint grid, kppa, energy, number of atoms, energy per atom, and difference in energy per atom.
    """

    # Get list of OSZICAR files
    OSZICAR_files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)) and file.startswith("OSZICAR")]
    conv_items = sorted([int(file.split(".")[1]) for file in OSZICAR_files])

    # Initialize lists to store data
    encut_list = []
    energy_list = []
    number_of_atoms_list = []
    kpoint_grid_list = []

    # Loop through each convergence item and extract data
    for item in conv_items:
        oszicar_path = os.path.join(path, f"OSZICAR.{item}")
        kpoints_path = os.path.join(path, f"KPOINTS.{item}")
        incar_path = os.path.join(path, f"INCAR.{item}")
        poscar_path = os.path.join(path, f"POSCAR.{item}")

        incar = Incar.from_file(incar_path)
        struct = Structure.from_file(poscar_path)
        oszicar = Oszicar(oszicar_path)
        kpoints = Kpoints.from_file(kpoints_path)

        encut_list.append(incar.get("ENCUT", None))
        energy_list.append(oszicar.final_energy)
        number_of_atoms_list.append(len(struct.sites))
        kpoint_grid_list.append([item for sublist in kpoints.kpts for item in sublist])

    # Calculate energy per atom
    energy_per_atom_list = [energy / num_atoms for energy, num_atoms in zip(energy_list, number_of_atoms_list)]

    # Calculate difference in energy per atom in meV
    difference_meV_per_atom_list = [(energy_per_atom_list[i] - energy_per_atom_list[i - 1]) * 1000 for i in range(1, len(energy_per_atom_list))]
    difference_meV_per_atom_list.insert(0, float("nan"))

    # Calculate kppa
    kppa_list = [np.prod(kpoint_grid) * num_atoms for kpoint_grid, num_atoms in zip(kpoint_grid_list, number_of_atoms_list)]

    # Create DataFrame
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
            title=dict(text=x_title, font=dict(size=22, color="rgb(0,0,0)")),
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
            title=dict(text=y_title, font=dict(size=22, color="rgb(0,0,0)")),
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
    """Makes a plot for ENCUT convergence using plotly.

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
    plot_format(fig, "ENCUT (eV)", "Energy (eV/atom)")

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


def calculate_encut_conv(path: str, plot: bool = True) -> tuple[pd.DataFrame, go.Figure]:
    """Convenience function to calculate the energy convergence with respect to ENCUT and plots the results.

    Args:
        path: path to the folder containing the ENCUT convergence calculation results.
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
    """Makes a plot for k-point convergence using plotly.

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


def calculate_kpoint_conv(path: str, plot: bool = True) -> tuple[pd.DataFrame, go.Figure]:
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
