"""
Module for calculating the electronic contributions to the Helmholtz free energy, entropy, and heat capacity.
"""

# Standard Library Imports
import os
import numpy as np

# Related third party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.constants
from scipy.special import expit
from scipy.optimize import bisect
from natsort import natsorted
from scipy.interpolate import UnivariateSpline

# Local application/library specific imports
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin

# DFTTK imports
from dfttk.plotly_format import plot_format
from dfttk.data_extraction import parse_doscar

BOLTZMANN_CONSTANT = (
    scipy.constants.Boltzmann / scipy.constants.electron_volt
)  # The Boltzmann constant in eV/K


# TODO: refactor into the ThermalElectronic class
def read_total_electron_dos(
    path: str,  # TODO: Update methods in other modules that depend on these arguments!
    folder_prefix: str = "elec",
    contcar_name: str = "CONTCAR.elec_dos",
    vasprun_name: str = "vasprun.xml.elec_dos",
    selected_volumes: np.ndarray = None,
    plot: bool = False,
) -> pd.DataFrame:
    """Reads the total electron DOS data from the VASP calculations for different volumes.

    Args:
        path (str): path to the directory containing the specific folders containing the CONTCAR, vasprun.xml, and DOSCAR files.
        folder_prefix (str, optional): prefix of the electronic folders. Defaults to "elec".
        contcar_name (str, optional): name of the CONTCAR file. Defaults to "CONTCAR.elec_dos".
        vasprun_name (str, optional): name of the vasprun.xml file. Defaults to "vasprun.xml.elec_dos".
        selected_volumes (np.ndarray, optional): list of selected volumes to keep the electron DOS data. Defaults to None.
        plot (bool, optional): plots the total electron DOS for different volumes. Defaults to False.

    Returns:
        pd.DataFrame: dataframe containing the electron DOS data.
    """

    # Get the list of electronic folders
    elec_folders = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder))
        and folder.startswith(folder_prefix)
    ]
    elec_folders = natsorted(elec_folders)

    # Initialize lists to store data
    volume_list = []
    num_atoms_list = []
    vasprun_energies_list = []
    vasprun_dos_list = []

    # Iterate over electronic folders to get the relevant electronic DOS data
    for elec_folder in elec_folders:

        # Get the volume from CONTCAR
        struct = Structure.from_file(os.path.join(path, elec_folder, contcar_name))
        volume = round(struct.volume, 6)
        volume_list.append(volume)

        # Get the number of atoms from vasprun.xml
        vasprun_path = os.path.join(path, elec_folder, vasprun_name)
        vasprun = Vasprun(vasprun_path)
        number_of_atoms = vasprun.final_structure.num_sites
        num_atoms_list.append(number_of_atoms)

        # Get the vasprun energies minus Fermi energy from vasprun.xml. For plotting.
        vasprun_energies = vasprun.complete_dos.energies - vasprun.efermi
        vasprun_energies_list.append(vasprun_energies)

        # Get the vasprun DOS from vasprun.xml. For plotting.
        try:
            # For spin polarized calculations
            vasprun_dos = (
                vasprun.complete_dos.densities[Spin.up]
                + vasprun.complete_dos.densities[Spin.down]
            )
        except:
            # For non-spin polarized calculations
            vasprun_dos = vasprun.complete_dos.densities[Spin.up]
        vasprun_dos_list.append(vasprun_dos)

    # Create a dataframe to store the electron DOS data
    electron_dos_data = pd.DataFrame(
        {
            "volumes": volume_list,
            "number_of_atoms": num_atoms_list,
            "energy_minus_fermi_energy": vasprun_energies_list,
            "total_dos": vasprun_dos_list,
        }
    )
    electron_dos_data = electron_dos_data.sort_values(by="volumes")
    electron_dos_data = electron_dos_data.reset_index(drop=True)

    # Filter the dataframe to only include selected volumes
    if selected_volumes is not None:
        electron_dos_data = electron_dos_data[
            electron_dos_data["volumes"].isin(selected_volumes)
        ]
        electron_dos_data = electron_dos_data.reset_index(drop=True)

    # Plot the total electron DOS for different volumes
    if plot:
        plot_total_electron_dos(electron_dos_data)

    return electron_dos_data


def plot_total_electron_dos(electron_dos_data: pd.DataFrame):
    """Plots the total electron DOS for different volumes.

    Args:
        electron_dos_data (pd.DataFrame): dataframe containing the electron DOS data.

    Returns:
        go.Figure: Plotly figure object.
    """

    fig = go.Figure()
    for i in range(len(electron_dos_data)):
        fig.add_trace(
            go.Scatter(
                x=electron_dos_data["energy_minus_fermi_energy_plot"].iloc[i],
                y=electron_dos_data["total_dos_plot"].iloc[i],
                mode="lines",
                name=f"{electron_dos_data['volumes'].iloc[i]} Å<sup>3</sup>",
                showlegend=True,
            )
        )
    plot_format(
        fig,
        xtitle="E - E<sub>F</sub> (eV)",
        ytitle=f"DOS (states/eV/{electron_dos_data['number_of_atoms'].iloc[i]} atoms)",
    )
    fig.show()
    return fig


def fit_electron_dos(
    energies: np.ndarray,
    dos: np.ndarray,
    energy_range: np.ndarray,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fits the electron DOS with a spline.

    Args:
        energies (np.ndarray): energy values for the electron DOS.
        dos (np.ndarray): electron DOS values.
        energy_range (np.ndarray): energy range to fit the electron DOS.
        resolution (float): energy resolution for the spline.

    Returns:
        tuple[np.ndarray, np.ndarray]: fitted energy and DOS values.
    """

    # Filter the energy and dos values within the energy range
    filtered_indices = (energies >= energy_range[0]) & (energies <= energy_range[1])
    filtered_energy = energies[filtered_indices]
    filtered_dos = dos[filtered_indices]

    # Fit the filtered energy and dos values with a spline
    spline = UnivariateSpline(filtered_energy, filtered_dos, s=0)
    energy_fit = np.arange(energy_range[0], energy_range[1] + resolution, resolution)
    dos_fit = spline(energy_fit)

    # Ensure that the DOS values are non-negative
    dos_fit[dos_fit < 0] = 0

    return energy_fit, dos_fit


def fermi_dirac_distribution(
    energies: np.ndarray,
    chemical_potential: float,
    temperature: float,
    plot: bool = False,
) -> np.ndarray:
    """Calculates the Fermi-Dirac distribution function given by the formula:
        f(E, mu, T) = 1 / (1 + exp((E - mu) / (k_B T)))

    Args:
        energies (np.ndarray): energy values for the electron DOS.
        chemical_potential (float): chemical potential for a given volume and temperature.
        temperature (float): temperature in K.
        plot (bool, optional): plots the Fermi-Dirac distribution function vs. energy for a
        given temperature and chemical potential. Defaults to False.

    Raises:
        ValueError: Temperature cannot be less than 0 K.

    Returns:
        np.ndarray: Fermi-Dirac distribution function values.
    """

    chemical_potential = float(chemical_potential)
    temperature = float(temperature)

    if temperature < 0:
        raise ValueError("Temperature cannot be less than 0 K")

    if temperature == 0:
        fermi_dist = np.where(energies <= chemical_potential, 1, 0)

    elif temperature > 0:
        # Note that expit(x) = 1/(1+exp(-x))
        fermi_dist = expit(
            -(energies - chemical_potential) / (BOLTZMANN_CONSTANT * temperature)
        )

    if plot:
        plot_fermi_dirac_distribution(
            energies, fermi_dist, chemical_potential, temperature
        )

    return fermi_dist


def plot_fermi_dirac_distribution(
    energies: np.ndarray,
    fermi_dist: np.ndarray,
    chemical_potential: float,
    temperature: float,
) -> go.Figure:
    """Plots the Fermi-Dirac distribution function vs. energy for a given temperature and
    chemical potential.

    Args:
        energy (np.ndarray): energy values for the electron DOS.
        fermi_dist (np.ndarray):  Fermi-Dirac distribution function values.
        chemical_potential (float): chemical potential for a given volume and temperature.
        temperature (float): temperature in K.
    """

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energies, y=fermi_dist, mode="lines"))
    fig.update_layout(
        title=dict(
            text=f"T = {temperature} K, &mu; = {chemical_potential} eV",
            font=dict(size=20, color="rgb(0,0,0)"),
        ),
        margin=dict(t=130),
    )
    plot_format(fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="f (E, T, V) ")
    fig.show()

    return fig


def calculate_num_electrons(
    energies: np.ndarray,
    dos: np.ndarray,
    chemical_potential: float,
    temperature: float,
) -> float:
    """Calculates the number of electrons for a given electronic DOS, chemical potential, and temperature using the formula:
        N = ∫ DOS(E) * f(E, mu, T) dE

    Args:
        energies (np.ndarray): energy values for the electron DOS.
        dos (np.ndarray): electron DOS values.
        chemical_potential (float): chemical potential for a given volume and temperature.
        temperature (float): temperature in K.

    Raises:
        ValueError: Temperature cannot be less than 0 K.

    Returns:
        float: number of electrons.
    """

    chemical_potential = float(chemical_potential)
    temperature = float(temperature)

    if temperature < 0:
        raise ValueError("Temperature cannot be less than 0 K")

    fermi_dist = fermi_dirac_distribution(energies, chemical_potential, temperature)
    integrand = dos * fermi_dist
    num_electrons = np.trapz(integrand, energies)

    return num_electrons


def calculate_chemical_potential(
    energies: np.ndarray,
    dos: np.ndarray,
    temperature: float,
    chemical_potential_range: np.ndarray = np.array([-0.1, 0.1]),
) -> float:
    """Calculates the chemical potential at a given electronic DOS, temperature, and volume
    such that the number of electrons is equal to that at 0 K. This is done by adjusting the chemical potential.

    Args:
        energies (np.ndarray): energy values for the electron DOS.
        dos (np.ndarray): electron DOS values.
        temperature (float): temperature in K.
        chemical_potential_range (np.ndarray, optional): range to search for the chemical potential. Defaults to np.array([-0.1, 0.1]).

    Returns:
        float: chemical potential at a given electronic DOS, temperature, and volume.
    """

    temperature = float(temperature)

    # Calculate the number of electrons at 0 K and at the given temperature with an initial guess of chemical potential = 0
    num_electrons_0K = round(
        calculate_num_electrons(
            energies=energies, dos=dos, chemical_potential=0, temperature=0
        )
    )
    num_electrons_guess = round(
        calculate_num_electrons(
            energies=energies, dos=dos, chemical_potential=0, temperature=temperature
        )
    )

    # If the number of electrons at the guess chemical potential is equal to that at 0 K (with rounding), return the guess chemical potential
    if num_electrons_guess == num_electrons_0K:
        chemical_potential = 0

    # Else use the bisection method to find the chemical potential that gives the correct number of electrons
    else:

        def electron_difference(chemical_potential):
            num_electrons = calculate_num_electrons(
                energies=energies,
                dos=dos,
                chemical_potential=chemical_potential,
                temperature=temperature,
            )
            return num_electrons - num_electrons_0K

        try:
            chemical_potential = bisect(
                electron_difference,
                chemical_potential_range[0],
                chemical_potential_range[1],
            )
        except ValueError as e:
            print(
                f"Warning: The chemical potential could not be found within the range {chemical_potential_range[0]} to {chemical_potential_range[1]} eV."
                "Consider increasing the chemical_potential_range."
            )
            chemical_potential = chemical_potential_range[1]

    return chemical_potential


def calculate_internal_energies(
    energies: np.ndarray,
    dos: np.ndarray,
    temperatures: np.ndarray,
    chemical_potentials: np.ndarray,
    resolution: float = 0.001,
    plot: bool = False,
    selected_temperatures: np.ndarray = None,
) -> np.ndarray:
    """Calculates the thermal electronic contribution to the internal energy for a given volume using the formula:
        U_el(T, V) = ∫ DOS(E) * f(E, mu, T) * E dE - ∫_(E<mu) DOS(E) * E dE

    Args:
        energies (np.ndarray): energy values from the electron DOS.
        dos (np.ndarray): electron DOS values.
        temperatures (np.ndarray): temperatures in K.
        chemical_potentials (np.ndarray): chemical potentials for each temperature.
        resolution (float, optional): energy resolution for the spline. Defaults to 0.001.
        plot (bool, optional): plots the integrand vs energy of the internal energy equation. Defaults to False.
        selected_temperatures (np.ndarray, optional): temperatures to plot the integrand vs energy of the internal energy equation. Defaults to None.

    Returns:
        np.ndarray: internal energy values.
    """

    # Fit the whole energy range of the electron DOS with a spline with the given resolution
    energies_fit, dos_fit = fit_electron_dos(
        energies, dos, [np.min(energies), np.max(energies)], resolution
    )

    # Initialize lists to store data
    integrand_1_list = []
    filtered_energies_list = []
    integrand_2_list = []
    internal_energies_list = []

    # Calculate the internal energy for each temperature
    for i, temperature in enumerate(temperatures):

        # Calculate the Fermi-Dirac distribution function at the given temperature and chemical potential
        chemical_potential = chemical_potentials[i]
        fermi_dist = fermi_dirac_distribution(
            energies_fit, chemical_potential, temperature
        )

        # Evaluate the first integral over the entire energy range
        integrand_1 = dos_fit * fermi_dist * energies_fit
        integrand_1_list.append(integrand_1)
        integral_1 = np.trapz(integrand_1, energies_fit)

        # Evaluate the second integral from -infinity to the chemical potential
        mask = energies_fit < chemical_potential
        filtered_energies = energies_fit[mask]
        filtered_energies_list.append(filtered_energies)
        filtered_dos = dos_fit[mask]

        integrand_2 = filtered_dos * filtered_energies
        integrand_2_list.append(integrand_2)
        integral_2 = np.trapz(integrand_2, filtered_energies)

        internal_energies = integral_1 - integral_2
        internal_energies_list.append(internal_energies)

    # Convert lists to numpy arrays
    integrand_1 = np.array(integrand_1_list)
    filtered_energies = np.array(filtered_energies_list)
    integrand_2 = np.array(integrand_2_list)
    internal_energies = np.array(internal_energies_list)

    # Plot the integrands if requested
    if plot:
        for selected_temperature in selected_temperatures:
            index = np.where(temperatures == selected_temperature)[0][0]

            plot_internal_energy_integral(
                energies_fit,
                integrand_1[index],
                filtered_energies[index],
                integrand_2[index],
                selected_temperature,
            )

    return internal_energies


def plot_internal_energy_integral(
    energies: np.ndarray,
    integrand_1: np.ndarray,
    filtered_energies: np.ndarray,
    integrand_2: np.ndarray,
    plot_temperature: float,
) -> go.Figure:
    """Plots the integrands vs energy of the internal energy equation.

    Args:
        energies (np.ndarray): energy values for the electron DOS.
        integrand_1 (np.ndarray): integrand 1 from the internal energy equation.
        filtered_energies (np.ndarray): filtered energy values for the electron DOS (where E < mu).
        integrand_2 (np.ndarray): integrand 2 from the internal energy equation.
        plot_temperature (float): temperature in K.

    Returns:
        go.Figure: Plotly figure object.
    """

    plot_temperature = float(plot_temperature)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energies, y=integrand_1, mode="markers"))
    fig.update_layout(
        title=dict(
            text=f"T = {plot_temperature} K",
            font=dict(size=20, color="rgb(0,0,0)"),
        ),
        margin=dict(t=130),
    )
    plot_format(
        fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="E<sub>el</sub> integrand 1"
    )
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_energies, y=integrand_2, mode="markers"))
    fig.update_layout(
        title=dict(
            text=f"T = {plot_temperature} K",
            font=dict(size=20, color="rgb(0,0,0)"),
        ),
        margin=dict(t=130),
    )
    plot_format(
        fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="E<sub>el</sub> integrand 2"
    )
    fig.show()

    return fig


def calculate_entropies(
    energies: np.ndarray,
    dos: np.ndarray,
    temperatures: np.ndarray,
    chemical_potentials: np.ndarray,
    energies_fit_range: np.ndarray = np.array([-2, 2]),
    resolution: float = 0.0001,
    plot: bool = False,
    selected_temperatures: np.ndarray = None,
) -> np.ndarray:
    """Calculates the thermal electronic contribution to the entropy for a given volume using the formula:
        S_el(T, V) = -k_B ∫ DOS(E) * [f(E, mu, T) * ln(f(E, mu, T)) + (1 - f(E, mu, T)) * ln(1 - f(E, mu, T))] dE

    Args:
        energies (np.ndarray): energy values for the electron DOS.
        dos (np.ndarray): electron DOS values.
        temperatures (np.ndarray): temperatures in K.
        chemical_potentials (np.ndarray): chemical potentials for each temperature.
        energies_fit_range (np.ndarray, optional): energy range to fit the electron DOS. Defaults to np.array([-2, 2]).
        resolution (float, optional): energy resolution for the spline. Defaults to 0.0001.
        plot (bool, optional): plots the integrand vs energy of the entropy equation. Defaults to False.
        selected_temperatures (np.ndarray, optional): temperatures to plot the integrand vs energy of the entropy equation. Defaults to None.

    Returns:
        np.ndarray: entropy values.
    """

    # Fit the electron DOS within the specified energy range with a spline
    energy_fit, dos_fit = fit_electron_dos(
        energies, dos, energies_fit_range, resolution
    )

    # Initialize lists to store data
    integrand_list = []
    entropies_list = []

    # Calculate the entropy for each temperature
    for i, temperature in enumerate(temperatures):
        chemical_potential = chemical_potentials[i]

        # Entropy is zero at 0 K
        if temperature == 0:
            entropy = 0
            entropies_list.append(entropy)

            integrand = np.zeros_like(energy_fit)
            integrand_list.append(integrand)

        # Calculate the entropy at finite temperatures
        elif temperature > 0:
            fermi_dist = fermi_dirac_distribution(
                energy_fit, chemical_potential, temperature
            )

            # At finite temperatures, f is never exactly 0 or 1, but due to lack of numerical precision, we may encounter these values.
            # The limit of f ln f + (1-f) ln (1-f) as f approaches 0 or 1 is 0
            # We use a mask to avoid log(0) issues
            mask = (fermi_dist == 0) | (fermi_dist == 1)
            integrand = np.zeros_like(fermi_dist)
            integrand[~mask] = dos_fit[~mask] * (
                fermi_dist[~mask] * np.log(fermi_dist[~mask])
                + (1 - fermi_dist[~mask]) * np.log(1 - fermi_dist[~mask])
            )
            integrand[mask] = 0
            integrand_list.append(integrand)

            entropy = -BOLTZMANN_CONSTANT * np.trapz(integrand, energy_fit)
            entropies_list.append(entropy)

        # Convert lists to numpy arrays
        integrand = np.array(integrand_list)
        entropies = np.array(entropies_list)

    # Plot the integrand if requested
    if plot:
        for plot_temperature in selected_temperatures:
            index = np.where(temperatures == plot_temperature)[0][0]
            plot_entropy_integral(energy_fit, integrand[index], plot_temperature)

    return entropies


def plot_entropy_integral(
    energies: np.ndarray, integrand: np.ndarray, plot_temperature: float
) -> go.Figure:
    """Plots the integrand vs energy of the entropy equation.

    Args:
        energies (np.ndarray): energy values for the electron DOS.
        integrand (np.ndarray): integrand from the entropy equation.
        plot_temperature (float): temperature in K.

    Returns:
        go.Figure: Plotly figure object.
    """

    plot_temperature = float(plot_temperature)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=energies, y=-BOLTZMANN_CONSTANT * integrand, mode="markers")
    )
    fig.update_layout(
        title=dict(
            text=f"T = {plot_temperature} K",
            font=dict(size=20, color="rgb(0,0,0)"),
        ),
        margin=dict(t=130),
    )
    plot_format(fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="S<sub>el</sub> integrand")
    fig.show()

    return fig


def calculate_heat_capacities(
    energies: np.ndarray,
    dos: np.ndarray,
    temperatures: np.ndarray,
    chemical_potentials: np.ndarray,
    energies_fit_range: np.ndarray = np.array([-2, 2]),
    resolution: float = 0.0001,
    plot=False,
    selected_temperatures: np.ndarray = None,
) -> np.array:
    """Calculates the thermal electronic contribution to the heat capacity for a given volume using the formula:
    Cv_el(T, V) = ∫ DOS(E) * f(E, mu, T) * (1 - f(E, mu, T)) * (E - mu)**2 / (k_B T**2) dE

    Args:
        energies (np.ndarray): energy values for the electron DOS.
        dos (np.ndarray): electron DOS values.
        temperatures (np.ndarray): temperatures in K.
        chemical_potentials (np.ndarray): chemical potentials for each temperature.
        energies_fit_range (np.ndarray, optional): energy range to fit the electron DOS. Defaults to np.array([-2, 2]).
        resolution (float, optional): energy resolution for the spline. Defaults to 0.0001.
        plot (bool, optional): plots the integrand vs energy of the heat capacity equation. Defaults to False.
        selected_temperatures (np.ndarray, optional): temperatures to plot the integrand vs energy of the heat capacity equation. Defaults to None.

    Returns:
        np.array: heat capacity values.
    """

    # Fit the electron DOS within the specified energy range with a spline
    energies_fit, dos_fit = fit_electron_dos(
        energies, dos, energies_fit_range, resolution
    )

    # Initialize lists to store data
    integrand_list = []
    heat_capacities_list = []

    # Calculate the heat capacity for each temperature
    for i, temperature in enumerate(temperatures):
        chemical_potential = chemical_potentials[i]

        # Heat capacity is zero at 0 K
        if temperature == 0:
            heat_capacity = 0
            heat_capacities_list.append(heat_capacity)

            integrand = np.zeros_like(energies_fit)
            integrand_list.append(integrand)

        # Calculate the heat capacity at finite temperatures
        elif temperature > 0:
            fermi_dist = fermi_dirac_distribution(
                energies_fit, chemical_potential, temperature
            )

            integrand = (
                dos_fit
                * fermi_dist
                * (1 - fermi_dist)
                * ((energies_fit - chemical_potential) / temperature) ** 2
                / BOLTZMANN_CONSTANT
            )
            integrand_list.append(integrand)

            heat_capacities = np.trapz(integrand, energies_fit)
            heat_capacities_list.append(heat_capacities)

    # Convert lists to numpy arrays
    integrand = np.array(integrand_list)
    heat_capacities = np.array(heat_capacities_list)

    # Plot the integrand if requested
    if plot:
        for selected_temperature in selected_temperatures:
            index = np.where(temperatures == selected_temperature)[0][0]
            plot_heat_capacity_integral(
                energies_fit, integrand[index], selected_temperature
            )

    return heat_capacities


def plot_heat_capacity_integral(
    energies: np.ndarray, integrand: np.ndarray, selected_temperature: float
) -> go.Figure:
    """Plots the integrand vs energy of the heat capacity equation.

    Args:
        energies (np.ndarray): energy values for the electron DOS.
        integrand (np.ndarray): integrand from the heat capacity equation.
        selected_temperature (float): temperature in K.

    Returns:
        go.Figure: Plotly figure object.
    """

    selected_temperature = float(selected_temperature)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energies, y=integrand, mode="markers"))
    fig.update_layout(
        title=dict(
            text=f"T = {selected_temperature} K",
            font=dict(size=20, color="rgb(0,0,0)"),
        ),
        margin=dict(t=130),
    )
    plot_format(
        fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="C<sub>v,el</sub> integrand"
    )
    fig.show()

    return fig


def calculate_helmholtz_energies(
    internal_energies: np.ndarray,
    entropies: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """Calculates the thermal electronic contribution to the Helmholtz free energy for a given volume using the formula:
        F_el(T, V) = U_el(T, V) - T * S_el(T, V)

    Args:
        internal_energies (np.ndarray): internal energy values.
        entropies (np.ndarray): entropy values.
        temperatures (np.ndarray): temperatures in K.

    Returns:
        np.ndarray: Helmholtz free energy values.
    """

    helmholtz_energies = internal_energies - temperatures * entropies

    return helmholtz_energies


def thermal_electronic(
    volumes: np.ndarray,
    temperatures: np.ndarray,
    energy_array: np.ndarray,
    dos_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the thermal electronic contributions to Helmholtz free energy, internal energy, entropy, and heat capacity.

    Args:
        volumes (np.ndarray): volumes.
        temperatures (np.ndarray): temperatures in K.
        energy_array (np.ndarray): a 2D array of energy values where each column corresponds to a volume.
        dos_array (np.ndarray): a 2D array of electron DOS values where each column corresponds to a volume.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        2D arrays of helmholtz energy, internal energy, entropy, heat capacity where each row corresponds to a temperature and each column corresponds to a volume.
    """

    # Initialize lists to store data
    chemical_potentials_list = []
    internal_energies_list = []
    entropies_list = []
    heat_capacities_list = []
    helmholtz_energies_list = []

    # Compute thermal electronic properties for each volume
    for i in range(len(volumes)):
        energies = energy_array[:, i]
        dos = dos_array[:, i]

        # For each volume, compute the thermal electronic properties at each temperature
        for temperature in temperatures:
            chemical_potential = calculate_chemical_potential(
                energies, dos, temperature
            )
            chemical_potentials_list.append(chemical_potential)
        chemical_potential_array = np.array(chemical_potentials_list)

        internal_energies = calculate_internal_energies(
            energies, dos, temperatures, chemical_potential_array
        )
        entropies = calculate_entropies(
            energies, dos, temperatures, chemical_potential_array
        )
        heat_capacities = calculate_heat_capacities(
            energies, dos, temperatures, chemical_potential_array
        )
        helmholtz_energies = calculate_helmholtz_energies(
            internal_energies, entropies, temperatures
        )

        internal_energies_list.append(internal_energies)
        entropies_list.append(entropies)
        heat_capacities_list.append(heat_capacities)
        helmholtz_energies_list.append(helmholtz_energies)

    # Convert lists to numpy arrays and transpose to have rows as temperatures and columns as volumes
    helmholtz_energies = np.array(helmholtz_energies_list).T
    internal_energies = np.array(internal_energies_list).T
    entropies = np.array(entropies_list).T
    heat_capacities = np.array(heat_capacities_list).T

    return (
        helmholtz_energies,
        internal_energies,
        entropies,
        heat_capacities,
    )


def fit_thermal_electronic(
    volumes: np.ndarray,
    volumes_fit: np.ndarray,
    temperatures: np.ndarray,
    helmholtz_energies: np.ndarray,
    entropies: np.ndarray,
    heat_capacities: np.ndarray,
    order: int = 1,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Fits the Helmholtz free energy, entropy, and heat capacity vs. volume for various fixed temperatures.

    Args:
        volumes (np.ndarray): volumes.
        volumes_fit (np.ndarray): 1D array of volumes to fit the properties to.
        temperatures (np.ndarray): temperatures in K.
        helmholtz_energies (np.ndarray): helmholtz energy values
        entropies (np.ndarray): entropy values
        heat_capacities (np.ndarray): heat capacity values
        order (int): order of the polynomial fit. Defaults to 1 (linear fit).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fitted volumes, fitted helmholtz energies, fitted entropies, fitted heat capacities, helmholtz free energy polynomials, entropy polynomials, heat capacity polynomials.
    """

    # Initialize lists to store data
    volume_fit_list = []
    helmholtz_energy_fit_list = []
    entropy_fit_list = []
    heat_capacity_fit_list = []
    helmholtz_energies_poly_coeffs = []
    entropies_poly_coeffs = []
    heat_capacities_poly_coeffs = []

    # Fit the properties vs. volume for each temperature
    for i in range(len(temperatures)):

        # Fit Helmholtz free energy, entropy, and heat capacity vs. volume with a polynomial of the specified order
        helmholtz_energy_coefficients = np.polyfit(
            volumes, helmholtz_energies[i], order
        )
        entropy_coefficients = np.polyfit(volumes, entropies[i], order)
        heat_capacity_coefficients = np.polyfit(volumes, heat_capacities[i], order)

        helmholtz_energies_poly_coeffs.append(helmholtz_energy_coefficients)
        entropies_poly_coeffs.append(entropy_coefficients)
        heat_capacities_poly_coeffs.append(heat_capacity_coefficients)

        helmholtz_energy_polynomial = np.poly1d(helmholtz_energy_coefficients)
        entropy_polynomial = np.poly1d(entropy_coefficients)
        heat_capacity_polynomial = np.poly1d(heat_capacity_coefficients)

        # Evaluate the polynomials at the specified fit volumes
        helmholtz_energy_fit = helmholtz_energy_polynomial(volumes_fit)
        entropy_fit = entropy_polynomial(volumes_fit)
        heat_capacity_fit = heat_capacity_polynomial(volumes_fit)

        volume_fit_list.append(volumes_fit)
        helmholtz_energy_fit_list.append(helmholtz_energy_fit)
        entropy_fit_list.append(entropy_fit)
        heat_capacity_fit_list.append(heat_capacity_fit)

    volumes_fit = np.array(volume_fit_list)
    helmholtz_energies_fit = np.array(helmholtz_energy_fit_list)
    entropies_fit = np.array(entropy_fit_list)
    heat_capacities_fit = np.array(heat_capacity_fit_list)
    helmholtz_energies_poly_coeffs = np.array(helmholtz_energies_poly_coeffs)
    entropies_poly_coeffs = np.array(entropies_poly_coeffs)
    heat_capacities_poly_coeffs = np.array(heat_capacities_poly_coeffs)

    return (
        volumes_fit,
        helmholtz_energies_fit,
        entropies_fit,
        heat_capacities_fit,
        helmholtz_energies_poly_coeffs,
        entropies_poly_coeffs,
        heat_capacities_poly_coeffs,
    )


def plot_thermal_electronic(
    number_of_atoms: int,
    volumes: np.ndarray,
    temperatures: np.ndarray,
    property: np.ndarray,
    property_name: str,
) -> go.Figure:
    """
    Plots the Helmholtz free energy, entropy, or heat capacity vs. temperature for various fixed volumes.

    Args:
        number_of_atoms (int): number of atoms the properties are per.
        volumes (np.ndarray): volumes.
        temperatures (np.ndarray): temperatures.
        property (np.ndarray): property values. Can be helmholtz energy, entropy, or heat capacity.
        property_name (str): name of the property to plot.
    Raises:
        ValueError: property_name must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'

    Returns:
        go.Figure: Plotly figure object.
    """

    valid_properties = {
        "helmholtz_energy": f"F<sub>el</sub> (eV/{number_of_atoms} atoms)",
        "entropy": f"S<sub>el</sub> (eV/K/{number_of_atoms} atoms)",
        "heat_capacity": f"C<sub>el</sub> (eV/K/{number_of_atoms} atoms)",
    }

    if property_name not in valid_properties:
        raise ValueError(
            "property_name must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'"
        )

    property = np.vstack(property)
    y_title = valid_properties[property_name]

    fig = go.Figure()
    for i, volume in enumerate(volumes):
        fig.add_trace(
            go.Scatter(
                x=temperatures,
                y=property[:, i],
                mode="lines",
                name=f"{volume} Å³",
                showlegend=True,
            )
        )
    plot_format(fig, "Temperature (K)", y_title)

    return fig


def plot_thermal_electronic_properties_fit(
    number_of_atoms: int,
    volumes: np.ndarray,
    temperatures: np.ndarray,
    property_name: str,
    property: np.ndarray,
    volumes_fit: np.ndarray,
    property_fit: np.ndarray,
    selected_temperatures: np.ndarray = None,
) -> go.Figure:
    """Plots the fitted Helmholtz free energy, entropy, or heat capacity vs. volume for various fixed temperatures.

    Args:
        number_of_atoms (int): number of atoms the properties are per.
        volumes (np.ndarray): volumes.
        property (np.ndarray): helmholtz energy, entropy, or heat capacity.
        volumes_fit (np.ndarray): fitted volumes.
        property_fit (np.ndarray): fitted helmholtz energy, entropy, or heat capacity.
        selected_temperatures (np.ndarray, optional): selected temperatures to plot. Defaults to None.

    Raises:
        ValueError: property_name must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'

    Returns:
        go.Figure: Plotly figure object.
    """

    if property_name not in ["helmholtz_energy", "entropy", "heat_capacity"]:
        raise ValueError(
            "property_name must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'"
        )

    if selected_temperatures is None:
        indices = np.linspace(0, len(temperatures) - 1, 5, dtype=int)
        selected_temperatures = np.array([temperatures[j] for j in indices])

    fig = go.Figure()
    colors = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]
    colors = [
        f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {1})"
        for color in colors
    ]

    for i, temperature in enumerate(selected_temperatures):
        index = np.where(temperatures == temperature)[0][0]
        x = volumes
        y = property[index]
        x_fit = volumes_fit
        y_fit = property_fit[index]

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                line=dict(color=color),
                legendgroup=f"{temperature} K",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                line=dict(color=color),
                name=f"{temperature} K",
                legendgroup=f"{temperature} K",
                showlegend=True,
            )
        )

    if property_name == "helmholtz_energy":
        y_title = f"F<sub>el</sub> (eV/{number_of_atoms} atoms)"
    elif property_name == "entropy":
        y_title = f"S<sub>el</sub> (eV/K/{number_of_atoms} atoms)"
    elif property_name == "heat_capacity":
        y_title = f"C<sub>v, el</sub> (eV/K/{number_of_atoms} atoms)"

    plot_format(fig, f"Volume (Å³/{number_of_atoms} atoms)", y_title)

    return fig
