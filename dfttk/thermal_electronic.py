"""
Module for calculating the thermal electronic contributions to Helmholtz energy, entropy, and heat capacity.
"""

# Standard Library Imports
import os

# Related third party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.constants
from scipy.special import expit
from natsort import natsorted
from scipy.interpolate import UnivariateSpline

# Local application/library specific imports
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin

# DFTTK imports
from dfttk.plotly_format import plot_format

BOLTZMANN_CONSTANT = (
    scipy.constants.Boltzmann / scipy.constants.electron_volt
)  # The Boltzmann constant in eV/K


def read_total_electron_dos(path: str, plot: bool = False) -> pd.DataFrame:
    """Reads the total electron DOS from vasprun.xml files.

    Args:
        path (str): path to the directory containing the elec_folders.
        plot (bool, optional): plots the total electron DOS for different volumes. Defaults to False.

    Returns:
        pd.DataFrame: dataframe containing the electron DOS data.
    """

    elec_folders = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder)) and folder.startswith("elec")
    ]
    elec_folders = natsorted(elec_folders)

    volume_list = []
    num_atoms_list = []
    energy_minus_fermi_energy_list = []
    total_dos_list = []
    for elec_folder in elec_folders:
        struct = Structure.from_file(
            os.path.join(path, elec_folder, "CONTCAR.elec_dos")
        )
        volume = round(struct.volume, 6)

        vasprun_path = os.path.join(path, elec_folder, "vasprun.xml.elec_dos")
        vasprun = Vasprun(vasprun_path)

        num_atoms = vasprun.final_structure.num_sites

        # TODO: Implement this for magnetic systems as well
        complete_dos = vasprun.complete_dos
        energy = complete_dos.energies
        total_dos = complete_dos.densities[Spin.up]

        fermi_energy = vasprun.efermi
        energy_minus_fermi_energy = energy - fermi_energy

        volume_list.append(volume)
        num_atoms_list.append(num_atoms)
        energy_minus_fermi_energy_list.append(energy_minus_fermi_energy)
        total_dos_list.append(total_dos)

    electron_dos_data = pd.DataFrame(
        {
            "volume": volume_list,
            "number_of_atoms": num_atoms_list,
            "energy_minus_fermi_energy": energy_minus_fermi_energy_list,
            "total_dos": total_dos_list,
        }
    )
    electron_dos_data = electron_dos_data.sort_values(by="volume")
    electron_dos_data = electron_dos_data.reset_index(drop=True)

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
                x=electron_dos_data["energy_minus_fermi_energy"].iloc[i],
                y=electron_dos_data["total_dos"].iloc[i],
                mode="lines",
                name=f"{electron_dos_data['volume'].iloc[i]} Å<sup>3</sup>",
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
    energy: np.ndarray,
    dos: np.ndarray,
    energy_range: list,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fits the electron DOS with a spline.

    Args:
        energy (np.ndarray): energy values from the electron DOS.
        dos (np.ndarray): electron DOS values.
        energy_range (list): energy range to fit the electron DOS.
        resolution (float): energy resolution for the spline.

    Returns:
        tuple[np.ndarray, np.ndarray]: fitted energy and DOS values.
    """

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    # Filter the energy and dos values within the energy range
    filtered_indices = (energy >= energy_range[0]) & (energy <= energy_range[1])
    filtered_energy = energy[filtered_indices]
    filtered_dos = dos[filtered_indices]

    # Fit the filtered energy and dos values with a spline
    spline = UnivariateSpline(filtered_energy, filtered_dos, s=0)
    energy_fit = np.arange(energy_range[0], energy_range[1] + resolution, resolution)
    dos_fit = spline(energy_fit)

    return energy_fit, dos_fit


def fermi_dirac_distribution(
    energy: np.ndarray,
    chemical_potential: float,
    temperature: float,
    plot: bool = False,
) -> np.ndarray:
    """Calculates the Fermi-Dirac distribution function.

    Args:
        energy (np.ndarray): energy values from the electron DOS.
        chemical_potential (float): chemical potential for a given volume and temperature.
        temperature (float): temperature range.
        plot (bool, optional): plots the Fermi-Dirac distribution function vs. energy for a
        given temperature and chemical potential. Defaults to False.

    Raises:
        ValueError: Temperature cannot be less than 0 K.

    Returns:
        np.ndarray: Fermi-Dirac distribution function values.
    """

    chemical_potential = float(chemical_potential)
    temperature = float(temperature)

    # Check if energy is a pandas Series and convert to NumPy array if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]

    if temperature < 0:
        raise ValueError("Temperature cannot be less than 0 K")

    if temperature == 0:
        fermi_dist = np.where(energy < chemical_potential, 1, 0)

    if temperature > 0:
        # Note that expit(x) = 1/(1+exp(-x))
        fermi_dist = expit(
            -(energy - chemical_potential) / (BOLTZMANN_CONSTANT * temperature)
        )

    if plot:
        plot_fermi_dirac_distribution(
            energy, chemical_potential, temperature, fermi_dist
        )

    return fermi_dist


def plot_fermi_dirac_distribution(
    energy: np.ndarray,
    chemical_potential: float,
    temperature: float,
    fermi_dist: np.ndarray,
) -> go.Figure:
    """Plots the Fermi-Dirac distribution function vs. energy for a given temperature and
    chemical potential.

    Args:
        energy (np.ndarray): energy values from the electron DOS.
        chemical_potential (float): chemical potential for a given volume and temperature.
        temperature (float): temperature in K.
        fermi_dist (np.ndarray):  Fermi-Dirac distribution function values.
    """

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energy, y=fermi_dist, mode="lines"))
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
    energy: np.ndarray,
    dos: np.ndarray,
    chemical_potential: float,
    temperature: float,
) -> float:
    """Calculates the number of electrons at a given electronic DOS, chemical potential, and temperature.

    Args:
        energy (np.ndarray): energy values from the electron DOS.
        dos (np.ndarray): electron DOS values.
        chemical_potential (float): chemical potential for a given volume and temperature.
        temperature (float): temperature.

    Raises:
        ValueError: Temperature cannot be less than 0 K.

    Returns:
        float: number of electrons.
    """

    chemical_potential = float(chemical_potential)
    temperature = float(temperature)

    if temperature < 0:
        raise ValueError("Temperature cannot be less than 0 K")

    fermi_dist = fermi_dirac_distribution(energy, chemical_potential, temperature)
    integrand = dos * fermi_dist
    num_electrons = np.trapz(integrand, energy)

    return num_electrons


def calculate_chemical_potential(
    energy: np.ndarray,
    dos: np.ndarray,
    temperature: float,
    min_chemical_potential: float = -0.01,
    max_chemical_potential: float = 0.01,
) -> float:
    """Calculates the chemical potential at a given electronic DOS, temperature, and volume.

    Args:
        energy (np.ndarray): energy values from the electron DOS.
        dos (np.ndarray): electron DOS values.
        temperature (float): temperature in K.
        min_chemical_potential (float, optional): starting search for the chemical potential. Defaults to -0.01.
        max_chemical_potential (float, optional): end search for the chemical potential. Defaults to 0.01.

    Returns:
        float: chemical potential at a given electronic DOS, temperature, and volume.
    """

    temperature = float(temperature)
    num_electrons_0K = round(calculate_num_electrons(energy, dos, 0, 0))

    # Find the chemical potential at temperature such that the number of electrons matches that at 0 K
    chemical_potential = min_chemical_potential
    num_electrons = calculate_num_electrons(
        energy, dos, chemical_potential, temperature
    )

    while (
        abs(num_electrons - num_electrons_0K) > 1e-3
        and chemical_potential < max_chemical_potential
    ):
        chemical_potential += 0.001
        num_electrons = calculate_num_electrons(
            energy, dos, chemical_potential, temperature
        )

    if chemical_potential == max_chemical_potential:
        print(
            f"Warning: The chemical potential is at the maximum value of {max_chemical_potential} eV. Consider increasing the maximum chemical potential."
        )

    return chemical_potential


def calculate_internal_energy(
    energy: np.ndarray,
    dos: np.ndarray,
    temperature_range: np.ndarray,
    resolution: float = 0.001,
    plot: bool = False,
    plot_temperatures: np.ndarray = None,
) -> list:
    """Calculates the thermal electronic contribution to the internal energy.

    Args:
        energy (np.ndarray): energy values from the electron DOS.
        dos (np.ndarray): electron DOS values.
        temperature_range (np.ndarray): temperatures in K.
        resolution (float, optional): energy resolution for the spline. Defaults to 0.001.
        plot (bool, optional): plots the integrand vs energy of the internal energy equation. Defaults to False.
        plot_temperatures (np.ndarray, optional): temperatures to plot the integrand vs energy of the internal energy equation. Defaults to None.

    Returns:
        list: thermal electronic contribution to the internal energy.
    """

    energy_fit, dos_fit = fit_electron_dos(
        energy, dos, [np.min(energy), np.max(energy)], resolution
    )
    integrand_1_list = []
    filtered_energy_list = []
    integrand_2_list = []
    internal_energy_list = []
    for temperature in temperature_range:
        chemical_potential = calculate_chemical_potential(
            energy_fit, dos_fit, temperature
        )
        fermi_dist = fermi_dirac_distribution(
            energy_fit, chemical_potential, temperature
        )

        integrand_1 = dos_fit * fermi_dist * energy_fit
        integrand_1_list.append(integrand_1)
        integral_1 = np.trapz(integrand_1, energy_fit)

        # Only integrate over energy levels less than the chemical potential
        mask = energy_fit < chemical_potential
        filtered_energy = energy_fit[mask]
        filtered_energy_list.append(filtered_energy)
        filtered_dos = dos_fit[mask]

        integrand_2 = filtered_dos * filtered_energy
        integrand_2_list.append(integrand_2)
        integral_2 = np.trapz(integrand_2, filtered_energy)

        internal_energy = integral_1 - integral_2
        internal_energy_list.append(internal_energy)

    if plot:
        for plot_temperature in plot_temperatures:
            index = np.where(temperature_range == plot_temperature)[0][0]

            plot_internal_energy_integral(
                energy_fit,
                integrand_1_list[index],
                filtered_energy_list[index],
                integrand_2_list[index],
                plot_temperature,
            )

    return internal_energy_list


def plot_internal_energy_integral(
    energy: np.ndarray,
    integrand_1: np.ndarray,
    filtered_energy: np.ndarray,
    integrand_2: np.ndarray,
    plot_temperature: float,
) -> go.Figure:
    """Plots the integrand vs energy of the internal energy equation.

    Args:
        energy (np.ndarray): energy values from the electron DOS.
        integrand_1 (np.ndarray): integrand 1 from the internal energy equation.
        filtered_energy (np.ndarray): filtered energy values from the electron DOS.
        integrand_2 (np.ndarray): integrand 2 from the internal energy equation.
        plot_temperature (float): temperature in K.

    Returns:
        go.Figure: Plotly figure object.
    """

    plot_temperature = float(plot_temperature)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energy, y=integrand_1, mode="markers"))
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
    fig.add_trace(go.Scatter(x=filtered_energy, y=integrand_2, mode="markers"))
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


def calculate_entropy(
    energy: np.ndarray,
    dos: np.ndarray,
    temperature_range: np.ndarray,
    energy_range: list = [-2, 2],
    resolution: float = 0.0001,
    plot: bool = False,
    plot_temperatures: np.ndarray = None,
) -> list:
    """Calculates the thermal electronic contribution to the entropy.

    Args:
        energy (np.ndarray): energy values from the electron DOS.
        dos (np.ndarray): electron DOS values.
        temperature_range (np.ndarray): temperatures in K.
        energy_range (list, optional): energy range to fit the electron DOS. Defaults to [-2, 2].
        resolution (float, optional): energy resolution for the spline. Defaults to 0.0001.
        plot (bool, optional): plots the integrand vs energy of the entropy equation. Defaults to False.
        plot_temperatures (np.ndarray, optional): temperatures to plot the integrand vs energy of the entropy equation. Defaults to None.

    Returns:
        list: thermal electronic contribution to the entropy.
    """

    energy_fit, dos_fit = fit_electron_dos(energy, dos, energy_range, resolution)

    integrand_list = []
    entropy_list = []
    for temperature in temperature_range:
        if temperature == 0:
            entropy = 0
            entropy_list.append(entropy)

            integrand = np.zeros_like(energy_fit)
            integrand_list.append(integrand)

        elif temperature > 0:
            chemical_potential = calculate_chemical_potential(energy, dos, temperature)
            fermi_dist = fermi_dirac_distribution(
                energy_fit, chemical_potential, temperature
            )

            # The limit of f ln f + (1-f) ln (1-f) as f approaches 0 or 1 is 0
            mask = (fermi_dist == 0) | (fermi_dist == 1)
            integrand = np.zeros_like(fermi_dist)
            integrand[~mask] = dos_fit[~mask] * (
                fermi_dist[~mask] * np.log(fermi_dist[~mask])
                + (1 - fermi_dist[~mask]) * np.log(1 - fermi_dist[~mask])
            )
            integrand[mask] = 0
            integrand_list.append(integrand)

            entropy = -BOLTZMANN_CONSTANT * np.trapz(integrand, energy_fit)
            entropy_list.append(entropy)

    if plot:
        for plot_temperature in plot_temperatures:
            index = np.where(temperature_range == plot_temperature)[0][0]
            plot_entropy_integral(energy_fit, integrand_list[index], plot_temperature)

    return entropy_list


def plot_entropy_integral(
    energy: np.ndarray, integrand: np.ndarray, plot_temperature: float
) -> go.Figure:
    """Plots the integrand vs energy of the entropy equation.

    Args:
        energy (np.ndarray): energy values from the electron DOS.
        integrand (np.ndarray): integrand from the entropy equation.
        plot_temperature (float): temperature in K.

    Returns:
        go.Figure: Plotly figure object.
    """

    plot_temperature = float(plot_temperature)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=energy, y=-BOLTZMANN_CONSTANT * integrand, mode="markers")
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

    return


def calculate_heat_capacity(
    energy: np.ndarray,
    dos: np.ndarray,
    temperature_range: np.ndarray,
    energy_range: list = [-2, 2],
    resolution: float = 0.0001,
    plot=False,
    plot_temperatures: np.ndarray = None,
) -> list:
    """Calculates the thermal electronic contribution to the heat capacity.

    Args:
        energy (np.ndarray): energy values from the electron DOS.
        dos (np.ndarray): electron DOS values.
        temperature_range (np.ndarray): temperatures in K.
        energy_range (list, optional): energy range to fit the electron DOS. Defaults to [-2, 2].
        resolution (float, optional): energy resolution for the spline. Defaults to 0.0001.
        plot (bool, optional): plots the integrand vs energy of the heat capacity equation. Defaults to False.
        plot_temperatures (np.ndarray, optional): temperatures to plot the integrand vs energy of the heat capacity equation. Defaults to None.

    Returns:
        list: thermal electronic contribution to the heat capacity.
    """

    energy_fit, dos_fit = fit_electron_dos(energy, dos, energy_range, resolution)

    integrand_list = []
    heat_capacity_list = []

    for temperature in temperature_range:
        if temperature == 0:
            heat_capacity = 0
            heat_capacity_list.append(heat_capacity)

            integrand = np.zeros_like(energy_fit)
            integrand_list.append(integrand)

        elif temperature > 0:
            chemical_potential = calculate_chemical_potential(energy, dos, temperature)
            fermi_dist = fermi_dirac_distribution(
                energy_fit, chemical_potential, temperature
            )

            # The limit of (1 / f - 1) * (f * (E - mu) / T) ** 2 as f approaches 0 or 1 is 0
            mask = (fermi_dist == 0) | (fermi_dist == 1)
            integrand = np.zeros_like(fermi_dist)
            integrand[~mask] = dos_fit[~mask] * (
                (1 / fermi_dist[~mask] - 1)
                * (
                    fermi_dist[~mask]
                    * (energy_fit[~mask] - chemical_potential)
                    / temperature
                )
                ** 2
                / BOLTZMANN_CONSTANT
            )
            integrand[mask] = 0
            integrand_list.append(integrand)

            heat_capacity = np.trapz(integrand, energy_fit)
            heat_capacity_list.append(heat_capacity)

    if plot:
        for plot_temperature in plot_temperatures:
            index = np.where(temperature_range == plot_temperature)[0][0]
            plot_heat_capacity_integral(
                energy_fit, integrand_list[index], plot_temperature
            )

    return heat_capacity_list


def plot_heat_capacity_integral(
    energy: np.ndarray, integrand: np.ndarray, plot_temperature: float
) -> go.Figure:
    """Plots the integrand vs energy of the heat capacity equation.

    Args:
        energy (np.ndarray): energy values from the electron DOS.
        integrand (np.ndarray): integrand from the heat capacity equation.
        plot_temperature (float): temperature in K.

    Returns:
        go.Figure: Plotly figure object.
    """

    plot_temperature = float(plot_temperature)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energy, y=integrand, mode="markers"))
    fig.update_layout(
        title=dict(
            text=f"T = {plot_temperature} K",
            font=dict(size=20, color="rgb(0,0,0)"),
        ),
        margin=dict(t=130),
    )
    plot_format(
        fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="C<sub>v,el</sub> integrand"
    )
    fig.show()

    return fig


def calculate_free_energy(
    energy: np.ndarray,
    dos: np.ndarray,
    temperature_range: np.ndarray,
) -> list:
    """Calculates the thermal electronic contribution to the Helmholtz energy.

    Args:
        energy (np.ndarray): energy values from the electron DOS.
        dos (np.ndarray): electron DOS values.
        temperature_range (np.ndarray): temperatures in K.

    Returns:
        list: thermal electronic contribution to the Helmholtz energy.
    """

    internal_energy_list = calculate_internal_energy(energy, dos, temperature_range)
    entropy_list = calculate_entropy(energy, dos, temperature_range)
    helmholtz_energy = internal_energy_list - temperature_range * entropy_list

    return helmholtz_energy.tolist()


def thermal_electronic(
    volumes: np.ndarray,
    temperature_range: np.ndarray,
    energy_array: np.ndarray,
    dos_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the thermal electronic contributions to Helmholtz energy, internal energy, entropy, and heat capacity.

    Args:
        volumes (np.ndarray): volumes.
        temperature_range (np.ndarray): temperatures in K.
        energy_array (np.ndarray): a 2D array of energy values where each column corresponds to a volume.
        dos_array (np.ndarray): a 2D array of electron DOS values where each column corresponds to a volume.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        2D arrays of helmholtz energy, internal energy, entropy, heat capacity where each row corresponds to a temperature and each column corresponds to a volume.
    """

    internal_energy_list = []
    entropy_list = []
    heat_capacity_list = []
    helmholtz_energy_list = []

    for i in range(len(volumes)):
        energy = energy_array[:, i]
        dos = dos_array[:, i]

        internal_energy = calculate_internal_energy(energy, dos, temperature_range)
        entropy = calculate_entropy(energy, dos, temperature_range)
        heat_capacity = calculate_heat_capacity(energy, dos, temperature_range)
        helmholtz_energy = calculate_free_energy(energy, dos, temperature_range)

        internal_energy_list.append(internal_energy)
        entropy_list.append(entropy)
        heat_capacity_list.append(heat_capacity)
        helmholtz_energy_list.append(helmholtz_energy)

    # Flatten the lists of lists using list comprehension
    internal_energy_list = [
        item for sublist in internal_energy_list for item in sublist
    ]
    entropy_list = [item for sublist in entropy_list for item in sublist]
    heat_capacity_list = [item for sublist in heat_capacity_list for item in sublist]
    helmholtz_energy_list = [
        item for sublist in helmholtz_energy_list for item in sublist
    ]

    num_temps = len(temperature_range)
    num_volumes = len(volumes)

    helmholtz_energy = np.reshape(helmholtz_energy_list, (num_volumes, num_temps))
    internal_energy = np.reshape(internal_energy_list, (num_volumes, num_temps))
    entropy = np.reshape(entropy_list, (num_volumes, num_temps))
    heat_capacity = np.reshape(heat_capacity_list, (num_volumes, num_temps))

    helmholtz_energy = helmholtz_energy.T
    internal_energy = internal_energy.T
    entropy = entropy.T
    heat_capacity = heat_capacity.T

    return (
        helmholtz_energy,
        internal_energy,
        entropy,
        heat_capacity,
    )


def fit_thermal_electronic(
    volumes: np.ndarray,
    temperatures: np.ndarray,
    helmholtz_energy: np.ndarray,
    entropy: np.ndarray,
    heat_capacity: np.ndarray,
    order: int,
) -> tuple[np.array, list, list, list, list, list, list]:
    """Fits the Helmholtz energy, entropy, and heat capacity vs. volume for various fixed temperatures.

    Args:
        volumes (np.ndarray): volumes.
        temperatures (np.ndarray): temperatures in K.
        helmholtz_energy (np.ndarray): helmholtz energies.
        entropy (np.ndarray): entropies.
        heat_capacity (np.ndarray): heat capacities.
        order (int): order of the polynomial fit.

    Returns:
        tuple[np.array, list, list, list, list, list, list]:
        fitted volumes, fitted helmholtz energies, entropies, heat capacities, helmholtz energy polynomials, entropy polynomials, heat capacity polynomials.
    """

    volume_fit_list = []
    helmholtz_energy_fit_list = []
    entropy_fit_list = []
    heat_capacity_fit_list = []
    helmholtz_energy_polynomial_list = []
    entropy_polynomial_list = []
    heat_capacity_polynomial_list = []

    for i in range(len(temperatures)):
        helmholtz_energy_coefficients = np.polyfit(volumes, helmholtz_energy[i], order)
        entropy_coefficients = np.polyfit(volumes, entropy[i], order)
        heat_capacity_coefficients = np.polyfit(volumes, heat_capacity[i], order)

        helmholtz_energy_polynomial = np.poly1d(helmholtz_energy_coefficients)
        entropy_polynomial = np.poly1d(entropy_coefficients)
        heat_capacity_polynomial = np.poly1d(heat_capacity_coefficients)

        helmholtz_energy_polynomial_list.append(helmholtz_energy_polynomial)
        entropy_polynomial_list.append(entropy_polynomial)
        heat_capacity_polynomial_list.append(heat_capacity_polynomial)

        volume_fit = np.linspace(min(volumes) * 0.98, max(volumes) * 1.02, 1000)
        helmholtz_energy_fit = helmholtz_energy_polynomial(volume_fit)
        entropy_fit = entropy_polynomial(volume_fit)
        heat_capacity_fit = heat_capacity_polynomial(volume_fit)

        volume_fit_list.append(volume_fit)
        helmholtz_energy_fit_list.append(helmholtz_energy_fit)
        entropy_fit_list.append(entropy_fit)
        heat_capacity_fit_list.append(heat_capacity_fit)

    return (
        volume_fit,
        helmholtz_energy_fit_list,
        entropy_fit_list,
        heat_capacity_fit_list,
        helmholtz_energy_polynomial_list,
        entropy_polynomial_list,
        heat_capacity_polynomial_list,
    )


def plot_thermal_electronic(
    number_of_atoms: int,
    volumes: list,  # TODO: change to np.ndarray
    temperatures: np.ndarray,
    property: np.ndarray,
    property_name: str,
) -> go.Figure:

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
    fig.show()
    return fig


def plot_thermal_electronic_properties_fit(
    number_of_atoms: int,
    volumes: list,  # TODO: change to np.ndarray
    temperatures: list,  # TODO: change to np.ndarray
    property_name: str,
    property: np.ndarray,
    volume_fit: np.ndarray,
    property_fit: list,
    selected_temperatures_plot: np.ndarray = None,
) -> go.Figure:
    """Plots the fitted Helmholtz energy, entropy, or heat capacity vs. volume for various fixed temperatures.

    Args:
        number_of_atoms (int): number of atoms the properties are per.
        volumes (list): volumes.
        property (np.ndarray): helmholtz energy, entropy, or heat capacity.
        volume_fit (np.ndarray): fitted volumes.
        property_fit (list): fitted helmholtz energy, entropy, or heat capacity.
        selected_temperatures_plot (np.ndarray, optional): selected temperatures to plot. Defaults to None.

    Raises:
        ValueError: property_name must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'

    Returns:
        go.Figure: Plotly figure object.
    """

    if property_name not in ["helmholtz_energy", "entropy", "heat_capacity"]:
        raise ValueError(
            "property_name must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'"
        )

    if selected_temperatures_plot is None:
        indices = np.linspace(0, len(temperatures) - 1, 5, dtype=int)
        selected_temperatures_plot = np.array([temperatures[j] for j in indices])

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

    for i, temperature in enumerate(selected_temperatures_plot):
        index = np.where(temperatures == temperature)[0][0]
        x = volumes
        y = property[index]
        x_fit = volume_fit
        y_fit = property_fit[index]

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                line=dict(color=color),
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
    fig.show()

    return fig
