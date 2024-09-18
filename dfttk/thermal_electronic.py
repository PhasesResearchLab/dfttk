"""
Calculates the thermal electronic contribution to the Helmholtz energy.
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

# Local application/library specific imports
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin

# DFTTK imports
from dfttk.data_extraction import extract_volume
from plotly_format import plot_format

BOLTZMANN_CONSTANT = (
    scipy.constants.Boltzmann / scipy.constants.electron_volt
)  # The Boltzmann constant in eV/K


def read_total_electron_dos(path, plot=False):

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
        volume = extract_volume(os.path.join(path, elec_folder, "CONTCAR.elec_dos"))

        vasprun_path = os.path.join(path, elec_folder, "vasprun.xml.elec_dos")
        vasprun = Vasprun(vasprun_path)

        volume = extract_volume(os.path.join(path, elec_folder, "CONTCAR.elec_dos"))
        num_atoms = vasprun.final_structure.num_sites

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

    return electron_dos_data


def fermi_dirac_distribution(energy, chemical_potential, temperature, plot=False):

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

    return fermi_dist


def plot_fermi_dirac_distribution(energy, chemical_potential, temperature):

    fermi_dist = fermi_dirac_distribution(energy, chemical_potential, temperature)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energy, y=fermi_dist, mode="lines"))
    plot_format(fig, xtitle="Energy (eV)", ytitle="Fermi-Dirac Distribution")
    fig.show()


def calculate_num_electrons(energy, dos, chemical_potential, temperature):

    if temperature < 0:
        raise ValueError("Temperature cannot be less than 0 K")

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    fermi_dist = fermi_dirac_distribution(energy, chemical_potential, temperature)
    integrand = dos * fermi_dist
    num_electrons = np.trapz(integrand, energy)

    return num_electrons


def calculate_chemical_potential(energy, dos, temperature):

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    num_electrons_0K = calculate_num_electrons(energy, dos, 0, 0)

    # Find the chemical potential at temperature such that the number of electrons matches that at 0 K
    chemical_potential = -0.2
    num_electrons = calculate_num_electrons(
        energy, dos, chemical_potential, temperature
    )
    while abs(num_electrons - num_electrons_0K) > 1e-2 and chemical_potential < 0.2:
        chemical_potential += 0.01
        num_electrons = calculate_num_electrons(
            energy, dos, chemical_potential, temperature
        )

    return chemical_potential


def calculate_internal_energy(energy, dos, temperature_range):

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    E_el_list = []
    for temperature in temperature_range:
        chemical_potential = calculate_chemical_potential(energy, dos, temperature)
        fermi_dist = fermi_dirac_distribution(energy, chemical_potential, temperature)

        integrand_1 = dos * fermi_dist * energy
        integral_1 = np.trapz(integrand_1, energy)

        # Only integrate over energy levels less than the chemical potential
        mask = energy < chemical_potential
        filtered_energy = energy[mask]
        filtered_dos = dos[mask]

        integrand_2 = filtered_dos * filtered_energy
        integral_2 = np.trapz(integrand_2, filtered_energy)

        E_el = integral_1 - integral_2
        E_el_list.append(E_el)

    return E_el_list


# TODO: Evaluate the quality of the integration
def calculate_entropy(energy, dos, temperature_range):

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    S_el_list = []
    for temperature in temperature_range:
        chemical_potential = calculate_chemical_potential(energy, dos, temperature)
        fermi_dist = fermi_dirac_distribution(energy, chemical_potential, temperature)

        # Suppress warnings for divide-by-zero and invalid value encountered in log and multiply
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = dos * (
                fermi_dist * np.log(fermi_dist)
                + (1 - fermi_dist) * np.log(1 - fermi_dist)
            )

        mask = ~np.isnan(integrand)

        filtered_integrand = integrand[mask]
        filtered_energy = energy[mask]

        S_el = -BOLTZMANN_CONSTANT * np.trapz(filtered_integrand, filtered_energy)
        S_el_list.append(S_el)

    return S_el_list


# TODO: Evaluate the quality of the integration
# TODO: Equation is from the MATLAB code. Double check this.
def calculate_heat_capacity(energy, dos, temperature_range):

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    Cv_el_list = []
    for temperature in temperature_range:
        chemical_potential = calculate_chemical_potential(energy, dos, temperature)
        fermi_dist = fermi_dirac_distribution(energy, chemical_potential, temperature)

        # Suppress warnings for divide-by-zero and invalid value encountered in log and multiply
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            integrand = (
                dos
                * (1 / fermi_dist - 1)
                * (fermi_dist * (energy - chemical_potential) / temperature) ** 2
                / BOLTZMANN_CONSTANT
            )

        mask = ~np.isnan(integrand) & ~np.isinf(integrand)

        filtered_energy = energy[mask]
        filtered_integrand = integrand[mask]

        Cv_el = np.trapz(filtered_integrand, filtered_energy)
        Cv_el_list.append(Cv_el)

    return Cv_el_list


def calculate_free_energy(energy, dos, temperature_range):

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    E_el_list = calculate_internal_energy(energy, dos, temperature_range)
    S_el_list = calculate_entropy(energy, dos, temperature_range)
    F_el = E_el_list - temperature_range * S_el_list

    return F_el.tolist()


def thermal_electronic(electron_dos_data, temperature_range, order=1, plot=True):

    E_el_list = []
    S_el_list = []
    Cv_el_list = []
    F_el_list = []

    volumes = electron_dos_data["volume"].unique()
    for volume in volumes:
        row = electron_dos_data.loc[electron_dos_data["volume"] == volume]
        energy = row["energy_minus_fermi_energy"]
        dos = row["total_dos"]

        E_el = calculate_internal_energy(energy, dos, temperature_range)
        S_el = calculate_entropy(energy, dos, temperature_range)
        Cv_el = calculate_heat_capacity(energy, dos, temperature_range)
        F_el = calculate_free_energy(energy, dos, temperature_range)

        E_el_list.append(E_el)
        S_el_list.append(S_el)
        Cv_el_list.append(Cv_el)
        F_el_list.append(F_el)

    # Flatten the lists of lists using list comprehension
    E_el_list = [item for sublist in E_el_list for item in sublist]
    S_el_list = [item for sublist in S_el_list for item in sublist]
    Cv_el_list = [item for sublist in Cv_el_list for item in sublist]
    F_el_list = [item for sublist in F_el_list for item in sublist]

    number_of_atoms = electron_dos_data["number_of_atoms"].unique()[0]
    thermal_electronic_properties = pd.DataFrame(
        {
            "number_of_atoms": np.repeat(
                number_of_atoms, len(temperature_range) * len(volumes)
            ),
            "volume": np.repeat(volumes, len(temperature_range)),
            "temperature": np.tile(temperature_range, len(volumes)),
            "E_el": E_el_list,
            "S_el": S_el_list,
            "Cv_el": Cv_el_list,
            "F_el": F_el_list,
        }
    )

    thermal_electronic_properties_fit = fit_thermal_electronic(
        thermal_electronic_properties, order
    )

    if plot == True:
        plot_thermal_electronic(thermal_electronic_properties)
        plot_thermal_electronic_properties_fit(thermal_electronic_properties_fit)

    return thermal_electronic_properties, thermal_electronic_properties_fit


def plot_thermal_electronic(thermal_electronic_properties):

    volumes = thermal_electronic_properties["volume"].unique()
    number_of_atoms = thermal_electronic_properties["number_of_atoms"].unique()[0]
    y_values = ["F_el", "S_el", "Cv_el"]
    for y_value in y_values:
        fig = go.Figure()
        for volume in volumes:
            temperature = thermal_electronic_properties[
                thermal_electronic_properties["volume"] == volume
            ]["temperature"]
            y_data = thermal_electronic_properties[
                thermal_electronic_properties["volume"] == volume
            ][y_value]

            fig.add_trace(
                go.Scatter(
                    x=temperature,
                    y=y_data,
                    mode="lines",
                    name=f"{volume} Å³",
                    showlegend=True,
                )
            )

        if y_value == "F_el":
            y_title = f"F<sub>el</sub> (eV/{number_of_atoms} atoms)"
        elif y_value == "S_el":
            y_title = f"S<sub>el</sub> (eV/K/{number_of_atoms} atoms)"
        elif y_value == "Cv_el":
            y_title = f"C<sub>v, el</sub> (eV/K/{number_of_atoms} atoms)"

        plot_format(fig, "Temperature (K)", y_title)
        fig.show()


def fit_thermal_electronic(thermal_electronic_properties, order):

    volume_fit_list = []
    F_el_fit_list = []
    S_el_fit_list = []
    Cv_el_fit_list = []
    F_el_polynomial_list = []
    S_el_polynomial_list = []
    Cv_el_polynomial_list = []

    thermal_electronic_properties_fit = thermal_electronic_properties.groupby(
        "temperature"
    ).agg(list)
    temperatures = thermal_electronic_properties_fit.index.tolist()

    for temperature in temperatures:
        volume = thermal_electronic_properties_fit.loc[temperature]["volume"]
        F_el = thermal_electronic_properties_fit.loc[temperature]["F_el"]
        S_el = thermal_electronic_properties_fit.loc[temperature]["S_el"]
        Cv_el = thermal_electronic_properties_fit.loc[temperature]["Cv_el"]

        F_el_coefficients = np.polyfit(volume, F_el, order)
        S_el_coefficients = np.polyfit(volume, S_el, order)
        Cv_el_coefficients = np.polyfit(volume, Cv_el, order)

        F_el_polynomial = np.poly1d(F_el_coefficients)
        S_el_polynomial = np.poly1d(S_el_coefficients)
        Cv_el_polynomial = np.poly1d(Cv_el_coefficients)

        F_el_polynomial_list.append(F_el_polynomial)
        S_el_polynomial_list.append(S_el_polynomial)
        Cv_el_polynomial_list.append(Cv_el_polynomial)

        volume_fit = np.linspace(min(volume) * 0.98, max(volume) * 1.02, 1000)
        F_el_fit = F_el_polynomial(volume_fit)
        S_el_fit = S_el_polynomial(volume_fit)
        Cv_el_fit = Cv_el_polynomial(volume_fit)

        volume_fit_list.append(volume_fit)
        F_el_fit_list.append(F_el_fit)
        S_el_fit_list.append(S_el_fit)
        Cv_el_fit_list.append(Cv_el_fit)

    thermal_electronic_properties_fit["number_of_atoms"] = (
        thermal_electronic_properties_fit["number_of_atoms"].values[0][0]
    )
    thermal_electronic_properties_fit["volume_fit"] = volume_fit_list
    thermal_electronic_properties_fit["F_el_fit"] = F_el_fit_list
    thermal_electronic_properties_fit["S_el_fit"] = S_el_fit_list
    thermal_electronic_properties_fit["Cv_el_fit"] = Cv_el_fit_list
    thermal_electronic_properties_fit["F_el_polynomial"] = F_el_polynomial_list
    thermal_electronic_properties_fit["S_el_polynomial"] = S_el_polynomial_list
    thermal_electronic_properties_fit["Cv_el_polynomial"] = Cv_el_polynomial_list

    return thermal_electronic_properties_fit


def plot_thermal_electronic_properties_fit(thermal_electronic_properties_fit):

    number_of_atoms = thermal_electronic_properties_fit["number_of_atoms"].iloc[0]
    temperature_list = thermal_electronic_properties_fit.index.values
    spaces = len(temperature_list) - 1
    step = int(spaces / 4)

    selected_temperatures = temperature_list[::step]
    if selected_temperatures[-1] != temperature_list[-1]:
        selected_temperatures = np.append(selected_temperatures, temperature_list[-1])

    y_values = [
        ("F_el", "F_el_fit"),
        ("S_el", "S_el_fit"),
        ("Cv_el", "Cv_el_fit"),
    ]
    for y_value, y_value_fit in y_values:
        fig = go.Figure()
        i = 0
        for temperature in selected_temperatures:
            x = thermal_electronic_properties_fit.loc[temperature]["volume"]
            y = thermal_electronic_properties_fit.loc[temperature][y_value]
            x_fit = thermal_electronic_properties_fit.loc[temperature]["volume_fit"]
            y_fit = thermal_electronic_properties_fit.loc[temperature][y_value_fit]

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
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    line=dict(color=colors[i]),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    line=dict(color=colors[i]),
                    name=f"{temperature} K",
                    showlegend=True,
                )
            )
            i += 1

        if y_value == "F_el":
            y_title = f"F<sub>el</sub> (eV/{number_of_atoms} atoms)"
        elif y_value == "S_el":
            y_title = f"S<sub>el</sub> (eV/K/{number_of_atoms} atoms)"
        elif y_value == "Cv_el":
            y_title = f"C<sub>v, el</sub> (eV/K/{number_of_atoms} atoms)"

        plot_format(fig, f"Volume (Å³/{number_of_atoms} atoms)", y_title)
        fig.show()
