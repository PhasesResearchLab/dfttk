"""
"Module for computing thermodynamic properties within the harmonic approximation using phonon data from VASP and YPHON."
"""

# Standard library imports
import os

# Related third party imports
import numpy as np
import pandas as pd
import scipy.constants
import plotly.graph_objects as go

# Local application/library specific imports
from dfttk.plotly_format import plot_format

EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3 = 160.21766208 GPa
BOLTZMANN_CONSTANT = (
    scipy.constants.Boltzmann / scipy.constants.electron_volt
)  # The Boltzmann constant in eV/K
PLANCK_CONSTANT = (
    scipy.constants.Planck / scipy.constants.electron_volt
)  # The Planck's constant in eVs


def load_phonon_dos(path: str) -> pd.DataFrame:
    """Loads the phonon DOS data from the vdos and volph files in the specified directory.

    Args:
        path (str):  path to the directory containing the vdos and volph files.

    Raises:
        ValueError: If the number of vdos files does not match the number of volph files.
        ValueError: If the indexes of vdos files do not match the indexes of volph files.

    Returns:
        pd.DataFrame: pandas dataframe containing the phonon DOS data.
    """

    file_list = os.listdir(path)
    vdos_files = [file for file in file_list if file.startswith("vdos_")]
    volph_files = [file for file in file_list if file.startswith("volph_")]
    vdos_files.sort()
    volph_files.sort()

    if len(vdos_files) != len(volph_files):
        raise ValueError(
            "The number of vdos files does not match the number of volph files."
        )

    vdos_indexes = [file.split("_")[1].split(".")[0] for file in vdos_files]
    volph_indexes = [file.split("_")[1].split(".")[0] for file in volph_files]
    if vdos_indexes != volph_indexes:
        raise ValueError(
            "The indexes of vdos files do not match the indexes of volph files."
        )

    dataframes = []
    for i in range(len(vdos_files)):
        volph_content = float(
            open(os.path.join(path, volph_files[i])).readline().strip()
        )
        df = pd.read_csv(
            os.path.join(path, vdos_files[i]),
            sep="\\s+",
            header=None,
            names=["frequency_hz", "dos_1_per_hz"],
        )
        df.insert(0, "volume_per_atom", volph_content)
        dataframes.append(df)

    vdos_data = pd.concat(dataframes)

    return vdos_data


def scale_phonon_dos(path: str, num_atoms: int = 5, plot: bool = False) -> pd.DataFrame:
    """Scales the area under the phonon DOS to 3N, where N is the number of atoms.
    YPHON normalizes the area to 3N.

    Args:
        path (str): path to the directory containing the vdos and volph files.
        num_atoms (int, optional): number of atoms to scale the phonon DOS to. Defaults to 5.
        plot (bool, optional): Defaults to False.

    Returns:
        pd.DataFrame: pandas dataframe containing the scaled phonon DOS data.
    """

    vdos_data = load_phonon_dos(path)

    # Count the area % of positive and negative frequencies
    area_count = ()
    volumes_per_atom = np.sort(vdos_data["volume_per_atom"].unique())
    for volume_per_atom in volumes_per_atom:
        vdos_data_total = vdos_data[vdos_data["volume_per_atom"] == volume_per_atom]
        vdos_data_negative = vdos_data_total[vdos_data_total["frequency_hz"] < 0]
        vdos_data_positive = vdos_data_total[vdos_data_total["frequency_hz"] >= 0]
        area_negative = np.trapz(
            vdos_data_negative["dos_1_per_hz"], vdos_data_negative["frequency_hz"]
        )
        area_positive = np.trapz(
            vdos_data_positive["dos_1_per_hz"], vdos_data_positive["frequency_hz"]
        )
        area_total = np.trapz(
            vdos_data_total["dos_1_per_hz"], vdos_data_total["frequency_hz"]
        )
        area_count += (
            (
                volume_per_atom,
                area_positive / area_total * 100,
                area_negative / area_total * 100,
            ),
        )

    # Count the original number of atoms before scaling
    original_atoms = []
    for volume_per_atom in volumes_per_atom:
        frequency = vdos_data[vdos_data["volume_per_atom"] == volume_per_atom][
            "frequency_hz"
        ]
        dos = vdos_data[vdos_data["volume_per_atom"] == volume_per_atom]["dos_1_per_hz"]
        area = np.trapz(dos, frequency)
        original_atoms.append(round(area / 3))

    # Remove all of the negative frequencies
    vdos_data_scaled = vdos_data[vdos_data["frequency_hz"] > 0].reset_index(drop=True)

    # Add a row of zero frequency and DOS to the beginning of each volume_per_atom
    final_df = pd.DataFrame(columns=vdos_data_scaled.columns)
    for volume_per_atom in volumes_per_atom:
        filtered_df = vdos_data_scaled[
            vdos_data_scaled["volume_per_atom"] == volume_per_atom
        ]
        new_row = pd.DataFrame(
            [[volume_per_atom, 0, 0]], columns=vdos_data_scaled.columns
        )
        filtered_df = pd.concat([new_row, filtered_df])
        if final_df.empty:
            final_df = filtered_df
        else:
            final_df = pd.concat([final_df, filtered_df])
    vdos_data_scaled = final_df.reset_index(drop=True)

    # Add number_of_atoms column
    vdos_data_scaled.insert(1, "number_of_atoms", num_atoms)

    # Scale the phonon DOS to 3N
    num_atoms_3N = num_atoms * 3
    for volume_per_atom in volumes_per_atom:
        frequency = vdos_data_scaled[
            vdos_data_scaled["volume_per_atom"] == volume_per_atom
        ]["frequency_hz"]
        dos = vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom][
            "dos_1_per_hz"
        ]
        area = np.trapz(dos, frequency)
        vdos_data_scaled.loc[
            vdos_data_scaled["volume_per_atom"] == volume_per_atom, "dos_1_per_hz"
        ] = (dos * num_atoms_3N / area)

    i = 0
    if plot == True:
        for volume_per_atom in volumes_per_atom:
            fig = go.Figure()

            frequency = vdos_data[vdos_data["volume_per_atom"] == volume_per_atom][
                "frequency_hz"
            ]
            dos = vdos_data[vdos_data["volume_per_atom"] == volume_per_atom][
                "dos_1_per_hz"
            ]
            scaled_frequency = vdos_data_scaled[
                vdos_data_scaled["volume_per_atom"] == volume_per_atom
            ]["frequency_hz"]
            scaled_dos = vdos_data_scaled[
                vdos_data_scaled["volume_per_atom"] == volume_per_atom
            ]["dos_1_per_hz"]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=frequency / 1e12,
                    y=dos * 1e12,
                    mode="lines",
                    name=f"Original - {original_atoms[i]} atoms",
                    showlegend=True,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=scaled_frequency / 1e12,
                    y=scaled_dos * 1e12,
                    mode="lines",
                    name=f"Scaled - {num_atoms} atoms",
                    showlegend=True,
                )
            )
            plot_format(fig, "Frequency (THz)", "DOS (1/THz)")
            fig.update_layout(
                title=dict(
                    text=f"Original volume: {volume_per_atom * original_atoms[i]} Å³/{original_atoms[i]} atoms \
                        <br> Scaled volume: {volume_per_atom * num_atoms} Å³/{num_atoms} atoms \
                        <br> (Area: {area_count[i][1]:.1f}% positive, {area_count[i][2]:.1f}% negative)",
                    font=dict(size=20, color="rgb(0,0,0)"),
                ),
                margin=dict(t=130),
            )
            i += 1
            fig.show()

    return vdos_data_scaled


def plot_phonon_dos(path: str, scale_atoms: int = 5):
    """Plot the scaled phonon DOS for multiple volumes.

    Args:
        path (str): path to the directory containing the vdos and volph files.
        scale_atoms (int, optional): Number of atoms to scale the phonon DOS to. Defaults to 5.
    """

    vdos_data_scaled = scale_phonon_dos(path, num_atoms=scale_atoms)
    volumes_per_atom = np.sort(vdos_data_scaled["volume_per_atom"].unique())

    fig = go.Figure()

    for volume_per_atom in volumes_per_atom:
        frequency = vdos_data_scaled[
            vdos_data_scaled["volume_per_atom"] == volume_per_atom
        ]["frequency_hz"]
        dos = vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom][
            "dos_1_per_hz"
        ]

        fig.add_trace(
            go.Scatter(
                x=frequency / 1e12,
                y=dos * 1e12,
                mode="lines",
                name=f"{volume_per_atom * scale_atoms} Å³",
                showlegend=True,
            )
        )
    plot_format(fig, "Frequency (THz)", "DOS (1/THz)")
    fig.update_layout(
        title=dict(
            text=f"Number of atoms = {scale_atoms}",
            font=dict(size=24, color="rgb(0,0,0)"),
        )
    )

    fig.show()


def harmonic(
    scale_atoms: int,
    volumes_per_atom: np.ndarray,
    temperatures: np.ndarray,
    frequency_array: np.ndarray,
    dos_array: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the thermodynamic properties from the phonon DOS data using the harmonic approximation.

    Args:
        scale_atoms (int): number of atoms to scale the thermodynamic properties to.
        volumes_per_atom (np.ndarray): volumes per atom corresponding to the phonon DOS data.
        temperatures (np.ndarray): temperatures in K.
        frequency_array (np.ndarray): a 2D array of frequencies in Hz where each column corresponds to a different volume per atom.
        dos_array (np.ndarray): a 2D array of DOS values in 1/Hz where each column corresponds to a different volume per atom.

    Returns:
        tuple: tuple containing the following:
            - volumes (np.ndarray): volumes corresponding to the thermodynamic properties.
            - helmholtz_energy (np.ndarray): a 2D array of vibrational helmholtz energy in eV/atom. Rows are temperatures and columns are volumes.
            - internal_energy (np.ndarray): a 2D array of vibrational internal energy in eV/atom. Rows are temperatures and columns are volumes.
            - entropy (np.ndarray): a 2D array of vibrational entropy in eV/K/atom. Rows are temperatures and columns are volumes.
            - heat_capacity (np.ndarray): a 2D array of vibrational heat capacity in eV/K/atom. Rows are temperatures and columns are volumes.
    """

    # Use these to evaluate the integrals
    frequency_diff_list = []
    frequency_mid_list = []
    dos_mid_list = []

    for i in range(len(volumes_per_atom)):

        frequency = frequency_array[:, i]
        frequency_current = frequency[1:]
        frequency_previous = frequency[:-1]

        frequency_diff_append = frequency_current - frequency_previous
        frequency_diff_list.append(np.insert(frequency_diff_append, 0, 0))

        frequency_mid = (frequency_current + frequency_previous) / 2
        frequency_mid_list.append(np.insert(frequency_mid, 0, 0))

        dos = dos_array[:, i]
        dos_current = dos[1:]
        dos_previous = dos[:-1]

        dos_mid_append = (dos_current + dos_previous) / 2
        dos_mid_list.append(np.insert(dos_mid_append, 0, 0))

    dos_mid_array = np.column_stack(dos_mid_list)
    frequency_diff_array = np.column_stack(frequency_diff_list)
    frequency_mid_array = np.column_stack(frequency_mid_list)

    helmholtz_energy_list = []
    internal_energy_list = []
    entropy_list = []
    heat_capacity_list = []

    for i in range(len(volumes_per_atom)):

        frequency_diff = frequency_diff_array[1:, i]
        frequency_mid = frequency_mid_array[1:, i]
        dos_mid = dos_mid_array[1:, i]

        for temperature in temperatures:
            if temperature == 0:
                integrand = (
                    PLANCK_CONSTANT / 2 * frequency_mid * dos_mid * frequency_diff
                )
                helmholtz_energy = np.sum(integrand) / 5 * scale_atoms
                helmholtz_energy_list.append(helmholtz_energy)

                internal_energy = helmholtz_energy
                internal_energy_list.append(internal_energy)

                entropy = 0
                entropy_list.append(entropy)

                heat_capacity = 0
                heat_capacity_list.append(heat_capacity)

            if temperature > 0:
                ratio = (PLANCK_CONSTANT * frequency_mid) / (
                    2 * BOLTZMANN_CONSTANT * temperature
                )
                differential = frequency_diff

                integrand = np.log(2 * np.sinh(ratio)) * dos_mid * differential
                helmholtz_energy = (
                    (BOLTZMANN_CONSTANT * temperature * np.sum(integrand))
                    / 5
                    * scale_atoms
                )
                helmholtz_energy_list.append(helmholtz_energy)

                integrand = (
                    frequency_mid
                    * np.cosh(ratio)
                    / np.sinh(ratio)
                    * dos_mid
                    * differential
                )
                internal_energy = (
                    (PLANCK_CONSTANT / 2 * np.sum(integrand)) / 5 * scale_atoms
                )
                internal_energy_list.append(internal_energy)

                integrand = (
                    (
                        ratio * np.cosh(ratio) / np.sinh(ratio)
                        - np.log(2 * np.sinh(ratio))
                    )
                    * dos_mid
                    * differential
                )
                entropy = (BOLTZMANN_CONSTANT * np.sum(integrand)) / 5 * scale_atoms
                entropy_list.append(entropy)

                integrand = (
                    ratio**2 * (1 / np.sinh(ratio)) ** 2 * dos_mid * differential
                )
                heat_capacity = (
                    (BOLTZMANN_CONSTANT * np.sum(integrand)) / 5 * scale_atoms
                )
                heat_capacity_list.append(heat_capacity)

    num_temps = len(temperatures)
    num_volumes = len(volumes_per_atom)

    helmholtz_energy = np.reshape(helmholtz_energy_list, (num_volumes, num_temps))
    internal_energy = np.reshape(internal_energy_list, (num_volumes, num_temps))
    entropy = np.reshape(entropy_list, (num_volumes, num_temps))
    heat_capacity = np.reshape(heat_capacity_list, (num_volumes, num_temps))

    helmholtz_energy = helmholtz_energy.T
    internal_energy = internal_energy.T
    entropy = entropy.T
    heat_capacity = heat_capacity.T

    return (
        volumes_per_atom * scale_atoms,
        helmholtz_energy,
        internal_energy,
        entropy,
        heat_capacity,
    )


def fit_harmonic(
    volumes: np.ndarray,
    temperatures: np.ndarray,
    helmholtz_energy: np.ndarray,
    entropy: np.ndarray,
    heat_capacity: np.ndarray,
    order: int,
) -> tuple[np.ndarray, list, list, list, list, list, list]:
    """For each fixed temperature, fits the thermodynamic properties vs. volume to a polynomial of a given order.

    Args:
        volumes (np.ndarray): volumes corresponding to the thermodynamic properties.
        temperatures (np.ndarray): temperatures in K.
        helmholtz_energy (np.ndarray): a 2D array of helmholtz energy. Rows are temperatures and columns are volumes.
        entropy (np.ndarray): a 2D array of entropy. Rows are temperatures and columns are volumes.
        heat_capacity (np.ndarray): a 2D array of heat capacity. Rows are temperatures and columns are volumes.
        order (int): order of the polynomial to fit.

    Returns:
        tuple[np.ndarray, list, list, list, list, list, list]:
        tuple containing the following:
            - volumes_fit (np.ndarray): volumes corresponding to the fitted thermodynamic properties.
            - helmholtz_energy_fit_list (list): a list of 1D arrays of fitted helmholtz energy. Each array corresponds to a temperature.
            - entropy_fit_list (list): a list of 1D arrays of fitted entropy. Each array corresponds to a temperature.
            - heat_capacity_fit_list (list): a list of 1D arrays of fitted heat capacity. Each array corresponds to a temperature.
            - helmholtz_energy_polynomial_list (list): a list of 1D polynomials of fitted helmholtz energy. Each polynomial corresponds to a temperature.
            - entropy_polynomial_list (list): a list of 1D polynomials of fitted entropy. Each polynomial corresponds to a temperature.
            - heat_capacity_polynomial_list (list): a list of 1D polynomials of fitted heat capacity. Each polynomial corresponds to a temperature.
    """

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

        volumes_fit = np.linspace(min(volumes) * 0.98, max(volumes) * 1.02, 1000)
        helmholtz_energy_fit = helmholtz_energy_polynomial(volumes_fit)
        entropy_fit = entropy_polynomial(volumes_fit)
        heat_capacity_fit = heat_capacity_polynomial(volumes_fit)

        helmholtz_energy_fit_list.append(helmholtz_energy_fit)
        entropy_fit_list.append(entropy_fit)
        heat_capacity_fit_list.append(heat_capacity_fit)

    return (
        volumes_fit,
        helmholtz_energy_fit_list,
        entropy_fit_list,
        heat_capacity_fit_list,
        helmholtz_energy_polynomial_list,
        entropy_polynomial_list,
        heat_capacity_polynomial_list,
    )


def plot_harmonic(
    scale_atoms: int,
    volumes: np.ndarray,
    temperatures: np.ndarray,
    property: np.ndarray,
    property_name: str,
) -> go.Figure:
    """Plots the thermodynamic properties from the harmonic approximation vs. temperature.

    Args:
        scale_atoms (int): number of atoms the thermodynamic properties are scaled to.
        volumes (np.ndarray): volumes corresponding to the thermodynamic properties.
        temperatures (np.ndarray): temperatures in K.
        property (np.ndarray): a 2D array of helmholtz energy, entropy, or heat capacity. Rows are temperatures and columns are volumes.
        property_name (str): helmholtz energy, entropy, or heat capacity.

    Raises:
        ValueError: If property_name is not one of 'helmholtz_energy', 'entropy', or 'heat_capacity'.

    Returns:
        go.Figure: plotly figure of the thermodynamic property vs. temperature.
    """

    valid_properties = {
        "helmholtz_energy": f"F<sub>vib</sub> (eV/{scale_atoms} atoms)",
        "entropy": f"S<sub>vib</sub> (eV/K/{scale_atoms} atoms)",
        "heat_capacity": f"C<sub>vib</sub> (eV/K/{scale_atoms} atoms)",
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
    

def plot_fit_harmonic(
    scale_atoms: int,
    volumes: np.ndarray,
    temperatures: np.ndarray,
    property_name: str,
    property: np.ndarray,
    volume_fit: np.ndarray,
    property_fit: list,
    selected_temperatures_plot: np.ndarray = None,
) -> go.Figure:

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

    for temperature in selected_temperatures_plot:
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
        y_title = f"F<sub>vib</sub> (eV/{scale_atoms} atoms)"
    elif property_name == "entropy":
        y_title = f"S<sub>vib</sub> (eV/K/{scale_atoms} atoms)"
    elif property_name == "heat_capacity":
        y_title = f"C<sub>vib</sub> (eV/K/{scale_atoms} atoms)"

    plot_format(fig, f"Volume (Å³/{scale_atoms} atoms)", y_title)
    fig.show()
    return fig
