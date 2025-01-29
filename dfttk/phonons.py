"""
Harmonic approximation using phonons from VASP and YPHON. 
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


# TODO: Make an error check to make sure there are equal number of vdos and volph files and all the indexes match
def load_phonon_dos(path: str) -> pd.DataFrame:
    """Load phonon DOS files processed by YPHON

    Args:
        path (str): path to the directory containing the vdos and volph files

    Returns:
        pd.DataFrame: pandas dataframe containing the phonon DOS data
    """

    file_list = os.listdir(path)
    vdos_files = [file for file in file_list if file.startswith("vdos_")]
    volph_files = [file for file in file_list if file.startswith("volph_")]
    vdos_files.sort()
    volph_files.sort()

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
        path (str): path to the directory containing the vdos and volph files
        num_atoms (int, optional): number of atoms to scale the phonon DOS to. Defaults to 5.
        plot (bool, optional): Defaults to False.

    Returns:
        pd.DataFrame: pandas dataframe containing the scaled phonon DOS data
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
    """Plot the scaled phonon DOS for multiple volumes

    Args:
        path (str): path to the directory containing the vdos and volph files
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

# TODO: Input should be - number_of_atoms, volume, frequency, dos
def harmonic(
    path: str,
    scale_atoms: int,
    temp_range: np.ndarray,
    order: int = 2,
) -> pd.DataFrame:
    """Calculate the harmonic properties at different volumes and temperatures

    Args:
        path (str): path to the directory containing the vdos and volph files
        scale_atoms (int): number of atoms to scale the thermodynamic properties to
        temp_range (list): temperature range to calculate the thermodynamic properties
        order (int, optional): order of the polynomial fit. Defaults to 1.
        plot (bool, optional): Defaults to True.
        selected_temperatures_plot (np.ndarray, optional): selected temperatures to plot. Defaults to None.

    Returns:
        pd.DataFrame: pandas dataframes containing the harmonic and fitted harmonic properties
    """

    vdos_data_scaled = scale_phonon_dos(path)
    volumes_per_atom = np.sort(vdos_data_scaled["volume_per_atom"].unique())

    # Use these to evaluate the integrals
    frequency_diff_list = []
    frequency_mid_list = []
    dos_mid_list = []

    for volume_per_atom in volumes_per_atom:
        frequency_diff_append = (
            vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom][
                "frequency_hz"
            ][1:].reset_index(drop=True)
            - vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom][
                "frequency_hz"
            ][:-1].reset_index(drop=True)
        ).tolist()
        frequency_diff_list.extend([0] + frequency_diff_append)

        frequency_mid_append = (
            (
                vdos_data_scaled[
                    vdos_data_scaled["volume_per_atom"] == volume_per_atom
                ]["frequency_hz"][1:].reset_index(drop=True)
                + vdos_data_scaled[
                    vdos_data_scaled["volume_per_atom"] == volume_per_atom
                ]["frequency_hz"][:-1].reset_index(drop=True)
            )
            / 2
        ).tolist()
        frequency_mid_list.extend([0] + frequency_mid_append)

        dos_mid_append = (
            (
                vdos_data_scaled[
                    vdos_data_scaled["volume_per_atom"] == volume_per_atom
                ]["dos_1_per_hz"][1:].reset_index(drop=True)
                + vdos_data_scaled[
                    vdos_data_scaled["volume_per_atom"] == volume_per_atom
                ]["dos_1_per_hz"][:-1].reset_index(drop=True)
            )
            / 2
        ).tolist()
        dos_mid_list.extend([0] + dos_mid_append)

    vdos_data_scaled["frequency_diff"] = pd.Series(frequency_diff_list)
    vdos_data_scaled["frequency_mid"] = pd.Series(frequency_mid_list)
    vdos_data_scaled["dos_mid"] = pd.Series(dos_mid_list)

    f_vib_list = []
    e_vib_list = []
    s_vib_list = []
    cv_vib_list = []

    for volume_per_atom in volumes_per_atom:
        frequency_diff = vdos_data_scaled[
            vdos_data_scaled["volume_per_atom"] == volume_per_atom
        ]["frequency_diff"].values[1:]
        frequency_mid = vdos_data_scaled[
            vdos_data_scaled["volume_per_atom"] == volume_per_atom
        ]["frequency_mid"].values[1:]
        dos_mid = vdos_data_scaled[
            vdos_data_scaled["volume_per_atom"] == volume_per_atom
        ]["dos_mid"].values[1:]

        for temp in temp_range:
            if temp == 0:
                integrand = (
                    PLANCK_CONSTANT / 2 * frequency_mid * dos_mid * frequency_diff
                )
                f_vib = np.sum(integrand) / 5 * scale_atoms
                f_vib_list.append(f_vib)

                e_vib = f_vib
                e_vib_list.append(e_vib)

                s_vib = 0
                s_vib_list.append(s_vib)

                cv_vib = 0
                cv_vib_list.append(cv_vib)

            if temp > 0:
                ratio = (PLANCK_CONSTANT * frequency_mid) / (
                    2 * BOLTZMANN_CONSTANT * temp
                )
                differential = frequency_diff

                integrand = np.log(2 * np.sinh(ratio)) * dos_mid * differential
                f_vib = (
                    (BOLTZMANN_CONSTANT * temp * np.sum(integrand)) / 5 * scale_atoms
                )
                f_vib_list.append(f_vib)

                integrand = (
                    frequency_mid
                    * np.cosh(ratio)
                    / np.sinh(ratio)
                    * dos_mid
                    * differential
                )
                e_vib = (PLANCK_CONSTANT / 2 * np.sum(integrand)) / 5 * scale_atoms
                e_vib_list.append(e_vib)

                integrand = (
                    (
                        ratio * np.cosh(ratio) / np.sinh(ratio)
                        - np.log(2 * np.sinh(ratio))
                    )
                    * dos_mid
                    * differential
                )
                s_vib = (BOLTZMANN_CONSTANT * np.sum(integrand)) / 5 * scale_atoms
                s_vib_list.append(s_vib)

                integrand = (
                    ratio**2 * (1 / np.sinh(ratio)) ** 2 * dos_mid * differential
                )
                cv_vib = (BOLTZMANN_CONSTANT * np.sum(integrand)) / 5 * scale_atoms
                cv_vib_list.append(cv_vib)

    harmonic_properties = pd.DataFrame(
        {
            "volume_per_atom": np.repeat(volumes_per_atom, len(temp_range)),
            "number_of_atoms": np.repeat(
                scale_atoms, len(volumes_per_atom) * len(temp_range)
            ),
            "volume": np.repeat(volumes_per_atom * scale_atoms, len(temp_range)),
            "temperature": np.tile(temp_range, len(volumes_per_atom)),
            "f_vib": f_vib_list,
            "e_vib": e_vib_list,
            "s_vib": s_vib_list,
            "cv_vib": cv_vib_list,
        }
    )

    temp_df = harmonic_properties.groupby("temperature").agg(list)
    f_vib = temp_df["f_vib"].values
    s_vib = temp_df["s_vib"].values
    cv_vib = temp_df["cv_vib"].values
    
    return harmonic_properties, f_vib, s_vib, cv_vib


def plot_harmonic(
    scale_atoms,
    temperature,
    y_data,
    volumes,
    property_to_plot: str):
    """Plots Fvib, Svib and Cvib as a function of temperature for different fixed volumes

    Args:
        harmonic_properties (pd.DataFrame): harmonic properties dataframe from the harmonic function
    """

    if property_to_plot not in ["f_vib", "s_vib", "cv_vib"]:
        raise ValueError("property_to_plot must be one of 'f_vib', 's_vib', or 'cv_vib'")
    
    y_value = property_to_plot
    y_data = np.vstack(y_data)
    
    fig = go.Figure()
    for i, volume in enumerate(volumes):
        
        fig.add_trace(
            go.Scatter(
                x=temperature,
                y=y_data[:, i],
                mode="lines",
                name=f"{volume} Å³",
                showlegend=True,
            )
        )

    if y_value == "f_vib":
        y_title = f"F<sub>vib</sub> (eV/{scale_atoms} atoms)"
    elif y_value == "s_vib":
        y_title = f"S<sub>vib</sub> (eV/K/{scale_atoms} atoms)"
    elif y_value == "cv_vib":
        y_title = f"C<sub>vib</sub> (eV/K/{scale_atoms} atoms)"

    plot_format(fig, "Temperature (K)", y_title)
    fig.show()
    return fig


def fit_harmonic(harmonic_properties: pd.DataFrame, order: int) -> pd.DataFrame:
    """Fits the harmonic properties to a polynomial function

    Args:
        harmonic_properties (pd.DataFrame): _description_
        order (int): order of the polynomial fit

    Returns:
        pd.DataFrame: pandas dataframe containing the fitted harmonic properties
    """

    volume_fit_list = []
    f_vib_fit_list = []
    s_vib_fit_list = []
    cv_vib_fit_list = []
    free_energy_polynomial_list = []
    entropy_polynomial_list = []
    heat_capacity_polynomial_list = []

    harmonic_properties_fit = harmonic_properties.groupby("temperature").agg(list)
    temperatures = harmonic_properties_fit.index.tolist()
    for temperature in temperatures:
        volume = harmonic_properties_fit.loc[temperature]["volume"]
        f_vib = harmonic_properties_fit.loc[temperature]["f_vib"]
        s_vib = harmonic_properties_fit.loc[temperature]["s_vib"]
        cv_vib = harmonic_properties_fit.loc[temperature]["cv_vib"]

        free_energy_coefficients = np.polyfit(volume, f_vib, order)
        entropy_coefficients = np.polyfit(volume, s_vib, order)
        heat_capacity_coefficients = np.polyfit(volume, cv_vib, order)

        free_energy_polynomial = np.poly1d(free_energy_coefficients)
        entropy_polynomial = np.poly1d(entropy_coefficients)
        heat_capacity_polynomial = np.poly1d(heat_capacity_coefficients)
        free_energy_polynomial_list.append(free_energy_polynomial)
        entropy_polynomial_list.append(entropy_polynomial)
        heat_capacity_polynomial_list.append(heat_capacity_polynomial)

        volume_fit = np.linspace(min(volume) * 0.98, max(volume) * 1.02, 1000)
        f_vib_fit = free_energy_polynomial(volume_fit)
        s_vib_fit = entropy_polynomial(volume_fit)
        cv_vib_fit = heat_capacity_polynomial(volume_fit)

        volume_fit_list.append(volume_fit)
        f_vib_fit_list.append(f_vib_fit)
        s_vib_fit_list.append(s_vib_fit)
        cv_vib_fit_list.append(cv_vib_fit)

    harmonic_properties_fit["volume_fit"] = volume_fit_list
    harmonic_properties_fit["f_vib_fit"] = f_vib_fit_list
    harmonic_properties_fit["s_vib_fit"] = s_vib_fit_list
    harmonic_properties_fit["cv_vib_fit"] = cv_vib_fit_list
    harmonic_properties_fit["f_vib_poly"] = free_energy_polynomial_list
    harmonic_properties_fit["s_vib_poly"] = entropy_polynomial_list
    harmonic_properties_fit["cv_vib_poly"] = heat_capacity_polynomial_list

    harmonic_properties_fit["number_of_atoms"] = harmonic_properties_fit[
        "number_of_atoms"
    ].values[0][0]
    harmonic_properties_fit = harmonic_properties_fit.drop(columns=["volume_per_atom"])

    return harmonic_properties_fit, volume_fit, f_vib_fit_list, s_vib_fit_list, cv_vib_fit_list
    # Continue modifying the outputs here!

def plot_fit_harmonic(
    scale_atoms,
    temperature_list,
    volume,
    property_to_plot,
    y_value,
    volume_fit,
    y_value_fit,
    selected_temperatures_plot: np.ndarray = None
):
    """Plots the fitted harmonic properties

    Args:
        harmonic_properties_fit (pd.DataFrame): fitted harmonic properties dataframe from the fit_harmonic function
        selected_temperatures_plot (np.ndarray, optional): selected temperatures to plot. Defaults to None.
    """

    if property_to_plot not in ["f_vib", "s_vib", "cv_vib"]:
        raise ValueError("property_to_plot must be one of 'f_vib', 's_vib', or 'cv_vib'")
    
    if selected_temperatures_plot is None:
        indices = np.linspace(0, len(temperature_list) - 1, 5, dtype=int)
        selected_temperatures_plot = np.array([temperature_list[j] for j in indices])
    
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
    i = 0
    for temperature in selected_temperatures_plot:
        index = np.where(temperature_list == temperature)[0][0]
        x = volume
        y = y_value[index]
        x_fit = volume_fit
        y_fit = y_value_fit[index]

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
        i += 1

    if property_to_plot == "f_vib":
        y_title = f"F<sub>vib</sub> (eV/{scale_atoms} atoms)"
    elif property_to_plot == "s_vib":
        y_title = f"S<sub>vib</sub> (eV/K/{scale_atoms} atoms)"
    elif property_to_plot == "cv_vib":
        y_title = f"C<sub>vib</sub> (eV/K/{scale_atoms} atoms)"

    plot_format(fig, f"Volume (Å³/{scale_atoms} atoms)", y_title)
    fig.show()
    return fig