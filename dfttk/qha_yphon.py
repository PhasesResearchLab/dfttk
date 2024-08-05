# Standard library imports
import os

# Related third party imports
import numpy as np
import pandas as pd
import scipy.constants
import plotly.graph_objects as go

# Local application/library specific imports
import dfttk.eos_fit as eos_fit

EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3  = 160.21766208 GPa


def plot_format(
    fig: go.Figure, xtitle: str, ytitle: str, width: int = 840, height: int = 600
):
    """Plot format using plotly

    Args:
        fig (go.Figure): plotly figure
        xtitle (str): title of x-axis
        ytitle (str): title of y-axis
        width (int, optional): plot width. Defaults to 840.
        height (int, optional): plot height. Defaults to 600.
    """

    fig.update_layout(
        font=dict(
            family="Devaju Sans",
        )
    )
    fig.update_xaxes(
        title=dict(
            text=xtitle,
            font=dict(size=22, color="rgb(0,0,0)"),
        )
    )
    fig.update_yaxes(title=dict(text=ytitle, font=dict(size=22, color="rgb(0,0,0)")))
    axis_params = dict(
        showline=True,
        linecolor="black",
        linewidth=1,
        ticks="outside",
        mirror="allticks",
        tickwidth=1,
        tickcolor="black",
        showgrid=False,
        tickfont=dict(color="rgb(0,0,0)", size=20),
    )
    fig.update_layout(
        plot_bgcolor="white",
        width=width,
        height=height,
        legend=dict(font=dict(size=20, color="black")),
        xaxis=axis_params,
        yaxis=axis_params,
    )


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


# TODO: Add the zero-point energy
def harmonic(
    path: str, scale_atoms: int, temp_range: list, order: int = 1, plot: bool = True
) -> pd.DataFrame:
    """Calculate the harmonic properties at different volumes and temperatures

    Args:
        path (str): path to the directory containing the vdos and volph files
        scale_atoms (int): number of atoms to scale the thermodynamic properties to
        temp_range (list): temperature range to calculate the thermodynamic properties
        order (int, optional): order of the polynomial fit. Defaults to 1.
        plot (bool, optional): Defaults to True.

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

    # Calculate the integrals for the free energy, internal energy, entropy, and heat capacity
    k_B = (
        scipy.constants.Boltzmann / scipy.constants.electron_volt
    )  # The Boltzmann constant in eV/K
    h = (
        scipy.constants.Planck / scipy.constants.electron_volt
    )  # The Planck's constant in eVs

    f_vib_list = []
    u_vib_list = []
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
            ratio = (h * frequency_mid) / (2 * k_B * temp)
            differential = frequency_diff

            integrand = np.log(2 * np.sinh(ratio)) * dos_mid * differential
            f_vib = (k_B * temp * np.sum(integrand)) / 5 * scale_atoms
            f_vib_list.append(f_vib)

            integrand = (
                frequency_mid * np.cosh(ratio) / np.sinh(ratio) * dos_mid * differential
            )
            u_vib = (h / 2 * np.sum(integrand)) / 5 * scale_atoms
            u_vib_list.append(u_vib)

            integrand = (
                (ratio * np.cosh(ratio) / np.sinh(ratio) - np.log(2 * np.sinh(ratio)))
                * dos_mid
                * differential
            )
            s_vib = (k_B * np.sum(integrand)) / 5 * scale_atoms
            s_vib_list.append(s_vib)

            integrand = ratio**2 * (1 / np.sinh(ratio)) ** 2 * dos_mid * differential
            cv_vib = (k_B * np.sum(integrand)) / 5 * scale_atoms
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
            "u_vib": u_vib_list,
            "s_vib": s_vib_list,
            "cv_vib": cv_vib_list,
        }
    )

    harmonic_properties_fit = fit_harmonic(harmonic_properties, order=order)

    if plot == True:
        plot_harmonic(harmonic_properties)
        plot_fit_harmonic(harmonic_properties_fit)

    return harmonic_properties, harmonic_properties_fit


def plot_harmonic(harmonic_properties: pd.DataFrame):
    """Plots Fvib, Svib and Cvib as a function of temperature for different fixed volumes

    Args:
        harmonic_properties (pd.DataFrame): harmonic properties dataframe from the harmonic function
    """

    volumes_per_atom = np.sort(harmonic_properties["volume_per_atom"].unique())
    scale_atoms = harmonic_properties["number_of_atoms"].unique()[0]
    y_values = ["f_vib", "s_vib", "cv_vib"]
    for y_value in y_values:
        fig = go.Figure()
        for volume_per_atom in volumes_per_atom:
            temperature = harmonic_properties[
                harmonic_properties["volume_per_atom"] == volume_per_atom
            ]["temperature"]
            y_data = harmonic_properties[
                harmonic_properties["volume_per_atom"] == volume_per_atom
            ][y_value]

            fig.add_trace(
                go.Scatter(
                    x=temperature,
                    y=y_data,
                    mode="lines",
                    name=f"{volume_per_atom * scale_atoms} Å³",
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
    harmonic_properties_fit["f_vib_polynomial"] = free_energy_polynomial_list
    harmonic_properties_fit["entropy_polynomial"] = entropy_polynomial_list
    harmonic_properties_fit["heat_capacity_polynomial"] = heat_capacity_polynomial_list

    harmonic_properties_fit["number_of_atoms"] = harmonic_properties_fit[
        "number_of_atoms"
    ].values[0][0]
    harmonic_properties_fit = harmonic_properties_fit.drop(columns=["volume_per_atom"])

    return harmonic_properties_fit


def plot_fit_harmonic(harmonic_properties_fit: pd.DataFrame):
    """Plots the fitted harmonic properties

    Args:
        harmonic_properties_fit (pd.DataFrame): fitted harmonic properties dataframe from the fit_harmonic function
    """

    scale_atoms = harmonic_properties_fit["number_of_atoms"].iloc[0]
    temperature_list = harmonic_properties_fit.index.values
    spaces = len(temperature_list) - 1
    step = int(spaces / 4)

    selected_temperatures = temperature_list[::step]
    if selected_temperatures[-1] != temperature_list[-1]:
        selected_temperatures = np.append(selected_temperatures, temperature_list[-1])

    y_values = [
        ("f_vib", "f_vib_fit"),
        ("s_vib", "s_vib_fit"),
        ("cv_vib", "cv_vib_fit"),
    ]
    for y_value, y_value_fit in y_values:
        fig = go.Figure()
        i = 0
        for temperature in selected_temperatures:
            x = harmonic_properties_fit.loc[temperature]["volume"]
            y = harmonic_properties_fit.loc[temperature][y_value]
            x_fit = harmonic_properties_fit.loc[temperature]["volume_fit"]
            y_fit = harmonic_properties_fit.loc[temperature][y_value_fit]

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

        if y_value == "f_vib":
            y_title = f"F<sub>vib</sub> (eV/{scale_atoms} atoms)"
        elif y_value == "s_vib":
            y_title = f"S<sub>vib</sub> (eV/K/{scale_atoms} atoms)"
        elif y_value == "cv_vib":
            y_title = f"C<sub>vib</sub> (eV/K/{scale_atoms} atoms)"

        plot_format(fig, f"Volume (Å³/{scale_atoms} atoms)", y_title)
        fig.show()


# TODO: At the moment, only supports BM4. Add other EOS. Also, add a way to choose the EOS.
def quasi_harmonic(
    eos_parameters_df: pd.DataFrame,
    harmonic_properties_fit: pd.DataFrame,
    P: int = 0,
    plot: bool = True,
    plot_type: str = "default",
) -> pd.DataFrame:
    """Calculates the quasi-harmonic properties

    Args:
        eos_parameters_df (pd.DataFrame): pandas dataframe containing the EOS parameters from the eos_fit.fit_to_all function
        harmonic_properties_fit (pd.DataFrame): pandas dataframe containing the fitted harmonic properties from the fit_harmonic function
        P (int, optional): Pressure in GPa. Defaults to 0.
        plot (bool, optional): Defaults to True.
        plot_type (str, optional): Type of plots to include. Defaults to 'default'.

    Returns:
        pd.DataFrame: pandas dataframe containing the quasi-harmonic properties
    """

    # TODO: Add a way to ensure that the number of atoms from eos_parameters_df and harmonic_properties_fit match
    eos_parameters_one_EOS = eos_parameters_df[eos_parameters_df["EOS"] == "BM4"]
    a = eos_parameters_one_EOS["a"].values[0]
    b = eos_parameters_one_EOS["b"].values[0]
    c = eos_parameters_one_EOS["c"].values[0]
    d = eos_parameters_one_EOS["d"].values[0]
    e = eos_parameters_one_EOS["e"].values[0]

    # Get the energy from the EOS fit
    volume_range = harmonic_properties_fit.iloc[0]["volume_fit"]
    energy_eos = eos_fit.BM4_equation(volume_range, a, b, c, d)

    # For each temperature, add energy_eos to f_vib_fit and fit to an EOS
    free_energy_list = []
    volume_range_list = []
    eos_constants_list = []
    V0_list = []
    F0_list = []
    B_list = []
    BP_list = []
    S0_list = []

    P = P / EV_PER_CUBIC_ANGSTROM_TO_GPA  # Convert GPa to eV/Å³
    temperature_list = harmonic_properties_fit.index.tolist()
    for temperature in temperature_list:

        f_vib_fit = harmonic_properties_fit.loc[temperature]["f_vib_fit"]
        free_energy = energy_eos + f_vib_fit + P * volume_range
        free_energy_list.append(free_energy)
        volume_range_list.append(volume_range)

        eos_constants, eos_parameters, _, _, _ = eos_fit.BM4(volume_range, free_energy)
        eos_constants_list.append(eos_constants)
        V0_list.append(eos_parameters[0])
        F0_list.append(eos_parameters[1])
        B_list.append(eos_parameters[2])
        BP_list.append(eos_parameters[3])

        entropy_polynomial = harmonic_properties_fit.loc[temperature][
            "entropy_polynomial"
        ]
        S0 = entropy_polynomial(eos_parameters[0])
        S0_list.append(S0)

    # Create a quasi-harmonic dataframe
    quasi_harmonic_properties = pd.DataFrame(
        data={
            "pressure": [P] * len(temperature_list),
            "number_of_atoms": [harmonic_properties_fit["number_of_atoms"].iloc[0]]
            * len(temperature_list),
            "temperature": temperature_list,
            "volume_range": volume_range_list,
            "free_energy": free_energy_list,
            "eos_constants": eos_constants_list,
            "V0": V0_list,
            "F0": F0_list,
            "B": B_list,
            "BP": BP_list,
            "S0": S0_list,
        }
    )

    # Calculate other properties using the finite difference method
    V0 = quasi_harmonic_properties["V0"].values
    S0 = quasi_harmonic_properties["S0"].values
    T = quasi_harmonic_properties["temperature"].values
    dV = V0[1:] - V0[:-1]
    dS = S0[1:] - S0[:-1]
    dT = T[1:] - T[:-1]

    CTE = (1 / V0[:-1]) * dV / dT * 1e6
    Cp = T[:-1] * dS / dT

    # Add a NaN value to the end of the CTE list
    CTE = np.append(CTE, np.nan)
    Cp = np.append(Cp, np.nan)

    quasi_harmonic_properties["H0"] = (
        quasi_harmonic_properties["F0"]
        + quasi_harmonic_properties["temperature"] * quasi_harmonic_properties["S0"]
    )
    quasi_harmonic_properties["CTE"] = CTE
    quasi_harmonic_properties["Cp"] = Cp

    if plot == True:
        plot_quasi_harmonic(quasi_harmonic_properties, plot_type)

    return quasi_harmonic_properties


def plot_quasi_harmonic(
    quasi_harmonic_properties: pd.DataFrame, plot_type: str = "default"
):
    """Plots the quasi-harmonic properties

    Args:
        quasi_harmonic_properties (pd.DataFrame): pandas dataframe containing the quasi-harmonic properties from the quasi_harmonic function
        plot_type (str, optional): Type of plots to include. Defaults to 'default'.
    """

    temperature_list = quasi_harmonic_properties["temperature"].values
    spaces = len(temperature_list) - 1
    step = int(spaces / 9)

    selected_temperatures = temperature_list[::step]
    if selected_temperatures[-1] != temperature_list[-1]:
        selected_temperatures = np.append(selected_temperatures, temperature_list[-1])

    scale_atoms = quasi_harmonic_properties["number_of_atoms"].iloc[0]

    if plot_type == "default" or plot_type == "all":
        # Free energy plot
        fig = go.Figure()
        for temperature in selected_temperatures:
            x = quasi_harmonic_properties[
                quasi_harmonic_properties["temperature"] == temperature
            ]["volume_range"].values[0]
            y = quasi_harmonic_properties[
                quasi_harmonic_properties["temperature"] == temperature
            ]["free_energy"].values[0]

            F0 = quasi_harmonic_properties[
                quasi_harmonic_properties["temperature"] == temperature
            ]["F0"].values[0]
            V0 = quasi_harmonic_properties[
                quasi_harmonic_properties["temperature"] == temperature
            ]["V0"].values[0]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    marker=dict(size=10),
                    name=f"{temperature} K",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[V0],
                    y=[F0],
                    mode="markers",
                    marker=dict(size=10, symbol="cross", color="black"),
                    showlegend=False,
                ),
            )
        plot_format(
            fig,
            f"Volume (Å³/{scale_atoms} atoms)",
            f"Free Energy (eV/{scale_atoms} atoms)",
            width=600,
            height=600,
        )
        fig.show()

        # Volume plot
        fig = go.Figure()
        x = temperature_list
        y = quasi_harmonic_properties["V0"].values

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                marker=dict(size=10),
                name=f"{temperature} K",
            ),
        )
        plot_format(
            fig,
            f"Temperature (K)",
            f"Volume (Å³/{scale_atoms} atoms)",
            width=600,
            height=600,
        )
        fig.show()

        # CTE plot
        fig = go.Figure()
        x = temperature_list
        y = quasi_harmonic_properties["CTE"].values

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                marker=dict(size=10),
                name=f"{temperature} K",
            ),
        )
        plot_format(
            fig,
            f"Temperature (K)",
            "CTE (10<sup>-6</sup> K<sup>-1</sup>)",
            width=600,
            height=600,
        )
        fig.show()

        # Entropy plot
        fig = go.Figure()
        x = temperature_list
        y = quasi_harmonic_properties["S0"].values

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                marker=dict(size=10),
                name=f"{temperature} K",
            ),
        )
        plot_format(
            fig,
            f"Temperature (K)",
            f"Entropy (eV/K/{scale_atoms} atoms)",
            width=600,
            height=600,
        )
        fig.show()

        # Cp plot
        fig = go.Figure()
        x = temperature_list
        y = quasi_harmonic_properties["Cp"].values

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                marker=dict(size=10),
                name=f"{temperature} K",
            ),
        )
        plot_format(
            fig,
            f"Temperature (K)",
            f"C<sub>p</sub> (eV/K/{scale_atoms} atoms)",
            width=600,
            height=600,
        )
        fig.show()

    if plot_type == "all":
        # Enthalpy plot
        fig = go.Figure()
        x = temperature_list
        y = quasi_harmonic_properties["H0"].values

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                marker=dict(size=10),
                name=f"{temperature} K",
            ),
        )
        plot_format(
            fig,
            f"Temperature (K)",
            f"Enthalpy (eV/{scale_atoms} atoms)",
            width=600,
            height=600,
        )
        fig.show()

        # Bulk modulus plot
        fig = go.Figure()
        x = temperature_list
        y = quasi_harmonic_properties["B"].values

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                marker=dict(size=10),
                name=f"{temperature} K",
            ),
        )
        plot_format(
            fig, f"Temperature (K)", "Bulk modulus (GPa)", width=600, height=600
        )
        fig.show()

        # Gibbs energy plot
        fig = go.Figure()
        x = temperature_list
        y = quasi_harmonic_properties["F0"].values

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                marker=dict(size=10),
                name=f"{temperature} K",
            ),
        )
        plot_format(
            fig,
            f"Temperature (K)",
            f"Gibbs energy (eV/{scale_atoms} atoms)",
            width=600,
            height=600,
        )
        fig.show()
