# Standard library imports
import os

# Related third party imports
import numpy as np
import pandas as pd
import scipy.constants
import plotly.graph_objects as go

# Local application/library specific imports
import dfttk.eos_fit as eos_fit
import dfttk.plotly_format as plot_format

EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3  = 160.21766208 GPa

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
