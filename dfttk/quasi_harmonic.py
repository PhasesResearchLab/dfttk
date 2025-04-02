"""
Quasiharmonic approximation module.
"""

# Related third party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Local application/library specific imports
import dfttk.eos_fit as eos_fit
from dfttk.plotly_format import plot_format

EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3  = 160.21766208 GPa

def process_quasi_harmonic(
    number_of_atoms: int,
    temperatures: np.ndarray, 
    volume_range: np.ndarray,
    energy_eos: np.ndarray,
    f_vib_fit = np.ndarray, 
    s_vib_fit = np.ndarray,  
    cv_vib_fit = np.ndarray,  
    f_el_fit: np.ndarray = None,
    s_el_fit: np.ndarray = None,
    cv_el_fit: np.ndarray = None,
    P: float = 0.0,
    eos: str = "BM4",
) -> pd.DataFrame:
    
    # For each temperature, add energy_eos to f_vib_fit, then fit to an EOS
    f_plus_pv_list = []
    volume_range_list = []
    eos_constants_list = []
    s_coefficients_list = []
    cv_coefficients_list = []
    V0_list = []
    F0_list = []
    B_list = []
    BP_list = []
    S0_list = []

    P = P / EV_PER_CUBIC_ANGSTROM_TO_GPA  # Convert GPa to eV/Å³
    
    # List of available EOS functions
    eos_fit_functions = {
        "mBM4": eos_fit.mBM4,
        "mBM5": eos_fit.mBM5,
        "BM4": eos_fit.BM4,
        "BM5": eos_fit.BM5,
        "LOG4": eos_fit.LOG4,
        "LOG5": eos_fit.LOG5,
        "murnaghan": eos_fit.murnaghan,
        "vinet": eos_fit.vinet,
        "morse": eos_fit.morse,
    }
    
    for index, temperature in enumerate(temperatures):
        # For each temperature, add f_vib and f_el to energy_eos, then fit to an EOS
        f_vib = f_vib_fit[index]
        f_plus_pv = energy_eos + f_vib + P * volume_range

        if f_el_fit != 0:
            f_el = f_el_fit[index]
            f_plus_pv += f_el

        f_plus_pv_list.append(f_plus_pv)
        volume_range_list.append(volume_range)

        try:
            eos_constants, eos_parameters, _, _, _ = eos_fit_functions[eos](
                volume_range, f_plus_pv
            )

        except RuntimeError as e:
            print(f"Error fitting EOS at {temperature} K: {e}")
            print(
                f"Suggestion: Try using a different EOS. Available options are: {list(eos_fit_functions.keys())}"
            )

        eos_constants_list.append(eos_constants)

        V0 = eos_parameters[0]
        F0 = eos_parameters[1]
        B = eos_parameters[2]
        BP = eos_parameters[3]

        V0_list.append(V0)
        F0_list.append(F0)
        B_list.append(B)
        BP_list.append(BP)

        # For each temperature, s = s_vib to s_el and cv = cv_vib to cv_el.
        # Find s and cv that corresponds to P.
        s_vib = s_vib_fit[index]
        cv_vib = cv_vib_fit[index]
        order = 2

        if s_el_fit != 0 and cv_el_fit != 0:
            s_el = s_el_fit[index]
            cv_el = cv_el_fit[index]

        elif s_el_fit == 0 and cv_el_fit == 0:
            s_el = 0
            cv_el = 0

        s = s_vib + s_el
        s_coefficients = np.polyfit(volume_range, s, order)
        s_coefficients_list.append(s_coefficients)
        s_poly = np.poly1d(s_coefficients)

        cv = cv_vib + cv_el
        cv_coefficients = np.polyfit(volume_range, cv, order)
        cv_coefficients_list.append(cv_coefficients)

        S0 = s_poly(V0)
        S0_list.append(S0)
    
    # Create a quasi-harmonic dataframe
    quasi_harmonic_properties = pd.DataFrame(
        data={
            "pressure": [P * EV_PER_CUBIC_ANGSTROM_TO_GPA] * len(temperatures),
            "number_of_atoms": [number_of_atoms] * len(temperatures),
            "temperature": temperatures,
            "volume_range": volume_range_list,
            "f_plus_pv": f_plus_pv_list,
            "eos_constants": eos_constants_list,
            "s_coefficients": s_coefficients_list,
            "cv_coefficients": cv_coefficients_list,
            "V0": V0_list,
            "G0": F0_list,
            "B": B_list,
            "BP": BP_list,
            "S0": S0_list,
        }
    )

    # Calculate other properties using the finite difference method 
    V0 = np.array(V0_list)
    S0 = np.array(S0_list)
    G0 = np.array(F0_list)
    T = temperatures
    
    dV = V0[1:] - V0[:-1]
    dS = S0[1:] - S0[:-1]
    dT = T[1:] - T[:-1]

    CTE = (1 / V0[:-1]) * dV / dT * 1e6
    Cp = T[:-1] * dS / dT

    # Add a value of 0 to the beginning of CTE and Cp
    CTE = np.insert(CTE, 0, 0)
    Cp = np.insert(Cp, 0, 0)

    H0 = G0 + T * S0
    
    quasi_harmonic_properties["H0"] = (
        quasi_harmonic_properties["G0"]
        + quasi_harmonic_properties["temperature"] * quasi_harmonic_properties["S0"]
    )
    quasi_harmonic_properties["CTE"] = CTE
    quasi_harmonic_properties["Cp"] = Cp

    f_plus_pv = np.array(f_plus_pv_list)
    eos_constants = np.array(eos_constants_list)
    s_coefficients = np.array(s_coefficients_list)
    cv_coefficients = np.array(cv_coefficients_list)
    V0 = np.array(V0_list)
    F0 = np.array(F0_list)
    B = np.array(B_list)
    BP = np.array(BP_list)
    S0 = np.array(S0_list)
    
    return f_plus_pv, eos_constants, s_coefficients, cv_coefficients, V0, F0, B, BP, S0, CTE, Cp, H0, quasi_harmonic_properties


def plot_quasi_harmonic(
    quasi_harmonic_properties: pd.DataFrame,
    plot_type: str,
    selected_temperatures_plot: list = None,
):
    """Plots the quasi-harmonic properties

    Args:
        quasi_harmonic_properties (pd.DataFrame): pandas dataframe containing the quasi-harmonic properties from the quasi_harmonic function
        plot_type (str, optional): Type of plots to include. Defaults to 'default'.
        selected_temperatures_plot (list, optional): List of selected temperatures to plot. Defaults to None.

    Returns:
        go.Figure: The plotly figure object for the specified plot type.
    """

    temperature_list = quasi_harmonic_properties["temperature"].values
    if selected_temperatures_plot is None:
        spaces = len(temperature_list) - 1
        step = max(1, int(spaces / 10))
        selected_temperatures = temperature_list[::step]
        if selected_temperatures[-1] != temperature_list[-1]:
            selected_temperatures = np.append(
                selected_temperatures, temperature_list[-1]
            )
    else:
        selected_temperatures = selected_temperatures_plot

    scale_atoms = quasi_harmonic_properties["number_of_atoms"].iloc[0]

    def create_plot(x, y, x_label, y_label):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", marker=dict(size=10)))
        plot_format(fig, x_label, y_label, width=650, height=600)
        fig.show()
        return fig

    if plot_type == "helmholtz_energy_pv":
        fig = go.Figure()
        for temperature in selected_temperatures:
            x = quasi_harmonic_properties[
                quasi_harmonic_properties["temperature"] == temperature
            ]["volume_range"].values[0]
            y = quasi_harmonic_properties[
                quasi_harmonic_properties["temperature"] == temperature
            ]["f_plus_pv"].values[0]
            G0 = quasi_harmonic_properties[
                quasi_harmonic_properties["temperature"] == temperature
            ]["G0"].values[0]
            V0 = quasi_harmonic_properties[
                quasi_harmonic_properties["temperature"] == temperature
            ]["V0"].values[0]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    marker=dict(size=10),
                    name=(
                        f"{int(temperature)} K"
                        if temperature % 1 == 0
                        else f"{temperature} K"
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[V0],
                    y=[G0],
                    mode="markers",
                    marker=dict(size=10, symbol="cross", color="black"),
                    showlegend=False,
                )
            )
        plot_format(
            fig,
            f"Volume (Å³/{scale_atoms} atoms)",
            f"F + PV (eV/{scale_atoms} atoms)",
            width=650,
            height=600,
        )
        fig.show()
        return fig

    else:
        fig = go.Figure()
        plot_mappings = {
            "volume": ("V0", f"Volume (Å³/{scale_atoms} atoms)"),
            "cte": ("CTE", "CTE (10<sup>-6</sup> K<sup>-1</sup>)"),
            "entropy": ("S0", f"Entropy (eV/K/{scale_atoms} atoms)"),
            "heat_capacity": ("Cp", f"C<sub>p</sub> (eV/K/{scale_atoms} atoms)"),
            "enthalpy": ("H0", f"Enthalpy (eV/{scale_atoms} atoms)"),
            "bulk_modulus": ("B", "Bulk modulus (GPa)"),
            "gibbs_energy": ("G0", f"Gibbs energy (eV/{scale_atoms} atoms)"),
        }

        if plot_type in plot_mappings:
            y_list, y_labels = plot_mappings[plot_type]
            x = temperature_list
            y = quasi_harmonic_properties[y_list].values
            fig = create_plot(x, y, "Temperature (K)", y_labels)
        return fig
