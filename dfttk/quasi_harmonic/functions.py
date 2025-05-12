"""
Functions for calculating quasiharmonic properties and plotting them.
"""

# Related third party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Local application/library specific imports
import dfttk.eos.functions as eos_functions
from dfttk.plotly_format import plot_format

EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3  = 160.21766208 GPa


def process_quasi_harmonic(
    temperatures: np.ndarray,
    volumes: np.ndarray,
    energy_eos: np.ndarray,
    f_vib_fit: np.ndarray,
    s_vib_fit: np.ndarray,
    cv_vib_fit: np.ndarray,
    f_el_fit: np.ndarray = None,
    s_el_fit: np.ndarray = None,
    cv_el_fit: np.ndarray = None,
    P: float = 0.0,
    eos: str = "BM4",
) -> tuple[
    np.ndarray,  # helmholtz_energy
    np.ndarray,  # helmholtz_eos_constants
    np.ndarray,  # entropy
    np.ndarray,  # entropy_poly_coeffs
    np.ndarray,  # heat_capacity
    np.ndarray,  # heat_capacity_poly_coeffs
    np.ndarray,  # helmholtz_energy_pv
    np.ndarray,  # helmholtz_pv_eos_constants
    np.ndarray,  # V0
    np.ndarray,  # G0
    np.ndarray,  # S0
    np.ndarray,  # H0
    np.ndarray,  # B
    np.ndarray,  # BP
    np.ndarray,  # CTE
    np.ndarray,  # LCTE
    np.ndarray,  # Cp
]:
    """Calculates the quasiharmonic properties.

    Args:
        temperatures (np.ndarray): Array of temperatures corresponding to vibrational and thermal electronic inputs.
        volumes (np.ndarray): Array of volumes corresponding to energy, vibrational, and thermal electronic inputs.
        energy_eos (np.ndarray): Array of energy values corresponding to the volume range.
        f_vib_fit (np.ndarray): Array of vibrational free energy values. Rows are temperatures, columns are volumes.
        s_vib_fit (np.ndarray): Array of vibrational entropy values. Rows are temperatures, columns are volumes.
        cv_vib_fit (np.ndarray): Array of vibrational heat capacity values. Rows are temperatures, columns are volumes.
        f_el_fit (np.ndarray, optional): Array of thermal electronic free energy values. Rows are temperatures, columns are volumes. Defaults to None.
        s_el_fit (np.ndarray, optional): Array of thermal electronic entropy values. Rows are temperatures, columns are volumes. Defaults to None.
        cv_el_fit (np.ndarray, optional): Array of thermal electronic heat capacity values. Rows are temperatures, columns are volumes. Defaults to None.
        P (float, optional): Pressure in GPa. Defaults to 0.0.
        eos (str, optional): Equation of state (EOS) to use for fitting F + PV. Defaults to "BM4".
        Available options are: "mBM4", "mBM5", "BM4", "BM5", "LOG4", "LOG5", "murnaghan", "vinet", "morse".

    Returns:
        tuple: A tuple containing the following calculated properties:
            - helmholtz_energy (np.ndarray): Helmholtz energy values. Rows are temperatures, columns are volumes.
            - helmholtz_eos_constants (np.ndarray): EOS constants for Helmholtz energy for each temperature.
            - entropy (np.ndarray): Entropy values. Rows are temperatures, columns are volumes.
            - entropy_poly_coeffs (np.ndarray): Polynomial coefficients for entropy for each temperature.
            - heat_capacity (np.ndarray): Heat capacity values. Rows are temperatures, columns are volumes.
            - heat_capacity_poly_coeffs (np.ndarray): Polynomial coefficients for heat capacity for each temperature.
            - helmholtz_energy_pv (np.ndarray): Helmholtz energy + PV values. Rows are temperatures, columns are volumes.
            - helmholtz_pv_eos_constants (np.ndarray): EOS constants for Helmholtz energy + PV for each temperature.
            - V0 (np.ndarray): Equilibrium volume values at P.
            - G0 (np.ndarray): Gibbs energy values at P.
            - S0 (np.ndarray): Entropy values at P.
            - H0 (np.ndarray): Enthalpy values at P.
            - B (np.ndarray): Bulk modulus values at P.
            - BP (np.ndarray): Pressure derivative of bulk modulus values at P.
            - CTE (np.ndarray): Coefficient of thermal expansion values at P.
            - LCTE (np.ndarray): Linear coefficient of thermal expansion values at P.
            - Cp (np.ndarray): Heat capacity values at P.
    """

    # Convert pressure to eV/Å³
    P = P / EV_PER_CUBIC_ANGSTROM_TO_GPA

    # List of available EOS functions
    eos_fit_functions = {
        "mBM4": eos_functions.mBM4,
        "mBM5": eos_functions.mBM5,
        "BM4": eos_functions.BM4,
        "BM5": eos_functions.BM5,
        "LOG4": eos_functions.LOG4,
        "LOG5": eos_functions.LOG5,
        "murnaghan": eos_functions.murnaghan,
        "vinet": eos_functions.vinet,
        "morse": eos_functions.morse,
    }

    # Initialize lists to store results
    (
        helmholtz_energy_list,
        helmholtz_energy_pv_list,
        helmholtz_constants_list,
        helmholtz_pv_constants_list,
    ) = ([], [], [], [])
    entropy_list, entropy_coeffs_list = [], []
    heat_capacity_list, heat_capacity_coeffs_list = [], []
    V0_list, G0_list, B_list, BP_list, S0_list = [], [], [], [], []

    for index, temperature in enumerate(temperatures):
        # Calculate helmholtz_energy(V,T) and helmholtz_energy_pv(V,T)
        f_vib = f_vib_fit[index]
        helmholtz_energy = energy_eos + f_vib
        helmholtz_energy_pv = helmholtz_energy + P * volumes
        if not np.all(f_el_fit == 0):
            f_el = f_el_fit[index]
            helmholtz_energy += f_el
            helmholtz_energy_pv += f_el
        helmholtz_energy_list.append(helmholtz_energy)
        helmholtz_energy_pv_list.append(helmholtz_energy_pv)

        # Fit EOS and extract constants and parameters
        try:
            helmholtz_eos_constants, _, _, _, _ = eos_fit_functions[eos](
                volumes, helmholtz_energy
            )
            helmholtz_pv_eos_constants, helmholtz_pv_parameters, _, _, _ = (
                eos_fit_functions[eos](volumes, helmholtz_energy_pv)
            )

        except RuntimeError as e:
            print(f"Error fitting EOS at {temperature} K: {e}")
            print(
                f"Suggestion: Try using a different EOS. Available options are: {list(eos_fit_functions.keys())}"
            )

        V0, F0, B, BP = helmholtz_pv_parameters[:4]
        V0_list.append(V0)
        G0_list.append(F0)
        B_list.append(B)
        BP_list.append(BP)
        helmholtz_constants_list.append(helmholtz_eos_constants)
        helmholtz_pv_constants_list.append(helmholtz_pv_eos_constants)

        # Calculate entropy(V,T) and heat capacity(V,T))
        s_vib = s_vib_fit[index]
        cv_vib = cv_vib_fit[index]

        if not np.all(s_el_fit == 0) and not np.all(cv_el_fit == 0):
            s_el = s_el_fit[index]
            cv_el = cv_el_fit[index]
        else:
            s_el = 0
            cv_el = 0

        # Calculate entropy at P which corresponds to V0
        order = 2
        entropy = s_vib + s_el
        entropy_list.append(entropy)
        entropy_poly_coeffs = np.polyfit(volumes, entropy, order)
        entropy_coeffs_list.append(entropy_poly_coeffs)
        entropy_poly = np.poly1d(entropy_poly_coeffs)

        # Calculate heat capacity at P which corresponds to V0
        heat_capacity = cv_vib + cv_el
        heat_capacity_list.append(heat_capacity)
        heat_capacity_poly_coeffs = np.polyfit(volumes, heat_capacity, order)
        heat_capacity_coeffs_list.append(heat_capacity_poly_coeffs)

        S0_list.append(entropy_poly(V0))

    # Convert lists to arrays
    V0, S0, G0 = np.array(V0_list), np.array(S0_list), np.array(G0_list)

    # Calculate finite difference properties - CTE, LCTE, and Cp
    T = temperatures
    dV = V0[1:] - V0[:-1]
    dS = S0[1:] - S0[:-1]
    dT = T[1:] - T[:-1]

    dV, dS, dT = np.diff(V0), np.diff(S0), np.diff(T)
    CTE = (1 / V0[:-1]) * dV / dT * 1e6
    Cp = T[:-1] * dS / dT

    # Add a value of 0 to the beginning of CTE and Cp
    CTE = np.insert(CTE, 0, 0)
    Cp = np.insert(Cp, 0, 0)

    # Calculate LCTE
    LCTE = CTE / 3

    # Calculate enthalpy
    H0 = G0 + T * S0

    # Convert results to arrays
    helmholtz_energy = np.array(helmholtz_energy_list)
    helmholtz_energy_pv = np.array(helmholtz_energy_pv_list)
    helmholtz_eos_constants = np.array(helmholtz_constants_list)
    helmholtz_pv_eos_constants = np.array(helmholtz_pv_constants_list)
    entropy = np.array(entropy_list)
    entropy_poly_coeffs = np.array(entropy_coeffs_list)
    heat_capacity = np.array(heat_capacity_list)
    heat_capacity_poly_coeffs = np.array(heat_capacity_coeffs_list)
    G0 = np.array(G0_list)
    B = np.array(B_list)
    BP = np.array(BP_list)

    return (
        helmholtz_energy,
        helmholtz_eos_constants,
        entropy,
        entropy_poly_coeffs,
        heat_capacity,
        heat_capacity_poly_coeffs,
        helmholtz_energy_pv,
        helmholtz_pv_eos_constants,
        V0,
        G0,
        S0,
        H0,
        B,
        BP,
        CTE,
        LCTE,
        Cp,
    )


def plot_quasi_harmonic(
    quasi_harmonic_output: tuple,
    temperatures: np.ndarray,
    volumes: np.ndarray,
    number_of_atoms: int,
    plot_type: str,
    selected_temperatures_plot: np.ndarray = None,
) -> go.Figure:
    """Plot the quasiharmonic properties.

    Args:
        quasi_harmonic_output (tuple): output tuple from process_quasi_harmonic function.
        temperatures (np.ndarray): temperatures corresponding to the quasiharmonic properties.
        volumes (np.ndarray): volume range corresponding to the quasiharmonic properties.
        number_of_atoms (int): number of atoms corresponding to the quasiharmonic properties.
        plot_type (str): helmholtz_energy_pv, volume, gibbs_energy, entropy, enthalpy, bulk_modulus, CTE, LCTE, heat_capacity.
        selected_temperatures_plot (np.ndarray, optional): temperatures to plot for helmholtz_energy_pv. If None, will select 11 evenly spaced temperatures. Defaults to None.

    Returns:
        go.Figure: Plotly figure object for the quasiharmonic properties based on the selected plot type.
    """

    # Unpack the quasi_harmonic_output tuple
    (
        helmholtz_energy_pv,
        V0,
        G0,
        B,
        S0,
        CTE,
        LCTE,
        Cp,
        H0,
    ) = quasi_harmonic_output

    # Selects a subset of temperatures for plotting, either based on 11 evenly spaced temperatures or the provided selected_temperatures_plot
    if selected_temperatures_plot is None:
        spaces = len(temperatures) - 1
        step = max(1, int(spaces / 10))
        selected_temperatures = temperatures[::step]
        if selected_temperatures[-1] != temperatures[-1]:
            selected_temperatures = np.append(selected_temperatures, temperatures[-1])
    else:
        selected_temperatures = selected_temperatures_plot

    # Plotly formatting function
    def create_plot(x, y, x_label, y_label):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", marker=dict(size=10)))
        plot_format(fig, x_label, y_label, width=650, height=600)
        fig.show()
        return fig

    # Plotting based on the selected plot type
    if plot_type == "helmholtz_energy_pv":
        fig = go.Figure()
        for index, temperature in enumerate(temperatures):
            if temperature in selected_temperatures:
                x = volumes
                y = helmholtz_energy_pv[index]
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
                        x=[V0[index]],
                        y=[G0[index]],
                        mode="markers",
                        marker=dict(size=10, symbol="cross", color="black"),
                        showlegend=False,
                    )
                )
        plot_format(
            fig,
            f"Volume (Å³/{number_of_atoms} atoms)",
            f"F + PV (eV/{number_of_atoms} atoms)",
            width=650,
            height=600,
        )
        fig.show()
        return fig

    else:
        fig = go.Figure()
        plot_mappings = {
            "volume": (V0, f"Volume (Å³/{number_of_atoms} atoms)"),
            "gibbs_energy": (G0, f"Gibbs energy (eV/{number_of_atoms} atoms)"),
            "entropy": (S0, f"Entropy (eV/K/{number_of_atoms} atoms)"),
            "enthalpy": (H0, f"Enthalpy (eV/{number_of_atoms} atoms)"),
            "bulk_modulus": (B, "Bulk modulus (GPa)"),
            "CTE": (CTE, "CTE (10⁻⁶ K⁻¹)"),
            "LCTE": (LCTE, "LCTE (10⁻⁶ K⁻¹)"),
            "heat_capacity": (Cp, f"Cₚ (eV/K/{number_of_atoms} atoms)"),
        }

        if plot_type in plot_mappings:
            y, y_label = plot_mappings[plot_type]
            x = temperatures
            fig = create_plot(x, y, "Temperature (K)", y_label)

        return fig
