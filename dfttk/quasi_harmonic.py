"""
QuasiHarmonic class for storing, processing, and plotting quasi-harmonic data.
"""

# Related third party imports
import numpy as np
import plotly.graph_objects as go

# Local application/library specific imports
import dfttk.eos.functions as eos_functions
from dfttk.plotly_format import plot_format

EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3 = 160.21766208 GPa


class QuasiHarmonic:
    """
    Stores, processes, and plots quasi-harmonic data for a given system.
    """

    def __init__(self, number_of_atoms: int, volumes: np.ndarray, temperatures: np.ndarray):
        """
        Args:
            number_of_atoms (int): Number of atoms in the system.
            volumes (np.ndarray): Array of volumes.
            temperatures (np.ndarray): Array of temperatures.
        """
        self.number_of_atoms = number_of_atoms
        self.volumes = np.asarray(volumes)
        self.temperatures = np.asarray(temperatures)
        self.methods: dict = {
            "debye": {},
            "debye_thermal_electronic": {},
            "phonons": {},
            "phonons_thermal_electronic": {},
        }

    def process(
        self,
        method: str,
        energy_eos: np.ndarray,
        vibrational_helmholtz_energy: np.ndarray,
        vibrational_entropy: np.ndarray,
        vibrational_heat_capacity: np.ndarray,
        electronic_helmholtz_energy: np.ndarray = None,
        electronic_entropy: np.ndarray = None,
        electronic_heat_capacity: np.ndarray = None,
        P: float = 0.0,
        eos_name: str = "BM4",
    ) -> None:
        """
        Calculates and stores the quasiharmonic properties for a given method.

        Args:
            method (str): Calculation method.
            energy_eos (np.ndarray): Static energies (shape: [n_volumes]).
            vibrational_helmholtz_energy (np.ndarray): Vibrational helmholtz energies (shape: [n_temps, n_volumes]).
            vibrational_entropy (np.ndarray): Vibrational entropies (shape: [n_temps, n_volumes]).
            vibrational_heat_capacity (np.ndarray): Vibrational heat capacities (shape: [n_temps, n_volumes]).
            electronic_helmholtz_energy (np.ndarray, optional): Electronic helmholtz energies (shape: [n_temps, n_volumes]). Defaults to None.
            electronic_entropy (np.ndarray, optional): Electronic entropies (shape: [n_temps, n_volumes]). Defaults to None.
            electronic_heat_capacity (np.ndarray, optional): Electronic heat capacities (shape: [n_temps, n_volumes]). Defaults to None.
            P (float, optional): Pressure in GPa. Defaults to 0.0.
            eos_name (str, optional): Equation of state to use. Defaults to "BM4".
        """

        # Convert P from GPa to eV/Å³
        P_eva3 = P / EV_PER_CUBIC_ANGSTROM_TO_GPA

        n_temps = len(self.temperatures)
        n_vols = len(self.volumes)

        # Check input shapes
        self._check_shape(energy_eos, (n_vols,), "energy_eos")
        self._check_shape(vibrational_helmholtz_energy, (n_temps, n_vols), "vibrational_helmholtz_energy")
        self._check_shape(vibrational_entropy, (n_temps, n_vols), "vibrational_entropy")
        self._check_shape(vibrational_heat_capacity, (n_temps, n_vols), "vibrational_heat_capacity")
        self._check_shape(electronic_helmholtz_energy, (n_temps, n_vols), "electronic_helmholtz_energy")
        self._check_shape(electronic_entropy, (n_temps, n_vols), "electronic_entropy")
        self._check_shape(electronic_heat_capacity, (n_temps, n_vols), "electronic_heat_capacity")

        # Check if P is non-negative
        if P < 0:
            raise ValueError(f"Pressure P should be non-negative, but got {P} GPa")

        # Prepare arrays for results
        helmholtz_energy = np.empty((n_temps, n_vols))
        helmholtz_energy_pv = np.empty((n_temps, n_vols))
        helmholtz_eos_constants = np.empty((n_temps, 5))
        helmholtz_pv_eos_constants = np.empty((n_temps, 5))
        entropy = np.empty((n_temps, n_vols))
        entropy_poly_coeffs = np.empty((n_temps, 3))
        heat_capacity = np.empty((n_temps, n_vols))
        heat_capacity_poly_coeffs = np.empty((n_temps, 3))
        V0 = np.empty(n_temps)
        G0 = np.empty(n_temps)
        S0 = np.empty(n_temps)
        H0 = np.empty(n_temps)
        B = np.empty(n_temps)
        BP = np.empty(n_temps)

        # Use zeros if electronic contributions are not provided
        if electronic_helmholtz_energy is None:
            electronic_helmholtz_energy = np.zeros_like(vibrational_helmholtz_energy)
        if electronic_entropy is None:
            electronic_entropy = np.zeros_like(vibrational_entropy)
        if electronic_heat_capacity is None:
            electronic_heat_capacity = np.zeros_like(vibrational_heat_capacity)

        for idx, T in enumerate(self.temperatures):
            f_vib = vibrational_helmholtz_energy[idx]
            s_vib = vibrational_entropy[idx]
            cv_vib = vibrational_heat_capacity[idx]
            f_el = electronic_helmholtz_energy[idx]
            s_el = electronic_entropy[idx]
            cv_el = electronic_heat_capacity[idx]

            # Calculate and store Helmholtz energies
            helmholtz = energy_eos + f_vib + f_el
            helmholtz_pv = helmholtz + P_eva3 * self.volumes
            helmholtz_energy[idx] = helmholtz
            helmholtz_energy_pv[idx] = helmholtz_pv

            # EOS fitting for F and F + PV
            eos_consts, _ = self._fit_eos(eos_name, self.volumes, helmholtz, T)
            pv_consts, pv_params = self._fit_eos(eos_name, self.volumes, helmholtz_pv, T)
            helmholtz_eos_constants[idx] = eos_consts
            helmholtz_pv_eos_constants[idx] = pv_consts
            V0[idx], G0[idx], B[idx], BP[idx] = pv_params[:4]

            # Thermodynamic properties at P (V0)
            entropy_row, entropy_coeffs, S0_val = self._calc_property_at_P(self.volumes, s_vib + s_el, V0[idx])
            heat_capacity_row, heat_capacity_coeffs, _ = self._calc_property_at_P(self.volumes, cv_vib + cv_el, V0[idx])
            entropy[idx] = entropy_row
            entropy_poly_coeffs[idx] = entropy_coeffs
            heat_capacity[idx] = heat_capacity_row
            heat_capacity_poly_coeffs[idx] = heat_capacity_coeffs
            S0[idx] = S0_val

        # Finite difference properties
        CTE, LCTE, Cp = self._finite_difference_properties(V0, S0, self.temperatures)
        H0 = G0 + self.temperatures * S0

        # Store results
        self.methods[method] = {
            "helmholtz_energy": {
                "eos_constants": {"eos_name": eos_name, **{f"{t}K": dict(zip("abcde", helmholtz_eos_constants[i])) for i, t in enumerate(self.temperatures)}},
                "values": helmholtz_energy,
            },
            "entropy": {
                "poly_coeffs": {f"{t}K": entropy_poly_coeffs[i] for i, t in enumerate(self.temperatures)},
                "values": entropy,
            },
            "heat_capacity": {
                "poly_coeffs": {f"{t}K": heat_capacity_poly_coeffs[i] for i, t in enumerate(self.temperatures)},
                "values": heat_capacity,
            },
            f"{int(P)}_GPa": {
                "helmholtz_energy_pv": {
                    "eos_constants": {"eos_name": eos_name, **{f"{t}K": dict(zip("abcde", helmholtz_pv_eos_constants[i])) for i, t in enumerate(self.temperatures)}},
                    "values": helmholtz_energy_pv,
                },
                "V0": V0,
                "G0": G0,
                "S0": S0,
                "H0": H0,
                "B": B,
                "BP": BP,
                "CTE": CTE,
                "LCTE": LCTE,
                "Cp": Cp,
            },
        }

    def plot(
        self,
        method: str,
        P: float,
        plot_type: str,
        selected_temperatures: np.ndarray = None,
    ) -> go.Figure:
        """
        Generate plots for the quasiharmonic data.

        Args:
            method (str): Calculation method.
            P (float): Pressure in GPa.
            plot_type (str): Type of plot to generate.
            selected_temperatures (np.ndarray, optional): Specific temperatures to plot for helmholtz_energy_pv.

        Returns:
            go.Figure: Plotly figure object for the selected data.
        """
        data = self.methods[method][f"{int(P)}_GPa"]
        V0, G0, S0, H0, B, BP, CTE, LCTE, Cp = (data["V0"], data["G0"], data["S0"], data["H0"], data["B"], data["BP"], data["CTE"], data["LCTE"], data["Cp"])
        helmholtz_energy_pv = data["helmholtz_energy_pv"]["values"]

        if selected_temperatures is None:
            spaces = len(self.temperatures) - 1
            step = max(1, int(spaces / 10))
            selected_temperatures = self.temperatures[::step]
            if selected_temperatures[-1] != self.temperatures[-1]:
                selected_temperatures = np.append(selected_temperatures, self.temperatures[-1])
        else:
            selected_temperatures = selected_temperatures

        def create_plot(x, y, x_label, y_label):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", marker=dict(size=10)))
            plot_format(fig, x_label, y_label, width=650, height=600)
            fig.show()
            return fig

        if plot_type == "helmholtz_energy_pv":
            fig = go.Figure()
            for idx, temperature in enumerate(self.temperatures):
                if temperature in selected_temperatures:
                    x = self.volumes
                    y = helmholtz_energy_pv[idx]
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            marker=dict(size=10),
                            name=f"{int(temperature)} K" if temperature % 1 == 0 else f"{temperature} K",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[V0[idx]],
                            y=[G0[idx]],
                            mode="markers",
                            marker=dict(size=10, symbol="cross", color="black"),
                            showlegend=False,
                        )
                    )
            plot_format(
                fig,
                f"Volume (Å³/{self.number_of_atoms} atoms)",
                f"F + PV (eV/{self.number_of_atoms} atoms)",
                width=650,
                height=600,
            )
            fig.show()
            return fig

        plot_mappings = {
            "volume": (V0, f"Volume (Å³/{self.number_of_atoms} atoms)"),
            "gibbs_energy": (G0, f"Gibbs energy (eV/{self.number_of_atoms} atoms)"),
            "entropy": (S0, f"Entropy (eV/K/{self.number_of_atoms} atoms)"),
            "enthalpy": (H0, f"Enthalpy (eV/{self.number_of_atoms} atoms)"),
            "bulk_modulus": (B, "Bulk modulus (GPa)"),
            "CTE": (CTE, "CTE (10⁻⁶ K⁻¹)"),
            "LCTE": (LCTE, "LCTE (10⁻⁶ K⁻¹)"),
            "heat_capacity": (Cp, f"Cₚ (eV/K/{self.number_of_atoms} atoms)"),
        }

        if plot_type in plot_mappings:
            y, y_label = plot_mappings[plot_type]
            x = self.temperatures
            fig = create_plot(x, y, "Temperature (K)", y_label)
            return fig

        # If plot_type is not recognized, return an empty figure
        return go.Figure()

    def _check_shape(
        self,
        arr: np.ndarray,
        expected_shape: tuple,
        name: str,
    ) -> None:
        """
        Check that an array has the expected shape.

        Args:
            arr (np.ndarray): The array to check.
            expected_shape (tuple): The expected shape.
            name (str): Name of the variable (for error messages).

        Raises:
            ValueError: If the array does not have the expected shape.
        """
        if arr is not None and arr.shape != expected_shape:
            raise ValueError(f"{name} should have shape {expected_shape}, but got {arr.shape}")

    def _fit_eos(
        self,
        eos_name: str,
        volumes: np.ndarray,
        energies: np.ndarray,
        temperature: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the equation of state (EOS) for the given energies and volumes.

        Args:
            eos_name (str): Name of the EOS function to use.
            volumes (np.ndarray): Array of volumes.
            energies (np.ndarray): Array of energies corresponding to volumes.
            temperature (float): Temperature (for error reporting).

        Returns:
            tuple: (eos_constants, eos_parameters)
        """
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
        try:
            eos_constants, eos_parameters, _, _, _ = eos_fit_functions[eos_name](volumes, energies)
        except RuntimeError as e:
            print(f"Error fitting EOS at {temperature} K: {e}")
            print(f"Suggestion: Try using a different EOS. Available options are: {list(eos_fit_functions.keys())}")
            eos_constants = eos_parameters = np.full(5, np.nan)
        return eos_constants, eos_parameters

    def _calc_property_at_P(
        self,
        volumes: np.ndarray,
        property_array: np.ndarray,
        V0: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Fit a quadratic polynomial to the property as a function of volume and evaluate at V0.

        Args:
            volumes (np.ndarray): Array of volumes.
            property_array (np.ndarray): Property values at each volume.
            V0 (float): Volume at which to evaluate the fitted polynomial.

        Returns:
            tuple: (property_array, coeffs, value_at_V0)
        """
        coeffs = np.polyfit(volumes, property_array, 2)
        poly = np.poly1d(coeffs)
        return property_array, coeffs, poly(V0)

    def _finite_difference_properties(
        self,
        V0: np.ndarray,
        S0: np.ndarray,
        T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate CTE, LCTE, and Cp using finite forward differences.

        Args:
            V0 (np.ndarray): Array of equilibrium volumes at each temperature.
            S0 (np.ndarray): Array of entropies at P/V0 for each temperature.
            T (np.ndarray): Array of temperatures.

        Returns:
            tuple: (CTE, LCTE, Cp)
        """
        dV = np.diff(V0)
        dS = np.diff(S0)
        dT = np.diff(T)
        CTE = (1 / V0[:-1]) * dV / dT * 1e6
        Cp = T[:-1] * dS / dT
        CTE = np.insert(CTE, 0, 0)
        Cp = np.insert(Cp, 0, 0)
        LCTE = CTE / 3
        return CTE, LCTE, Cp
