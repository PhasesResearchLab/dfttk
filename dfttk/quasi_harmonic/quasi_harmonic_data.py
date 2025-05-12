"""
QuasiHarmonicData class for storing, processing, and plotting quasiharmonic data.
"""

# Related third party imports
import numpy as np
import plotly.graph_objects as go

# Local application/library specific imports
from dfttk.quasi_harmonic.functions import (
    process_quasi_harmonic,
    plot_quasi_harmonic,
)

EV_PER_CUBIC_ANGSTROM_TO_GPA = (
    160.21766208  # Conversion factor: 1 eV/Å^3 = 160.21766208 GPa
)


class QuasiHarmonicData:
    """
    A class to store, process, and plot quasiharmonic data.
    """

    def __init__(self):
        """
        Initialize the QuasiHarmonicData object with default attributes.
        """
        self.number_of_atoms: int = None
        self.temperatures: np.ndarray = None
        self.volumes: np.ndarray = None
        self.methods: dict = {
            "debye": {},
            "debye_thermal_electronic": {},
            "phonons": {},
            "phonons_thermal_electronic": {},
        }

    def get_quasi_harmonic_data(
        self,
        method: str,
        eos: str,
        number_of_atoms: int,
        temperatures: np.ndarray,
        volumes: np.ndarray,
        energy_eos: np.ndarray,
        f_vib_fit: np.ndarray,
        s_vib_fit: np.ndarray,
        cv_vib_fit: np.ndarray,
        f_el_fit=None,
        s_el_fit=None,
        cv_el_fit=None,
        P: float = 0,
    ) -> None:
        """
        Process and store quasiharmonic data for a given method.

        Args:
            method (str): Calculation method (e.g., "debye", "debye_thermal_electronic", "phonons", "phonons_thermal_electronic").
            eos (str): Equation of state (e.g., "BM4").
            number_of_atoms (int): Number of atoms for the configuration.
            temperatures (np.ndarray): Array of temperatures.
            volumes (np.ndarray): Array of volumes.
            energy_eos (np.ndarray): Energy values for the equation of state.
            f_vib_fit (np.ndarray): Vibrational free energy fit.
            s_vib_fit (np.ndarray): Vibrational entropy fit.
            cv_vib_fit (np.ndarray): Vibrational heat capacity fit.
            f_el_fit (np.ndarray, optional): Electronic free energy fit. Defaults to None.
            s_el_fit (np.ndarray, optional): Electronic entropy fit. Defaults to None.
            cv_el_fit (np.ndarray, optional): Electronic heat capacity fit. Defaults to None.
            P (float, optional): Pressure in GPa. Defaults to 0.
        """
        # Store basic attributes
        self.number_of_atoms = number_of_atoms
        self.temperatures = temperatures
        self.volumes = volumes

        # Process the quasiharmonic data using the provided inputs
        (
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
        ) = process_quasi_harmonic(
            temperatures=temperatures,
            volumes=volumes,
            energy_eos=energy_eos,
            f_vib_fit=f_vib_fit,
            s_vib_fit=s_vib_fit,
            cv_vib_fit=cv_vib_fit,
            f_el_fit=f_el_fit,
            s_el_fit=s_el_fit,
            cv_el_fit=cv_el_fit,
            P=P,
            eos=eos,
        )

        # Store Helmholtz energy data
        helmholtz_energy_data = {
            "eos_constants": {
                "eos_name": eos,
            },
            "values": helmholtz_energy,
        }
        for temp, constants in zip(self.temperatures, helmholtz_eos_constants):
            helmholtz_energy_data["eos_constants"][f"{temp}K"] = {
                "a": constants[0],
                "b": constants[1],
                "c": constants[2],
                "d": constants[3],
                "e": constants[4],
            }

        # Store Helmholtz energy + PV data
        helmholtz_energy_pv_data = {
            "eos_constants": {
                "eos_name": eos,
            },
            "values": helmholtz_energy_pv,
        }
        for temp, constants in zip(self.temperatures, helmholtz_pv_eos_constants):
            helmholtz_energy_pv_data["eos_constants"][f"{temp}K"] = {
                "a": constants[0],
                "b": constants[1],
                "c": constants[2],
                "d": constants[3],
                "e": constants[4],
            }

        # Store entropy data
        entropy_data = {
            "poly_coeffs": {},
            "values": entropy,
        }
        for temp, entropy_coeff in zip(self.temperatures, entropy_poly_coeffs):
            entropy_data["poly_coeffs"][f"{temp}K"] = entropy_coeff

        # Store heat capacity data
        heat_capacity_data = {
            "poly_coeffs": {},
            "values": heat_capacity,
        }
        for temp, cv_coeff in zip(self.temperatures, heat_capacity_poly_coeffs):
            heat_capacity_data["poly_coeffs"][f"{temp}K"] = cv_coeff

        # Store all processed data in the methods dictionary
        self.methods[method] = {
            "helmholtz_energy": helmholtz_energy_data,
            "entropy": entropy_data,
            "heat_capacity": heat_capacity_data,
            f"{int(P)}_GPa": {
                "helmholtz_energy_pv": helmholtz_energy_pv_data,
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
        selected_temperatures_plot: np.ndarray = None,
    ) -> go.Figure:
        """
        Generate plots for the quasiharmonic data.

        Args:
            method (str): Calculation method (e.g., "debye", "debye_thermal_electronic", "phonons", "phonons_thermal_electronic").
            P (float): Pressure in GPa.
            plot_type (str, optional): Type of plot to generate. Options include "helmholtz_energy_pv", "volume", "gibbs_energy", "entropy", "enthalpy", "bulk_modulus",
            "CTE", "LCTE", "heat_capacity".
            selected_temperatures_plot (np.ndarray, optional): Specific temperatures to plot for helmholtz_energy_pv. If None, plot 11 evenly spaced temperatures
            between the minimum and maximum of the provided temperatures. Defaults to None.

        Returns:
            go.Figure: Plotly figure object for the selected data.
        """
        # Retrieve the tuple output from the quasi-harmonic data
        quasi_harmonic_output = (
            self.methods[method][f"{int(P)}_GPa"]["helmholtz_energy_pv"]["values"],
            self.methods[method][f"{int(P)}_GPa"]["V0"],
            self.methods[method][f"{int(P)}_GPa"]["G0"],
            self.methods[method][f"{int(P)}_GPa"]["B"],
            self.methods[method][f"{int(P)}_GPa"]["S0"],
            self.methods[method][f"{int(P)}_GPa"]["CTE"],
            self.methods[method][f"{int(P)}_GPa"]["LCTE"],
            self.methods[method][f"{int(P)}_GPa"]["Cp"],
            self.methods[method][f"{int(P)}_GPa"]["H0"],
        )

        # Generate the plot using the plot_quasi_harmonic function
        fig = plot_quasi_harmonic(
            quasi_harmonic_output=quasi_harmonic_output,
            temperatures=self.temperatures,
            volumes=self.volumes,
            number_of_atoms=self.number_of_atoms,
            plot_type=plot_type,
            selected_temperatures_plot=selected_temperatures_plot,
        )
        return fig
