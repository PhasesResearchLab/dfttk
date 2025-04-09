"""
Quasiharmonic approximation module.
"""

# Related third party imports
import numpy as np
import plotly.graph_objects as go

# Local application/library specific imports
from dfttk.quasi_harmonic.functions import (
    process_quasi_harmonic,
    plot_quasi_harmonic,
)

EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3  = 160.21766208 GPa


class QuasiHarmonicData:
    def __init__(self):
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
        temperatures,
        volume_range: np.ndarray,
        energy_eos: np.ndarray,
        f_vib_fit=None,
        s_vib_fit=None,
        cv_vib_fit=None,
        f_el_fit=None,
        s_el_fit=None,
        cv_el_fit=None,
        P: float = 0,
    ) -> None:

        (
            f_plus_pv,
            eos_constants,
            s_coefficients,
            cv_coefficients,
            V0,
            G0,
            B,
            BP,
            S0,
            CTE,
            Cp,
            H0,
        ) = process_quasi_harmonic(
            temperatures=temperatures,
            volume_range=volume_range,
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

        self.method = method
        self.pressure = P
        self.number_of_atoms = number_of_atoms
        self.temperatures = temperatures
        self.volumes = volume_range
        eos_constants = eos_constants

        self.helmholtz_energy = {
            "eos_parameters": {
                "eos_name": eos,
            }
        }
        for temp, constants in zip(self.temperatures, eos_constants):
            self.helmholtz_energy["eos_parameters"][f"{temp}K"] = {
                "a": constants[0],
                "b": constants[1],
                "c": constants[2],
                "d": constants[3],
                "e": constants[4],
            }
        self.entropy = {"polynomial_coefficients": {}}
        for temp, entropy in zip(self.temperatures, s_coefficients):
            self.entropy["polynomial_coefficients"][f"{temp}K"] = entropy

        self.heat_capacity = {"polynomial_coefficients": {}}
        for temp, cv in zip(self.temperatures, cv_coefficients):
            self.heat_capacity["polynomial_coefficients"][f"{temp}K"] = cv

        self.methods[method][P] = {
            "helmholtz_energy": self.helmholtz_energy,
            "entropy": self.entropy,
            "heat_capacity": self.heat_capacity,
            "f_plus_pv": f_plus_pv,
            "eos_constants": eos_constants,
            "s_coefficients": s_coefficients,
            "cv_coefficients": cv_coefficients,
            "V0": V0,
            "G0": G0,
            "B": B,
            "BP": BP,
            "S0": S0,
            "CTE": CTE,
            "Cp": Cp,
            "H0": H0,
        }

    def plot(
        self,
        method: str,
        P: float,
        plot_type: str = "default",
        selected_temperatures_plot: np.ndarray = None,
    ) -> go.Figure:

        pressure = P

        # Retrieve the tuple output from the quasi-harmonic data
        quasi_harmonic_output = (
            self.methods[method][pressure]["f_plus_pv"],
            self.methods[method][pressure]["eos_constants"],
            self.methods[method][pressure]["s_coefficients"],
            self.methods[method][pressure]["cv_coefficients"],
            self.methods[method][pressure]["V0"],
            self.methods[method][pressure]["G0"],
            self.methods[method][pressure]["B"],
            self.methods[method][pressure]["BP"],
            self.methods[method][pressure]["S0"],
            self.methods[method][pressure]["CTE"],
            self.methods[method][pressure]["Cp"],
            self.methods[method][pressure]["H0"],
        )

        fig = plot_quasi_harmonic(
            quasi_harmonic_output=quasi_harmonic_output,
            temperatures=self.temperatures,
            volume_range=self.volumes,
            number_of_atoms=self.number_of_atoms,
            plot_type=plot_type,
            selected_temperatures_plot=selected_temperatures_plot,
        )
        return fig


