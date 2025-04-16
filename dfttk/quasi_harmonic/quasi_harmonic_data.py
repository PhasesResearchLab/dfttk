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
        volumes: np.ndarray,
        energy_eos: np.ndarray,
        f_vib_fit=None,
        s_vib_fit=None,
        cv_vib_fit=None,
        f_el_fit=None,
        s_el_fit=None,
        cv_el_fit=None,
        P: float = 0,
    ) -> None:

        self.method = method
        self.number_of_atoms = number_of_atoms
        self.temperatures = temperatures
        self.volumes = volumes
        
        (
            f_plus_pv,
            eos_constants,
            s,
            s_coefficients,
            cv,
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

        helmholtz_energy = {
            "eos_parameters": {
                "eos_name": eos,
            }
        }
        for temp, constants in zip(self.temperatures, eos_constants):
            helmholtz_energy["eos_parameters"][f"{temp}K"] = {
                "a": constants[0],
                "b": constants[1],
                "c": constants[2],
                "d": constants[3],
                "e": constants[4],
            }
        entropy = {"polynomial_coefficients": {}}
        for temp, entropy_coeff in zip(self.temperatures, s_coefficients):
            entropy["polynomial_coefficients"][f"{temp}K"] = entropy_coeff

        heat_capacity = {"polynomial_coefficients": {}}
        for temp, cv_coeff in zip(self.temperatures, cv_coefficients):
            heat_capacity["polynomial_coefficients"][f"{temp}K"] = cv_coeff

        self.methods[method][P] = {
            "helmholtz_energy": helmholtz_energy,
            "entropy": entropy,
            "heat_capacity": heat_capacity,
            "f_plus_pv": f_plus_pv,
            "eos_constants": eos_constants,
            "s": s,
            "s_coefficients": s_coefficients,
            "cv": cv,
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

        # Retrieve the tuple output from the quasi-harmonic data
        quasi_harmonic_output = (
            self.methods[method][P]["f_plus_pv"],
            self.methods[method][P]["eos_constants"],
            self.methods[method][P]["s_coefficients"],
            self.methods[method][P]["cv_coefficients"],
            self.methods[method][P]["V0"],
            self.methods[method][P]["G0"],
            self.methods[method][P]["B"],
            self.methods[method][P]["BP"],
            self.methods[method][P]["S0"],
            self.methods[method][P]["CTE"],
            self.methods[method][P]["Cp"],
            self.methods[method][P]["H0"],
        )

        fig = plot_quasi_harmonic(
            quasi_harmonic_output=quasi_harmonic_output,
            temperatures=self.temperatures,
            volumes=self.volumes,
            number_of_atoms=self.number_of_atoms,
            plot_type=plot_type,
            selected_temperatures_plot=selected_temperatures_plot,
        )
        return fig


