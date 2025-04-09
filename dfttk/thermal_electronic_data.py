"""
This module contains the ThermalElectronicData class, which is used to process and analyze thermal electronic data from VASP calculations.
"""

# Standard Library Imports
import os
import numpy as np

# Related third party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from natsort import natsorted


# Local application/library specific imports
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar

# DFTTK imports
from dfttk.thermal_electronic_functions import (
    thermal_electronic,
    fit_thermal_electronic,
    read_total_electron_dos,
    plot_total_electron_dos,
    plot_thermal_electronic,
    plot_thermal_electronic_properties_fit,
)


class ThermalElectronicData:
    def __init__(self, path: str):
        self.path = path
        self.incars: list[dict] = []
        self.kpoints: Kpoints = None
        self.potcar: Potcar = None
        self.electron_dos_data: pd.DataFrame = None
        self.number_of_atoms: int = None
        self.volumes: np.ndarray = None
        self.temperatures: np.ndarray = None
        self.helmholtz_energy: np.ndarray = None
        self.internal_energy: np.ndarray = None
        self.entropy: np.ndarray = None
        self.heat_capacity: np.ndarray = None
        self.helmholtz_energy_fit: dict = None
        self.entropy_fit: dict = None
        self.heat_capacity_fit: dict = None
        self.structures: list[Structure] = []

    def get_total_electron_dos(self) -> None:
        self.electron_dos_data = read_total_electron_dos(self.path)

    def _get_elec_folders(self) -> list[str]:
        return natsorted([f for f in os.listdir(self.path) if f.startswith("elec_")])

    def get_vasp_input(self, volumes: list[float] = None) -> None:
        elec_folders = self._get_elec_folders()
        incar_keys = ["elec_dos"]

        if volumes is not None:
            volumes_set = {round(volume, 2) for volume in volumes}
            elec_folders = [
                elec_folder
                for elec_folder in elec_folders
                if os.path.exists(
                    os.path.join(self.path, elec_folder, "CONTCAR.elec_dos")
                )
                and round(
                    Structure.from_file(
                        os.path.join(self.path, elec_folder, "CONTCAR.elec_dos")
                    ).volume,
                    2,
                )
                in volumes_set
            ]

        for elec_folder in elec_folders:
            incar_data = {
                key: Incar.from_file(
                    os.path.join(self.path, elec_folder, f"INCAR.{key}")
                )
                for key in incar_keys
            }
            self.incars.append(incar_data)

        if elec_folders:
            self.kpoints = Kpoints.from_file(
                os.path.join(self.path, elec_folders[0], "KPOINTS.elec_dos")
            )

        try:
            self.potcar = Potcar.from_file(os.path.join(self.path, "POTCAR"))
        except FileNotFoundError:
            self.potcar = None

    def get_thermal_electronic_data(
        self,
        temperature_range: np.ndarray,
        order: int,
    ):
        self.structures = [
            Structure.from_file(
                os.path.join(self.path, elec_folder, "CONTCAR.elec_dos")
            )
            for elec_folder in self._get_elec_folders()
        ]

        self.get_total_electron_dos()
        volumes = self.electron_dos_data["volume"].unique()
        self.number_of_atoms = self.electron_dos_data["number_of_atoms"].unique()[0]

        energy_array = [
            self.electron_dos_data[self.electron_dos_data["volume"] == volume][
                "energy_minus_fermi_energy"
            ].values[0]
            for volume in volumes
        ]
        dos_array = [
            self.electron_dos_data[self.electron_dos_data["volume"] == volume][
                "total_dos"
            ].values[0]
            for volume in volumes
        ]

        energy_array = np.column_stack(energy_array)
        dos_array = np.column_stack(dos_array)

        f_el, e_el, s_el, cv_el = thermal_electronic(
            volumes,
            temperature_range,
            energy_array,
            dos_array,
        )
        self.f_el = f_el
        self.e_el = e_el
        self.s_el = s_el
        self.cv_el = cv_el
        self.volumes = volumes
        self.temperatures = temperature_range

        volume_fit, f_el_fit, s_el_fit, cv_el_fit, f_el_poly, s_el_poly, cv_el_poly = (
            fit_thermal_electronic(
                self.volumes, self.temperatures, f_el, s_el, cv_el, order
            )
        )

        self.volume_fit = volume_fit
        self.f_el_fit = f_el_fit
        self.s_el_fit = s_el_fit
        self.cv_el_fit = cv_el_fit
        self.f_el_poly = f_el_poly
        self.s_el_poly = s_el_poly
        self.cv_el_poly = cv_el_poly

        self.helmholtz_energy = {}
        self.internal_energy = {}
        self.entropy = {}
        self.heat_capacity = {}

        self.helmholtz_energy = {
            f"{temp}K": f_el[i] for i, temp in enumerate(self.temperatures)
        }
        self.internal_energy = {
            f"{temp}K": e_el[i] for i, temp in enumerate(self.temperatures)
        }
        self.entropy = {f"{temp}K": s_el[i] for i, temp in enumerate(self.temperatures)}
        self.heat_capacity = {
            f"{temp}K": cv_el[i] for i, temp in enumerate(self.temperatures)
        }

        self.helmholtz_energy_fit = {
            "polynomial_coefficients": {
                f"{temp}K": coeff for temp, coeff in zip(self.temperatures, f_el_poly)
            }
        }
        self.entropy_fit = {
            "polynomial_coefficients": {
                f"{temp}K": coeff for temp, coeff in zip(self.temperatures, s_el_poly)
            }
        }
        self.heat_capacity_fit = {
            "polynomial_coefficients": {
                f"{temp}K": coeff for temp, coeff in zip(self.temperatures, cv_el_poly)
            }
        }

    def plot(
        self, property_to_plot: str, selected_temperatures_plot: np.ndarray = None
    ) -> tuple[go.Figure, go.Figure]:

        property_mapping = {
            "helmholtz_energy": "f_el",
            "entropy": "s_el",
            "heat_capacity": "cv_el",
        }

        if property_to_plot not in property_mapping:
            raise ValueError(f"Invalid property_to_plot: {property_to_plot}")

        property_name = property_mapping[property_to_plot]
        property_data = getattr(self, property_name)
        property_fit_data = getattr(self, f"{property_name}_fit")

        fig = plot_thermal_electronic(
            number_of_atoms=self.number_of_atoms,
            volumes=self.volumes,
            temperatures=self.temperatures,
            property=property_data,
            property_name=property_to_plot,
        )

        fig_fit = plot_thermal_electronic_properties_fit(
            number_of_atoms=self.number_of_atoms,
            volumes=self.volumes,
            temperatures=self.temperatures,
            property_name=property_to_plot,
            property=property_data,
            volume_fit=self.volume_fit,
            property_fit=property_fit_data,
            selected_temperatures_plot=selected_temperatures_plot,
        )

        return fig, fig_fit

    def plot_electron_dos(self):
        fig = plot_total_electron_dos(self.electron_dos_data)
        return fig

