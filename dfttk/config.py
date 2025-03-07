"""
Configuration class for DFTTK.
"""

# Standard Library Imports
import os
import json
import subprocess
import importlib.resources
from datetime import datetime
from collections import OrderedDict

# Third-Party Library Imports
import numpy as np
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
from natsort import natsorted
import plotly.graph_objects as go
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar

# DFTTK Module Imports
import dfttk.vasp_input as vasp_input
from dfttk.aggregate_extraction import (
    extract_configuration_data,
    calculate_encut_conv,
    calculate_kpoint_conv,
)
from dfttk.eos_fit import (
    fit_to_eos,
    plot_ev,
    assign_colors_to_configs,
    assign_marker_symbols_to_configs,
)
from dfttk.debye import process_debye_gruneisen, plot_debye
from dfttk.workflows import process_phonon_dos_YPHON
from dfttk.phonons import (
    harmonic,
    fit_harmonic,
    scale_phonon_dos,
    plot_phonon_dos,
    plot_harmonic,
    plot_fit_harmonic,
)
from dfttk.thermal_electronic import (
    thermal_electronic,
    fit_thermal_electronic,
    read_total_electron_dos,
    plot_total_electron_dos,
    plot_thermal_electronic,
    plot_thermal_electronic_properties_fit,
)
from dfttk.quasi_harmonic import process_quasi_harmonic, plot_quasi_harmonic


class MetaData:
    def __init__(
        self,
        affiliation: str = "DFTTK",
        mpdd_id: ObjectId = None,
        parent_database: str = None,
        parent_database_id: str = None,
        parent_database_url: str = None,
        comment: str = None,
    ):
        self.affiliation = affiliation
        self.mpdd_id = mpdd_id
        self.parent_database = parent_database
        self.parent_database_id = parent_database_id
        self.parent_database_url = parent_database_url
        self.comment = comment


class EvCurvesData:
    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name
        self.incars = []
        self.kpoints = None
        self.potcar = None
        self.number_of_atoms = None
        self.volumes = []
        self.energies = []
        self.atomic_masses = None
        self.average_mass = None
        self.total_magnetic_moment = None
        self.magnetic_ordering = None
        self.mag_data = []
        self.energy_volume_df = None
        self.eos_parameters = {
            "eos_name": None,
            "a": None,
            "b": None,
            "c": None,
            "d": None,
            "e": None,
            "V0": None,
            "E0": None,
            "B": None,
            "BP": None,
            "B2P": None,
        }
        self.relaxed_structures = []
        self.starting_poscar = None

    def _get_volume_folders(self) -> list[str]:
        return natsorted([f for f in os.listdir(self.path) if f.startswith("vol_")])

    def get_vasp_input(self, volumes: list[float] = None):
        vol_folders = self._get_volume_folders()
        incar_keys = ["1relax", "2relax", "3static"]

        if volumes is not None:
            volumes = {round(volume, 2) for volume in volumes}
            vol_folders = [
                vol_folder
                for vol_folder in vol_folders
                if os.path.exists(
                    os.path.join(self.path, vol_folder, "CONTCAR.3static")
                )
                and round(
                    Structure.from_file(
                        os.path.join(self.path, vol_folder, "CONTCAR.3static")
                    ).volume,
                    2,
                )
                in volumes
            ]

        for vol_folder in vol_folders:
            incar_data = {
                key: Incar.from_file(
                    os.path.join(self.path, vol_folder, f"INCAR.{key}")
                )
                for key in incar_keys
            }
            self.incars.append(incar_data)

        self.kpoints = Kpoints.from_file(os.path.join(self.path, "KPOINTS"))

        try:
            self.potcar = Potcar.from_file(os.path.join(self.path, "POTCAR"))
        except FileNotFoundError:
            self.potcar = None

        self.starting_poscar = Structure.from_file(os.path.join(self.path, "POSCAR"))

    def get_energy_volume_data(
        self,
        volumes: list[float] = None,
        outcar_name: str = "OUTCAR.3static",
        oszicar_name: str = "OSZICAR.3static",
        contcar_name: str = "CONTCAR.3static",
        collect_mag_data: bool = False,
        magmom_tolerance: float = 1e-12,
        total_magnetic_moment_tolerance: float = 1e-12,
        mass_average: str = "geometric",
    ) -> None:
        (
            self.number_of_atoms,
            all_volumes,
            all_energies,
            self.atomic_masses,
            self.average_mass,
            all_mag_data_list,
            all_total_magnetic_moments,
            all_magnetic_orderings,
        ) = extract_configuration_data(
            self.path,
            outcar_name,
            oszicar_name,
            contcar_name,
            collect_mag_data,
            magmom_tolerance,
            total_magnetic_moment_tolerance,
            mass_average,
        )

        self.volumes = all_volumes
        self.energies = all_energies
        self.mag_data = all_mag_data_list.tolist()
        self.total_magnetic_moment = all_total_magnetic_moments
        self.magnetic_ordering = all_magnetic_orderings

        transformed_data = [
            [{f"{item[0]}": {item[2]: item[1]}} for item in sublist]
            for sublist in self.mag_data
        ]
        transformed_data = {
            str(index): {f"{item[0]}": {item[2]: item[1]} for item in sublist}
            for index, sublist in enumerate(self.mag_data)
        }
        self.mag_data = transformed_data

        if volumes is not None:
            volumes_set = set(volumes)
            filtered_indices = [
                i for i, v in enumerate(all_volumes) if v in volumes_set
            ]
            self.volumes = np.array(all_volumes)[filtered_indices]
            self.energies = np.array(all_energies)[filtered_indices]
            self.mag_data = np.array(all_mag_data_list)[filtered_indices]
            self.total_magnetic_moment = np.array(all_total_magnetic_moments)[
                filtered_indices
            ]
            self.magnetic_ordering = np.array(all_magnetic_orderings)[filtered_indices]

        vol_folders = self._get_volume_folders()
        if volumes is not None:
            volumes_set = {round(volume, 2) for volume in volumes}
            vol_folders = [
                vol_folder
                for vol_folder in vol_folders
                if os.path.exists(os.path.join(self.path, vol_folder, contcar_name))
                and round(
                    Structure.from_file(
                        os.path.join(self.path, vol_folder, contcar_name)
                    ).volume,
                    2,
                )
                in volumes_set
            ]

        self.relaxed_structures = [
            Structure.from_file(os.path.join(self.path, vol_folder, contcar_name))
            for vol_folder in vol_folders
        ]

    def fit_energy_volume_data(
        self,
        eos_name: str = "BM4",
        volume_min: float = None,
        volume_max: float = None,
        num_volumes: int = 1000,
    ) -> None:
        eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = (
            fit_to_eos(
                self.volumes,
                self.energies,
                eos_name,
                volume_min,
                volume_max,
                num_volumes,
            )
        )

        self.eos_parameters = {
            "eos_name": eos_name,
            "a": eos_constants[0],
            "b": eos_constants[1],
            "c": eos_constants[2],
            "d": eos_constants[3],
            "e": eos_constants[4],
            "V0": eos_parameters[0],
            "E0": eos_parameters[1],
            "B": eos_parameters[2],
            "BP": eos_parameters[3],
            "B2P": eos_parameters[4],
        }

    def plot(
        self,
        volume_min: float = None,
        volume_max: float = None,
        num_volumes: int = 1000,
        eos_name: str = "BM4",
        highlight_minimum: bool = True,
        per_atom: bool = False,
        title: str = None,
        show_fig: bool = True,
        cmap: str = "plotly",
        marker_alpha: float = 1.0,
        marker_size: int = 10,
    ) -> go.Figure:
        fig = plot_ev(
            name=self.name,
            number_of_atoms=self.number_of_atoms,
            volumes=self.volumes,
            energies=self.energies,
            volume_min=volume_min,
            volume_max=volume_max,
            num_volumes=num_volumes,
            eos_name=eos_name,
            highlight_minimum=highlight_minimum,
            per_atom=per_atom,
            title=title,
            show_fig=show_fig,
            cmap=cmap,
            marker_alpha=marker_alpha,
            marker_size=marker_size,
        )

        return fig


class DebyeData:
    def __init__(self):
        self.debye_df: pd.DataFrame = None
        self.number_of_atoms: int = None
        self.scaling_factor: float = None
        self.gruneisen_x: float = None
        self.temperatures: np.ndarray = None
        self.volumes: np.ndarray = None
        self.free_energy: np.ndarray = None
        self.entropy: np.ndarray = None
        self.heat_capacity: np.ndarray = None

    def get_debye_gruneisen_data(
        self,
        number_of_atoms: int,
        volumes: np.ndarray,
        average_mass: float,
        volume_0: float,
        bulk_modulus: float,
        bulk_modulus_prime: float,
        scaling_factor: float = 0.617,
        gruneisen_x: float = 1,
        temperatures: np.ndarray = np.linspace(0, 1000, 101),
    ) -> None:
        volumes = np.linspace(0.98 * min(volumes), 1.02 * max(volumes), 1000)
        (
            number_of_atoms,
            scaling_factor,
            gruneisen_x,
            temperatures,
            volumes,
            f_vib,
            s_vib,
            cv_vib,
        ) = process_debye_gruneisen(
            number_of_atoms,
            volumes,
            average_mass,
            volume_0,
            bulk_modulus,
            bulk_modulus_prime,
            scaling_factor,
            gruneisen_x,
            temperatures,
        )

        self.number_of_atoms = number_of_atoms
        self.scaling_factor = scaling_factor
        self.gruneisen_x = gruneisen_x
        self.temperatures = temperatures
        self.volumes = volumes
        self.free_energy = f_vib
        self.entropy = s_vib
        self.heat_capacity = cv_vib

        # Temporary until I fix the quasi_harmonic module
        self.debye_df = pd.DataFrame(
            {
                "number_of_atoms": number_of_atoms,
                "temperatures": temperatures,
                "scaling_factor": [scaling_factor] * len(temperatures),
                "gruneisen_x": [gruneisen_x] * len(temperatures),
                "volume": [volumes] * len(temperatures),
                "f_vib": [col for col in f_vib],
                "s_vib": [col for col in s_vib],
                "cv_vib": [col for col in cv_vib],
            }
        )

    def plot(
        self,
        property: str,
        temperatures: np.ndarray = None,
        volumes: np.ndarray = None,
    ) -> tuple[go.Figure, go.Figure]:

        fig_t, fig_v = plot_debye(
            property_to_plot=property,
            number_of_atoms=self.number_of_atoms,
            temperatures=self.temperatures,
            volumes=self.volumes,
            f_vib=self.free_energy,
            s_vib=self.entropy,
            cv_vib=self.heat_capacity,
            selected_temperatures_plot=temperatures,
            selected_volumes=volumes,
        )

        return fig_t, fig_v


class PhononsData:
    def __init__(self, path: str):
        self.path = path
        self.incars: list[dict] = []
        self.kpoints: Kpoints = None
        self.potcar: Potcar = None
        self.phonon_structures: list[Structure] = []
        self.number_of_atoms: int = None
        self.temperatures: np.ndarray = None
        self.volumes: np.ndarray = None
        self.helmholtz_energy: np.ndarray = None
        self.internal_energy: np.ndarray = None
        self.entropy: np.ndarray = None
        self.heat_capacity: np.ndarray = None
        self.helmholtz_energy_fit: dict = None
        self.entropy_fit: dict = None
        self.heat_capacity_fit: dict = None
        self.harmonic_fit_df: pd.DataFrame = None
        self.f_vib: np.ndarray = None
        self.s_vib: np.ndarray = None
        self.cv_vib: np.ndarray = None
        self.f_vib_fit: np.ndarray = None
        self.s_vib_fit: np.ndarray = None
        self.cv_vib_fit: np.ndarray = None
        self.volume_fit: np.ndarray = None

    def process_phonon_dos(self):
        process_phonon_dos_YPHON(self.path)

    def _get_phonon_folders(self):
        return natsorted([f for f in os.listdir(self.path) if f.startswith("phonon_")])

    def get_vasp_input(self, volumes: list[float] = None):
        phonon_folders = self._get_phonon_folders()
        incar_keys = ["1relax", "2phonons"]

        if volumes is not None:
            volumes_set = {round(volume, 2) for volume in volumes}
            phonon_folders = [
                phonon_folder
                for phonon_folder in phonon_folders
                if os.path.exists(
                    os.path.join(self.path, phonon_folder, "CONTCAR.2phonons")
                )
                and round(
                    Structure.from_file(
                        os.path.join(self.path, phonon_folder, "CONTCAR.2phonons")
                    ).volume,
                    2,
                )
                in volumes_set
            ]

        for phonon_folder in phonon_folders:
            incar_data = {}
            for key in incar_keys:
                try:
                    incar_data[key] = Incar.from_file(
                        os.path.join(self.path, phonon_folder, f"INCAR.{key}")
                    )
                except FileNotFoundError:
                    if key == "1relax":
                        continue
                    else:
                        raise
            self.incars.append(incar_data)

            structure = Structure.from_file(
                os.path.join(self.path, phonon_folder, "CONTCAR.2phonons")
            )
            self.phonon_structures.append(structure)

        if phonon_folders:
            self.kpoints = Kpoints.from_file(
                os.path.join(self.path, phonon_folders[0], "KPOINTS.2phonons")
            )

        try:
            self.potcar = Potcar.from_file(os.path.join(self.path, "POTCAR"))
        except FileNotFoundError:
            self.potcar = None

    @staticmethod
    def pad_arrays(arrays, pad_value=0, pad_type="constant"):
        max_length = max(len(arr) for arr in arrays)
        padded_arrays = []
        for arr in arrays:
            if pad_type == "constant":
                padded_arr = np.pad(
                    arr, (0, max_length - len(arr)), constant_values=pad_value
                )
            elif pad_type == "increasing":
                increment = arr[-1] - arr[-2]
                pad_values = np.arange(
                    arr[-1] + increment,
                    arr[-1] + increment * (max_length - len(arr) + 1),
                    increment,
                )
                padded_arr = np.concatenate([arr, pad_values])
            padded_arrays.append(padded_arr)
        return np.column_stack(padded_arrays)

    def get_harmonic_data(
        self,
        scale_atoms: int,
        temperatures: np.ndarray,
        order: int,
    ) -> None:

        yphon_results_path = os.path.join(self.path, "YPHON_results")
        vdos_data_scaled = scale_phonon_dos(yphon_results_path)
        volumes_per_atom = np.sort(vdos_data_scaled["volume_per_atom"].unique())
        frequency_array = []
        dos_array = []

        frequency_array = [
            vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom][
                "frequency_hz"
            ].values
            for volume_per_atom in volumes_per_atom
        ]
        dos_array = [
            vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom][
                "dos_1_per_hz"
            ].values
            for volume_per_atom in volumes_per_atom
        ]

        frequency_array = self.pad_arrays(frequency_array, pad_type="increasing")
        dos_array = self.pad_arrays(dos_array, pad_value=0, pad_type="constant")

        self.temperatures = temperatures
        volumes, f_vib, e_vib, s_vib, cv_vib = harmonic(
            scale_atoms,
            volumes_per_atom,
            temperatures,
            frequency_array,
            dos_array,
        )

        self.number_of_atoms = scale_atoms
        self.volumes = volumes
        self.f_vib = f_vib
        self.s_vib = s_vib
        self.cv_vib = cv_vib

        (
            volume_fit,
            f_vib_fit,
            s_vib_fit,
            cv_vib_fit,
            f_vib_poly,
            s_vib_poly,
            cv_vib_poly,
        ) = fit_harmonic(
            self.volumes, self.temperatures, self.f_vib, self.s_vib, self.cv_vib, order
        )

        self.f_vib_fit = f_vib_fit
        self.s_vib_fit = s_vib_fit
        self.cv_vib_fit = cv_vib_fit
        self.volume_fit = volume_fit

        self.helmholtz_energy = {
            f"{temp}K": f_vib[i] for i, temp in enumerate(self.temperatures)
        }
        self.internal_energy = {
            f"{temp}K": e_vib[i] for i, temp in enumerate(self.temperatures)
        }
        self.entropy = {
            f"{temp}K": s_vib[i] for i, temp in enumerate(self.temperatures)
        }
        self.heat_capacity = {
            f"{temp}K": cv_vib[i] for i, temp in enumerate(self.temperatures)
        }

        self.helmholtz_energy_fit = {
            "polynomial_coefficients": {
                f"{temp}K": coeff for temp, coeff in zip(self.temperatures, f_vib_poly)
            }
        }
        self.entropy_fit = {
            "polynomial_coefficients": {
                f"{temp}K": coeff for temp, coeff in zip(self.temperatures, s_vib_poly)
            }
        }
        self.heat_capacity_fit = {
            "polynomial_coefficients": {
                f"{temp}K": coeff for temp, coeff in zip(self.temperatures, cv_vib_poly)
            }
        }

        # Temporary harmonic_fit_df for qha.
        harmonic_fit_df = (
            pd.DataFrame(
                {
                    "number_of_atoms": [self.number_of_atoms] * len(self.temperatures),
                    "temperatures": self.temperatures,
                    "f_vib_poly": f_vib_poly,
                    "s_vib_poly": s_vib_poly,
                    "cv_vib_poly": cv_vib_poly,
                }
            )
            .groupby("temperatures")
            .agg(list)
        )

        # Remove the outer layer of lists
        for col in ["number_of_atoms", "f_vib_poly", "s_vib_poly", "cv_vib_poly"]:
            harmonic_fit_df[col] = harmonic_fit_df[col].apply(lambda x: x[0])

        self.harmonic_fit_df = harmonic_fit_df

    def plot_scaled_dos(self, number_of_atoms: int, plot: bool = True) -> None:
        yphon_results_path = os.path.join(self.path, "YPHON_results")
        scale_phonon_dos(yphon_results_path, number_of_atoms, plot)

    def plot_multiple_dos(self, number_of_atoms: int) -> None:
        yphon_results_path = os.path.join(self.path, "YPHON_results")
        plot_phonon_dos(yphon_results_path, number_of_atoms)

    def plot_harmonic(
        self, property_to_plot: str, selected_temperatures_plot: np.ndarray = None
    ) -> tuple[go.Figure, go.Figure]:

        property_mapping = {
            "helmholtz_energy": "f_vib",
            "entropy": "s_vib",
            "heat_capacity": "cv_vib",
        }

        if property_to_plot not in property_mapping:
            raise ValueError(f"Invalid property_to_plot: {property_to_plot}")

        property_name = property_mapping[property_to_plot]
        property_data = getattr(self, property_name)
        property_fit_data = getattr(self, f"{property_name}_fit")

        fig_harmonic = plot_harmonic(
            scale_atoms=self.number_of_atoms,
            volumes=self.volumes,
            temperatures=self.temperatures,
            property=property_data,
            property_name=property_to_plot,
        )

        fig_fit_harmonic = plot_fit_harmonic(
            scale_atoms=self.number_of_atoms,
            volumes=self.volumes,
            temperatures=self.temperatures,
            property_name=property_to_plot,
            property=property_data,
            volume_fit=self.volume_fit,
            property_fit=property_fit_data,
            selected_temperatures_plot=selected_temperatures_plot,
        )
        return fig_harmonic, fig_fit_harmonic


class ThermalElectronicData:
    def __init__(self, path: str):
        self.path = path
        self.incars: list[dict] = []
        self.kpoints: Kpoints = None
        self.potcar: Potcar = None
        self.electron_dos_data: pd.DataFrame = None
        self.thermal_electronic_fit_df: pd.DataFrame = None
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

        # Temporary df for qha
        thermal_electronic_fit_df = (
            pd.DataFrame(
                {
                    "number_of_atoms": [self.number_of_atoms] * len(self.temperatures),
                    "temperatures": self.temperatures,
                    "f_el_poly": f_el_poly,
                    "s_el_poly": s_el_poly,
                    "cv_el_poly": cv_el_poly,
                }
            )
            .groupby("temperatures")
            .agg(list)
        )

        # Remove the outer layer of lists
        for col in ["number_of_atoms", "f_el_poly", "s_el_poly", "cv_el_poly"]:
            thermal_electronic_fit_df[col] = thermal_electronic_fit_df[col].apply(
                lambda x: x[0]
            )

        self.thermal_electronic_fit_df = thermal_electronic_fit_df

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


class QuasiHarmonicData:
    def __init__(self):
        self.number_of_atoms: int = None
        self.temperatures: np.ndarray = None
        self.volumes: np.ndarray = None
        self.methods: dict = {
            "debye": {},
            "debye + thermal_electronic": {},
            "phonons": {},
            "phonons + thermal_electronic": {},
        }

    def get_quasi_harmonic_data(
        self,
        method: str,
        eos: str,
        num_atoms_eos: int,
        volume_range: np.ndarray,
        eos_constants: list,
        harmonic_properties_fit: pd.DataFrame = None,
        debye_properties: pd.DataFrame = None,
        thermal_electronic_properties_fit: pd.DataFrame = None,
        P: float = 0,
        apply_smoothing: bool = False,
        smoothing_window_length: int = 21,
        smoothing_polyorder: int = 2,
    ) -> None:

        quasi_harmonic_properties = process_quasi_harmonic(
            num_atoms_eos=num_atoms_eos,
            volume_range=volume_range,
            eos_constants=eos_constants,
            apply_smoothing=apply_smoothing,
            smoothing_window_length=smoothing_window_length,
            smoothing_polyorder=smoothing_polyorder,
            harmonic_properties_fit=harmonic_properties_fit,
            debye_properties=debye_properties,
            thermal_electronic_properties_fit=thermal_electronic_properties_fit,
            P=P,
            eos=eos,
        )

        self.quasi_harmonic_df = quasi_harmonic_properties
        self.method = method
        self.pressure = quasi_harmonic_properties["pressure"].values.tolist()[0]
        self.number_of_atoms = int(
            quasi_harmonic_properties["number_of_atoms"].values.tolist()[0]
        )
        self.temperatures = quasi_harmonic_properties["temperature"].values.tolist()
        self.volumes = quasi_harmonic_properties["volume_range"].values[0].tolist()
        eos_constants = [
            arr.tolist() for arr in quasi_harmonic_properties["eos_constants"]
        ]
        s_coefficients = [
            arr.tolist() for arr in quasi_harmonic_properties["s_coefficients"]
        ]

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

        cv_coefficients = [
            arr.tolist() for arr in quasi_harmonic_properties["cv_coefficients"]
        ]
        self.heat_capacity = {"polynomial_coefficients": {}}
        for temp, cv in zip(self.temperatures, cv_coefficients):
            self.heat_capacity["polynomial_coefficients"][f"{temp}K"] = cv

        self.methods[method][P] = {
            "quasi_harmonic_df": self.quasi_harmonic_df,
            "helmholtz_energy": self.helmholtz_energy,
            "entropy": self.entropy,
            "heat_capacity": self.heat_capacity,
        }

    def plot(
        self,
        method: str,
        pressure: float,
        plot_type: str = "default",
        selected_temperatures_plot: list = None,
    ) -> go.Figure:

        fig = plot_quasi_harmonic(
            quasi_harmonic_properties=self.methods[method][pressure][
                "quasi_harmonic_df"
            ],
            plot_type=plot_type,
            selected_temperatures_plot=selected_temperatures_plot,
        )
        return fig


# Continue refactoring here!
class Configuration:
    def __init__(self, path, name, alias: str = None, multiplicity: int = None):
        self.path = path
        self.name = name
        self.alias = alias
        self.multiplicity = multiplicity
        self.job_script = {}
        self.vasp_cmd = None
        self.ev_curves_settings_data = {}
        self.phonons_settings_data = {}
        self.thermal_electronic_settings_data = {}
        self.ev_curves_job_script = {}
        self.phonons_job_script = {}
        self.thermal_electronic_job_script = {}

    def set_vasp_cmd(self, vasp_cmd: list[str]) -> None:
        self.vasp_cmd = vasp_cmd

    # TODO: At the moment this only works for the Bridges-2 template. Modify it to make it more general.
    def read_job_script(self, template: str) -> None:
        templates_map = {
            "bridges2": "bridges2.json",
        }
        if template in templates_map:
            with importlib.resources.path(
                "dfttk.job_templates", templates_map[template]
            ) as job_script_path:
                with open(job_script_path, "r") as file:
                    self.job_script = json.load(file)

    def modify_job_script(self, key, value, position=None, action="add") -> None:
        if key in self.job_script and key != "commands":
            self.job_script[key] = value
        elif key == "commands":
            if action == "add":
                if position is None:
                    self.job_script["commands"].append(value)
                else:
                    self.job_script["commands"].insert(position, value)
            elif action == "remove" and position is not None:
                if 0 <= position < len(self.job_script["commands"]):
                    self.job_script["commands"].pop(position)

    def write_job_script(self, job_script_file="job.sh") -> None:
        job_script_path = os.path.join(self.path, job_script_file)
        with open(job_script_path, "w") as file:
            file.write("#!/bin/bash\n")
            file.write(f"#SBATCH --job-name={self.job_script['job_name']}\n")
            file.write(f"#SBATCH -A {self.job_script['account']}\n")
            file.write(f"#SBATCH -p {self.job_script['partition']}\n")
            file.write(f"#SBATCH -N {self.job_script['nodes']}\n")
            file.write(
                f"#SBATCH --ntasks-per-node={self.job_script['ntasks_per_node']}\n"
            )
            file.write(f"#SBATCH -t {self.job_script['time']}\n")
            file.write(f"#SBATCH -o {self.job_script['output_file']}\n")
            file.write(f"#SBATCH -e {self.job_script['error_file']}\n\n")
            for command in self.job_script["commands"]:
                file.write(f"{command}\n")

    def add_metadata(
        self,
        mpdd_id=None,
        parent_database=None,
        parent_database_id=None,
        parent_database_url=None,
        comment=None,
        affiliation="DFTTK",
    ) -> None:

        self.metadata = MetaData(
            affiliation,
            mpdd_id,
            parent_database,
            parent_database_id,
            parent_database_url,
            comment,
        )

    def run_volume_relax(
        self,
        material_type: str,
        encut: int = 520,
        kppa: int = 4000,
        magmom_fm: bool = False,
        potcar_functional: str = "PBE_54",
        incar_functional: str = "PBE",
        other_settings: dict = {},
    ) -> None:

        vasp_input.volume_relax_set(
            self.path,
            material_type,
            encut,
            kppa,
            magmom_fm,
            potcar_functional,
            incar_functional,
            other_settings,
        )

        subprocess.run(["sbatch", "job.sh"], cwd=self.path)

    # TODO: add a way to select the custodian handlers
    def run_conv_test(
        self,
        encut: int = 520,
        kppa: int = 4000,
        magmom_fm: bool = False,
        potcar_functional: str = "PBE_54",
        incar_functional: str = "PBE",
        other_settings: dict = {},
        encut_list: list[int] = [
            270,
            320,
            370,
            420,
            470,
            520,
            570,
            620,
            670,
            720,
            770,
            820,
        ],
        kppa_list: list[float] = [
            1000,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
        ],
        force_gamma: bool = True,
        backup: bool = False,
        max_errors: int = 10,
    ) -> None:

        vasp_input.conv_set(
            self.path,
            encut,
            kppa,
            magmom_fm,
            potcar_functional,
            incar_functional,
            other_settings,
        )

        # Prepare the run_dfttk.py script
        with open(os.path.join(self.path, "run_dfttk.py"), "w") as file:
            file.write("import os\n")
            file.write("from custodian.vasp.handlers import VaspErrorHandler\n")
            file.write("import dfttk.workflows as workflows\n")
            file.write("subset = list(VaspErrorHandler.error_msgs.keys())\n")
            file.write("handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]\n")
            file.write(f"vasp_cmd = {self.vasp_cmd}\n")
            file.write(
                f"workflows.encut_conv_test(os.getcwd(), vasp_cmd, handlers, encut_list={encut_list}, backup={backup}, max_errors={max_errors})\n"
            )
            file.write(
                f"workflows.kpoints_conv_test(os.getcwd(), vasp_cmd, handlers, kppa_list={kppa_list}, force_gamma={force_gamma}, backup={backup}, max_errors={max_errors})\n"
            )

        # Run the job
        subprocess.run(["sbatch", "job.sh"], cwd=self.path)

    def analyze_encut_conv(self, plot: bool = True) -> tuple[pd.DataFrame, go.Figure]:
        encut_conv_path = os.path.join(self.path, "encut_conv")
        encut_conv_df, fig = calculate_encut_conv(encut_conv_path, plot)

        return encut_conv_df, fig

    def analyze_kpoints_conv(self, plot: bool = True) -> tuple[pd.DataFrame, go.Figure]:
        kpoints_conv_path = os.path.join(self.path, "kpoints_conv")
        kpoints_conv_df, fig = calculate_kpoint_conv(kpoints_conv_path, plot)

        return kpoints_conv_df, fig

    def ev_curves_settings(
        self,
        material_type: str,
        volumes: list[float],
        encut: int = 520,
        kppa: int = 4000,
        magmom_fm: bool = False,
        potcar_functional: str = "PBE_54",
        incar_functional: str = "PBE",
        other_settings: dict = {},
        restarting: bool = False,
        keep_wavecar: bool = False,
        keep_chgcar: bool = False,
        copy_magmom: bool = False,
        default_settings: bool = True,
        override_2relax: list = None,
        override_3static: list = None,
        max_errors: int = 10,
    ) -> None:

        self.ev_curves_settings_data = {
            "material_type": material_type,
            "volumes": volumes,
            "encut": encut,
            "kppa": kppa,
            "magmom_fm": magmom_fm,
            "potcar_functional": potcar_functional,
            "incar_functional": incar_functional,
            "other_settings": other_settings,
            "restarting": restarting,
            "keep_wavecar": keep_wavecar,
            "keep_chgcar": keep_chgcar,
            "copy_magmom": copy_magmom,
            "default_settings": default_settings,
            "override_2relax": override_2relax,
            "override_3static": override_3static,
            "max_errors": max_errors,
        }

        self.ev_curves_job_script = self.job_script

    # TODO: add a way to select the custodian handlers
    def run_ev_curves(self) -> None:

        # Ensure ev_curves_settings_data is set and not empty
        if not self.ev_curves_settings_data:
            raise AttributeError(
                "EV curves settings data not set. Please call ev_curves_settings() first."
            )

        # Prepare the VASP input files
        vasp_input.ev_curve_set(
            self.path,
            material_type=self.ev_curves_settings_data["material_type"],
            encut=self.ev_curves_settings_data["encut"],
            kppa=self.ev_curves_settings_data["kppa"],
            magmom_fm=self.ev_curves_settings_data["magmom_fm"],
            potcar_functional=self.ev_curves_settings_data["potcar_functional"],
            incar_functional=self.ev_curves_settings_data["incar_functional"],
            other_settings=self.ev_curves_settings_data["other_settings"],
        )

        # Prepare the run_dfttk.py script
        run_dfttk_script = f"""
import os
from custodian.vasp.handlers import VaspErrorHandler
import dfttk.workflows as workflows

subset = list(VaspErrorHandler.error_msgs.keys())
handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]
vasp_cmd = {self.vasp_cmd}
volumes = {self.ev_curves_settings_data["volumes"]}

workflows.ev_curve_series(
    os.getcwd(),
    volumes,
    vasp_cmd,
    handlers,
    restarting={self.ev_curves_settings_data["restarting"]},
    keep_wavecar={self.ev_curves_settings_data["keep_wavecar"]},
    keep_chgcar={self.ev_curves_settings_data["keep_chgcar"]},
    copy_magmom={self.ev_curves_settings_data["copy_magmom"]},
    default_settings={self.ev_curves_settings_data["default_settings"]},
    override_2relax={self.ev_curves_settings_data["override_2relax"]},
    override_3static={self.ev_curves_settings_data["override_3static"]},
    max_errors={self.ev_curves_settings_data["max_errors"]}
)
workflows.custodian_errors_location(os.getcwd())
workflows.NELM_reached(os.getcwd())
""".strip()

        with open(os.path.join(self.path, "run_dfttk.py"), "w") as file:
            file.write(run_dfttk_script)

        # Run the job
        subprocess.run(["sbatch", "job.sh"], cwd=self.path)

    def process_ev_curves(
        self,
        volumes: list[float] = None,
        outcar_name: str = "OUTCAR.3static",
        oszicar_name: str = "OSZICAR.3static",
        contcar_name: str = "CONTCAR.3static",
        collect_mag_data: bool = False,
        magmom_tolerance: float = 1e-12,
        total_magnetic_moment_tolerance: float = 1e-12,
        mass_average: str = "geometric",
        eos_name: str = "BM4",
        volume_min: float = None,
        volume_max: float = None,
        num_volumes: int = 1000,
    ) -> None:

        # Initialize EvCurvesData
        self.ev_curves = EvCurvesData(self.path, self.name)

        # Get VASP input
        self.ev_curves.get_vasp_input(volumes)

        # Get energy-volume data
        self.ev_curves.get_energy_volume_data(
            volumes=volumes,
            outcar_name=outcar_name,
            oszicar_name=oszicar_name,
            contcar_name=contcar_name,
            collect_mag_data=collect_mag_data,
            magmom_tolerance=magmom_tolerance,
            total_magnetic_moment_tolerance=total_magnetic_moment_tolerance,
            mass_average=mass_average,
        )

        # Fit energy-volume data
        self.ev_curves.fit_energy_volume_data(
            eos_name=eos_name,
            volume_min=volume_min,
            volume_max=volume_max,
            num_volumes=num_volumes,
        )

    def process_debye(
        self,
        scaling_factor: float = 0.617,
        gruneisen_x: float = 1,
        temperatures: np.array = np.linspace(0, 1000, 101),
    ):
        self.debye = DebyeData()

        self.debye.get_debye_gruneisen_data(
            self.ev_curves.number_of_atoms,
            self.ev_curves.volumes,
            self.ev_curves.average_mass,
            self.ev_curves.eos_parameters["V0"],
            self.ev_curves.eos_parameters["B"],
            self.ev_curves.eos_parameters["BP"],
            scaling_factor,
            gruneisen_x,
            temperatures,
        )

    def phonons_settings(
        self,
        phonon_volumes: list[float],
        kppa: float,
        scaling_matrix: tuple[tuple[int]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        copy_magmom: bool = False,
        backup: bool = False,
        max_errors: int = 10,
        relax: bool = True,
    ) -> None:

        self.phonons_settings_data = {
            "phonon_volumes": phonon_volumes,
            "kppa": kppa,
            "scaling_matrix": scaling_matrix,
            "copy_magmom": copy_magmom,
            "backup": backup,
            "max_errors": max_errors,
            "relax": relax,
        }
        self.phonons_job_script = self.job_script

    def run_phonons(self, run_file: str = "run_dfttk_phonons.py") -> None:

        # Ensure phonons_settings_data is set and not empty
        if not self.phonons_settings_data:
            raise AttributeError(
                "Phonons settings data not set. Please call phonons_settings() first."
            )

        # Prepare the run_file script
        run_dfttk_script = f"""
import os
from custodian.vasp.handlers import VaspErrorHandler
import dfttk.workflows as workflows

subset = list(VaspErrorHandler.error_msgs.keys())
handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]
vasp_cmd = {self.vasp_cmd}
copy_magmom = {self.phonons_settings_data["copy_magmom"]}
backup = {self.phonons_settings_data["backup"]}
max_errors = {self.phonons_settings_data["max_errors"]}
relax = {self.phonons_settings_data["relax"]}
phonon_volumes = {self.phonons_settings_data["phonon_volumes"]}
scaling_matrix = {self.phonons_settings_data["scaling_matrix"]}
kppa = {self.phonons_settings_data["kppa"]}

workflows.phonons_parallel(os.getcwd(), phonon_volumes, kppa, 'job.sh', scaling_matrix=scaling_matrix)
""".strip()
        with open(os.path.join(self.path, run_file), "w") as file:
            file.write(run_dfttk_script)

        # Run the phonon jobs in parallel
        subprocess.run(["python", run_file], cwd=self.path)
        os.remove(os.path.join(self.path, run_file))

    def generate_phonon_dos(self):
        self.phonons = PhononsData(self.path)
        self.phonons.process_phonon_dos()

    def process_phonons(
        self,
        scale_atoms: int,
        temperatures: np.ndarray,
        volumes: list[float] = None,
        order: int = 2,
    ):
        self.phonons = PhononsData(self.path)
        self.phonons.get_vasp_input(volumes)
        self.phonons.get_harmonic_data(
            scale_atoms,
            temperatures,
            order=order,
        )

    def thermal_electronic_settings(
        self,
        volumes: list[float],
        kppa: float,
        scaling_matrix: tuple[tuple[int]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        NEDOS: int = 10001,
        backup: bool = False,
        max_errors: int = 10,
    ) -> None:

        self.thermal_electronic_settings_data = {
            "volumes": volumes,
            "kppa": kppa,
            "scaling_matrix": scaling_matrix,
            "NEDOS": NEDOS,
            "backup": backup,
            "max_errors": max_errors,
        }
        self.thermal_electronic_job_script = self.job_script

    def run_thermal_electronic(self, run_file: str = "run_dfttk_thermal_electronic.py"):
        # Ensure thermal_electronic_settings_data is set and not empty
        if not self.thermal_electronic_settings_data:
            raise AttributeError(
                "Thermal electronic settings data not set. Please call thermal_electronic_settings() first."
            )

        # Prepare the run_file script
        run_dfttk_script = f"""
import os
from custodian.vasp.handlers import VaspErrorHandler
import dfttk.workflows as workflows

subset = list(VaspErrorHandler.error_msgs.keys())
handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]
vasp_cmd = {self.vasp_cmd}
volumes = {self.thermal_electronic_settings_data["volumes"]}
scaling_matrix = {self.thermal_electronic_settings_data["scaling_matrix"]}
kppa = {self.thermal_electronic_settings_data["kppa"]}
NEDOS = {self.thermal_electronic_settings_data["NEDOS"]}
backup = {self.thermal_electronic_settings_data["backup"]}
max_errors = {self.thermal_electronic_settings_data["max_errors"]}

workflows.elec_dos_parallel(os.getcwd(), volumes, kppa, 'job.sh', scaling_matrix=scaling_matrix)
""".strip()

        with open(os.path.join(self.path, run_file), "w") as file:
            file.write(run_dfttk_script)

        # Run the phonon jobs in parallel
        subprocess.run(["python", run_file], cwd=self.path)

        # Delete the run_file script
        os.remove(os.path.join(self.path, run_file))

    def process_thermal_electronic(
        self,
        temperature_range: np.ndarray,
        volumes: list[float] = None,
        order: int = 1,
    ):
        self.thermal_electronic = ThermalElectronicData(self.path)
        self.thermal_electronic.get_vasp_input(volumes)
        self.thermal_electronic.get_thermal_electronic_data(
            temperature_range,
            order=order,
        )

    def add_experiments(self, experiments: dict):
        self.experiments = experiments

    def process_qha(
        self,
        method: str,
        volume_range: np.ndarray,
        P: float = 0,
        apply_smoothing: bool = False,
        smoothing_window_length: int = 21,
        smoothing_polyorder: int = 2,
    ):
        if not hasattr(self, "qha"):
            self.qha = QuasiHarmonicData()

        if method == "debye":
            debye_properties = self.debye.debye_df
            harmonic_properties_fit = None
            thermal_electronic_properties_fit = None
        elif method == "debye + thermal_electronic":
            debye_properties = self.debye.debye_df
            harmonic_properties_fit = None
            thermal_electronic_properties_fit = (
                self.thermal_electronic.thermal_electronic_fit_df
            )
        elif method == "phonons":
            debye_properties = None
            harmonic_properties_fit = self.phonons.harmonic_fit_df
            thermal_electronic_properties_fit = None
        elif method == "phonons + thermal_electronic":
            debye_properties = None
            harmonic_properties_fit = self.phonons.harmonic_fit_df
            thermal_electronic_properties_fit = (
                self.thermal_electronic.thermal_electronic_fit_df
            )
        else:
            raise ValueError(f"Unknown option: {method}")

        eos = self.ev_curves.eos_parameters["eos_name"]
        eos_constants = np.array(
            [
                self.ev_curves.eos_parameters["a"],
                self.ev_curves.eos_parameters["b"],
                self.ev_curves.eos_parameters["c"],
                self.ev_curves.eos_parameters["d"],
                self.ev_curves.eos_parameters["e"],
            ]
        )
        self.qha.get_quasi_harmonic_data(
            method,
            eos,
            self.ev_curves.number_of_atoms,
            volume_range,
            eos_constants,
            harmonic_properties_fit=harmonic_properties_fit,
            debye_properties=debye_properties,
            thermal_electronic_properties_fit=thermal_electronic_properties_fit,
            P=P,
            apply_smoothing=apply_smoothing,
            smoothing_window_length=smoothing_window_length,
            smoothing_polyorder=smoothing_polyorder,
        )

    def replace_keys(self, d, key_mapping):
        if isinstance(d, dict):
            return {key_mapping.get(k, k): self.replace_keys(v, key_mapping) for k, v in d.items()}
        elif isinstance(d, list):
            return [self.replace_keys(i, key_mapping) for i in d]
        else:
            return d
                
    def to_mongodb(self, connection_string: str, db_name: str, collection_name: str):
        self.cluster = MongoClient(connection_string)
        self.db = self.cluster[db_name]
        self.collection = self.db[collection_name]

        document = {
            "metadata": {
                "MPDDId": None,
                "parentDatabase": None,
                "parentDatabaseId": None,
                "parentDatabaseURL": None,
                "affiliation": "DFTTK",
                "comment": None,
                "created": datetime.utcnow(),
                "lastModified": datetime.utcnow(),
            },
            "configuration": {
                "alias": self.alias,
                "multiplicity": self.multiplicity,
                "reducedFormula": (
                    self.ev_curves.relaxed_structures[0].composition.reduced_formula
                    if hasattr(self, "ev_curves")
                    else None
                ),
                "nComponents": (
                    len(self.ev_curves.relaxed_structures[0].composition.elements)
                    if hasattr(self, "ev_curves")
                    else None
                ),
                "numberOfAtoms": (
                    self.ev_curves.number_of_atoms
                    if hasattr(self, "ev_curves")
                    else None
                ),
            },
        }

        # Update metadata with actual values if they exist
        if hasattr(self, "metadata"):
            document["metadata"].update(
                {
                    "MPDDId": self.metadata.mpdd_id,
                    "parentDatabase": self.metadata.parent_database,
                    "parentDatabaseId": self.metadata.parent_database_id,
                    "parentDatabaseURL": self.metadata.parent_database_url,
                    "affiliation": self.metadata.affiliation,
                    "comment": self.metadata.comment,
                }
            )

        if hasattr(self, "ev_curves"):
            eos_parameters = self.ev_curves.eos_parameters.copy()
            ev_curves_settings_copy = self.ev_curves_settings_data.copy()
            ev_curves_settings_copy.pop("volumes")
            number_of_atoms = self.ev_curves.number_of_atoms

            eos_parameters_ordered = OrderedDict()
            eos_parameters_ordered["eosName"] = eos_parameters.pop("eos_name")
            eos_parameters_ordered.update(eos_parameters)

            document["evCurves"] = {
                "input": {
                    "jobScript": self.ev_curves_job_script,
                    "settings": ev_curves_settings_copy,
                    "poscar": self.ev_curves.starting_poscar.as_dict(),
                    "incars": self.ev_curves.incars,
                    "kpoints": self.ev_curves.kpoints.as_dict(),
                    "potcar": self.ev_curves.potcar.as_dict(),
                },
                "output": {
                    "scaleAtoms": number_of_atoms,
                    "volumes": self.ev_curves.volumes.tolist(),
                    "energies": self.ev_curves.energies.tolist(),
                    "relaxedStructures": [
                        s.as_dict() for s in self.ev_curves.relaxed_structures
                    ],
                    "totalMagneticMoments": (
                        self.ev_curves.total_magnetic_moment
                        if isinstance(self.ev_curves.total_magnetic_moment, list)
                        else self.ev_curves.total_magnetic_moment.tolist()
                    ),
                    "magneticOrderings": (
                        self.ev_curves.magnetic_ordering
                        if isinstance(self.ev_curves.magnetic_ordering, list)
                        else self.ev_curves.magnetic_ordering.tolist()
                    ),
                    "magData": self.ev_curves.mag_data,
                    "eosParameters": eos_parameters_ordered,
                },
            }

        if hasattr(self, "debye"):
            document["debye"] = {
                "scalingFactor": self.debye.scaling_factor,
                "gruneisenX": self.debye.gruneisen_x,
            }

        # Function to recursively convert numpy.poly1d objects to lists of coefficients
        def convert_poly1d(obj):
            if isinstance(obj, np.poly1d):
                return obj.coefficients.tolist()
            elif isinstance(obj, dict):
                return {k: convert_poly1d(v) for k, v in obj.items()}
            return obj

        if hasattr(self, "phonons"):
            phonons_settings_copy = self.phonons_settings_data.copy()
            phonons_settings_copy.pop("phonon_volumes")
            phonons_settings_copy.pop("relax")
            
            temperatures = self.phonons.temperatures
            min_temperature = min(temperatures)
            max_temperature = max(temperatures)
            num_temperatures = len(temperatures)
            
            helmholtz_energy = convert_poly1d(self.phonons.helmholtz_energy_fit)
            entropy = convert_poly1d(self.phonons.entropy_fit)
            heat_capacity = convert_poly1d(self.phonons.heat_capacity_fit)
            
            key_mapping = {
                "polynomial_coefficients": "polynomialCoefficients"
            }
            helmholtz_energy = self.replace_keys(helmholtz_energy, key_mapping)
            entropy = self.replace_keys(entropy, key_mapping)
            heat_capacity = self.replace_keys(heat_capacity, key_mapping)
            
            document["phonons"] = {
                "input": {
                    "jobScript": self.phonons_job_script,
                    "settings": phonons_settings_copy,
                    "incars": self.phonons.incars,
                    "kpoints": self.phonons.kpoints.as_dict(),
                    "potcar": self.phonons.potcar.as_dict(),
                },
                "output": {
                    "scaleAtoms": self.phonons.number_of_atoms,
                    "volumes": self.phonons.volumes.tolist(),
                    "phononStructures": [
                        s.as_dict() for s in self.phonons.phonon_structures
                    ],
                    "temperatures": {
                        "min": min_temperature.tolist(),
                        "max": max_temperature.tolist(),
                        "number": num_temperatures,
                    },
                    "helmholtzEnergy": helmholtz_energy,
                    "entropy": entropy,
                    "heatCapacity": heat_capacity,
                },
            }

        if hasattr(self, "thermal_electronic"):
            thermal_electronic_settings_copy = (
                self.thermal_electronic_settings_data.copy()
            )
            
            temperatures = self.thermal_electronic.temperatures
            min_temperature = min(temperatures)
            max_temperature = max(temperatures)
            num_temperatures = len(temperatures)
            
            helmholtz_energy = convert_poly1d(self.thermal_electronic.helmholtz_energy_fit)
            entropy = convert_poly1d(self.thermal_electronic.entropy_fit)
            heat_capacity = convert_poly1d(self.thermal_electronic.heat_capacity_fit)
            
            key_mapping = {
                "polynomial_coefficients": "polynomialCoefficients"
            }
            helmholtz_energy = self.replace_keys(helmholtz_energy, key_mapping)
            entropy = self.replace_keys(entropy, key_mapping)
            heat_capacity = self.replace_keys(heat_capacity, key_mapping)
            
            thermal_electronic_settings_copy.pop("volumes")
            document["thermalElectronic"] = {
                "input": {
                    "jobScript": self.thermal_electronic_job_script,
                    "settings": thermal_electronic_settings_copy,
                    "incars": self.thermal_electronic.incars,
                    "kpoints": self.thermal_electronic.kpoints.as_dict(),
                    "potcar": self.thermal_electronic.potcar.as_dict(),
                },
                "output": {
                    "scaleAtoms": self.thermal_electronic.number_of_atoms.tolist(),
                    "volumes": [
                        round(s.volume, 2) for s in self.thermal_electronic.structures
                    ],
                    "structures": [
                        s.as_dict() for s in self.thermal_electronic.structures
                    ],
                    "temperatures": {
                        "min": min_temperature.tolist(),
                        "max": max_temperature.tolist(),
                        "number": num_temperatures,
                    },
                    "helmholtzEnergy": helmholtz_energy,
                    "entropy": entropy,
                    "heatCapacity": heat_capacity,
                },
            }

        if hasattr(self, "qha"):
            key_mapping = {
                "debye": "debye",
                "debye + thermal_electronic": "debyeThermalElectronic",
                "phonons": "phonons",
                "phonons + thermal_electronic": "phononsThermalElectronic",
                "helmholtz_energy": "helmholtzEnergy",
                "heat_capacity": "heatCapacity",
                "eos_parameters": "eosParameters",
                "eos_name": "eosName",
                "polynomial_coefficients": "polynomialCoefficients"
            }
            methods_copy = {
                key_mapping.get(method, method): {
                    str(P) + " GPa": {k: v for k, v in data.items() if k != "quasi_harmonic_df"}
                    for P, data in pressures.items()
                }
                for method, pressures in self.qha.methods.items()
            }

            methods_copy = self.replace_keys(methods_copy, key_mapping)
            
            volumes = self.qha.volumes
            min_volume = min(volumes)
            max_volume = max(volumes)
            num_volumes = len(volumes)

            temperatures = self.qha.temperatures
            min_temperature = min(temperatures)
            max_temperature = max(temperatures)
            num_temperatures = len(temperatures)
            
            document["qha"] = {
                "scaleAtoms": self.qha.number_of_atoms,
                "volumes": {
                    "min": min_volume,
                    "max": max_volume,
                    "number": num_volumes,
                },
                "temperatures": {
                    "min": min_temperature,
                    "max": max_temperature,
                    "number": num_temperatures,
                },
                "methods": methods_copy,
            }

        if hasattr(self, "experiments"):
            document["experiments"] = self.experiments

        # Determine the field to use for comparison
        comparison_field = "metadata.parentDatabaseId"
        comparison_value = document["metadata"]["parentDatabaseId"]

        if comparison_value is None:
            comparison_field = "metadata.comment"
            comparison_value = document["metadata"]["comment"]

        # Check if a document with the same comparison field exists
        existing_doc = self.collection.find_one({comparison_field: comparison_value})

        if existing_doc:
            # Preserve the original "created" field while updating the rest
            document["metadata"]["created"] = existing_doc["metadata"]["created"]
            self.collection.update_one({"_id": existing_doc["_id"]}, {"$set": document})
        else:
            # Add the "created" timestamp for new documents
            document["metadata"]["created"] = datetime.utcnow()
            self.collection.insert_one(document)

        # Close the MongoDB connection
        self.cluster.close()


def plot_multiple_ev(
    config_objects: dict[str, Configuration],
    config_names: list[str],
    volume_min: float = None,
    volume_max: float = None,
    num_volumes: int = 1000,
    eos_name: str = "BM4",
    highlight_minimum: bool = True,
    per_atom: bool = False,
    title: str = None,
    show_fig: bool = False,
    cmap: str = "plotly",
    marker_alpha: float = 1,
    marker_size: int = 10,
):

    combined_fig = go.Figure()

    config_colors = assign_colors_to_configs(
        config_names, alpha=marker_alpha, cmap=cmap
    )
    config_symbols = assign_marker_symbols_to_configs(config_names)

    for config_name in config_names:
        fig = config_objects[config_name].ev_curves.plot(
            volume_min=volume_min,
            volume_max=volume_max,
            num_volumes=num_volumes,
            eos_name=eos_name,
            highlight_minimum=highlight_minimum,
            per_atom=per_atom,
            cmap=cmap,
            marker_alpha=marker_alpha,
            marker_size=marker_size,
            show_fig=False,
        )
        if fig is not None:
            for trace in fig.data:
                if trace.name != "minimum":
                    trace.marker.color = config_colors[config_name]
                    trace.line.color = config_colors[config_name]
                    trace.marker.symbol = config_symbols[config_name]
                combined_fig.add_trace(trace)
            combined_fig.update_layout(fig.layout)

    if title:
        combined_fig.update_layout(title=title)

    if show_fig:
        combined_fig.show()

    return combined_fig
