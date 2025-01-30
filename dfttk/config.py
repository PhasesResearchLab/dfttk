"""
Configuration class for DFTTK.
"""

# Standard Library Imports
import os
import json
import subprocess
import importlib.resources
import numpy as np

# Third-Party Library Imports
import pandas as pd
from pymongo import MongoClient
from natsort import natsorted
import plotly.graph_objects as go
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar

# DFTTK imports
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
    read_total_electron_dos,
    plot_thermal_electronic,
    plot_thermal_electronic_properties_fit,
)
from dfttk.quasi_harmonic import process_quasi_harmonic, plot_quasi_harmonic


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

    def _get_volume_folders(self):
        return natsorted([f for f in os.listdir(self.path) if f.startswith("vol_")])

    def get_vasp_input(self, volumes: list[float] = None):
        vol_folders = self._get_volume_folders()
        incar_keys = ["1relax", "2relax", "3static"]
        if volumes is not None:
            volumes = [round(volume, 2) for volume in volumes]
            filtered_vol_folders = []
            for vol_folder in vol_folders:
                contcar_path = os.path.join(self.path, vol_folder, "CONTCAR.3static")
                if os.path.exists(contcar_path):
                    structure = Structure.from_file(contcar_path)
                    if round(structure.volume, 2) in volumes:
                        filtered_vol_folders.append(vol_folder)
            vol_folders = filtered_vol_folders

        for vol_folder in vol_folders:
            incar_data = {}
            for key in incar_keys:
                file_path = os.path.join(self.path, vol_folder, f"INCAR.{key}")
                incar_data[key] = Incar.from_file(file_path)
            self.incars.append(incar_data)

        self.kpoints = Kpoints.from_file(os.path.join(self.path, "KPOINTS"))

        try:
            self.potcar = Potcar.from_file(os.path.join(self.path, "POTCAR"))
        except FileNotFoundError:
            self.potcar = None

    def get_energy_volume_data(
        self,
        volumes,
        outcar_name: str = "OUTCAR.3static",
        oszicar_name: str = "OSZICAR.3static",
        contcar_name: str = "CONTCAR.3static",
        collect_mag_data: bool = False,
        magmom_tolerance: float = 1e-12,
        total_magnetic_moment_tolerance: float = 1e-12,
        mass_average: str = "geometric",
    ) -> None:
        (
            number_of_atoms,
            all_volumes,
            all_energies,
            atomic_masses,
            average_mass,
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

        self.number_of_atoms = number_of_atoms
        self.volumes = all_volumes
        self.energies = all_energies
        self.atomic_masses = atomic_masses
        self.average_mass = average_mass
        self.mag_data = all_mag_data_list
        self.total_magnetic_moment = all_total_magnetic_moments
        self.magnetic_ordering = all_magnetic_orderings

        if volumes is not None:
            filtered_indices = [i for i, v in enumerate(all_volumes) if v in volumes]
            self.volumes = np.array(all_volumes)[filtered_indices]
            self.energies = np.array(all_energies)[filtered_indices]
            self.mag_data = np.array(all_mag_data_list)[filtered_indices]
            self.total_magnetic_moment = np.array(all_total_magnetic_moments)[
                filtered_indices
            ]
            self.magnetic_ordering = np.array(all_magnetic_orderings)[filtered_indices]

        vol_folders = self._get_volume_folders()
        if volumes is not None:
            volumes = [round(volume, 2) for volume in volumes]
            filtered_vol_folders = []
            for vol_folder in vol_folders:
                contcar_path = os.path.join(self.path, vol_folder, contcar_name)
                if os.path.exists(contcar_path):
                    structure = Structure.from_file(contcar_path)
                    if round(structure.volume, 2) in volumes:
                        filtered_vol_folders.append(vol_folder)
            vol_folders = filtered_vol_folders

        for vol_folder in vol_folders:
            contcar_path = os.path.join(self.path, vol_folder, contcar_name)
            structure = Structure.from_file(contcar_path)
            self.relaxed_structures.append(structure)

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
        a = eos_constants[0]
        b = eos_constants[1]
        c = eos_constants[2]
        d = eos_constants[3]
        e = eos_constants[4]
        V0 = eos_parameters[0]
        E0 = eos_parameters[1]
        B = eos_parameters[2]
        BP = eos_parameters[3]
        B2P = eos_parameters[4]

        self.eos_parameters = {
            "eos_name": eos_name,
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "V0": V0,
            "E0": E0,
            "B": B,
            "BP": BP,
            "B2P": B2P,
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
        marker_alpha: float = 1,
        marker_size: int = 10,
    ):
        fig = plot_ev(
            self.name,
            self.number_of_atoms,
            self.volumes,
            self.energies,
            volume_min,
            volume_max,
            num_volumes,
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
        self.debye_df = None
        self.number_of_atoms = None
        self.scaling_factor = None
        self.gruneisen_x = None
        self.temperatures = None
        self.volumes = None
        self.free_energy = None
        self.entropy = None
        self.heat_capacity = None

    def get_debye_gruneisen_data(
        self,
        number_of_atoms: int,
        volumes: np.array,
        average_mass: float,
        volume_0: float,
        bulk_modulus: float,
        bulk_modulus_prime: float,
        scaling_factor: float = 0.617,
        gruneisen_x: float = 1,
        temperatures: np.array = np.linspace(0, 1000, 101),
    ):
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
        debye_df = pd.DataFrame(
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

        self.debye_df = debye_df

    def plot(self, property, temperatures: np.array = None, volumes: np.array = None):

        fig_t, fig_v = plot_debye(
            property,
            self.number_of_atoms,
            self.temperatures,
            self.volumes,
            self.free_energy,
            self.entropy,
            self.heat_capacity,
            temperatures,
            volumes,
        )

        return fig_t, fig_v


class PhononsData:
    def __init__(self, path: str):
        self.path = path
        self.incars = []
        self.kpoints = None
        self.potcar = None
        self.phonon_structures = []
        self.number_of_atoms = None
        self.temperatures = None
        self.volumes = None
        self.helmholtz_energy = None
        self.internal_energy = None
        self.entropy = None
        self.heat_capacity = None
        self.helmholtz_energy_fit = None
        self.entropy_fit = None
        self.heat_capacity_fit = None
        self.harmonic_df = None
        self.harmonic_fit_df = None
        self.f_vib = None
        self.s_vib = None
        self.cv_vib = None
        self.f_vib_fit = None
        self.s_vib_fit = None
        self.cv_vib_fit = None
        self.volume_fit = None

    def process_phonon_dos(self):
        process_phonon_dos_YPHON(self.path)

    def _get_phonon_folders(self):
        return natsorted([f for f in os.listdir(self.path) if f.startswith("phonon_")])

    def get_vasp_input(self, volumes: list[float] = None):
        phonon_folders = self._get_phonon_folders()
        incar_keys = ["1relax", "2phonons"]

        if volumes is not None:
            volumes = [round(volume, 2) for volume in volumes]
            filtered_phonon_folders = []
            for phonon_folder in phonon_folders:
                contcar_path = os.path.join(
                    self.path, phonon_folder, "CONTCAR.2phonons"
                )
                if os.path.exists(contcar_path):
                    structure = Structure.from_file(contcar_path)
                    if round(structure.volume, 2) in volumes:
                        filtered_phonon_folders.append(phonon_folder)
            phonon_folders = filtered_phonon_folders

        for phonon_folder in phonon_folders:
            incar_data = {}
            for key in incar_keys:
                file_path = os.path.join(self.path, phonon_folder, f"INCAR.{key}")
                incar_data[key] = Incar.from_file(file_path)
            self.incars.append(incar_data)

            structure = Structure.from_file(
                os.path.join(self.path, phonon_folder, "CONTCAR.2phonons")
            )
            self.phonon_structures.append(structure)

        self.kpoints = Kpoints.from_file(
            os.path.join(self.path, phonon_folders[0], "KPOINTS.2phonons")
        )
        try:
            self.potcar = Potcar.from_file(os.path.join(self.path, "POTCAR"))
        except FileNotFoundError:
            self.potcar = None

    def get_harmonic_data(
        self,
        scale_atoms: int,
        temperatures: np.ndarray,
        order: int,
    ):

        yphon_results_path = os.path.join(self.path, "YPHON_results")
        vdos_data_scaled = scale_phonon_dos(yphon_results_path)
        volumes_per_atom = np.sort(vdos_data_scaled["volume_per_atom"].unique())
        frequency_array = []
        dos_array = []
        for volume_per_atom in volumes_per_atom:
            frequency = vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom][
                "frequency_hz"].values
            frequency_array.append(frequency)
            
            dos = vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom]["dos_1_per_hz"].values
            dos_array.append(dos)
        frequency_array = np.column_stack(frequency_array)
        dos_array = np.column_stack(dos_array)
        self.temperatures = temperatures
        (
            volumes,
            f_vib,
            e_vib,
            s_vib,
            cv_vib,
        ) = harmonic(
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
        ) = fit_harmonic(self.volumes,
                         self.temperatures,
                         self.f_vib,
                         self.s_vib,
                         self.cv_vib,
                         order)

        self.f_vib_fit = f_vib_fit
        self.s_vib_fit = s_vib_fit
        self.cv_vib_fit = cv_vib_fit
        self.volume_fit = volume_fit

        self.helmholtz_energy = {}
        self.internal_energy = {}
        self.entropy = {}
        self.heat_capacity = {}
        for i, temp in enumerate(self.temperatures):
            self.helmholtz_energy[f"{temp}K"] = self.f_vib[i]
            self.internal_energy[f"{temp}K"] = e_vib[i]
            self.entropy[f"{temp}K"] = s_vib[i]
            self.heat_capacity[f"{temp}K"] = cv_vib[i]

        self.helmholtz_energy_fit = {"polynomial_coefficients": {}}
        fvib_coefficients = [arr for arr in f_vib_poly]
        for temp, coefficients in zip(self.temperatures, fvib_coefficients):
            self.helmholtz_energy_fit["polynomial_coefficients"][
                f"{temp}K"
            ] = coefficients

        self.entropy_fit = {"polynomial_coefficients": {}}
        svib_coefficients = [arr for arr in s_vib_poly]
        for temp, coefficients in zip(self.temperatures, svib_coefficients):
            self.entropy_fit["polynomial_coefficients"][f"{temp}K"] = coefficients

        self.heat_capacity_fit = {"polynomial_coefficients": {}}
        cvib_coefficients = [arr for arr in cv_vib_poly]
        for temp, coefficients in zip(self.temperatures, cvib_coefficients):
            self.heat_capacity_fit["polynomial_coefficients"][f"{temp}K"] = coefficients

        # Temporary harmonic_fit_df for qha.
        harmonic_fit_df = pd.DataFrame(
            {
                "number_of_atoms": self.number_of_atoms,
                "temperatures": self.temperatures,
                "f_vib_poly": f_vib_poly,
                "s_vib_poly": s_vib_poly,
                "cv_vib_poly": cv_vib_poly,
            }
        )
        harmonic_fit_df = harmonic_fit_df.groupby("temperatures").agg(list)
        # Remove the outer layer of lists
        harmonic_fit_df["number_of_atoms"] = harmonic_fit_df["number_of_atoms"].apply(
            lambda x: x[0]
        )
        harmonic_fit_df["f_vib_poly"] = harmonic_fit_df["f_vib_poly"].apply(
            lambda x: x[0]
        )
        harmonic_fit_df["s_vib_poly"] = harmonic_fit_df["s_vib_poly"].apply(
            lambda x: x[0]
        )
        harmonic_fit_df["cv_vib_poly"] = harmonic_fit_df["cv_vib_poly"].apply(
            lambda x: x[0]
        )
        self.harmonic_fit_df = harmonic_fit_df

    def plot_scaled_dos(self, num_atoms: int, plot=True):
        yphon_results_path = os.path.join(self.path, "YPHON_results")
        scale_phonon_dos(yphon_results_path, num_atoms, plot)

    def plot_multiple_dos(self, num_atoms: int):
        yphon_results_path = os.path.join(self.path, "YPHON_results")
        plot_phonon_dos(yphon_results_path, num_atoms)

    def plot_harmonic(
        self, property_to_plot, selected_temperatures_plot: np.ndarray = None
    ):
        property_mapping = {
            "helmholtz_energy": "f_vib",
            "entropy": "s_vib",
            "heat_capacity": "cv_vib"
        }
        if property_to_plot in property_mapping:
            property_name = property_mapping[property_to_plot]
            property_data = getattr(self, property_name)
            property_fit_data = getattr(self, f"{property_name}_fit")

        fig_harmonic = plot_harmonic(
            self.number_of_atoms,
            self.volumes,
            self.temperatures,
            property_data,
            property_to_plot,
        )

        fig_fit_harmonic = plot_fit_harmonic(
            self.number_of_atoms,
            self.volumes,
            self.temperatures,
            property_to_plot,
            property_data,
            self.volume_fit,
            property_fit_data,
            selected_temperatures_plot,
        )
        return fig_harmonic, fig_fit_harmonic


class ThermalElectronicData:
    def __init__(self, path: str):
        self.path = path
        self.incars = []
        self.kpoints = None
        self.potcar = None
        self.electron_dos_data = None
        self.thermal_electronic_df = None
        self.thermal_electronic_fit_df = None
        self.number_of_atoms = None
        self.volumes = None
        self.temperatures = None
        self.helmholtz_energy = None
        self.internal_energy = None
        self.entropy = None
        self.heat_capacity = None
        self.helmholtz_energy_fit = None
        self.entropy_fit = None
        self.heat_capacity_fit = None

    def get_total_electron_dos(self):
        self.electron_dos_data = read_total_electron_dos(self.path)

    def _get_elec_folders(self):
        return natsorted([f for f in os.listdir(self.path) if f.startswith("elec_")])

    def get_vasp_input(self, volumes: list[float] = None):
        elec_folders = self._get_elec_folders()
        incar_keys = ["elec_dos"]

        if volumes is not None:
            volumes = [round(volume, 2) for volume in volumes]
            filtered_elec_folders = []
            for elec_folder in elec_folders:
                contcar_path = os.path.join(self.path, elec_folder, "CONTCAR.elec_dos")
                if os.path.exists(contcar_path):
                    structure = Structure.from_file(contcar_path)
                    if round(structure.volume, 2) in volumes:
                        filtered_elec_folders.append(elec_folder)
            elec_folders = filtered_elec_folders

        for elec_folder in elec_folders:
            incar_data = {}
            for key in incar_keys:
                file_path = os.path.join(self.path, elec_folder, f"INCAR.{key}")
                incar_data[key] = Incar.from_file(file_path)
            self.incars.append(incar_data)

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
        self.get_total_electron_dos()
        thermal_electronic_df, thermal_electronic_fit_df = thermal_electronic(
            self.electron_dos_data,
            temperature_range,
            order,
        )

        self.thermal_electronic_df = thermal_electronic_df
        self.thermal_electronic_fit_df = thermal_electronic_fit_df
        self.number_of_atoms = int(thermal_electronic_df["number_of_atoms"].values[0])
        self.volumes = thermal_electronic_df["volume"].unique().tolist()
        self.temperatures = thermal_electronic_df["temperature"].unique().tolist()

        self.helmholtz_energy = {}
        self.internal_energy = {}
        self.entropy = {}
        self.heat_capacity = {}
        for temp in self.temperatures:
            self.helmholtz_energy[f"{temp}K"] = thermal_electronic_df[
                thermal_electronic_df["temperature"] == temp
            ]["f_el"].values.tolist()
            self.internal_energy[f"{temp}K"] = thermal_electronic_df[
                thermal_electronic_df["temperature"] == temp
            ]["e_el"].values.tolist()
            self.entropy[f"{temp}K"] = thermal_electronic_df[
                thermal_electronic_df["temperature"] == temp
            ]["s_el"].values.tolist()
            self.heat_capacity[f"{temp}K"] = thermal_electronic_df[
                thermal_electronic_df["temperature"] == temp
            ]["cv_el"].values.tolist()

        self.helmholtz_energy_fit = {"polynomial_coefficients": {}}
        fel_coefficients = [
            arr.coeffs.tolist() for arr in self.thermal_electronic_fit_df["f_el_poly"]
        ]
        for temp, coefficients in zip(self.temperatures, fel_coefficients):
            self.helmholtz_energy_fit["polynomial_coefficients"][
                f"{temp}K"
            ] = coefficients

        self.entropy_fit = {"polynomial_coefficients": {}}
        sel_coefficients = [
            arr.coeffs.tolist() for arr in self.thermal_electronic_fit_df["s_el_poly"]
        ]
        for temp, coefficients in zip(self.temperatures, sel_coefficients):
            self.entropy_fit["polynomial_coefficients"][f"{temp}K"] = coefficients

        self.heat_capacity_fit = {"polynomial_coefficients": {}}
        cvel_coefficients = [
            arr.coeffs.tolist() for arr in self.thermal_electronic_fit_df["cv_el_poly"]
        ]
        for temp, coefficients in zip(self.temperatures, cvel_coefficients):
            self.heat_capacity_fit["polynomial_coefficients"][f"{temp}K"] = coefficients

    def plot(self, property_to_plot, selected_temperatures_plot: np.ndarray = None):
        fig = plot_thermal_electronic(self.thermal_electronic_df, property_to_plot)
        fig_fit = plot_thermal_electronic_properties_fit(
            self.thermal_electronic_fit_df,
            property_to_plot,
            selected_temperatures_plot,
        )

        return fig, fig_fit


class QuasiHarmonicData:
    def __init__(self):
        self.number_of_atoms = None
        self.temperatures = None
        self.volumes = None
        self.methods = {
            "debye": {},
            "debye + thermal_electronic": {},
            "phonons": {},
            "phonons + thermal_electronic": {},
        }

    def get_quasi_harmonic_data(
        self,
        method,
        eos: str,
        num_atoms_eos: int,
        volume_range: np.ndarray,
        eos_constants,
        harmonic_properties_fit: pd.DataFrame = None,
        debye_properties: pd.DataFrame = None,
        thermal_electronic_properties_fit: pd.DataFrame = None,
        P: float = 0,
    ) -> pd.DataFrame:

        quasi_harmonic_properties = process_quasi_harmonic(
            num_atoms_eos,
            volume_range,
            eos_constants,
            harmonic_properties_fit,
            debye_properties,
            thermal_electronic_properties_fit,
            P,
            eos,
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
    ):

        plot_quasi_harmonic(
            self.methods[method][pressure]["quasi_harmonic_df"],
            plot_type,
            selected_temperatures_plot,
        )


class Configuration:
    def __init__(self, path, name, multiplicity=None):
        self.path = path
        self.name = name
        self.multiplicity = multiplicity
        self.batch_script = {}
        self.template = None
        self.vasp_cmd = None

    def set_vasp_cmd(self, vasp_cmd: list[str]):
        self.vasp_cmd = vasp_cmd

    def read_batch_script(self, template: str):
        self.template = template
        templates_map = {
            "bridges2": "bridges2.json",
        }
        if template in templates_map:
            with importlib.resources.path(
                "dfttk.job_templates", templates_map[template]
            ) as batch_script_path:
                with open(batch_script_path, "r") as file:
                    self.batch_script = json.load(file)

    def modify_batch_script(self, key, value, position=None, action="add"):
        if key in self.batch_script and key != "commands":
            self.batch_script[key] = value
        elif key == "commands":
            if action == "add":
                if position is None:
                    self.batch_script["commands"].append(value)
                else:
                    self.batch_script["commands"].insert(position, value)
            elif action == "remove" and position is not None:
                if 0 <= position < len(self.batch_script["commands"]):
                    self.batch_script["commands"].pop(position)

    def write_batch_script(self, batch_script_file="job.sh"):
        batch_script_path = os.path.join(self.path, batch_script_file)
        if self.template == "bridges2":
            with open(batch_script_path, "w") as file:
                file.write("#!/bin/bash\n")
                file.write(f"#SBATCH --job-name={self.batch_script['job_name']}\n")
                file.write(f"#SBATCH -A {self.batch_script['account']}\n")
                file.write(f"#SBATCH -p {self.batch_script['partition']}\n")
                file.write(f"#SBATCH -N {self.batch_script['nodes']}\n")
                file.write(
                    f"#SBATCH --ntasks-per-node={self.batch_script['ntasks_per_node']}\n"
                )
                file.write(f"#SBATCH -t {self.batch_script['time']}\n")
                file.write(f"#SBATCH -o {self.batch_script['output_file']}\n")
                file.write(f"#SBATCH -e {self.batch_script['error_file']}\n\n")
                for command in self.batch_script["commands"]:
                    file.write(f"{command}\n")

    def run_volume_relax(
        self,
        material_type: str,
        encut: int = 520,
        kppa: int = 4000,
        magmom_fm: bool = False,
        potcar_functional: str = "PBE_54",
        incar_functional: str = "PBE",
        other_settings: dict = {},
    ):

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
    ):

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

    def analyze_encut_conv(self, plot: bool = True):
        encut_conv_path = os.path.join(self.path, "encut_conv")
        encut_conv_df, fig = calculate_encut_conv(encut_conv_path, plot)

        return encut_conv_df, fig

    def analyze_kpoints_conv(self, plot: bool = True):
        kpoints_conv_path = os.path.join(self.path, "kpoints_conv")
        kpoints_conv_df, fig = calculate_kpoint_conv(kpoints_conv_path, plot)

        return kpoints_conv_df, fig

    # TODO: add a way to select the custodian handlers
    def run_ev_curves(
        self,
        material_type: str,
        volumes: list[float],
        encut: int = 520,
        kppa: int = 4000,
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

        # Prepare the VASP input files
        vasp_input.ev_curve_set(
            self.path,
            material_type=material_type,
            encut=encut,
            kppa=kppa,
            potcar_functional=potcar_functional,
            incar_functional=incar_functional,
            other_settings=other_settings,
        )

        # Prepare the run_dfttk.py script
        with open(os.path.join(self.path, "run_dfttk.py"), "w") as file:
            file.write("import os\n")
            file.write("from custodian.vasp.handlers import VaspErrorHandler\n")
            file.write("import dfttk.workflows as workflows\n")
            file.write("subset = list(VaspErrorHandler.error_msgs.keys())\n")
            file.write("handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]\n")
            file.write(f"vasp_cmd = {self.vasp_cmd}\n")
            file.write(f"volumes = {volumes} \n")
            file.write(
                f"workflows.ev_curve_series(os.getcwd(), volumes, vasp_cmd, handlers, restarting={restarting}, keep_wavecar={keep_wavecar}, keep_chgcar={keep_chgcar}, copy_magmom={copy_magmom}, default_settings={default_settings}, override_2relax={override_2relax}, override_3static={override_3static}, max_errors={max_errors})\n"
            )
            file.write("workflows.custodian_errors_location(os.getcwd())\n")
            file.write("workflows.NELM_reached(os.getcwd())")

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
    ):
        self.ev_curves = EvCurvesData(self.path, self.name)
        self.ev_curves.get_vasp_input(volumes)
        self.ev_curves.get_energy_volume_data(
            volumes,
            outcar_name,
            oszicar_name,
            contcar_name,
            collect_mag_data,
            magmom_tolerance,
            total_magnetic_moment_tolerance,
            mass_average,
        )
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

    def run_phonons(
        self,
        phonon_volumes: list[float],
        kppa: float,
        run_file: str = "run_dfttk_phonons.py",
        scaling_matrix: tuple[tuple[int]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ):

        # Prepare the run_file script
        with open(os.path.join(self.path, run_file), "w") as file:
            file.write("import os\n")
            file.write("from custodian.vasp.handlers import VaspErrorHandler\n")
            file.write("import dfttk.workflows as workflows\n")
            file.write("subset = list(VaspErrorHandler.error_msgs.keys())\n")
            file.write("handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]\n")
            file.write(f"vasp_cmd = {self.vasp_cmd}\n")
            file.write(f"phonon_volumes = {phonon_volumes} \n")
            file.write(f"scaling_matrix = {scaling_matrix} \n")
            file.write(f"kppa = {kppa} \n")
            file.write(
                f"workflows.phonons_parallel(os.getcwd(), phonon_volumes, kppa, 'job.sh', scaling_matrix = scaling_matrix)\n"
            )
            file.write("workflows.custodian_errors_location(os.getcwd())\n")
            file.write("workflows.NELM_reached(os.getcwd())\n")

        # Run the phonon jobs in parallel
        subprocess.run(["python", run_file], cwd=self.path)

        # Delete the run_file script
        os.remove(os.path.join(self.path, run_file))

    def generate_phonon_dos(self):
        self.phonons = PhononsData()
        self.phonons.process_phonon_dos(self.path)

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

    def run_thermal_electronic(
        self,
        volumes: list[float],
        kppa: float,
        run_file: str = "run_dfttk_thermal_electronic.py",
        scaling_matrix: tuple[tuple[int]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ):

        # Prepare the run_file script
        with open(os.path.join(self.path, run_file), "w") as file:
            file.write("import os\n")
            file.write("from custodian.vasp.handlers import VaspErrorHandler\n")
            file.write("import dfttk.workflows as workflows\n")
            file.write("subset = list(VaspErrorHandler.error_msgs.keys())\n")
            file.write("handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]\n")
            file.write(f"vasp_cmd = {self.vasp_cmd}\n")
            file.write(f"volumes = {volumes} \n")
            file.write(f"scaling_matrix = {scaling_matrix} \n")
            file.write(f"kppa = {kppa} \n")
            file.write(
                f"workflows.elec_dos_parallel(os.getcwd(), volumes, kppa, 'job.sh')\n"
            )
            file.write("workflows.custodian_errors_location(os.getcwd())\n")
            file.write("workflows.NELM_reached(os.getcwd())\n")

        # Run the phonon jobs in parallel
        subprocess.run(["python", run_file], cwd=self.path)

        # Delete the run_file script
        os.remove(os.path.join(self.path, run_file))

    def process_thermal_electronic(
        self,
        temperature_range: np.ndarray,
        volumes: list[float] = None,
        order: int = 2,
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
        )

    def to_mongodb(self, connection_string: str, db_name: str, collection_name: str):
        self.cluster = MongoClient(connection_string)
        self.db = self.cluster[db_name]
        self.collection = self.db[collection_name]

        document = {
            "name": self.name,
            "reduced_formula": self.ev_curves.relaxed_structures[
                0
            ].composition.reduced_formula,
            "multiplicity": self.multiplicity,
        }

        if hasattr(self, "ev_curves"):
            document["ev_curves"] = {
                "vasp_input": {
                    "incars": self.ev_curves.incars,
                    "kpoints": self.ev_curves.kpoints.as_dict(),
                    "potcar": self.ev_curves.potcar.as_dict(),
                },
                "relaxed_structures": [
                    s.as_dict() for s in self.ev_curves.relaxed_structures
                ],
                "number_of_atoms": self.ev_curves.number_of_atoms,
                "volumes": self.ev_curves.volumes,
                "energies": self.ev_curves.energies,
                "total_magnetic_moment": self.ev_curves.total_magnetic_moment,
                "magnetic_ordering": self.ev_curves.magnetic_ordering,
                "mag_data": self.ev_curves.mag_data,
                "eos_parameters": self.ev_curves.eos_parameters,
            }

        if hasattr(self, "debye"):
            document["debye"] = {
                "number_of_atoms": self.debye.number_of_atoms,
                "scaling_factor": self.debye.scaling_factor,
                "gruneisen_x": self.debye.gruneisen_x,
            }

        if hasattr(self, "phonons"):
            document["phonons"] = {
                "vasp_input": {
                    "incars": self.phonons.incars,
                    "kpoints": self.phonons.kpoints.as_dict(),
                    "potcar": self.phonons.potcar.as_dict(),
                },
                "phonon_structures": [
                    s.as_dict() for s in self.phonons.phonon_structures
                ],
                "number_of_atoms": self.phonons.number_of_atoms,
                "volumes": self.phonons.volumes,
                "temperatures": self.phonons.temperatures,
                "helmholtz_energy": self.phonons.helmholtz_energy_fit,
                "entropy": self.phonons.entropy_fit,
                "heat_capacity": self.phonons.heat_capacity_fit,
            }

        if hasattr(self, "thermal_electronic"):
            document["thermal_electronic"] = {
                "vasp_input": {
                    "incars": self.thermal_electronic.incars,
                    "kpoints": self.thermal_electronic.kpoints.as_dict(),
                    "potcar": self.thermal_electronic.potcar.as_dict(),
                },
                "number_of_atoms": self.thermal_electronic.number_of_atoms,
                "volumes": self.thermal_electronic.volumes,
                "temperatures": self.thermal_electronic.temperatures,
                "helmholtz_energy": self.thermal_electronic.helmholtz_energy_fit,
                "entropy": self.thermal_electronic.entropy_fit,
                "heat_capacity": self.thermal_electronic.heat_capacity_fit,
            }

        if hasattr(self, "qha"):
            methods_copy = {
                method: {
                    str(P)
                    + " GPa": {
                        k: v for k, v in data.items() if k != "quasi_harmonic_df"
                    }
                    for P, data in pressures.items()
                }
                for method, pressures in self.qha.methods.items()
            }
            document["qha"] = {
                "number_of_atoms": self.qha.number_of_atoms,
                "volumes": self.qha.volumes,
                "temperatures": self.qha.temperatures,
                "methods": methods_copy,
            }

        if hasattr(self, "experiments"):
            document["experiments"] = self.experiments

        self.collection.insert_one(document)


def plot_multiple_ev(
    config_objects: dict[str, Configuration],
    config_names: list[str],
    volume_min: float,
    volume_max: float,
    num_volumes: int,
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
            volume_min,
            volume_max,
            num_volumes,
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
