"""
Configuration class for DFTTK.
"""

# Standard Library Imports
import os
import json
import subprocess
import importlib.resources
import numpy as np
from collections import namedtuple

# Third-Party Library Imports
import pandas as pd
from pymongo import MongoClient
from natsort import natsorted
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
    mBM4_equation,
    mBM5_equation,
    BM4_equation,
    BM5_equation,
    LOG4_equation,
    LOG5_equation,
    murnaghan_equation,
    vinet_equation,
    morse_equation,
    fit_to_all_eos,
    plot_ev,
)
from dfttk.debye import process_debye_gruneisen, plot_debye
from dfttk.workflows import process_phonon_dos_YPHON
from dfttk.phonons import (
    harmonic,
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
    def __init__(self, path: str):
        self.path = path
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
        self.eos_parameters_df = None  # Temporary
        self.relaxed_structures = []
        self.volumes = None
        self.eos_energies = None

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
        self.potcar = Potcar.from_file(os.path.join(self.path, "POTCAR"))

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
        self.energy_volume_df = extract_configuration_data(
            self.path,
            outcar_name,
            oszicar_name,
            contcar_name,
            collect_mag_data,
            magmom_tolerance,
            total_magnetic_moment_tolerance,
            mass_average,
        )

        if volumes is not None:
            volumes = [round(volume, 2) for volume in volumes]
            self.energy_volume_df["volume"] = self.energy_volume_df["volume"].round(2)
            self.energy_volume_df = self.energy_volume_df[
                self.energy_volume_df["volume"].isin(volumes)
            ]

        self.number_of_atoms = self.energy_volume_df["number_of_atoms"].values.tolist()[
            0
        ]
        self.volumes = self.energy_volume_df["volume"].values.tolist()
        self.energies = self.energy_volume_df["energy"].values.tolist()
        self.atomic_masses = self.energy_volume_df["atomic_masses"].values[0]
        self.average_mass = self.energy_volume_df["average_mass"].values[0]

        if collect_mag_data:
            self.total_magnetic_moment = self.energy_volume_df[
                "total_magnetic_moment"
            ].values.tolist()
            self.magnetic_ordering = self.energy_volume_df[
                "magnetic_ordering"
            ].values.tolist()
            self.mag_data = [
                series.astype({"#_of_ion": str}).set_index("#_of_ion").to_dict()
                for series in self.energy_volume_df["mag_data"].values.tolist()
            ]
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
        __, eos_parameters_df = fit_to_all_eos(
            self.energy_volume_df, volume_min, volume_max, num_volumes
        )
        one_eos_parameters_df = eos_parameters_df[eos_parameters_df["eos"] == eos_name]
        a = one_eos_parameters_df["a"].values[0]
        b = one_eos_parameters_df["b"].values[0]
        c = one_eos_parameters_df["c"].values[0]
        d = one_eos_parameters_df["d"].values[0]
        e = one_eos_parameters_df["e"].values[0]
        V0 = one_eos_parameters_df["V0"].values[0]
        E0 = one_eos_parameters_df["E0"].values[0]
        B = one_eos_parameters_df["B"].values[0]
        BP = one_eos_parameters_df["BP"].values[0]
        B2P = one_eos_parameters_df["B2P"].values[0]
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
        self.eos_parameters_df = eos_parameters_df

    # TODO: this method might not be necessary
    def get_eos_energies(
        self, volume_min: float, volume_max: float, num_volumes: int = 1000
    ) -> np.ndarray:
        if not hasattr(self, "eos_parameters"):
            raise ValueError(
                "EOS parameters not found. Please run fit_energy_volume_data first."
            )

        volume_range = np.linspace(volume_min, volume_max, num_volumes)
        eos_name = self.eos_parameters["eos_name"]
        a = self.eos_parameters["a"]
        b = self.eos_parameters["b"]
        c = self.eos_parameters["c"]
        d = self.eos_parameters["d"]
        e = self.eos_parameters["e"]
        V0 = self.eos_parameters["V0"]
        E0 = self.eos_parameters["E0"]
        B = self.eos_parameters["B"]
        BP = self.eos_parameters["BP"]

        eos_equations = {
            "mBM4": mBM4_equation,
            "mBM5": mBM5_equation,
            "BM4": BM4_equation,
            "BM5": BM5_equation,
            "LOG4": LOG4_equation,
            "LOG5": LOG5_equation,
            "murnaghan": murnaghan_equation,
            "vinet": vinet_equation,
            "morse": morse_equation,
        }

        if eos_name == "mBM4" or eos_name == "BM4" or eos_name == "LOG4":
            eos_energies = eos_equations[eos_name](volume_range, a, b, c, d)
        elif eos_name == "mBM5" or eos_name == "BM5" or eos_name == "LOG5":
            eos_energies = eos_equations[eos_name](volume_range, a, b, c, d, e)
        elif eos_name == "murnaghan" or eos_name == "vinet" or eos_name == "morse":
            eos_energies = eos_equations[eos_name](volume_range, V0, E0, B, BP)

        self.volumes = volume_range
        self.eos_energies = eos_energies

    def plot(
        self,
        eos_name: str = "BM4",
        highlight_minimum: bool = True,
        per_atom: bool = False,
        title: str = None,
        show_fig: bool = True,
        cmap: str = "plotly",
        marker_alpha: float = 1,
        marker_size: int = 10,
    ) -> None:
        plot_ev(
            self.energy_volume_df,
            eos_name=eos_name,
            highlight_minimum=highlight_minimum,
            per_atom=per_atom,
            title=title,
            show_fig=show_fig,
            cmap=cmap,
            marker_alpha=marker_alpha,
            marker_size=marker_size,
        )


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
        energy_volume_df: pd.DataFrame,
        eos_parameters_df: pd.DataFrame,
        scaling_factor: float = 0.617,
        gruneisen_x: float = 1,
        volumes: np.array = None,
        temperatures: np.array = np.linspace(0, 1000, 101),
        eos: str = "BM4",
        plot=None,
        selected_temperatures_plot: np.array = None,
    ):

        debye_df = process_debye_gruneisen(
            energy_volume_df,
            eos_parameters_df,
            scaling_factor,
            gruneisen_x,
            volumes,
            temperatures,
            eos,
            plot,
            selected_temperatures_plot,
        )
        self.debye_df = debye_df

        self.number_of_atoms = int(debye_df["number_of_atoms"].values.tolist()[0])
        self.scaling_factor = scaling_factor
        self.gruneisen_x = gruneisen_x
        self.temperatures = debye_df["temperatures"].values.tolist()
        self.volumes = debye_df["volume"][0].tolist()
        self.free_energy = debye_df["f_vib"].apply(lambda x: x.tolist()).tolist()
        self.entropy = debye_df["s_vib"].apply(lambda x: x.tolist()).tolist()
        self.heat_capacity = debye_df["cv_vib"].apply(lambda x: x.tolist()).tolist()

    def plot(
        self, selected_temperatures: np.array = None, selected_volumes: np.array = None
    ):

        config = self.debye_df["config"].values.tolist()[0]
        plot_debye(
            config,
            self.debye_df,
            selected_temperatures,
            selected_volumes,
        )


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
        self.internal_energy_fit = None
        self.entropy_fit = None
        self.harmonic_df = None
        self.harmonic_fit_df = None

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
        self.potcar = Potcar.from_file(os.path.join(self.path, "POTCAR"))

    def get_harmonic_data(
        self,
        scale_atoms: int,
        temp_range: list,
        order: int,
        plot: bool,
        selected_temperatures_plot: np.ndarray,
    ):

        yphon_results_path = os.path.join(self.path, "YPHON_results")
        harmonic_df, harmonic_fit_df = harmonic(
            yphon_results_path,
            scale_atoms,
            temp_range,
            order,
            plot,
            selected_temperatures_plot,
        )

        self.harmonic_df = harmonic_df
        self.harmonic_fit_df = harmonic_fit_df

        self.number_of_atoms = int(harmonic_df["number_of_atoms"].values[0])
        self.temperatures = harmonic_df["temperature"].unique().tolist()
        self.volumes = harmonic_df["volume"].unique().tolist()

        self.helmholtz_energy = {}
        self.internal_energy = {}
        self.entropy = {}
        self.heat_capacity = {}
        for temp in self.temperatures:
            self.helmholtz_energy[f"{temp}K"] = self.harmonic_df[
                self.harmonic_df["temperature"] == temp
            ]["f_vib"].values.tolist()
            self.internal_energy[f"{temp}K"] = self.harmonic_df[
                self.harmonic_df["temperature"] == temp
            ]["e_vib"].values.tolist()
            self.entropy[f"{temp}K"] = self.harmonic_df[
                self.harmonic_df["temperature"] == temp
            ]["s_vib"].values.tolist()
            self.heat_capacity[f"{temp}K"] = self.harmonic_df[
                self.harmonic_df["temperature"] == temp
            ]["cv_vib"].values.tolist()

        self.helmholtz_energy_fit = {"polynomial_coefficients": {}}
        fvib_coefficients = [
            arr.coeffs.tolist() for arr in self.harmonic_fit_df["f_vib_poly"]
        ]
        for temp, coefficients in zip(self.temperatures, fvib_coefficients):
            self.helmholtz_energy_fit["polynomial_coefficients"][
                f"{temp}K"
            ] = coefficients

        self.entropy_fit = {"polynomial_coefficients": {}}
        svib_coefficients = [
            arr.coeffs.tolist() for arr in self.harmonic_fit_df["s_vib_poly"]
        ]
        for temp, coefficients in zip(self.temperatures, svib_coefficients):
            self.entropy_fit["polynomial_coefficients"][f"{temp}K"] = coefficients

        self.heat_capacity_fit = {"polynomial_coefficients": {}}
        cvib_coefficients = [
            arr.coeffs.tolist() for arr in self.harmonic_fit_df["cv_vib_poly"]
        ]
        for temp, coefficients in zip(self.temperatures, cvib_coefficients):
            self.heat_capacity_fit["polynomial_coefficients"][f"{temp}K"] = coefficients

    def plot_scaled_dos(self, num_atoms: int, plot=True):
        yphon_results_path = os.path.join(self.path, "YPHON_results")
        scale_phonon_dos(yphon_results_path, num_atoms, plot)

    def plot_multiple_dos(self, num_atoms: int):
        yphon_results_path = os.path.join(self.path, "YPHON_results")
        plot_phonon_dos(yphon_results_path, num_atoms)

    def plot_harmonic(self, selected_temperatures_plot: np.ndarray = None):
        plot_harmonic(self.harmonic_df)
        plot_fit_harmonic(self.harmonic_fit_df, selected_temperatures_plot)


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
                contcar_path = os.path.join(
                    self.path, elec_folder, "CONTCAR.elec_dos"
                )
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
        self.potcar = Potcar.from_file(os.path.join(self.path, "POTCAR"))
        
    def get_thermal_electronic_data(
        self,
        temperature_range: np.ndarray,
        order: int,
        plot: bool,
        selected_temperatures_plot: np.ndarray,
    ):
        self.get_total_electron_dos()
        thermal_electronic_df, thermal_electronic_fit_df = thermal_electronic(
            self.electron_dos_data,
            temperature_range,
            order,
            plot,
            selected_temperatures_plot,
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

    def plot(self, selected_temperatures_plot: np.ndarray = None):
        plot_thermal_electronic(
            self.thermal_electronic_df
        )
        plot_thermal_electronic_properties_fit(
            self.thermal_electronic_fit_df,
            selected_temperatures_plot,
        )

# TODO: incorporate other pressures
class QuasiHarmonicData:
    def get_quasi_harmonic_data(
        self,
        method,
        eos: str,
        volume_range: np.ndarray,
        eos_parameters_df: pd.DataFrame,
        harmonic_properties_fit: pd.DataFrame = None,
        debye_properties: pd.DataFrame = None,
        thermal_electronic_properties_fit: pd.DataFrame = None,
        P: int = 0,
        plot: bool = False,
        plot_type: str = "default",
        selected_temperatures_plot: list = None,
    ) -> pd.DataFrame:

        quasi_harmonic_properties = process_quasi_harmonic(
            volume_range,
            eos_parameters_df,
            harmonic_properties_fit,
            debye_properties,
            thermal_electronic_properties_fit,
            P,
            eos,
            plot,
            plot_type,
            selected_temperatures_plot,
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

    def plot(self, plot_type: str = "default", selected_temperatures_plot: list = None):
        plot_quasi_harmonic(
            self.quasi_harmonic_df,
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
    ):
        self.ev_curves = EvCurvesData(self.path)
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
        self.ev_curves.fit_energy_volume_data(eos_name=eos_name)

    def process_debye(
        self,
        scaling_factor: float = 0.617,
        gruneisen_x: float = 1,
    ):
        energy_volume_df = self.ev_curves.energy_volume_df
        eos_parameters_df = self.ev_curves.eos_parameters_df
        eos = self.ev_curves.eos_parameters["eos_name"]
        self.debye = DebyeData()
        self.debye.get_debye_gruneisen_data(
            energy_volume_df,
            eos_parameters_df,
            scaling_factor=scaling_factor,
            gruneisen_x=gruneisen_x,
            eos=eos,
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
        temp_range: list,
        volumes: list[float] = None,
        order: int = 2,
        plot: bool = False,
        selected_temperatures_plot: np.ndarray = None,
    ):
        self.phonons = PhononsData(self.path)
        self.phonons.get_vasp_input(volumes)
        self.phonons.get_harmonic_data(
            scale_atoms,
            temp_range,
            order=order,
            plot=plot,
            selected_temperatures_plot=selected_temperatures_plot,
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
        plot: bool = False,
        selected_temperatures_plot: np.ndarray = None,
    ):
        self.thermal_electronic = ThermalElectronicData(self.path)
        self.thermal_electronic.get_vasp_input(volumes)
        self.thermal_electronic.get_thermal_electronic_data(
            temperature_range,
            order=order,
            plot=plot,
            selected_temperatures_plot=selected_temperatures_plot,
        )

    def process_qha(
        self,
        method: str,
        volume_range: np.ndarray,
        P: int = 0,
        plot: bool = False,
        plot_type: str = "default",
        selected_temperatures_plot: list = None,
    ):
        self.qha = QuasiHarmonicData()

        if method == "debye":
            debye_properties = self.debye.debye_df
            harmonic_properties_fit = None
            thermal_electronic_properties_fit = None
        else:
            raise ValueError(f"Unknown option: {method}")

        eos = self.ev_curves.eos_parameters["eos_name"]
        self.qha.get_quasi_harmonic_data(
            method,
            eos,
            volume_range,
            self.ev_curves.eos_parameters_df,
            harmonic_properties_fit=harmonic_properties_fit,
            debye_properties=debye_properties,
            thermal_electronic_properties_fit=thermal_electronic_properties_fit,
            P=P,
            plot=plot,
            plot_type=plot_type,
            selected_temperatures_plot=selected_temperatures_plot,
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
            document["qha"] = {
                "method": self.qha.method,
                "number_of_atoms": self.qha.number_of_atoms,
                "volumes": self.qha.volumes,
                "temperatures": self.qha.temperatures,
                "helmholtz_energy": self.qha.helmholtz_energy,
                "entropy": self.qha.entropy,
                "heat_capacity": self.qha.heat_capacity,
            }

        self.collection.insert_one(document)


def plot_multiple_ev(
    config_objects: dict[str, Configuration],
    config_names: list[str],
    eos_name: str = "BM4",
    highlight_minimum: bool = True,
    per_atom: bool = False,
    title: str = None,
    show_fig: bool = False,
    cmap: str = "plotly",
    marker_alpha: float = 1,
    marker_size: int = 10,
):

    dataframes = []
    for config_name in config_names:
        dataframes.append(config_objects[config_name].ev_curves.energy_volume_df)
    dataframes = pd.concat(dataframes)

    fig = plot_ev(
        dataframes,
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