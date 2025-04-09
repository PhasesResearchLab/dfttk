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
    calculate_encut_conv,
    calculate_kpoint_conv,
)
import dfttk.eos_functions as eos_functions
from dfttk.eos_fit import (
    assign_colors_to_configs,
    assign_marker_symbols_to_configs,
)
from dfttk.ev_curve_data import EvCurveData
from dfttk.debye_data import DebyeData
from dfttk.phonon_data import PhononData

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
    """A class to store metadata information for a configuration."""

    def __init__(
        self,
        affiliation: str = "DFTTK",
        mpdd_id: ObjectId = None,
        parent_database: str = None,
        parent_database_id: str = None,
        parent_database_url: str = None,
        comment: str = None,
    ) -> None:
        """Initialize the MetaData object with the given attributes.

        Args:
            affiliation (str, optional): Database affiliation. Defaults to "DFTTK".
            mpdd_id (ObjectId, optional): The MongoDB ObjectId for the MPDD entry. Defaults to None.
            parent_database (str, optional): The name of the parent database. Defaults to None.
            parent_database_id (str, optional): The ID of the parent database entry. Defaults to None.
            parent_database_url (str, optional): The URL of the parent database. Defaults to None.
            comment (str, optional): Additional comments. Defaults to None.
        """

        self.affiliation = affiliation
        self.mpdd_id = mpdd_id
        self.parent_database = parent_database
        self.parent_database_id = parent_database_id
        self.parent_database_url = parent_database_url
        self.comment = comment

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


class Configuration:
    def __init__(self, path, name, alias: str = None, multiplicity: int = None):
        self.path = path
        self.name = name
        self.alias = alias
        self.multiplicity = multiplicity
        self.job_script = {}
        self.vasp_cmd = None
        self.ev_curve_settings_data = {}
        self.phonons_settings_data = {}
        self.thermal_electronic_settings_data = {}
        self.ev_curve_job_script = {}
        self.phonons_job_script = {}
        self.thermal_electronic_job_script = {}

    def set_vasp_cmd(self, vasp_cmd: list[str]) -> None:
        self.vasp_cmd = vasp_cmd

    def read_job_script(self, template: str) -> None:
        templates_map = {
            "slurm": "slurm.json",
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

    def ev_curve_settings(
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

        self.ev_curve_settings_data = {
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

        self.ev_curve_job_script = self.job_script

    # TODO: add a way to select the custodian handlers
    def run_ev_curve(self) -> None:

        # Ensure ev_curve_settings_data is set and not empty
        if not self.ev_curve_settings_data:
            raise AttributeError(
                "EV curves settings data not set. Please call ev_curve_settings() first."
            )

        # Prepare the VASP input files
        vasp_input.ev_curve_set(
            self.path,
            material_type=self.ev_curve_settings_data["material_type"],
            encut=self.ev_curve_settings_data["encut"],
            kppa=self.ev_curve_settings_data["kppa"],
            magmom_fm=self.ev_curve_settings_data["magmom_fm"],
            potcar_functional=self.ev_curve_settings_data["potcar_functional"],
            incar_functional=self.ev_curve_settings_data["incar_functional"],
            other_settings=self.ev_curve_settings_data["other_settings"],
        )

        # Prepare the run_dfttk.py script
        run_dfttk_script = f"""
import os
from custodian.vasp.handlers import VaspErrorHandler
import dfttk.workflows as workflows

subset = list(VaspErrorHandler.error_msgs.keys())
handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]
vasp_cmd = {self.vasp_cmd}
volumes = {self.ev_curve_settings_data["volumes"]}

workflows.ev_curve_series(
    os.getcwd(),
    volumes,
    vasp_cmd,
    handlers,
    restarting={self.ev_curve_settings_data["restarting"]},
    keep_wavecar={self.ev_curve_settings_data["keep_wavecar"]},
    keep_chgcar={self.ev_curve_settings_data["keep_chgcar"]},
    copy_magmom={self.ev_curve_settings_data["copy_magmom"]},
    default_settings={self.ev_curve_settings_data["default_settings"]},
    override_2relax={self.ev_curve_settings_data["override_2relax"]},
    override_3static={self.ev_curve_settings_data["override_3static"]},
    max_errors={self.ev_curve_settings_data["max_errors"]}
)
workflows.custodian_errors_location(os.getcwd())
workflows.NELM_reached(os.getcwd())
""".strip()

        with open(os.path.join(self.path, "run_dfttk.py"), "w") as file:
            file.write(run_dfttk_script)

        # Run the job
        subprocess.run(["sbatch", "job.sh"], cwd=self.path)

    def process_ev_curve(
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

        # Initialize EvCurveData
        self.ev_curve = EvCurveData(self.path, self.name)

        # Get VASP input
        self.ev_curve.get_vasp_input(volumes)

        # Get energy-volume data
        self.ev_curve.get_energy_volume_data(
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
        self.ev_curve.fit_energy_volume_data(
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
            self.ev_curve.number_of_atoms,
            self.ev_curve.volumes,
            self.ev_curve.average_mass,
            self.ev_curve.eos_parameters["V0"],
            self.ev_curve.eos_parameters["B"],
            self.ev_curve.eos_parameters["BP"],
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
        self.phonons = PhononData(self.path)
        self.phonons.process_phonon_dos()

    def process_phonons(
        self,
        scale_atoms: int,
        temperatures: np.ndarray,
        volumes: list[float] = None,
        order: int = 2,
    ):
        self.phonons = PhononData(self.path)
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
    ):

        eos = self.ev_curve.eos_parameters["eos_name"]
        a = self.ev_curve.eos_parameters["a"]
        b = self.ev_curve.eos_parameters["b"]
        c = self.ev_curve.eos_parameters["c"]
        d = self.ev_curve.eos_parameters["d"]
        e = self.ev_curve.eos_parameters["e"]

        # Get the EOS energy at 0 K corresponding to the volume range
        equation_functions = {
            "mBM4": eos_functions.mBM4_equation,
            "mBM5": eos_functions.mBM5_equation,
            "BM4": eos_functions.BM4_equation,
            "BM5": eos_functions.BM5_equation,
            "LOG4": eos_functions.LOG4_equation,
            "LOG5": eos_functions.LOG5_equation,
        }
        if eos == "mBM4" or eos == "BM4" or eos == "LOG4":
            energy_eos = equation_functions[eos](volume_range, a, b, c, d)
        elif eos == "mBM5" or eos == "BM5" or eos == "LOG5":
            energy_eos = equation_functions[eos](volume_range, a, b, c, d, e)

        phonons_f_vib_fit = []
        phonons_s_vib_fit = []
        phonons_cv_vib_fit = []
        debye_f_vib = []
        debye_s_vib = []
        debye_cv_vib = []
        f_el_fit = []
        s_el_fit = []
        cv_el_fit = []
        phonon_temperatures_list = self.phonons.temperatures

        # Convert the list to a 2D NumPy array
        phonons_f_vib_fit = np.vstack(self.phonons.f_vib_fit)
        phonons_s_vib_fit = np.vstack(self.phonons.s_vib_fit)
        phonons_cv_vib_fit = np.vstack(self.phonons.cv_vib_fit)
        debye_f_vib = self.debye.free_energy
        debye_s_vib = self.debye.entropy
        debye_cv_vib = self.debye.heat_capacity
        f_el_fit = np.vstack(self.thermal_electronic.f_el_fit)
        s_el_fit = np.vstack(self.thermal_electronic.s_el_fit)
        cv_el_fit = np.vstack(self.thermal_electronic.cv_el_fit)

        if not hasattr(self, "qha"):
            self.qha = QuasiHarmonicData()

        if method == "debye":
            f_vib_fit = debye_f_vib
            s_vib_fit = debye_s_vib
            cv_vib_fit = debye_cv_vib
            f_el_fit = 0
            s_el_fit = 0
            cv_el_fit = 0
        elif method == "debye_thermal_electronic":
            f_vib_fit = debye_f_vib
            s_vib_fit = debye_s_vib
            cv_vib_fit = debye_cv_vib
        elif method == "phonons":
            f_vib_fit = phonons_f_vib_fit
            s_vib_fit = phonons_s_vib_fit
            cv_vib_fit = phonons_cv_vib_fit
            f_el_fit = 0
            s_el_fit = 0
            cv_el_fit = 0
        elif method == "phonons_thermal_electronic":
            f_vib_fit = phonons_f_vib_fit
            s_vib_fit = phonons_s_vib_fit
            cv_vib_fit = phonons_cv_vib_fit
        else:
            raise ValueError(f"Unknown option: {method}")

        self.qha.get_quasi_harmonic_data(
            method,
            eos,
            self.ev_curve.number_of_atoms,
            phonon_temperatures_list,
            volume_range,
            energy_eos,
            f_vib_fit=f_vib_fit,
            s_vib_fit=s_vib_fit,
            cv_vib_fit=cv_vib_fit,
            f_el_fit=f_el_fit,
            s_el_fit=s_el_fit,
            cv_el_fit=cv_el_fit,
            P=P,
        )

    def replace_keys(self, d, key_mapping):
        if isinstance(d, dict):
            return {
                key_mapping.get(k, k): self.replace_keys(v, key_mapping)
                for k, v in d.items()
            }
        elif isinstance(d, list):
            return [self.replace_keys(i, key_mapping) for i in d]
        else:
            return d

    # TODO: Fix this!
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
                    self.ev_curve.relaxed_structures[0].composition.reduced_formula
                    if hasattr(self, "ev_curve")
                    else None
                ),
                "nComponents": (
                    len(self.ev_curve.relaxed_structures[0].composition.elements)
                    if hasattr(self, "ev_curve")
                    else None
                ),
                "numberOfAtoms": (
                    self.ev_curve.number_of_atoms if hasattr(self, "ev_curve") else None
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

        if hasattr(self, "ev_curve"):
            eos_parameters = self.ev_curve.eos_parameters.copy()
            ev_curve_settings_copy = self.ev_curve_settings_data.copy()
            ev_curve_settings_copy.pop("volumes")
            number_of_atoms = self.ev_curve.number_of_atoms

            eos_parameters_ordered = OrderedDict()
            eos_parameters_ordered["eosName"] = eos_parameters.pop("eos_name")
            eos_parameters_ordered.update(eos_parameters)

            document["evCurve"] = {
                "input": {
                    "jobScript": self.ev_curve_job_script,
                    "settings": ev_curve_settings_copy,
                    "poscar": self.ev_curve.starting_poscar.as_dict(),
                    "incars": self.ev_curve.incars,
                    "kpoints": self.ev_curve.kpoints.as_dict(),
                    "potcar": self.ev_curve.potcar.as_dict(),
                },
                "output": {
                    "scaleAtoms": number_of_atoms,
                    "volumes": self.ev_curve.volumes.tolist(),
                    "energies": self.ev_curve.energies.tolist(),
                    "relaxedStructures": [
                        s.as_dict() for s in self.ev_curve.relaxed_structures
                    ],
                    "totalMagneticMoments": (
                        self.ev_curve.total_magnetic_moment
                        if isinstance(self.ev_curve.total_magnetic_moment, list)
                        else self.ev_curve.total_magnetic_moment.tolist()
                    ),
                    "magneticOrderings": (
                        self.ev_curve.magnetic_ordering
                        if isinstance(self.ev_curve.magnetic_ordering, list)
                        else self.ev_curve.magnetic_ordering.tolist()
                    ),
                    "magData": self.ev_curve.mag_data,
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

            key_mapping = {"polynomial_coefficients": "polynomialCoefficients"}
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

            helmholtz_energy = convert_poly1d(
                self.thermal_electronic.helmholtz_energy_fit
            )
            entropy = convert_poly1d(self.thermal_electronic.entropy_fit)
            heat_capacity = convert_poly1d(self.thermal_electronic.heat_capacity_fit)

            key_mapping = {"polynomial_coefficients": "polynomialCoefficients"}
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
        """
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
                "polynomial_coefficients": "polynomialCoefficients",
            }
            methods_copy = {
                key_mapping.get(method, method): {
                    str(P)
                    + " GPa": {
                        k: v for k, v in data.items() if k != "quasi_harmonic_df"
                    }
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
        """
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
        fig = config_objects[config_name].ev_curve.plot(
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
