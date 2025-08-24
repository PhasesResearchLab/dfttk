"""
Configuration class for DFTTK.

Manages all settings, input generation, job submission, and data processing
for high-throughput DFT calculations of the Helmholtz energy and related
thermodynamic properties. Supports automated workflows for volume relaxation,
convergence tests, energy-volume curves, phonon calculations, Debye model,
thermal electronic contributions, and quasi-harmonic analysis using VASP and DFTTK.
"""

# Standard Library Imports
import os
import re
import subprocess
from collections import OrderedDict
from datetime import datetime

# Third-Party Library Imports
import numpy as np
import pandas as pd
from bson import ObjectId
from custodian.vasp.handlers import VaspErrorHandler
from pymongo import MongoClient
import plotly.graph_objects as go

# DFTTK Imports
import dfttk.vasp_input as vasp_input
from dfttk.aggregate_extraction import calculate_encut_conv, calculate_kpoint_conv
from dfttk.debye.debye_gruneisen import DebyeGruneisen
import dfttk.eos.functions as eos_functions
from dfttk.eos.ev_curve_data import EvCurveData
from dfttk.eos.fit import assign_colors_to_configs, assign_marker_symbols_to_configs
from dfttk.phonon.yphon_phonon_data import YphonPhononData
from dfttk.quasi_harmonic import QuasiHarmonic
from dfttk.thermal_electronic.thermal_electronic_data import ThermalElectronicData
from dfttk.workflows import SingleJobWorkflow


class MetaData:
    """A class to store metadata information for a configuration."""

    def __init__(
        self,
        vasp_version: str = None,
        mpdd_id: ObjectId = None,
        parent_database: str = None,
        parent_database_id: str = None,
        parent_database_url: str = None,
        affiliation: str = "DFTTK",
        comment: str = None,
    ) -> None:
        """Initialize the MetaData object with the given attributes.

        Args:
            vasp_version (str, optional): Version of VASP used. Defaults to None.
            affiliation (str, optional): Database affiliation. Defaults to "DFTTK".
            mpdd_id (ObjectId, optional): The MongoDB ObjectId for the MPDD entry. Defaults to None.
            parent_database (str, optional): The name of the parent database. Defaults to None.
            parent_database_id (str, optional): The ID of the parent database entry. Defaults to None.
            parent_database_url (str, optional): The URL of the parent database. Defaults to None.
            comment (str, optional): Additional comments. Defaults to None.
        """

        self.vasp_version = vasp_version
        self.affiliation = affiliation
        self.mpdd_id = mpdd_id
        self.parent_database = parent_database
        self.parent_database_id = parent_database_id
        self.parent_database_url = parent_database_url
        self.comment = comment
        

class Configuration:
    """
    Main configuration class for automating DFTTK workflows for Helmholtz energy calculations.

    Args:
        path (str): Path to the working directory.
        name (str): Name for this configuration.
        vasp_cmd (list[str]): The command and arguments to run VASP, e.g., ['mpirun', 'vasp_std'].
        alias (str, optional): Alias for the configuration, used for easier identification. Defaults to None.
        multiplicity (int, optional): Configuration multiplicity. Defaults to None.
    """

    def __init__(
        self,
        path: str,
        name: str,
        vasp_cmd: list[str],
        alias: str = None,
        multiplicity: int = None,
    ):
        self.path = path
        self.name = name
        self.alias = alias
        self.multiplicity = multiplicity
        self.vasp_cmd = vasp_cmd

        # Settings for different workflows
        self.ev_curve_settings_data: dict = {}
        self.phonons_settings_data: dict = {}
        self.thermal_electronic_settings_data: dict = {}

        # Placeholders for results/objects
        self.ev_curve = None
        self.phonons = None
        self.thermal_electronic = None
        self.debye = None
        self.qha = None
        self.experiments = None
        self.metadata = None

    def add_metadata(
        self,
        vasp_version: str = None,
        mpdd_id: ObjectId = None,
        parent_database: str = None,
        parent_database_id: str = None,
        parent_database_url: str = None,
        affiliation: str = "DFTTK",
        comment: str = None,
    ) -> None:
        """
        Add metadata information to the configuration.

        Args:
            vasp_version (str, optional): Version of VASP used. Defaults to None.
            mpdd_id (ObjectId, optional): The MongoDB ObjectId for the MPDD entry. Defaults to None.
            parent_database (str, optional): The name of the parent database. Defaults to None.
            parent_database_id (str, optional): The ID of the parent database entry. Defaults to None.
            parent_database_url (str, optional): The URL of the parent database. Defaults to None.
            affiliation (str, optional): Database affiliation. Defaults to "DFTTK".
            comment (str, optional): Additional comments. Defaults to None.
        """
        
        self.metadata = MetaData(
            vasp_version,
            mpdd_id,
            parent_database,
            parent_database_id,
            parent_database_url,
            affiliation,
            comment,
        )

    def run_volume_relax(
        self,
        material_type: str,
        error_msgs: list[str] = list(VaspErrorHandler.error_msgs.keys()),
        max_errors: int = 10,
        encut: int = 520,
        kppa: int = 4000,
        magmom_fm: bool = False,
        potcar_functional: str = "PBE_54",
        incar_functional: str = "PBE",
        other_settings: dict = None,
        vaspjob_kwargs: dict = None,
        custodian_kwargs: dict = None,
    ) -> None:
        """
        Set up and submit a volume relaxation job.

        Args:
            material_type (str): Type of material (e.g., "metal" or "non_metal").
            error_msgs (list[str], optional): List of error messages for VaspErrorHandler. Defaults to all known messages.
            max_errors (int, optional): Maximum number of errors allowed before Custodian stops. Defaults to 10.
            encut (int, optional): Plane-wave cutoff energy. Defaults to 520.
            kppa (int, optional): K-points per reciprocal atom. Defaults to 4000.
            magmom_fm (bool, optional): Use ferromagnetic magmom. Defaults to False.
            potcar_functional (str, optional): POTCAR functional. Defaults to "PBE_54".
            incar_functional (str, optional): INCAR functional. Defaults to "PBE".
            other_settings (dict, optional): Additional VASP INCAR/KPOINTS settings. Defaults to None.
            vaspjob_kwargs (dict, optional): Additional keyword arguments to pass to VaspJob (e.g., output_file, stderr_file).
            custodian_kwargs (dict, optional): Additional keyword arguments to pass to Custodian (e.g., scratch_dir, gzipped_output).

        Returns:
            None
        """
        # Write VASP input files
        if other_settings is None:
            other_settings = {}

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
        
        # Write run_dfttk.py script
        workflow = SingleJobWorkflow(
            path=self.path,
            vasp_cmd=self.vasp_cmd,
            error_msgs=error_msgs,
            max_errors=max_errors,
            vaspjob_kwargs=vaspjob_kwargs,
            custodian_kwargs=custodian_kwargs,
            )
        workflow.write_run_dfttk()
        
        # Submit the job using SLURM
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
        encut_list: list[int] = [270, 320, 370, 420, 470, 520, 570, 620, 670, 720, 770, 820],
        kppa_list: list[float] = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        force_gamma: bool = True,
        backup: bool = False,
        max_errors: int = 10,
    ) -> None:
        """
        Set up and submit convergence tests for ENCUT and k-point mesh.

        Args:
            encut (int, optional): Plane-wave cutoff energy. Defaults to 520.
            kppa (int, optional): K-points per reciprocal atom. Defaults to 4000.
            magmom_fm (bool, optional): Use ferromagnetic magmom. Defaults to False.
            potcar_functional (str, optional): POTCAR functional. Defaults to "PBE_54".
            incar_functional (str, optional): INCAR functional. Defaults to "PBE".
            other_settings (dict, optional): Additional VASP settings. Defaults to empty dict.
            encut_list (list[int], optional): List of ENCUT values to test. Defaults to [270, ..., 820].
            kppa_list (list[float], optional): List of KPPA values to test. Defaults to [1000, ..., 10000].
            force_gamma (bool, optional): Force gamma-centered mesh. Defaults to True.
            backup (bool, optional): Enable backup. Defaults to False.
            max_errors (int, optional): Maximum number of errors allowed. Defaults to 10.
        """
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
        """
        Analyze the ENCUT convergence test results.

        Args:
            plot (bool, optional): Whether to generate a plot of the results. Defaults to True.

        Returns:
            tuple[pd.DataFrame, go.Figure]: DataFrame of ENCUT convergence data and the corresponding plotly Figure.
        """
        encut_conv_path = os.path.join(self.path, "encut_conv")
        encut_conv_df, fig = calculate_encut_conv(encut_conv_path, plot)
        return encut_conv_df, fig

    def analyze_kpoints_conv(self, plot: bool = True) -> tuple[pd.DataFrame, go.Figure]:
        """
        Analyze the k-point convergence test results.

        Args:
            plot (bool, optional): Whether to generate a plot of the results. Defaults to True.

        Returns:
            tuple[pd.DataFrame, go.Figure]: DataFrame of k-point convergence data and the corresponding plotly Figure.
        """
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
        """
        Set the settings for the energy-volume (E-V) curve workflow.

        Args:
            material_type (str): Type of material (e.g., "metal" or "non_metal").
            volumes (list[float]): List of volumes to use for the E-V curve.
            encut (int, optional): Plane-wave cutoff energy. Defaults to 520.
            kppa (int, optional): K-points per reciprocal atom. Defaults to 4000.
            magmom_fm (bool, optional): Use ferromagnetic magmom. Defaults to False.
            potcar_functional (str, optional): POTCAR functional. Defaults to "PBE_54".
            incar_functional (str, optional): INCAR functional. Defaults to "PBE".
            other_settings (dict, optional): Additional VASP settings. Defaults to empty dict.
            restarting (bool, optional): Whether to restart from previous calculations. Defaults to False.
            keep_wavecar (bool, optional): Keep WAVECAR file. Defaults to False.
            keep_chgcar (bool, optional): Keep CHGCAR file. Defaults to False.
            copy_magmom (bool, optional): Copy magmom from previous runs. Defaults to False.
            default_settings (bool, optional): Use default settings. Defaults to True.
            override_2relax (list, optional): Override settings for 2nd relaxation. Defaults to None.
            override_3static (list, optional): Override settings for 3rd static run. Defaults to None.
            max_errors (int, optional): Maximum number of errors allowed. Defaults to 10.
        """
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

    # TODO: add a way to select the custodian handlers
    def run_ev_curve(self) -> None:
        """
        Set up and submit the energy-volume (E-V) curve workflow.

        This method prepares VASP input files and generates a run script for the E-V curve series
        using the settings stored in `self.ev_curve_settings_data`. It then submits the job using SLURM.

        Raises:
            AttributeError: If E-V curve settings have not been set (call `ev_curve_settings()` first).
        """
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
        incar_keys: list[str] = ['1relax', '2relax', '3static'],
        incar_names: list[str] = ["INCAR.1relax", "INCAR.2relax", "INCAR.3static"],
        kpoints_keys: list[str] = ['1relax', '2relax', '3static'],
        kpoints_names: list[str] = ["KPOINTS.1relax", "KPOINTS.2relax", "KPOINTS.3static"],
        selected_volumes: list[float] = None,
        read_initial_poscar: bool = True,
        outcar_name: str = "OUTCAR.3static",
        oszicar_name: str = "OSZICAR.3static",
        contcar_name: str = "CONTCAR.3static",
        collect_mag_data: bool = False,
        magmom_tolerance: float = 0.05,
        total_magnetic_moment_tolerance: float = 0.1,
        mass_average: str = "geometric",
        eos_name: str = "BM4",
        volume_min: float = None,
        volume_max: float = None,
        num_volumes: int = 1000,
    ) -> None:
        """
        Process the energy-volume (E-V) curve data for the configuration.

        This method initializes the EvCurveData object, extracts VASP input and output data,
        and fits the energy-volume data to an equation of state.

        Args:
            incar_keys (list[str], optional): List of INCAR keys for dictionary keys. Defaults to ["1relax", "2relax", "3static"].
            incar_names (list[str], optional): List of INCAR names to read. Defaults to ["INCAR.1relax", "INCAR.2relax", "INCAR.3static"].
            kpoints_keys (list[str], optional): List of KPOINTS keys for dictionary keys. Defaults to ["1relax", "2relax", "3static"].
            kpoints_names (list[str], optional): List of KPOINTS names to read. Defaults to ["KPOINTS.1relax", "KPOINTS.2relax", "KPOINTS.3static"].
            selected_volumes (list[float], optional): List of volumes to process. Defaults to None.
            read_initial_poscar (bool, optional): Whether to read the initial POSCAR file. Defaults to True.
            outcar_name (str, optional): Name of the OUTCAR file. Defaults to "OUTCAR.3static".
            oszicar_name (str, optional): Name of the OSZICAR file. Defaults to "OSZICAR.3static".
            contcar_name (str, optional): Name of the CONTCAR file. Defaults to "CONTCAR.3static".
            collect_mag_data (bool, optional): Whether to collect magnetic data. Defaults to False.
            magmom_tolerance (float, optional): Tolerance for magnetic moment. Defaults to 0.05.
            total_magnetic_moment_tolerance (float, optional): Tolerance for total magnetic moment. Defaults to 0.1.
            mass_average (str, optional): Method for mass averaging. Defaults to "geometric".
            eos_name (str, optional): Name of the equation of state to fit. Defaults to "BM4".
            volume_min (float, optional): Minimum volume for fitting. Defaults to None.
            volume_max (float, optional): Maximum volume for fitting. Defaults to None.
            num_volumes (int, optional): Number of volumes for fitting. Defaults to 1000.
        """
        # Initialize EvCurveData
        self.ev_curve = EvCurveData(self.path, self.name)

        # Get VASP input
        self.ev_curve.get_vasp_input(
            incar_keys=incar_keys,
            incar_names=incar_names,
            kpoints_keys=kpoints_keys,
            kpoints_names=kpoints_names,
            contcar_name=contcar_name,
            selected_volumes=selected_volumes,
            read_initial_poscar=read_initial_poscar
        )

        # Get energy-volume data
        self.ev_curve.get_energy_volume_data(
            selected_volumes=selected_volumes,
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
        gruneisen_x: float = 2/3,
        temperatures: np.array = np.linspace(0, 1000, 101),
    ):
        """
        Process the Debye-Grüneisen model for the configuration.

        This method initializes the DebyeGruneisen object and computes thermodynamic properties
        over a range of volumes and temperatures using the Debye model.

        Args:
            scaling_factor (float, optional): Scaling factor for the Debye temperature. Defaults to 0.617.
            gruneisen_x (float, optional): Grüneisen parameter exponent. Defaults to 2/3.
            temperatures (np.array, optional): Array of temperatures to evaluate. Defaults to np.linspace(0, 1000, 101).
        """
        volumes = np.linspace(0.98 * min(self.ev_curve.volumes), 1.02 * max(self.ev_curve.volumes), 1000)
        self.debye = DebyeGruneisen()
        self.debye.process(
            number_of_atoms=self.ev_curve.number_of_atoms,
            volumes=volumes,
            temperatures=temperatures,
            atomic_mass=self.ev_curve.average_mass,
            V0=self.ev_curve.eos_parameters["V0"],
            B=self.ev_curve.eos_parameters["B"],
            BP=self.ev_curve.eos_parameters["BP"],
            scaling_factor=scaling_factor,
            gruneisen_x=gruneisen_x,
        )
        
    def phonons_settings(
        self,
        phonon_volumes: list[float],
        kppa: float,
        scaling_matrix: tuple[tuple[int]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        copy_magmom: bool = True,
        backup: bool = False,
        max_errors: int = 10,
        relax: bool = True,
    ) -> None:
        """
        Set the settings for the phonon calculation workflow.

        Args:
            phonon_volumes (list[float]): List of volumes to use for phonon calculations.
            kppa (float): K-points per reciprocal atom.
            scaling_matrix (tuple[tuple[int]], optional): Supercell scaling matrix. Defaults to identity matrix.
            copy_magmom (bool, optional): Copy magmom from previous runs. Defaults to True.
            backup (bool, optional): Enable backup. Defaults to False.
            max_errors (int, optional): Maximum number of errors allowed. Defaults to 10.
            relax (bool, optional): Whether to relax the structure before phonon calculation. Defaults to True.
        """
        self.phonons_settings_data = {
            "phonon_volumes": phonon_volumes,
            "kppa": kppa,
            "scaling_matrix": scaling_matrix,
            "copy_magmom": copy_magmom,
            "backup": backup,
            "max_errors": max_errors,
            "relax": relax,
        }

    def run_phonons(self, run_file: str = "run_dfttk_phonons.py") -> None:
        """
        Set up and submit the phonon calculation workflow.

        This method generates a Python script to run the phonon calculations in parallel
        using the settings stored in `self.phonons_settings_data`, then executes the script.

        Args:
            run_file (str, optional): Name of the temporary Python script to run. Defaults to "run_dfttk_phonons.py".

        Raises:
            AttributeError: If phonons settings have not been set (call `phonons_settings()` first).
        """
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
        """
        Generate the phonon density of states (DOS) using YPHON. 

        This method initializes the YphonPhononData object and processes the phonon DOS
        using the results in the working directory.
        """
        self.phonons = YphonPhononData(self.path)
        self.phonons.process_phonon_dos()

    def process_phonons(
        self,
        number_of_atoms: int,
        temperatures: np.ndarray,
        selected_volumes: list[float] = None,
        order: int = 2,
    ):
        """
        Process phonon data for the configuration.

        This method initializes the YphonPhononData object, extracts VASP input and output data,
        and processes harmonic phonon data for the specified number of atoms, temperatures, and order of fitting.

        Args:
            number_of_atoms (int): Number of atoms to scale for the phonon calculation.
            temperatures (np.ndarray): Array of temperatures to evaluate.
            selected_volumes (list[float], optional): List of volumes to process. Defaults to None.
            order (int, optional): Order of polynomial fitting. Defaults to 2.
        """
        self.phonons = YphonPhononData(self.path)
        self.phonons.get_vasp_input(selected_volumes)
        self.phonons.get_harmonic_data(
            number_of_atoms,
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
        """
        Set the settings for the thermal electronic calculation workflow.

        Args:
            volumes (list[float]): List of volumes to use for the calculation.
            kppa (float): K-points per reciprocal atom.
            scaling_matrix (tuple[tuple[int]], optional): Supercell scaling matrix. Defaults to identity matrix.
            NEDOS (int, optional): Number of points in the DOS. Defaults to 10001.
            backup (bool, optional): Enable backup. Defaults to False.
            max_errors (int, optional): Maximum number of errors allowed. Defaults to 10.
        """
        self.thermal_electronic_settings_data = {
            "volumes": volumes,
            "kppa": kppa,
            "scaling_matrix": scaling_matrix,
            "NEDOS": NEDOS,
            "backup": backup,
            "max_errors": max_errors,
        }

    def run_thermal_electronic(self, run_file: str = "run_dfttk_thermal_electronic.py"):
        """
        Set up and submit the thermal electronic calculation workflow.

        This method generates a Python script to run the thermal electronic calculations in parallel
        using the settings stored in `self.thermal_electronic_settings_data`, then executes the script.

        Args:
            run_file (str, optional): Name of the temporary Python script to run. Defaults to "run_dfttk_thermal_electronic.py".

        Raises:
            AttributeError: If thermal electronic settings have not been set (call `thermal_electronic_settings()` first).
        """
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
        selected_volumes: list[float] = None,
        order: int = 1,
    ):
        """
        Process thermal electronic data for the configuration.

        This method initializes the ThermalElectronicData object, extracts VASP input and output data,
        and processes thermal electronic data for the specified temperature range and order of fitting. 

        Args:
            temperature_range (np.ndarray): Array of temperatures to evaluate.
            selected_volumes (list[float], optional): List of volumes to process. Defaults to None.
            order (int, optional): Order of polynomial fitting. Defaults to 1.
        """
        self.thermal_electronic = ThermalElectronicData(self.path)
        self.thermal_electronic.get_vasp_input(selected_volumes)
        self.thermal_electronic.get_thermal_electronic_data(
            temperature_range,
            order=order,
        )

    def process_qha(
        self,
        method: str,
        volume_range: np.ndarray,
        P: float = 0.00,
    ):
        """
        Perform quasi-harmonic approximation (QHA) calculations for a range of volumes.
        This method calculates thermodynamic quantities (Helmholtz free energy, entropy,
        and heat capacity) as functions of volume and temperature, and determines
        pressure-related properties using the specified QHA method.

        Args:
            method (str): The method to use for QHA. Options are:
                - "debye"
                - "debye_thermal_electronic"
                - "phonons"
                - "phonons_thermal_electronic"
            volume_range (np.ndarray): Array of volumes to evaluate.
            P (float, optional): Pressure in GPa. Defaults to 0.00.

        Raises:
            AttributeError: If required data is missing, including:
                - If the energy-volume curve is not processed.
                - If the Debye-Grüneisen model is not processed (for "debye" or "debye_thermal_electronic").
                - If the phonons data is not processed (for "phonons" or "phonons_thermal_electronic").
                - If the thermal electronic data is not processed (for "debye_thermal_electronic" or "phonons_thermal_electronic").
            ValueError: If the temperature arrays for Debye/phonons and thermal electronic do not match
                (for "debye_thermal_electronic" or "phonons_thermal_electronic"), or if an unknown method is provided.
        """

        # If ev_curve is not processed, raise an error
        if self.ev_curve is None:
            raise AttributeError("Energy-volume curve not processed. Call process_ev_curve() first.")
        
         # Get the EOS energy at 0 K corresponding to the volume range
        eos = self.ev_curve.eos_parameters["eos_name"]
        a = self.ev_curve.eos_parameters["a"]
        b = self.ev_curve.eos_parameters["b"]
        c = self.ev_curve.eos_parameters["c"]
        d = self.ev_curve.eos_parameters["d"]
        e = self.ev_curve.eos_parameters["e"]

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

        # For first time, initialize the QuasiHarmonic object
        if self.qha is None:
            if method in ("debye", "debye_thermal_electronic"):
                
                # If Debye-Grüneisen model is not processed, raise an error
                if self.debye is None:
                    raise AttributeError("Debye-Grüneisen model not processed. Call process_debye() first.")
                if method == "debye":
                    self.debye.temperatures = np.array([int(t) if isinstance(t, float) and t.is_integer() else t for t in self.debye.temperatures]) # temp fix
                    self.qha = QuasiHarmonic(self.ev_curve.number_of_atoms, volume_range, temperatures=self.debye.temperatures)
                elif method == "debye_thermal_electronic":
                    # If thermal electronic data is not processed, raise an error
                    if self.thermal_electronic is None:
                        raise AttributeError("Thermal electronic data not processed. Call process_thermal_electronic() first.")
                    # If the temperatures of debye and thermal electronic do not match, raise an error
                    if not np.array_equal(self.debye.temperatures, self.thermal_electronic.temperatures):
                        raise ValueError("Debye and thermal electronic temperatures do not match.")
                    self.debye.temperatures = np.array([int(t) if isinstance(t, float) and t.is_integer() else t for t in self.debye.temperatures]) # temp fix
                    self.qha = QuasiHarmonic(self.ev_curve.number_of_atoms, volume_range, temperatures=self.debye.temperatures)
            elif method in ("phonons", "phonons_thermal_electronic"): 
                # If phonons data is not processed, raise an error
                if self.phonons is None:
                    raise AttributeError("Phonon data not processed. Call process_phonons() first.")
                if method == "phonons":
                    self.qha = QuasiHarmonic(self.ev_curve.number_of_atoms, volume_range, temperatures=self.phonons.temperatures)
                elif method == "phonons_thermal_electronic":
                    # If thermal electronic data is not processed, raise an error
                    if self.thermal_electronic is None:
                        raise AttributeError("Thermal electronic data not processed. Call process_thermal_electronic() first.")
                    # If the temperatures of phonons and thermal electronic do not match, raise an error
                    if not np.array_equal(self.phonons.temperatures, self.thermal_electronic.temperatures):
                        raise ValueError("Phonons and thermal electronic temperatures do not match.")
                    self.qha = QuasiHarmonic(self.ev_curve.number_of_atoms, volume_range, temperatures=self.phonons.temperatures)

        # TODO: make tests for uneven volume ranges
        if method in ("debye", "debye_thermal_electronic"):
            # If debye is processed with the same volume range, use the precomputed values
            if np.array_equal(self.debye.volumes, volume_range):
                f_vib_fit = self.debye.helmholtz_energies
                s_vib_fit = self.debye.entropies
                cv_vib_fit = self.debye.heat_capacities
            # If debye is not processed with the same volume range, compute the new values    
            else:
                new_debye = DebyeGruneisen()
                new_debye.process(
                    number_of_atoms=self.ev_curve.number_of_atoms,
                    volumes=volume_range,
                    temperatures=self.debye.temperatures,
                    atomic_mass=self.ev_curve.average_mass,
                    V0=self.ev_curve.eos_parameters["V0"],
                    B=self.ev_curve.eos_parameters["B"],
                    BP=self.ev_curve.eos_parameters["BP"],
                    scaling_factor=self.debye.scaling_factor,
                    gruneisen_x=self.debye.gruneisen_x,
                )
                f_vib_fit = new_debye.helmholtz_energies
                s_vib_fit = new_debye.entropies
                cv_vib_fit = new_debye.heat_capacities
            if method == "debye":
                f_el_fit = None
                s_el_fit = None
                cv_el_fit = None
            elif method == "debye_thermal_electronic":
                # If thermal electronic data is processed with the same volume range, use the precomputed values
                if np.array_equal(self.thermal_electronic.volume_fit, volume_range):
                    f_el_fit = np.vstack(self.thermal_electronic.f_el_fit)
                    s_el_fit = np.vstack(self.thermal_electronic.s_el_fit)
                    cv_el_fit = np.vstack(self.thermal_electronic.cv_el_fit)
                # If thermal electronic data is not processed with the same volume range, compute the new values
                else:
                    f_el_fit = np.array([coeff_row(volume_range) for coeff_row in self.thermal_electronic.f_el_poly])
                    s_el_fit = np.array([coeff_row(volume_range) for coeff_row in self.thermal_electronic.s_el_poly])
                    cv_el_fit = np.array([coeff_row(volume_range) for coeff_row in self.thermal_electronic.cv_el_poly])
                
        elif method in ("phonons", "phonons_thermal_electronic"): 
            # If phonons are processed with the same volume range, use the precomputed values
            if np.array_equal(self.phonons.volumes_fit, volume_range):
                f_vib_fit = self.phonons.helmholtz_energies_fit
                s_vib_fit = self.phonons.entropies_fit
                cv_vib_fit = self.phonons.heat_capacities_fit
            # If phonons are not processed with the same volume range, compute the new values
            else:
                f_vib_fit = np.array([np.polyval(coeff_row, volume_range) for coeff_row in self.phonons.helmholtz_energies_poly_coeffs])
                s_vib_fit = np.array([np.polyval(coeff_row, volume_range) for coeff_row in self.phonons.entropies_poly_coeffs])
                cv_vib_fit = np.array([np.polyval(coeff_row, volume_range) for coeff_row in self.phonons.heat_capacities_poly_coeffs])
            if method == "phonons":
                f_el_fit = None
                s_el_fit = None
                cv_el_fit = None
            elif method == "phonons_thermal_electronic":
                # If thermal electronic data is processed with the same volume range, use the precomputed values
                if np.array_equal(self.thermal_electronic.volume_fit, volume_range):
                    f_el_fit = np.vstack(self.thermal_electronic.f_el_fit)
                    s_el_fit = np.vstack(self.thermal_electronic.s_el_fit)
                    cv_el_fit = np.vstack(self.thermal_electronic.cv_el_fit)
                # If thermal electronic data is not processed with the same volume range, compute the new values
                else:
                    f_el_fit = np.array([coeff_row(volume_range) for coeff_row in self.thermal_electronic.f_el_poly])
                    s_el_fit = np.array([coeff_row(volume_range) for coeff_row in self.thermal_electronic.s_el_poly])
                    cv_el_fit = np.array([coeff_row(volume_range) for coeff_row in self.thermal_electronic.cv_el_poly])
        else:
            raise ValueError(f"Unknown option: {method}")

        self.qha.process(
            method=method,
            energy_eos=energy_eos,
            vibrational_helmholtz_energy=f_vib_fit,
            vibrational_entropy=s_vib_fit,
            vibrational_heat_capacity=cv_vib_fit,
            electronic_helmholtz_energy=f_el_fit,
            electronic_entropy=s_el_fit,
            electronic_heat_capacity=cv_el_fit,
            P=P,
            eos_name=eos,
        )
        
    def _replace_keys(self, d, key_mapping):
        if isinstance(d, dict):
            new_dict = {}
            for k, v in d.items():
                if isinstance(k, str) and re.match(r"^\d+_GPa$", k):
                    new_key = "p" + k.replace("_GPa", "GPa")
                else:
                    new_key = key_mapping.get(k, k)
                new_dict[new_key] = self._replace_keys(v, key_mapping)
            return new_dict
        elif isinstance(d, list):
            return [self._replace_keys(i, key_mapping) for i in d]
        else:
            return d

    def to_mongodb(self, connection_string: str, db_name: str, collection_name: str, insert: bool = True) -> dict:
        """
        This method initializes the MongoDB client, selects the specified database and collection,
        and prepares the object for subsequent data insertion or updates.

        Args:
            connection_string (str): The MongoDB connection string.
            db_name (str): The name of the MongoDB database to use.
            collection_name (str): The name of the collection within the database.
            insert (bool, optional): Whether to insert/update the document in MongoDB. Defaults to True.
        
        Returns:
            dict: A dictionary representation of the configuration object, ready for insertion into MongoDB.
        """
        self.cluster = MongoClient(connection_string)
        self.db = self.cluster[db_name]
        self.collection = self.db[collection_name]

        document = {
            "metadata": {
                "vaspVersion": None,
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
                "name": self.name,
                "alias": self.alias,
                "multiplicity": self.multiplicity,
                "reducedFormula": (
                    self.ev_curve.relaxed_structures[0].composition.reduced_formula
                    if getattr(self, "ev_curve", None) is not None
                    else None
                ),
                "nComponents": (
                    len(self.ev_curve.relaxed_structures[0].composition.elements)
                    if getattr(self, "ev_curve", None) is not None
                    else None
                ),
                "numberOfAtoms": (
                    self.ev_curve.number_of_atoms if getattr(self, "ev_curve", None) is not None else None
                ),
            },
        }

        # Update metadata with actual values if they exist
        if getattr(self, "metadata", None) is not None:
            document["metadata"].update(
                {
                    "vaspVersion": self.metadata.vasp_version,
                    "MPDDId": self.metadata.mpdd_id,
                    "parentDatabase": self.metadata.parent_database,
                    "parentDatabaseId": self.metadata.parent_database_id,
                    "parentDatabaseURL": self.metadata.parent_database_url,
                    "affiliation": self.metadata.affiliation,
                    "comment": self.metadata.comment,
                }
            )

        if getattr(self, "ev_curve", None) is not None:
            eos_parameters = self.ev_curve.eos_parameters.copy()
            number_of_atoms = self.ev_curve.number_of_atoms

            eos_parameters_ordered = OrderedDict()
            eos_parameters_ordered["eosName"] = eos_parameters.pop("eos_name")
            eos_parameters_ordered.update(eos_parameters)

            document["evCurve"] = {
                "input": {
                    "initialPoscar": self.ev_curve.initial_poscar.as_dict() if self.ev_curve.initial_poscar is not None else None,
                    "incars": self.ev_curve.incars,
                    "kpoints": [
                        {key: kp.as_dict() for key, kp in kpoints_dict.items()}
                        for kpoints_dict in self.ev_curve.kpoints
                    ],
                    "potcar": self.ev_curve.potcar.as_dict() if self.ev_curve.potcar is not None else None,
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
        
        if getattr(self, "debye", None) is not None:
            document["debye"] = {
                "atomicMass": self.debye.atomic_mass,
                "V0": self.debye.V0,
                "B": self.debye.B,
                "BP": self.debye.BP,
                "scalingFactor": self.debye.scaling_factor,
                "gruneisenX": self.debye.gruneisen_x,
            }
        
        # Function to recursively convert numpy.poly1d objects to lists of coefficients
        def convert_poly1d(obj):
            if isinstance(obj, np.poly1d):
                return obj.coefficients.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_poly1d(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_poly1d(v) for v in obj]
            return obj

        if getattr(self, "phonons", None) is not None:
            temperatures = self.phonons.temperatures
            min_temperature = min(temperatures)
            max_temperature = max(temperatures)
            num_temperatures = len(temperatures)

            helmholtz_energy = convert_poly1d(self.phonons._helmholtz_energies_fit_to_db)
            entropy = convert_poly1d(self.phonons._entropies_fit_to_db)
            heat_capacity = convert_poly1d(self.phonons._heat_capacities_fit_to_db)

            key_mapping = {"poly_coeffs": "polyCoeffs"}
            helmholtz_energy = self._replace_keys(helmholtz_energy, key_mapping)
            entropy = self._replace_keys(entropy, key_mapping)
            heat_capacity = self._replace_keys(heat_capacity, key_mapping)

            document["phonons"] = {
                "input": {
                    "incars": self.phonons.incars,
                    "kpoints": [
                        {key: kp.as_dict() for key, kp in kpoints_dict.items()}
                        for kpoints_dict in self.phonons.kpoints
                    ],
                    "potcar": self.phonons.potcar.as_dict() if self.phonons.potcar is not None else None,
                },
                "output": {
                    "phononStructures": [
                        s.as_dict() for s in self.phonons.phonon_structures
                    ],
                    "scaleAtoms": self.phonons.number_of_atoms,
                    "volumes": self.phonons.volumes.tolist(),
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
        
        if getattr(self, "thermal_electronic", None) is not None:
            temperatures = self.thermal_electronic.temperatures
            min_temperature = min(temperatures)
            max_temperature = max(temperatures)
            num_temperatures = len(temperatures)

            helmholtz_energy = convert_poly1d(
                self.thermal_electronic.helmholtz_energy_fit
            )
            entropy = convert_poly1d(self.thermal_electronic.entropy_fit)
            heat_capacity = convert_poly1d(self.thermal_electronic.heat_capacity_fit)

            key_mapping = {"polynomial_coefficients": "polyCoeffs", "elec_dos": "elecDos"}
            helmholtz_energy = self._replace_keys(helmholtz_energy, key_mapping)
            entropy = self._replace_keys(entropy, key_mapping)
            heat_capacity = self._replace_keys(heat_capacity, key_mapping)
            kpoints = [{key: kp.as_dict() for key, kp in kpoints_dict.items()}
                        for kpoints_dict in self.thermal_electronic.kpoints]
            kpoints = self._replace_keys(kpoints, key_mapping)

            document["thermalElectronic"] = {
                "input": {
                    "incars": self.thermal_electronic.incars,
                    "kpoints": kpoints,
                    "potcar": self.thermal_electronic.potcar.as_dict() if self.thermal_electronic.potcar is not None else None,
                },
                "output": {
                    "elecStructures": [
                        s.as_dict() for s in self.thermal_electronic.structures
                    ],
                    "scaleAtoms": self.thermal_electronic.number_of_atoms.tolist(),
                    "volumes": [
                        round(s.volume, 2) for s in self.thermal_electronic.structures
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
        
        if getattr(self, "qha", None) is not None:
            key_mapping = {
                "debye": "debye",
                "debye_thermal_electronic": "debyeThermalElectronic",
                "phonons": "phonons",
                "phonons_thermal_electronic": "phononsThermalElectronic",
                "helmholtz_energy": "helmholtzEnergy",
                "heat_capacity": "heatCapacity",
                "eos_parameters": "eosParameters",
                "eos_constants": "eosConstants",
                "eos_name": "eosName",
                "poly_coeffs": "polyCoeffs",
            }

            def remove_values_and_convert_arrays(d):
                if isinstance(d, dict):
                    return {k: remove_values_and_convert_arrays(v) for k, v in d.items() if k != "values"}
                elif isinstance(d, list):
                    return [remove_values_and_convert_arrays(v) for v in d]
                elif isinstance(d, np.ndarray):
                    return d.tolist()
                else:
                    return d

            methods_copy = {}
            for method, properties in self.qha.methods.items():
                methods_copy[method] = {}
                for key, value in properties.items():
                    methods_copy[method][key] = remove_values_and_convert_arrays(value)
            
            methods_copy = self._replace_keys(methods_copy, key_mapping)

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
                    "min": float(min_temperature),
                    "max": float(max_temperature),
                    "number": int(num_temperatures),
                },
                "methods": methods_copy,
            }
        
        if getattr(self, "experiments", None) is not None:
            document["experiments"] = self.experiments

        if insert:
            # Use configuration.name as the unique identifier for upsert
            unique_field = "configuration.name"
            unique_value = document["configuration"]["name"]

            existing_doc = self.collection.find_one({unique_field: unique_value})

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

        return document

    def add_experiments(self, experiments: dict):
        self.experiments = experiments

def plot_multiple_ev(
    config_objects: dict[str, Configuration],
    config_names: list[str],
    highlight_minimum: bool = True,
    per_atom: bool = False,
    title: str = None,
    cmap: str = "plotly",
    marker_alpha: float = 1,
    marker_size: int = 10,
):
    """
    Plot multiple energy-volume (E-V) curves on a single figure.

    Args:
        config_objects (dict[str, Configuration]): Dictionary mapping configuration names to Configuration objects.
        config_names (list[str]): List of configuration names to plot.
        highlight_minimum (bool, optional): Whether to highlight the minimum energy point. Defaults to True.
        per_atom (bool, optional): Whether to plot energy per atom. Defaults to False.
        title (str, optional): Title for the plot. Defaults to None.
        cmap (str, optional): Colormap to use for the plot. Defaults to "plotly".
        marker_alpha (float, optional): Alpha (opacity) for markers. Defaults to 1.
        marker_size (int, optional): Size of the markers. Defaults to 10.

    Returns:
        plotly.graph_objects.Figure: The combined Plotly figure containing all E-V curves.
    """
    combined_fig = go.Figure()

    config_colors = assign_colors_to_configs(
        config_names, alpha=marker_alpha, cmap=cmap
    )
    config_symbols = assign_marker_symbols_to_configs(config_names)

    for config_name in config_names:
        fig = config_objects[config_name].ev_curve.plot(
            highlight_minimum=highlight_minimum,
            per_atom=per_atom,
            cmap=cmap,
            marker_alpha=marker_alpha,
            marker_size=marker_size,
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

    return combined_fig
