"""
Module for managing energy-volume (E-V) data for a configuration using the EvCurveData class.
"""

# Related third party imports
import os
import numpy as np
import plotly.graph_objects as go
from natsort import natsorted
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar

from dfttk.aggregate_extraction import extract_configuration_data
from dfttk.eos_fit import fit_to_eos, plot_ev

class EvCurveData:
    """A class for handling energy-volume (E-V) curve data for a configuration."""

    def __init__(self, path: str, name: str) -> None:
        """Initialize the EvCurveData object.

        Args:
            path (str): path to the directory containing the vol_* folders.
            name (str): name of the configuration.
        """

        self.path = path
        self.name = name

        # VASP input files
        self.incars = []
        self.kpoints = None
        self.potcar = None
        self.starting_poscar = None

        # VASP output files
        self.relaxed_structures = []

        # E-V data
        self.number_of_atoms = None
        self.volumes = []
        self.energies = []
        self.atomic_masses = None
        self.average_mass = None
        self.total_magnetic_moment = None
        self.magnetic_ordering = None
        self.mag_data = []
        self.eos_parameters = {
            key: None
            for key in [
                "eos_name",
                "a",
                "b",
                "c",
                "d",
                "e",
                "V0",
                "E0",
                "B",
                "BP",
                "B2P",
            ]
        }

    def _get_volume_folders(self) -> list[str]:
        """Gets the list of vol_* folders in the specified path and sorts them.

        Returns:
            list[str]: list of vol_* folders sorted in natural order.
        """

        return natsorted([f for f in os.listdir(self.path) if f.startswith("vol_")])

    def get_vasp_input(self, volumes: list[float] = None) -> None:
        """Gets the VASP input files from the specified path and stores them in the class attributes.

        Args:
            volumes (list[float], optional): List of volumes to filter the folders. If None, all folders are considered. Defaults to None.
        """

        # Get the list of volume folders
        vol_folders = self._get_volume_folders()

        # Filter volume folders based on the provided volumes
        if volumes is not None:
            volumes_set = {round(volume, 2) for volume in volumes}
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
                in volumes_set
            ]

        # Read the INCAR files for each volume folder
        incar_keys = ["1relax", "2relax", "3static"]
        for vol_folder in vol_folders:
            incar_data = {
                key: Incar.from_file(
                    os.path.join(self.path, vol_folder, f"INCAR.{key}")
                )
                for key in incar_keys
            }
            self.incars.append(incar_data)

        # Read the KPOINTS file
        self.kpoints = Kpoints.from_file(os.path.join(self.path, "KPOINTS"))

        # Read the POTCAR file
        try:
            self.potcar = Potcar.from_file(os.path.join(self.path, "POTCAR"))
        except FileNotFoundError:
            self.potcar = None

        # Read the starting POSCAR file
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
        """Gets the energy-volume data from the specified path and stores them in the class attributes.

        Args:
            volumes (list[float], optional): List of volumes to filter the folders. If None, all folders are considered. Defaults to None.
            outcar_name (str, optional): Path to the OUTCAR file. Defaults to "OUTCAR.3static".
            oszicar_name (str, optional): Path to the OSZICAR file. Defaults to "OSZICAR.3static".
            contcar_name (str, optional): Path to the CONTCAR file. Defaults to "CONTCAR.3static".
            collect_mag_data (bool, optional): Whether to collect magnetic data. Defaults to False.
            magmom_tolerance (float, optional): Tolerance for magnetic moment. Defaults to 1e-12.
            total_magnetic_moment_tolerance (float, optional): Tolerance for total magnetic moment. Defaults to 1e-12.
            mass_average (str, optional): Available options are "arithmetic", "geometric", or "harmonic". Defaults to "geometric".
        """

        # Extract configuration data
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

        # Store extracted data
        self.volumes = all_volumes
        self.energies = all_energies
        self.mag_data = {
            str(index): {f"{item[0]}": {item[2]: item[1]} for item in sublist}
            for index, sublist in enumerate(all_mag_data_list.tolist())
        }
        self.total_magnetic_moment = all_total_magnetic_moments
        self.magnetic_ordering = all_magnetic_orderings

        # Filter data by volumes if provided
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

        # Get the volume folders
        vol_folders = self._get_volume_folders()

        # Filter volume folders based on the provided volumes
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

        # Read the relaxed structures from the CONTCAR files
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
        """Fit the energy-volume data to an equation of state (EOS).

        Args:
            eos_name (str, optional): Available options are "mBM4", "mBM5", "BM4", "BM5", "LOG4", "LOG5", "vinet", "murnaghan", and "morse". Defaults to "BM4".
            volume_min (float, optional): Minimum volume for fitted EOS. Defaults to None.
            volume_max (float, optional): Maximum volume for fitted EOS. Defaults to None.
            num_volumes (int, optional): Number of volumes for fitted EOS. Defaults to 1000.
        """

        eos_constants, eos_parameters, *_ = fit_to_eos(
            self.volumes,
            self.energies,
            eos_name,
            volume_min,
            volume_max,
            num_volumes,
        )

        self.eos_parameters = {
            "eos_name": eos_name,
            **dict(zip(["a", "b", "c", "d", "e"], eos_constants)),
            **dict(zip(["V0", "E0", "B", "BP", "B2P"], eos_parameters)),
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
        """Plots the energy-volume data and the fitted EOS.

        Args:
            volume_min (float, optional): Minimum volume for fitted EOS. Defaults to None.
            volume_max (float, optional): Maximum volume for fitted EOS. Defaults to None.
            num_volumes (int, optional): Number of volumes for fitted EOS. Defaults to 1000.
            eos_name (str, optional): Available options are "mBM4", "mBM5", "BM4", "BM5", "LOG4", "LOG5", "vinet", "murnaghan", and "morse". Defaults to "BM4".
            highlight_minimum (bool, optional): Whether to highlight the minimum energy. Defaults to True.
            per_atom (bool, optional): Whether to plot the energy per atom. Defaults to False.
            title (str, optional): Whether to add a title to the plot. Defaults to None.
            show_fig (bool, optional): Whether to show the plot. Defaults to True.
            cmap (str, optional): Color map for the plot. Defaults to "plotly".
            marker_alpha (float, optional): Transparency of the markers. Defaults to 1.0.
            marker_size (int, optional): Size of the markers. Defaults to 10.

        Returns:
            go.Figure: Plotly figure object containing the energy-volume plot.
        """

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