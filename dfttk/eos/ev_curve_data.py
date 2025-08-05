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

# DFTTK imports
from dfttk.magnetism import get_magnetic_structure
from dfttk.aggregate_extraction import extract_configuration_data
from dfttk.eos.fit import EOSFitter


class EvCurveData:
    """Class for handling energy-volume (E-V) curve data for a configuration."""

    def __init__(self, path: str, name: str) -> None:
        """
        Initialize the EvCurveData object.

        Args:
            path (str): Path to the directory containing the vol_* folders.
            name (str): Name of the configuration.
        """

        self.path = path
        self.name = name

        # VASP input files
        self.incars = []
        self.kpoints = []
        self.potcar = None
        self.initial_poscar = None

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
        self.eos_parameters = {key: None for key in ["eos_name", "a", "b", "c", "d", "e", "V0", "E0", "B", "BP", "B2P"]}

    def _get_volume_folders(self) -> list[str]:
        """
        Get the list of vol_* folders in the specified path, sorted in natural order.

        Returns:
            list[str]: List of vol_* folders sorted in natural order.
        """

        return natsorted([f for f in os.listdir(self.path) if f.startswith("vol_")])

    def get_vasp_input(self, incar_keys: list[str] = ["1relax", "2relax", "3static"], incar_names: list[str] = ["INCAR.1relax", "INCAR.2relax", "INCAR.3static"], kpoints_keys: list[str] = ["1relax", "2relax", "3static"], kpoints_names: list[str] = ["KPOINTS.1relax", "KPOINTS.2relax", "KPOINTS.3static"], contcar_name: str = "CONTCAR.3static", selected_volumes: list[float] = None, read_initial_poscar: bool = True) -> None:
        """
        Get the VASP input files from the specified path and store them in the class attributes.

        Args:
            incar_keys (list[str], optional): List of INCAR keys for dictionary keys. Defaults to ["1relax", "2relax", "3static"].
            incar_names (list[str], optional): List of INCAR names to read. Defaults to ["INCAR.1relax", "INCAR.2relax", "INCAR.3static"].
            kpoints_keys (list[str], optional): List of KPOINTS keys for dictionary keys. Defaults to ["1relax", "2relax", "3static"].
            kpoints_names (list[str], optional): List of KPOINTS names to read. Defaults to ["KPOINTS.1relax", "KPOINTS.2relax", "KPOINTS.3static"].
            contcar_name (str, optional): Name of the CONTCAR file. Defaults to "CONTCAR.3static".
            selected_volumes (list[float], optional): List of volumes to filter the folders. If None, all folders are considered.
            read_initial_poscar (bool, optional): Whether to read the initial POSCAR file. Defaults to True.
        """

        # Get the list of volume folders
        vol_folders = self._get_volume_folders()

        # Filter volume folders based on the provided selected_volumes
        if selected_volumes is not None:
            volumes_set = {round(volume, 2) for volume in selected_volumes}
            vol_folders = [
                vol_folder
                for vol_folder in vol_folders
                if os.path.exists(os.path.join(self.path, vol_folder, contcar_name))
                and round(
                    Structure.from_file(os.path.join(self.path, vol_folder, contcar_name)).volume,
                    2,
                )
                in volumes_set
            ]

        # Read the INCAR files for each volume folder
        for vol_folder in vol_folders:
            incar_data = {}
            for key, name in zip(incar_keys, incar_names):
                incar_data[key] = Incar.from_file(os.path.join(self.path, vol_folder, name))
            self.incars.append(incar_data)

        # Read the KPOINTS files for each volume folder
        for vol_folder in vol_folders:
            kpoints_data = {}
            for key, name in zip(kpoints_keys, kpoints_names):
                kpoints_data[key] = Kpoints.from_file(os.path.join(self.path, vol_folder, name))
            self.kpoints.append(kpoints_data)

        # Read the POTCAR file
        try:
            self.potcar = Potcar.from_file(os.path.join(self.path, vol_folders[0], "POTCAR"))
        except FileNotFoundError:
            self.potcar = None

        # Read the starting POSCAR file
        if read_initial_poscar:
            self.initial_poscar = Structure.from_file(os.path.join(self.path, "POSCAR"))

    def get_energy_volume_data(
        self,
        selected_volumes: list[float] = None,
        outcar_name: str = "OUTCAR.3static",
        oszicar_name: str = "OSZICAR.3static",
        contcar_name: str = "CONTCAR.3static",
        collect_mag_data: bool = False,
        magmom_tolerance: float = 0.01,
        total_magnetic_moment_tolerance: float = 0.01,
        mass_average: str = "geometric",
    ) -> None:
        """
        Get the energy-volume data from the specified path and store them in the class attributes.

        Args:
            selected_volumes (list[float], optional): List of volumes to filter the folders. If None, all folders are considered.
            outcar_name (str, optional): Name of the OUTCAR file. Defaults to "OUTCAR.3static".
            oszicar_name (str, optional): Name of the OSZICAR file. Defaults to "OSZICAR.3static".
            contcar_name (str, optional): Name of the CONTCAR file. Defaults to "CONTCAR.3static".
            collect_mag_data (bool, optional): Whether to collect magnetic data. Defaults to False.
            magmom_tolerance (float, optional): Tolerance for magnetic moment. Defaults to 0.01.
            total_magnetic_moment_tolerance (float, optional): Tolerance for total magnetic moment. Defaults to 0.01.
            mass_average (str, optional): "arithmetic", "geometric", or "harmonic". Defaults to "geometric".
        """

        # Extract configuration data
        (
            self.number_of_atoms,
            all_volumes,
            all_energies,
            self.atomic_masses,
            self.average_mass,
            all_mag_data_array,
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

        # Filter data by selected_volumes if provided
        if selected_volumes is not None:
            volumes_set = set(selected_volumes)
            filtered_indices = [i for i, v in enumerate(all_volumes) if v in volumes_set]
            self.volumes = np.array(all_volumes)[filtered_indices]
            self.energies = np.array(all_energies)[filtered_indices]
            self.mag_data = all_mag_data_array[filtered_indices] if len(all_mag_data_array) > 0 else []
            self.total_magnetic_moment = np.array(all_total_magnetic_moments[filtered_indices]) if len(all_total_magnetic_moments) > 0 else []
            self.magnetic_ordering = np.array(all_magnetic_orderings)[filtered_indices] if len(all_magnetic_orderings) > 0 else []

        if selected_volumes is None:
            self.total_magnetic_moment = all_total_magnetic_moments
            self.magnetic_ordering = all_magnetic_orderings

        # Convert all_mag_data_array to a list of dicts (one per volume)
        if selected_volumes is None:
            self.mag_data = []
            mag_data_source = all_mag_data_array
        else:
            mag_data_source = self.mag_data
            self.mag_data = []
        for mag_data_per_volume in mag_data_source:
            element_dict = {}
            for atom in mag_data_per_volume:
                # atom is typically (atom_index, magmom, element) or (atom_index, element, magmom)
                # Adjust the indices below if needed
                if isinstance(atom, dict):
                    element = atom.get("element")
                    magmom = atom.get("magmom")
                else:
                    # Try to infer element and magmom position
                    if isinstance(atom[1], str):
                        element = atom[1]
                        magmom = atom[2]
                    else:
                        element = atom[2]
                        magmom = atom[1]
                element_dict.setdefault(element, []).append(magmom)
            self.mag_data.append(element_dict)

        # Get the volume folders
        vol_folders = self._get_volume_folders()

        # Filter volume folders based on the provided selected_volumes
        if selected_volumes is not None:
            volumes_set = {round(volume, 2) for volume in selected_volumes}
            vol_folders = [
                vol_folder
                for vol_folder in vol_folders
                if os.path.exists(os.path.join(self.path, vol_folder, contcar_name))
                and round(
                    Structure.from_file(os.path.join(self.path, vol_folder, contcar_name)).volume,
                    2,
                )
                in volumes_set
            ]

        if collect_mag_data:
            # Read the magnetic structures from the CONTCAR files
            try:
                self.relaxed_structures = [
                    get_magnetic_structure(
                        os.path.join(self.path, vol_folder, contcar_name),
                        os.path.join(self.path, vol_folder, outcar_name),
                    )
                    for vol_folder in vol_folders
                ]
            except:
                # Read the relaxed structures from the CONTCAR files
                self.relaxed_structures = [Structure.from_file(os.path.join(self.path, vol_folder, contcar_name)) for vol_folder in vol_folders]
        else:
            # Read the relaxed structures from the CONTCAR files
            self.relaxed_structures = [Structure.from_file(os.path.join(self.path, vol_folder, contcar_name)) for vol_folder in vol_folders]

        # TODO: this might be a temporary fix for an empty magnetic_ordering issue. Revisit EOSFitter.
        # Ensure empty arrays are converted to empty lists for compatibility
        if isinstance(self.magnetic_ordering, np.ndarray) and self.magnetic_ordering.size == 0:
            self.magnetic_ordering = []
        if isinstance(self.total_magnetic_moment, np.ndarray) and self.total_magnetic_moment.size == 0:
            self.total_magnetic_moment = []

    def fit_energy_volume_data(
        self,
        eos_name: str = "BM4",
        volume_min: float = None,
        volume_max: float = None,
        num_volumes: int = 1000,
    ) -> None:
        """
        Fit the energy-volume data to an equation of state (EOS).

        Args:
            eos_name (str, optional): EOS function to use. Options: "mBM4", "mBM5", "BM4", "BM5", "LOG4", "LOG5", "vinet", "murnaghan", "morse". Defaults to "BM4".
            volume_min (float, optional): Minimum volume for fitted EOS. Defaults to None.
            volume_max (float, optional): Maximum volume for fitted EOS. Defaults to None.
            num_volumes (int, optional): Number of volumes for fitted EOS. Defaults to 1000.

        Raises:
            RuntimeError: If energy-volume data has not been loaded. You must call get_energy_volume_data() first.
        """

        if len(self.volumes) == 0 or len(self.energies) == 0:
            raise RuntimeError("You must call get_energy_volume_data() before fit_energy_volume_data().")
        self._fitter = EOSFitter(self.name, self.number_of_atoms, self.volumes, self.energies)
        self._fitter.fit(eos_name=eos_name, volume_min=volume_min, volume_max=volume_max, num_volumes=num_volumes)

        eos_constants = self._fitter.eos_constants
        eos_parameters = self._fitter.eos_parameters

        self.eos_parameters = {
            "eos_name": eos_name,
            **dict(zip(["a", "b", "c", "d", "e"], eos_constants)),
            **dict(zip(["V0", "E0", "B", "BP", "B2P"], eos_parameters)),
        }

    def plot(
        self,
        highlight_minimum: bool = True,
        per_atom: bool = False,
        title: str = None,
        cmap: str = "plotly",
        marker_alpha: float = 1.0,
        marker_size: int = 10,
    ) -> go.Figure:
        """
        Plot the energy-volume data and the fitted EOS.

        Args:
            highlight_minimum (bool, optional): Whether to highlight the minimum energy. Defaults to True.
            per_atom (bool, optional): Whether to plot the energy per atom. Defaults to False.
            title (str, optional): Title for the plot. Defaults to None.
            cmap (str, optional): Color map for the plot. Defaults to "plotly".
            marker_alpha (float, optional): Transparency of the markers. Defaults to 1.0.
            marker_size (int, optional): Size of the markers. Defaults to 10.

        Returns:
            go.Figure: Plotly figure object containing the energy-volume plot.
        """
        if not hasattr(self, "_fitter") or self._fitter is None:
            raise RuntimeError("You must call fit_energy_volume_data() before plotting.")
        fig = self._fitter.plot(
            highlight_minimum=highlight_minimum,
            per_atom=per_atom,
            title=title,
            cmap=cmap,
            marker_alpha=marker_alpha,
            marker_size=marker_size,
        )
        return fig
