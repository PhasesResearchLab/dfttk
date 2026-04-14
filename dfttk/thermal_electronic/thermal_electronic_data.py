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
from dfttk.thermal_electronic import ThermalElectronic


class ThermalElectronicData:
    """
    A convenience class used by the Configuration class.

    Uses the ThermalElectronic class to process thermal electronic data from a set of VASP
    calculations at different volumes and generate plots. It also includes a built-in method
    to store the VASP input files used for the electron DOS calculations.

    Args:
        path: Path to the directory containing the electron DOS VASP calculations.

    Attributes:
        path: Path to the directory containing electronic DOS data.
        incars: List of INCAR objects for each volume.
        kpoints: List of KPOINTS objects for each volume.
        potcar: POTCAR object.
        structures: Relaxed structures for each volume.

        number_of_atoms: Number of atoms used in the DOS calculations.
        volumes: Array of volumes for each structure, shape (n_volumes,), in Å³.
        temperatures: Array of temperatures, shape (n_temperatures,), in K.

        helmholtz_energies:
            Helmholtz free energy as a function of temperature and volume,
            shape (n_temperatures, n_volumes), in eV.
        internal_energies:
            Internal energy as a function of temperature and volume,
            shape (n_temperatures, n_volumes), in eV.
        entropies:
            Entropy as a function of temperature and volume,
            shape (n_temperatures, n_volumes), in eV/K.
        heat_capacities:
            Heat capacity as a function of temperature and volume,
            shape (n_temperatures, n_volumes), in eV/K.

        volumes_fit:
            Volumes used for polynomial fits, shape (n_volumes_fit,), in Å³.
        helmholtz_energies_fit:
            Polynomial-fitted Helmholtz free energies as a function of
            temperature and fitted volume, shape (n_temperatures, n_volumes_fit), in eV.
        entropies_fit:
            Polynomial-fitted entropies as a function of temperature and
            fitted volume, shape (n_temperatures, n_volumes_fit), in eV/K.
        heat_capacities_fit:
            Polynomial-fitted heat capacities as a function of temperature and
            fitted volume, shape (n_temperatures, n_volumes_fit), in eV/K.

        helmholtz_energies_poly_coeffs:
            Polynomial coefficients for Helmholtz free energy fits as a function
            of volume, shape (n_temperatures, order + 1).
        entropies_poly_coeffs:
            Polynomial coefficients for entropy fits as a function of volume,
            shape (n_temperatures, order + 1).
        heat_capacities_poly_coeffs:
            Polynomial coefficients for heat capacity fits as a function of
            volume, shape (n_temperatures, order + 1).

        _helmholtz_energies_fit_to_db:
            Reformatted helmholtz_energies_poly_coeffs for database export.
        _entropies_fit_to_db:
            Reformatted entropies_poly_coeffs for database export.
        _heat_capacities_fit_to_db:
            Reformatted heat_capacities_poly_coeffs for database export.
    """

    def __init__(self, path: str):

        self.path = path
        self.incars: list[dict] = []
        self.kpoints: list[dict] = []
        self.potcar: Potcar = None
        self.structures: list[Structure] = []

        self.number_of_atoms: int = None
        self.volumes: np.ndarray = None
        self.energies_list: list = None
        self.dos_list: list = None
        
        self.temperatures: np.ndarray = None
        self.helmholtz_energies: np.ndarray = None
        self.internal_energies: np.ndarray = None
        self.entropies: np.ndarray = None
        self.heat_capacities: np.ndarray = None

        self.volumes_fit: np.ndarray = None
        self.helmholtz_energies_fit: np.ndarray = None
        self.entropies_fit: np.ndarray = None
        self.heat_capacities_fit: np.ndarray = None
        self.helmholtz_energies_poly_coeffs: np.ndarray = None
        self.entropies_poly_coeffs: np.ndarray = None
        self.heat_capacities_poly_coeffs: np.ndarray = None

        self._helmholtz_energies_fit_to_db: dict = None
        self._entropies_fit_to_db: dict = None
        self._heat_capacities_fit_to_db: dict = None

    def _get_elec_folders(self, folder_prefix: str = "elec") -> list[str]:
        """
        Get the list of folders with prefix in the specified path, sorted in natural order.

        Args:
            folder_prefix: Prefix of the electronic folders. Defaults to ``"elec"``.

        Returns: list of folder names with prefix sorted in natural order.
        """

        return natsorted(
            [f for f in os.listdir(self.path) if f.startswith(folder_prefix)]
        )

    def get_vasp_data(
        self,
        incar_keys: list[str] = ["elec_dos"],
        incar_names: list[str] = ["INCAR.elec_dos"],
        kpoints_keys: list[str] = ["elec_dos"],
        kpoints_names: list[str] = ["KPOINTS.elec_dos"],
        contcar_name: str = "CONTCAR.elec_dos",
        selected_volumes: np.ndarray | None = None,
        selected_folders: list[str] | None = None,
        folder_prefix: str = "elec",
    ) -> None:
        """
        Get relevant VASP data from the specified path and folders and store them in
        the class attributes.

        Args:
            incar_keys: List of INCAR keys for dictionary keys. Defaults to ["elec_dos"].
            incar_names: List of INCAR files to read. Defaults to ["INCAR.elec_dos"].
            kpoints_keys: List of KPOINTS keys for dictionary keys. Defaults to ["elec_dos"].
            kpoints_names: List of KPOINTS files to read. Defaults to ["KPOINTS.elec_dos"].
            contcar_name: Name of the CONTCAR file to read. Defaults to "CONTCAR.elec_dos".
            selected_volumes: NumPy array of selected volumes to keep the electron
                DOS data, or None to keep all volumes. Defaults to None.
            selected_folders: List of selected folders to keep the electron DOS
                data. Defaults to None.
            folder_prefix: Prefix of the electronic folders. Defaults to ``"elec"``.

        Raises:
            ValueError: If both `selected_volumes` and `selected_folders` are provided,
                as they are mutually exclusive.
        """

        # Get the list of electronic DOS folders
        elec_folders = self._get_elec_folders(folder_prefix=folder_prefix)

        if selected_volumes is not None and selected_folders is not None:
            raise ValueError(
                "selected_volumes and selected_folders are mutually exclusive. "
                "Provide only one of them."
            )
            
        # Filter electronic DOS folders based on selected volumes if provided
        if selected_volumes is not None:
            volumes_set = {round(volume, 2) for volume in selected_volumes}
            elec_folders = [
                elec_folder
                for elec_folder in elec_folders
                if os.path.exists(os.path.join(self.path, elec_folder, contcar_name))
                and round(
                    Structure.from_file(
                        os.path.join(self.path, elec_folder, contcar_name)
                    ).volume,
                    2,
                )
                in volumes_set
            ]

        # Iterate over the requested folders; if none are provided, use all detected elec folders
        folders_to_process = natsorted(
            elec_folders if selected_folders is None else selected_folders
        )
        
        # Read the INCAR, KPOINTS, and structures for each electronic DOS folder
        for elec_folder in folders_to_process:
            incar_data = {}
            for key, name in zip(incar_keys, incar_names):
                incar_data[key] = Incar.from_file(
                    os.path.join(self.path, elec_folder, name)
                )
            self.incars.append(incar_data)

            kpoints_data = {}
            for key, name in zip(kpoints_keys, kpoints_names):
                kpoints_data[key] = Kpoints.from_file(
                    os.path.join(self.path, elec_folder, name)
                )
            self.kpoints.append(kpoints_data)

            structure = Structure.from_file(
                os.path.join(self.path, elec_folder, contcar_name)
            )
            self.structures.append(structure)

        # Read the POTCAR file
        try:
            self.potcar = Potcar.from_file(os.path.join(self.path, "POTCAR"))
        except FileNotFoundError:
            self.potcar = None

    def get_thermal_electronic_data(
        self,
        volumes_fit: np.ndarray,
        temperatures: np.ndarray,
        order: int = 1,
        folder_prefix: str = "elec",
        vasprun_name: str = "vasprun.xml.elec_dos",
        selected_volumes: np.ndarray | None = None,
        selected_folders: list[str] | None = None,
    ):
        """
        Calls the ThermalElectronic class to read the total electron DOS data, compute
        the thermodynamic properties, and generate plots.

        Args:
            volumes_fit: Volumes used for polynomial fits, shape (n_volumes_fit,), in Å³.
            temperatures: Array of temperatures, shape (n_temperatures,), in K.
            order: Order of the polynomial fit. Defaults to 1 (linear fit).
            folder_prefix: Prefix of the electronic folders. Defaults to ``"elec"``.
            vasprun_name: Name of the vasprun.xml file. Defaults to ``"vasprun.xml.elec_dos"``.
            selected_volumes: Array of volumes to process. Defaults to None.
            selected_folders: List of selected folders to keep the electron DOS data. Defaults to None.
        """

        # Initialize ThermalElectronic object
        self.te = ThermalElectronic()

        # Load the total electron DOS data for all or selected volumes
        self.te.read_total_electron_dos(
            path=self.path,
            folder_prefix=folder_prefix,
            vasprun_name=vasprun_name,
            selected_volumes=selected_volumes,
            selected_folders=selected_folders,
        )
        self.number_of_atoms = self.te.number_of_atoms
        self.volumes = self.te.volumes
        self.energies_list = self.te.energies_list
        self.dos_list = self.te.dos_list

        # Compute and fit the thermodynamic properties
        self.te.process(temperatures=temperatures)
        self.te.fit(volumes_fit=volumes_fit, order=order)

        self.temperatures = self.te.temperatures
        self.helmholtz_energies = self.te.helmholtz_energies
        self.internal_energies = self.te.internal_energies
        self.entropies = self.te.entropies
        self.heat_capacities = self.te.heat_capacities

        self.volumes_fit = self.te.volumes_fit
        self.helmholtz_energies_fit = self.te.helmholtz_energies_fit
        self.entropies_fit = self.te.entropies_fit
        self.heat_capacities_fit = self.te.heat_capacities_fit
        self.helmholtz_energies_poly_coeffs = self.te.helmholtz_energies_poly_coeffs
        self.entropies_poly_coeffs = self.te.entropies_poly_coeffs
        self.heat_capacities_poly_coeffs = self.te.heat_capacities_poly_coeffs

        self._helmholtz_energies_fit_to_db = {
            "poly_coeffs": {
                f"{temp}K": coeff
                for temp, coeff in zip(
                    self.temperatures, self.helmholtz_energies_poly_coeffs
                )
            }
        }
        self._entropies_fit_to_db = {
            "poly_coeffs": {
                f"{temp}K": coeff
                for temp, coeff in zip(self.temperatures, self.entropies_poly_coeffs)
            }
        }
        self._heat_capacities_fit_to_db = {
            "poly_coeffs": {
                f"{temp}K": coeff
                for temp, coeff in zip(
                    self.temperatures, self.heat_capacities_poly_coeffs
                )
            }
        }

    def plot_total_dos(self) -> go.Figure:
        """
        Plots the total electron DOS for different volumes.

        Returns: Plotly figure object containing the total electron DOS curves for
            the different volumes.
        """

        fig = self.te.plot_total_dos()
        return fig

    def plot_vt(self, type: str, selected_temperatures: np.ndarray = None) -> go.Figure:
        """
        Plots thermal electronic properties as a function of temperature or volume.

        Args:
            type:
                Must be one of the following values:
                ``'helmholtz_energy_vs_temperature'``, ``'entropy_vs_temperature'``,
                ``'heat_capacity_vs_temperature'``, ``'helmholtz_energy_vs_volume'``,
                ``'entropy_vs_volume'``, or ``'heat_capacity_vs_volume'``.
            selected_temperatures:
                Selected temperatures to use for volume plots, shape
                (n_selected_temperatures,). Defaults to None.

        Returns: Plotly figure object containing the requested thermal electronic
            property curves.
        """

        fig = self.te.plot_vt(
            type=type,
            selected_temperatures=selected_temperatures,
        )
        return fig
