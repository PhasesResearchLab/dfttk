"""
ThermalElectronicData class for storing and processing thermal electronic data from VASP calculations.
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
from dfttk.thermal_electronic.thermal_electronic import ThermalElectronic

class ThermalElectronicData:
    """
    Class for storing, processing, and plotting thermal electronic data from VASP calculations.

    Attributes:
        path (str): Path to the directory containing thermal electronic calculation results.
        incars (list[dict]): List of INCAR dictionaries for each electron DOS calculation.
        kpoints (Kpoints): KPOINTS object for the electron DOS calculations.
        potcar (Potcar): POTCAR object for the electron DOS calculations.
        structures (list[Structure]): List of Pymatgen Structure objects for each electron DOS calculation.
        electron_dos_data (pd.DataFrame): dataframe containing all electron DOS for different volumes.
        number_of_atoms (int): Number of atoms corresponding to the structures used in the electron DOS calculations.
        volumes (np.ndarray): Array of volumes for each structure, shape (n_volumes,).
        temperatures (np.ndarray): Array of temperatures used for thermodynamic calculations, shape (n_temperatures,).
        helmholtz_energies (np.ndarray): Helmholtz free energies (eV/atom), shape (n_temperatures, n_volumes).
        internal_energies (np.ndarray): Internal energies (eV/atom), shape (n_temperatures, n_volumes).
        entropies (np.ndarray): Entropies (eV/K/atom), shape (n_temperatures, n_volumes).
        heat_capacities (np.ndarray): Heat capacities (eV/K/atom), shape (n_temperatures, n_volumes).
        volumes_fit (np.ndarray): Volumes used for polynomial fits, shape (n_volumes_fit,).
        helmholtz_energies_fit (np.ndarray): Fitted Helmholtz free energies (n_temperatures, n_volumes_fit).
        entropies_fit (np.ndarray): Fitted entropies (n_temperatures, n_volumes_fit).
        heat_capacities_fit (np.ndarray): Fitted heat capacity data (n_temperatures, n_volumes_fit).
        helmholtz_energies_poly_coeffs (np.ndarray): Polynomial coefficients for Helmholtz energy fits.
        entropies_poly_coeffs (np.ndarray): Polynomial coefficients for entropy fits.
        heat_capacities_poly_coeffs (np.ndarray): Polynomial coefficients for heat capacity fits.
        _helmholtz_energies_fit_to_db (dict): Fitted Helmholtz energy data for database export.
        _entropies_fit_to_db (dict): Fitted entropy data for database export.
        _heat_capacities_fit_to_db (dict): Fitted heat capacity data for database export.
    """

    def __init__(self, path: str):
        """
        Initialize a ThermalElectronicData object.

        Args:
            path (str): Path to the directory containing the electron DOS VASP calculations.
        """

        self.path = path
        self.incars: list[dict] = []
        self.kpoints: list[dict] = []
        self.potcar: Potcar = None
        self.structures: list[Structure] = []
        self.electron_dos_data: pd.DataFrame = None

        self.number_of_atoms: int = None
        self.volumes: np.ndarray = None
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
            folder_prefix (str, optional): prefix of the folders to search for. Defaults to "elec".

        Returns:
            list[str]: list of folder names with prefix sorted in natural order.
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
        selected_volumes: list[float] = None,
        folder_prefix: str = "elec",
    ) -> None:
        """
        Get the VASP input files from the specified path and folders and store them in the class attributes.

        Args:
            incar_keys (list[str], optional): List of INCAR keys for dictionary keys. Defaults to ["elec_dos"].
            incar_names (list[str], optional): List of INCAR names to read. Defaults to ["INCAR.elec_dos"].
            kpoints_keys (list[str], optional): List of KPOINTS keys for dictionary keys. Defaults to ["elec_dos"].
            kpoints_names (list[str], optional): List of KPOINTS names to read. Defaults to ["KPOINTS.elec_dos"].
            contcar_name (str, optional): Name of the CONTCAR file. Defaults to "CONTCAR.elec_dos".
            selected_volumes (list[float], optional): _description_. Defaults to None.
            folder_prefix (str, optional): _description_. Defaults to "elec".
        """

        # Get the list of electronic DOS folders
        elec_folders = self._get_elec_folders(folder_prefix=folder_prefix)

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

        # Read the INCAR, KPOINTS, and structures for each electronic DOS folder
        for elec_folder in elec_folders:
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
        selected_volumes: np.ndarray = None,
        folder_prefix: str = "elec",
    ):
        """
        Loads and processes the thermal electronic data from the VASP calculations to compute thermodynamic properties.

        Args:
            volumes_fit (np.ndarray): 1D array of volumes for fitting the thermodynamic properties.
            temperatures (np.ndarray): 1D array of temperatures for computing thermodynamic properties.
            order (int): Order of the polynomial fit for the thermodynamic properties.
            selected_volumes (np.ndarray, optional): list of selected volumes to keep the electron DOS data. Defaults to None.
            folder_prefix (str, optional): prefix of the folders containing the vasprun.xml files. Defaults to "elec".
        """
        # Initialize ThermalElectronic object
        self.te = ThermalElectronic()

        # Load the total electron DOS data for all or selected volumes
        self.te.read_total_electron_dos(
            path=self.path,
            folder_prefix=folder_prefix,
            selected_volumes=selected_volumes,
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
                for temp, coeff in zip(self.temperatures, self.heat_capacities_poly_coeffs)
            }
        }

    def plot(
        self, property: str, selected_temperatures: np.ndarray = None
    ) -> tuple[go.Figure, go.Figure]:
        """
        Plot the thermodynamic properties and their polynomial fits.

        Args:
            property (str): Property to plot ('helmholtz_energy', 'entropy', or 'heat_capacity').
            selected_temperatures (np.ndarray, optional): Temperatures to plot for the fit.

        Raises:
            ValueError: If the property is not one of 'helmholtz_energy', 'entropy', or 'heat_capacity'.
            AttributeError: If the required attributes have not been calculated.

        Returns:
            tuple[go.Figure, go.Figure]: (property vs. temperature, property fit vs. volume)
        """

        property_mapping = {
            "helmholtz_energy": "helmholtz_energies",
            "entropy": "entropies",
            "heat_capacity": "heat_capacities",
        }

        if property not in property_mapping:
            raise ValueError(f"Invalid property_to_plot: {property}")

        property_name = property_mapping[property]
        # Check if the required attribute is calculated (not None)
        if (
            getattr(self, property_name) is None
            or getattr(self, f"{property_name}_fit") is None
        ):
            raise AttributeError(
                f"Attribute '{property_name}' and '{property_name}_fit' have not been calculated. Run get_thermal_electronic_data() in ThermalElectronicData or process_thermal_electronic() in Configuration."
            )
        # TODO: have to fix this!
        fig = self.te.plot_vt(
            property=property,
        )

        fig_fit = self.te.plot_vt(
            property=property,
            selected_temperatures=selected_temperatures,
        )

        return fig, fig_fit

    def plot_electron_dos(self) -> go.Figure:
        """
        Plots the total electron DOS for multiple volumes

        Returns:
            go.Figure: The figure object for the plot.
        """

        fig = self.te.plot_total_dos()
        return fig
