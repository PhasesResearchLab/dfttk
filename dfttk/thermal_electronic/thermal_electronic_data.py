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
from dfttk.thermal_electronic.functions import (
    thermal_electronic,
    fit_thermal_electronic,
    read_total_electron_dos,
    plot_total_electron_dos,
    plot_thermal_electronic,
    plot_thermal_electronic_properties_fit,
)


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
        _heat_capacities_fit_to_db (dict): Fitted heat capacity data for database export.d
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

    def get_total_electron_dos(
        self,
        selected_volumes: np.array = None,
        plot: bool = False,
        folder_prefix: str = "elec",
    ) -> None:
        """
        Reads the total electron DOS from vasprun.xml files in the specified path and folders
        and stores it in the electron_dos_data attribute as a pandas DataFrame.

        Args:
            selected_volumes (np.array, optional): list of selected volumes to keep the electron DOS data. Defaults to None.
            plot (bool, optional): if True, plots the total electron DOS for different volumes. Defaults to False.
            folder_prefix (str, optional): prefix of the folders containing the vasprun.xml files. Defaults to "elec".
        """
        self.electron_dos_data = read_total_electron_dos(
            self.path,
            selected_volumes=selected_volumes,
            plot=plot,
            folder_prefix=folder_prefix,
        )

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

        # Load the total electron DOS data for all or selected volumes
        self.get_total_electron_dos(
            selected_volumes=selected_volumes, folder_prefix=folder_prefix
        )
        self.number_of_atoms = self.electron_dos_data["number_of_atoms"].unique()[0]
        volumes = self.electron_dos_data["volumes"].unique()
        self.volumes = volumes
        self.temperatures = temperatures

        # Get the energy and DOS arrays for each volume
        energy_array = [
            self.electron_dos_data[self.electron_dos_data["volumes"] == volume][
                "energy_minus_fermi_energy"
            ].values[0]
            for volume in volumes
        ]
        dos_array = [
            self.electron_dos_data[self.electron_dos_data["volumes"] == volume][
                "total_dos"
            ].values[0]
            for volume in volumes
        ]
        energy_array = np.column_stack(energy_array)
        dos_array = np.column_stack(dos_array)

        # Compute the thermodynamic properties
        helmholtz_energies, internal_energies, entropies, heat_capacities = (
            thermal_electronic(
                volumes,
                temperatures,
                energy_array,
                dos_array,
            )
        )
        self.helmholtz_energies = helmholtz_energies
        self.internal_energies = internal_energies
        self.entropies = entropies
        self.heat_capacities = heat_capacities

        # Fit the thermodynamic properties vs. volumes for fixed temperatures
        (
            volumes_fit,
            helmholtz_energies_fit,
            entropies_fit,
            heat_capacities_fit,
            helmholtz_energies_poly_coeffs,
            entropies_poly_coeffs,
            heat_capacities_poly_coeffs,
        ) = fit_thermal_electronic(
            volumes=self.volumes,
            volumes_fit=volumes_fit,
            temperatures=self.temperatures,
            helmholtz_energies=helmholtz_energies,
            entropies=entropies,
            heat_capacities=heat_capacities,
            order=order,
        )

        self.volumes_fit = volumes_fit
        self.helmholtz_energies_fit = helmholtz_energies_fit
        self.entropies_fit = entropies_fit
        self.heat_capacities_fit = heat_capacities_fit
        self.helmholtz_energies_poly_coeffs = helmholtz_energies_poly_coeffs
        self.entropies_poly_coeffs = entropies_poly_coeffs
        self.heat_capacities_poly_coeffs = heat_capacities_poly_coeffs

        self._helmholtz_energies_fit_to_db = {
            "poly_coeffs": {
                f"{temp}K": coeff
                for temp, coeff in zip(
                    self.temperatures, helmholtz_energies_poly_coeffs
                )
            }
        }
        self._entropies_fit_to_db = {
            "poly_coeffs": {
                f"{temp}K": coeff
                for temp, coeff in zip(self.temperatures, entropies_poly_coeffs)
            }
        }
        self._heat_capacities_fit_to_db = {
            "poly_coeffs": {
                f"{temp}K": coeff
                for temp, coeff in zip(self.temperatures, heat_capacities_poly_coeffs)
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

        property_data = getattr(self, property_name)
        property_fit_data = getattr(self, f"{property_name}_fit")

        fig = plot_thermal_electronic(
            number_of_atoms=self.number_of_atoms,
            volumes=self.volumes,
            temperatures=self.temperatures,
            property=property_data,
            property_name=property,
        )

        fig_fit = plot_thermal_electronic_properties_fit(
            number_of_atoms=self.number_of_atoms,
            volumes=self.volumes,
            temperatures=self.temperatures,
            property_name=property,
            property=property_data,
            volumes_fit=self.volumes_fit,
            property_fit=property_fit_data,
            selected_temperatures_plot=selected_temperatures,
        )

        return fig, fig_fit

    def plot_electron_dos(self) -> go.Figure:
        """
        Plots the total electron DOS for multiple volumes

        Returns:
            go.Figure: The figure object for the plot.
        """

        fig = plot_total_electron_dos(self.electron_dos_data)
        return fig
