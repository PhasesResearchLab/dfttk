'''
YphonPhononData class for storing and processing phonon data from VASP and YPHON calculations.
'''

# Standard library imports
import os

# Third-party imports
import numpy as np
from natsort import natsorted
import plotly.graph_objects as go
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar

# DFTTK imports
from dfttk.workflows import process_phonon_dos_YPHON
from dfttk.phonon.harmonic_phonon_yphon import HarmonicPhononYphon

class YphonPhononData:
    """
    Class for storing and processing phonon data from VASP and YPHON calculations.

    This class manages VASP input/output, loads and processes YPHON phonon DOS results,
    and provides methods for calculating and plotting thermodynamic properties using the
    harmonic approximation.

    Attributes:
        path (str): Path to the directory containing phonon calculation results.
        incars (list[dict]): List of INCAR dictionaries for each phonon calculation.
        kpoints (Kpoints): KPOINTS object for the calculations.
        potcar (Potcar): POTCAR object for the calculations.
        phonon_structures (list[Structure]): List of Pymatgen Structure objects for each phonon calculation.
        number_of_atoms (int): Number of atoms used for scaling the phonon DOS.
        volumes (np.ndarray): Array of volumes for each structure, shape (n_volumes,).
        temperatures (np.ndarray): Array of temperatures used for thermodynamic calculations, shape (n_temperatures,).
        helmholtz_energy (np.ndarray): Helmholtz free energy (eV/atom), shape (n_temperatures, n_volumes).
        internal_energy (np.ndarray): Internal energy (eV/atom), shape (n_temperatures, n_volumes).
        entropy (np.ndarray): Entropy (eV/K/atom), shape (n_temperatures, n_volumes).
        heat_capacity (np.ndarray): Heat capacity (eV/K/atom), shape (n_temperatures, n_volumes).
        volumes_fit (np.ndarray): Volumes used for polynomial fits, shape (n_volumes_fit,).
        helmholtz_energy_fit (dict): Fitted Helmholtz energy data.
        entropy_fit (dict): Fitted entropy data.
        heat_capacity_fit (dict): Fitted heat capacity data.
        helmholtz_energy_poly_coeffs (np.ndarray): Polynomial coefficients for Helmholtz energy fits.
        entropy_poly_coeffs (np.ndarray): Polynomial coefficients for entropy fits.
        heat_capacity_poly_coeffs (np.ndarray): Polynomial coefficients for heat capacity fits.
        _harmonic_phonon (HarmonicPhononYphon): Cached instance for calculations and plotting.
        _helmholtz_energy_to_db (dict): Helmholtz energy data formatted for database export.
        _internal_energy_to_db (dict): Internal energy data formatted for database export.
        _entropy_to_db (dict): Entropy data formatted for database export.
        _heat_capacity_to_db (dict): Heat capacity data formatted for database export.
        _helmholtz_energy_fit_to_db (dict): Fitted Helmholtz energy data for database export.
        _entropy_fit_to_db (dict): Fitted entropy data for database export.
        _heat_capacity_fit_to_db (dict): Fitted heat capacity data for database export.
    """

    def __init__(self, path: str):
        """
        Initialize a YphonPhononData object.

        Args:
            path (str): Path to the directory containing phonon calculation results.
        """
        
        self.path = path
        self.incars: list[dict] = []
        self.kpoints: Kpoints = None
        self.potcar: Potcar = None
        self.phonon_structures: list[Structure] = []
        
        self.number_of_atoms: int = None
        self.volumes: np.ndarray = None
        self.temperatures: np.ndarray = None
        self.helmholtz_energy: np.ndarray = None
        self.internal_energy: np.ndarray = None
        self.entropy: np.ndarray = None
        self.heat_capacity: np.ndarray = None
        
        self.volumes_fit: np.ndarray = None
        self.helmholtz_energy_fit: dict = None
        self.entropy_fit: dict = None
        self.heat_capacity_fit: dict = None
        self.helmholtz_energy_poly_coeffs = None
        self.entropy_poly_coeffs = None
        self.heat_capacity_poly_coeffs = None
        self._harmonic_phonon = None  # cache the instance

    def process_phonon_dos(self):
        """
        Run YPHON post-processing on the phonon calculation directory.
        """
        process_phonon_dos_YPHON(self.path)

    def _get_phonon_folders(self):
        """
        Get a sorted list of phonon calculation subfolders in the current path.

        Returns:
            list[str]: Sorted list of phonon folder names.
        """
        return natsorted([f for f in os.listdir(self.path) if f.startswith("phonon_")])

    def get_vasp_input(self, volumes: list[float] = None):
        """
        Load VASP files (INCAR, KPOINTS, POTCAR, CONTCAR) for each phonon folder.

        Args:
            volumes (list[float], optional): If provided, only load structures with these volumes.
        """
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

    def get_harmonic_data(
        self,
        number_of_atoms: int,
        temperatures: np.ndarray,
        order: int,
    ) -> None:
        """
        Load, scale, and process YPHON phonon DOS data and compute thermodynamic properties.

        Args:
            number_of_atoms (int): Number of atoms to scale the phonon DOS to.
            temperatures (np.ndarray): Array of temperatures (K) for property calculation.
            order (int): Polynomial order for fitting thermodynamic properties.
        """
        yphon_results_path = os.path.join(self.path, "YPHON_results")
        self._harmonic_phonon = HarmonicPhononYphon()
        hp = self._harmonic_phonon
        hp.load_dos(yphon_results_path)
        hp.scale_dos(number_of_atoms)
        self.number_of_atoms = number_of_atoms
        self.volumes = hp.volumes
        self.temperatures = temperatures
        hp.calculate_harmonic(temperatures)
        self.helmholtz_energy = hp.helmholtz_energy
        self.internal_energy = hp.internal_energy
        self.entropy = hp.entropy
        self.heat_capacity = hp.heat_capacity
        hp.fit_harmonic(order=order)
        self.volumes_fit = hp.volumes_fit
        self.helmholtz_energy_fit = hp.helmholtz_energy_fit
        self.entropy_fit = hp.entropy_fit
        self.heat_capacity_fit = hp.heat_capacity_fit
        self.helmholtz_energy_poly_coeffs = hp.helmholtz_energy_poly_coeffs
        self.entropy_poly_coeffs = hp.entropy_poly_coeffs
        self.heat_capacity_poly_coeffs = hp.heat_capacity_poly_coeffs

        self._helmholtz_energy_to_db = {
            f"{temp}K": self.helmholtz_energy[i] for i, temp in enumerate(self.temperatures)
        }
        self._internal_energy_to_db = {
            f"{temp}K": self.internal_energy[i] for i, temp in enumerate(self.temperatures)
        }
        self._entropy_to_db = {
            f"{temp}K": self.entropy[i] for i, temp in enumerate(self.temperatures)
        }
        self._heat_capacity_to_db = {
            f"{temp}K": self.heat_capacity[i] for i, temp in enumerate(self.temperatures)
        }
        self._helmholtz_energy_fit_to_db = {
            "poly_coeffs": {
                f"{temp}K": coeff for temp, coeff in zip(self.temperatures, self.helmholtz_energy_poly_coeffs)
            }
        }
        self._entropy_fit_to_db = {
            "poly_coeffs": {
                f"{temp}K": coeff for temp, coeff in zip(self.temperatures, self.entropy_poly_coeffs)
            }
        }
        self._heat_capacity_fit_to_db = {
            "poly_coeffs": {
                f"{temp}K": coeff for temp, coeff in zip(self.temperatures, self.heat_capacity_poly_coeffs)
            }
        }

    def plot_scaled_dos(self, number_of_atoms: int, plot: bool = True) -> None:
        """
        Plot the scaled phonon DOS for the specified number of atoms.

        Args:
            number_of_atoms (int): Number of atoms to scale the DOS to.
            plot (bool): If True, display the plot.
        
        Raises:
            RuntimeError: If get_harmonic_data() has not been called.
        """
        if self._harmonic_phonon is None:
            raise RuntimeError("Call get_harmonic_data() before plotting.")
        self._harmonic_phonon.scale_dos(number_of_atoms=number_of_atoms, plot=plot)

    def plot_multiple_dos(self, number_of_atoms: int) -> None:
        """
        Plot the scaled phonon DOS for multiple volumes.

        Args:
            number_of_atoms (int): Number of atoms to scale the DOS to.
        
        Raises:
            RuntimeError: If get_harmonic_data() has not been called.
        """
        if self._harmonic_phonon is None:
            raise RuntimeError("Call get_harmonic_data() before plotting.")
        self._harmonic_phonon.plot_dos()

    def plot_harmonic(
        self, property: str, selected_temperatures: np.ndarray = None
    ) -> tuple[go.Figure, go.Figure]:
        """
        Plot harmonic thermodynamic properties and their polynomial fits.

        Args:
            property (str): Property to plot ('helmholtz_energy', 'entropy', or 'heat_capacity').
            selected_temperatures (np.ndarray, optional): Temperatures to plot for the fit.

        Returns:
            tuple[go.Figure, go.Figure]: (property vs. temperature, property fit vs. volume)
        
        Raises:
            RuntimeError: If get_harmonic_data() has not been called.
            ValueError: If property is not a valid option.
        """
        if self._harmonic_phonon is None:
            raise RuntimeError("Call get_harmonic_data() before plotting.")
        properties = ["helmholtz_energy", "entropy", "heat_capacity"]
        if property not in properties:
            raise ValueError(f"Invalid property_to_plot: {property}")
        fig_harmonic = self._harmonic_phonon.plot_harmonic(property_name=property)
        fig_fit_harmonic = self._harmonic_phonon.plot_fit_harmonic(
            property_name=property,
            selected_temperatures_plot=selected_temperatures,
        )
        return fig_harmonic, fig_fit_harmonic

