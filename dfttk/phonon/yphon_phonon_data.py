"""
YphonPhononData class for storing and processing phonon data from VASP and YPHON calculations.
"""

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
        helmholtz_energies (np.ndarray): Helmholtz free energy (eV/atom), shape (n_temperatures, n_volumes).
        internal_energies (np.ndarray): Internal energy (eV/atom), shape (n_temperatures, n_volumes).
        entropies (np.ndarray): Entropy (eV/K/atom), shape (n_temperatures, n_volumes).
        heat_capacities (np.ndarray): Heat capacity (eV/K/atom), shape (n_temperatures, n_volumes).
        volumes_fit (np.ndarray): Volumes used for polynomial fits, shape (n_volumes_fit,).
        helmholtz_energies_fit (dict): Fitted Helmholtz energy data.
        entropies_fit (dict): Fitted entropy data.
        heat_capacities_fit (dict): Fitted heat capacity data.
        helmholtz_energies_poly_coeffs (np.ndarray): Polynomial coefficients for Helmholtz energy fits.
        entropies_poly_coeffs (np.ndarray): Polynomial coefficients for entropy fits.
        heat_capacities_poly_coeffs (np.ndarray): Polynomial coefficients for heat capacity fits.
        _harmonic_phonon (HarmonicPhononYphon): Cached instance for calculations and plotting.
    """

    def __init__(self, path: str):
        """
        Initialize a YphonPhononData object.

        Args:
            path (str): Path to the directory containing phonon calculation results.
        """

        self.path = path
        self.incars: list[dict] = []
        self.kpoints: list[dict] = []
        self.potcar: Potcar = None
        self.phonon_structures: list[Structure] = []

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
        self._harmonic_phonon = None

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

    def get_vasp_input(self, selected_phonon_volumes: np.ndarray = None) -> None:
        """
        Load VASP files (INCAR, KPOINTS, POTCAR, CONTCAR) for each phonon folder.

        Args:
            selected_phonon_volumes (np.ndarray, optional): If provided, only load structures with these phonon volumes.
        """
        phonon_folders = self._get_phonon_folders()
        incar_keys = ["1relax", "2phonons"]
        kpoints_keys = ["1relax", "2phonons"]

        if selected_phonon_volumes is not None:
            # Filter phonon folders based on selected volumes
            volumes_set = {round(volume, 2) for volume in selected_phonon_volumes}
            phonon_folders = [
                phonon_folder
                for phonon_folder in phonon_folders
                if os.path.exists(os.path.join(self.path, phonon_folder, "CONTCAR.2phonons"))
                and round(
                    Structure.from_file(os.path.join(self.path, phonon_folder, "CONTCAR.2phonons")).volume,
                    2,
                )
                in volumes_set
            ]

        volumes_per_atom = []
        for phonon_folder in phonon_folders:
            incar_data = {}
            for key in incar_keys:
                try:
                    incar_data[key] = Incar.from_file(os.path.join(self.path, phonon_folder, f"INCAR.{key}"))
                except FileNotFoundError:
                    if key == "1relax":
                        continue
                    else:
                        raise
            self.incars.append(incar_data)

            structure = Structure.from_file(os.path.join(self.path, phonon_folder, "CONTCAR.2phonons"))
            self.phonon_structures.append(structure)
            phonon_atoms = structure.num_sites
            volumes_per_atom.append(round(structure.volume / phonon_atoms, 2))

            kpoints_data = {}
            for key in kpoints_keys:
                try:
                    kpoints_data[key] = Kpoints.from_file(os.path.join(self.path, phonon_folder, f"KPOINTS.{key}"))
                except FileNotFoundError:
                    if key == "1relax":
                        continue
                    else:
                        raise
            self.kpoints.append(kpoints_data)
        self._volumes_per_atom = np.array(volumes_per_atom)

        try:
            first_phonon_folder = phonon_folders[0]
            potcar_path = os.path.join(self.path, first_phonon_folder, "POTCAR")
            self.potcar = Potcar.from_file(potcar_path)
        except FileNotFoundError:
            self.potcar = None

    def get_harmonic_data(
        self,
        number_of_atoms: int,
        temperatures: np.ndarray,
        order: int = 2,
        selected_volumes: np.ndarray = None,
    ) -> None:
        """
        Load, scale, and process YPHON phonon DOS data and compute thermodynamic properties.

        Args:
            number_of_atoms (int): Number of atoms to scale the phonon DOS to.
            temperatures (np.ndarray): Array of temperatures (K) for property calculation.
            order (int): Polynomial order for fitting thermodynamic properties. Defaults to 2.
            selectedd_volumes (np.ndarray, optional): If provided, only calculate properties using these volumes. The volumes are volumes per atom multiplied by the number of atoms.
        """
        yphon_results_path = os.path.join(self.path, "YPHON_results")
        self._harmonic_phonon = HarmonicPhononYphon()
        hp = self._harmonic_phonon
        hp.load_dos(yphon_results_path)
        hp.scale_dos(number_of_atoms)
        self.number_of_atoms = number_of_atoms
        self.volumes = self._volumes_per_atom * self.number_of_atoms
        self.temperatures = temperatures
        hp.calculate_harmonic(temperatures=temperatures, selected_volumes=selected_volumes)
        self.helmholtz_energies = hp.helmholtz_energies
        self.internal_energies = hp.internal_energies
        self.entropies = hp.entropies
        self.heat_capacities = hp.heat_capacities
        hp.fit_harmonic(order=order)
        self.volumes_fit = hp.volumes_fit
        self.helmholtz_energies_fit = hp.helmholtz_energies_fit
        self.entropies_fit = hp.entropies_fit
        self.heat_capacities_fit = hp.heat_capacities_fit
        self.helmholtz_energies_poly_coeffs = hp.helmholtz_energies_poly_coeffs
        self.entropies_poly_coeffs = hp.entropies_poly_coeffs
        self.heat_capacities_poly_coeffs = hp.heat_capacities_poly_coeffs

    def plot_scaled_dos(self, number_of_atoms: int, plot: bool = True) -> go.Figure:
        """
        Plot the scaled phonon DOS for the specified number of atoms.

        Args:
            number_of_atoms (int): Number of atoms to scale the DOS to.
            plot (bool): If True, display the plot.

        Returns:
            go.Figure: The Plotly figure object for the scaled DOS.

        Raises:
            RuntimeError: If get_harmonic_data() has not been called.
        """
        if self._harmonic_phonon is None:
            raise RuntimeError("Call get_harmonic_data() before plotting.")
        return self._harmonic_phonon.scale_dos(number_of_atoms=number_of_atoms, plot=plot)

    def plot_multiple_dos(self) -> go.Figure:
        """
        Plot the scaled phonon DOS for multiple volumes.

        Returns:
            go.Figure: The Plotly figure object for the multiple DOS plot.

        Raises:
            RuntimeError: If get_harmonic_data() has not been called.
        """
        if self._harmonic_phonon is None:
            raise RuntimeError("Call get_harmonic_data() before plotting.")
        return self._harmonic_phonon.plot_dos()

    def plot_harmonic(self, property: str, selected_temperatures: np.ndarray = None) -> tuple[go.Figure, go.Figure]:
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
        fig_harmonic = self._harmonic_phonon.plot_harmonic(property=property)
        fig_fit_harmonic = self._harmonic_phonon.plot_fit_harmonic(
            property=property,
            selected_temperatures=selected_temperatures,
        )
        return fig_harmonic, fig_fit_harmonic
