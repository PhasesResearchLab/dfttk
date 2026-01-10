# Standard Library Imports
import os
import warnings
import numpy as np

# Related third party imports
import numpy as np
import plotly.graph_objects as go
import scipy.constants
from scipy.special import expit
from scipy.optimize import bisect
from natsort import natsorted
from scipy.interpolate import UnivariateSpline

# Local application/library specific imports
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin

# DFTTK imports
from dfttk.plotly_format import plot_format

BOLTZMANN_CONSTANT = (
    scipy.constants.Boltzmann / scipy.constants.electron_volt
)  # The Boltzmann constant in eV/K


# TODO: Update methods in other modules to reflect all updates made here!
class ThermalElectronic:
    """
    A class for reading or setting electronic density-of-states (DOS) data,
    calculating thermal electronic properties, and generating plots.

    Typical usage:

        1. Load electronic DOS data from VASP calculations for multiple volumes
        using `read_total_electron_dos()`, or provide DOS data directly with
        `set_total_electron_dos()`.\n
        2. Compute thermal electronic contributions (Helmholtz free energy,
        internal energy, entropy, and heat capacity) using `process()` and `fit()`.\n
        3. Visualize results with the provided plotting methods.

    Additional intermediate methods are available for calculating the chemical
    potential, fitting the DOS, computing the Fermi-Dirac distribution, and
    related quantities.

    Attributes:
        path (str): Path to the directory containing electronic DOS data.
        number_of_atoms (int): Number of atoms used in the DOS calculations.
        nelect (int): Total number of electrons in the DOS data.

        volumes (np.ndarray):
            Array of volumes for each structure, shape (n_volumes,), in Å³.
        energies_list (list[np.ndarray]):
            List of arrays of electronic energies referenced to the Fermi level
            (:math:`E - E_F`) for each volume, in eV.
        dos_list (list[np.ndarray]):
            List of arrays of electronic DOS values for each volume, in states/eV.

        temperatures (np.ndarray):
            Array of temperatures, shape (n_temperatures,), in K.
        helmholtz_energies (np.ndarray):
            Helmholtz free energy as a function of temperature and volume,
            shape (n_temperatures, n_volumes), in eV.
        internal_energies (np.ndarray):
            Internal energy as a function of temperature and volume,
            shape (n_temperatures, n_volumes), in eV.
        entropies (np.ndarray):
            Entropy as a function of temperature and volume,
            shape (n_temperatures, n_volumes), in eV/K.
        heat_capacities (np.ndarray):
            Heat capacity as a function of temperature and volume,
            shape (n_temperatures, n_volumes), in eV/K.

        volumes_fit (np.ndarray):
            Volumes used for polynomial fits, shape (n_volumes_fit,), in Å³.
        helmholtz_energies_fit (np.ndarray):
            Polynomial-fitted Helmholtz free energies as a function of
            temperature and fitted volume, shape (n_temperatures, n_volumes_fit), in eV.
        entropies_fit (np.ndarray):
            Polynomial-fitted entropies as a function of temperature and
            fitted volume, shape (n_temperatures, n_volumes_fit), in eV/K.
        heat_capacities_fit (np.ndarray):
            Polynomial-fitted heat capacities as a function of temperature and
            fitted volume, shape (n_temperatures, n_volumes_fit), in eV/K.

        helmholtz_energies_poly_coeffs (np.ndarray):
            Polynomial coefficients for Helmholtz free energy fits as a function
            of volume, shape (n_temperatures, order + 1).
        entropies_poly_coeffs (np.ndarray):
            Polynomial coefficients for entropy fits as a function of volume,
            shape (n_temperatures, order + 1).
        heat_capacities_poly_coeffs (np.ndarray):
            Polynomial coefficients for heat capacity fits as a function of
            volume, shape (n_temperatures, order + 1).
    """

    def __init__(self):
        self.path = None
        self.number_of_atoms = None
        self.nelect = None
        self.volumes = None
        self.energies_list = None
        self.dos_list = None

        self.temperatures = None
        self.helmholtz_energies = None
        self.internal_energies = None
        self.entropies = None
        self.heat_capacities = None

        self.volumes_fit = None
        self.helmholtz_energies_fit = None
        self.entropies_fit = None
        self.heat_capacities_fit = None
        self.helmholtz_energies_poly_coeffs = None
        self.entropies_poly_coeffs = None
        self.heat_capacities_poly_coeffs = None

    def _get_elec_folders(self, path: str, folder_prefix: str = "elec") -> list[str]:
        """
        Get the list of folders with the specified prefix in the given path,
        sorted in natural order.

        Args:
            path (str): Path to the directory containing the folders.
            folder_prefix (str, optional): Prefix of the folders to search for.
                Defaults to ``"elec"``.

        Returns:
            list[str]: List of folder names with the given prefix, sorted in
                natural order.
        """
        return natsorted([f for f in os.listdir(path) if f.startswith(folder_prefix)])

    def read_total_electron_dos(
        self,
        path: str,
        folder_prefix: str = "elec",
        vasprun_name: str = "vasprun.xml.elec_dos",
        selected_volumes: np.ndarray = None,
    ) -> None:
        """
        Reads the total electron DOS data from VASP calculations for different volumes.

        Args:
            path (str):
                Path to the directory containing the specific folders with
                CONTCAR and vasprun.xml files.
            folder_prefix (str, optional):
                Prefix of the electronic folders. Defaults to ``"elec"``.
            contcar_name (str, optional):
                Name of the CONTCAR file. Defaults to ``"CONTCAR.elec_dos"``.
            vasprun_name (str, optional):
                Name of the vasprun.xml file. Defaults to ``"vasprun.xml.elec_dos"``.
            selected_volumes (np.ndarray, optional):
                List of selected volumes to keep the electron DOS data. Defaults
                to None.

        Raises:
            ValueError:
                If selected volumes are not found.
            ValueError:
                If the number of atoms is not the same for all volumes.
            ValueError:
                If the number of electrons is not the same for all volumes.
        """

        self.path = path

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Get the electronic folders
            elec_folders = self._get_elec_folders(
                path=path, folder_prefix=folder_prefix
            )

            # Initialize lists to store data
            volumes_list = []
            num_atoms_list = []
            nelect_list = []
            energies_list = []
            dos_list = []

            # Iterate over electronic folders to get the relevant electronic DOS data
            for elec_folder in elec_folders:

                # Get the volumes, number of atoms, and number of electrons from vasprun.xml
                vasprun_path = os.path.join(path, elec_folder, vasprun_name)
                vasprun = Vasprun(vasprun_path)
                volume = round(vasprun.final_structure.volume, 6)
                volumes_list.append(volume)
                number_of_atoms = vasprun.final_structure.num_sites
                num_atoms_list.append(number_of_atoms)
                nelect = vasprun.parameters["NELECT"]
                nelect_list.append(nelect)

                # Get the vasprun energies minus Fermi energy from vasprun.xml.
                vasprun_energies = vasprun.complete_dos.energies - vasprun.efermi
                energies_list.append(vasprun_energies)

                # Get the vasprun DOS from vasprun.xml.
                try:
                    # For spin polarized calculations
                    vasprun_dos = (
                        vasprun.complete_dos.densities[Spin.up]
                        + vasprun.complete_dos.densities[Spin.down]
                    )
                except:
                    # For non-spin polarized calculations
                    vasprun_dos = vasprun.complete_dos.densities[Spin.up]
                dos_list.append(vasprun_dos)

            # Get the sorted indices based on volumes_list
            sorted_indices = np.argsort(volumes_list)

            # Sort all lists using the sorted indices
            volumes_list = [volumes_list[i] for i in sorted_indices]
            num_atoms_list = [num_atoms_list[i] for i in sorted_indices]
            nelect_list = [nelect_list[i] for i in sorted_indices]
            energies_list = [energies_list[i] for i in sorted_indices]
            dos_list = [dos_list[i] for i in sorted_indices]

            # Filter values to only include selected volumes
            if selected_volumes is not None:
                # Check for missing volumes
                missing = [v for v in selected_volumes if v not in volumes_list]
                if missing:
                    raise ValueError(
                        f"The following selected volumes were not found: {missing}"
                    )

                filtered_volume_list = []
                filtered_num_atoms_list = []
                filtered_nelect_list = []
                filtered_vasprun_energies_list = []
                filtered_vasprun_dos_list = []

                for i in range(len(volumes_list)):
                    if volumes_list[i] in selected_volumes:
                        filtered_volume_list.append(volumes_list[i])
                        filtered_num_atoms_list.append(num_atoms_list[i])
                        filtered_nelect_list.append(nelect_list[i])
                        filtered_vasprun_energies_list.append(energies_list[i])
                        filtered_vasprun_dos_list.append(dos_list[i])

                volumes_list = filtered_volume_list
                num_atoms_list = filtered_num_atoms_list
                energies_list = filtered_vasprun_energies_list
                dos_list = filtered_vasprun_dos_list

            self.number_of_atoms = np.unique(num_atoms_list)
            # If the number of atoms is not the same for all volumes, raise an error
            if len(self.number_of_atoms) > 1:
                raise ValueError("Number of atoms is not the same for all volumes.")
            self.number_of_atoms = int(self.number_of_atoms[0])

            self.nelect = np.unique(nelect_list)
            # If the number of electrons is not the same for all volumes, raise an error
            if len(self.nelect) > 1:
                raise ValueError("Number of electrons is not the same for all volumes.")
            self.nelect = int(self.nelect[0])

            self.volumes = np.array(volumes_list)
            self.energies_list = energies_list
            self.dos_list = dos_list

    def set_total_electron_dos(
        self,
        number_of_atoms: int,
        volumes: np.ndarray,
        energies_list: list[np.ndarray],
        dos_list: list[np.ndarray],
    ) -> None:
        """
        Set the total electron DOS directly.

        Args:
            number_of_atoms (int):
                Number of atoms corresponding to the DOS data.
            volumes (np.ndarray):
                1D array of volumes, shape (n_volumes,), in Å³.
            energies_list (list[np.ndarray]):
                List of 1D arrays of energies referenced to the Fermi level
                (:math:`E - E_F`) for each volume, in eV.
            dos_list (list[np.ndarray]):
                List of 1D arrays of DOS values for each volume, in states/eV.

        Raises:
            ValueError:
                Lengths of volumes, energies_list, and dos_list must be the same.
        """

        self.number_of_atoms = number_of_atoms

        # Check that the lengths of volumes, energies_list, and dos_list are the same
        if not (len(volumes) == len(energies_list) == len(dos_list)):
            raise ValueError(
                "Lengths of volumes, energies_list, and dos_list must be the same."
            )

        self.volumes = volumes
        self.energies_list = energies_list
        self.dos_list = dos_list

    def process(
        self,
        temperatures: np.ndarray,
    ) -> None:
        """
        Calculates the thermal electronic contributions to Helmholtz free energy,
        internal energy, entropy, and heat capacity.

        Args:
            volumes_fit (np.ndarray):
                1D array of volumes used for fitting the properties, shape
                (n_volumes_fit,), in Å³.
            temperatures (np.ndarray):
                1D array of temperatures in K, shape (n_temperatures,).
            order (int):
                Order of the polynomial fit. Defaults to 1 (linear fit).

        Raises:
            ValueError:
                If DOS data is not found.
        """

        self.temperatures = temperatures

        # If dos_list is None, raise an error
        if self.dos_list is None:
            raise ValueError(
                "DOS data not found. Please read or set the total electron DOS first "
                "using read_total_electron_dos() or set_total_electron_dos()."
            )

        # Initialize lists to store data
        internal_energies_list = []
        entropies_list = []
        heat_capacities_list = []
        helmholtz_energies_list = []

        # Compute thermal electronic properties for each volume
        for i in range(len(self.volumes)):
            energies = self.energies_list[i]
            dos = self.dos_list[i]

            # For each volume, compute the thermal electronic properties at each temperature
            internal_energies = self.calculate_internal_energies(
                energies, dos, temperatures
            )
            entropies = self.calculate_entropies(energies, dos, temperatures)
            heat_capacities = self.calculate_heat_capacities(
                energies, dos, temperatures
            )
            helmholtz_energies = ThermalElectronic.calculate_helmholtz_energies(
                internal_energies, entropies, temperatures
            )

            internal_energies_list.append(internal_energies)
            entropies_list.append(entropies)
            heat_capacities_list.append(heat_capacities)
            helmholtz_energies_list.append(helmholtz_energies)

        # Convert lists to numpy arrays and transpose to have rows as temperatures and columns as volumes
        self.helmholtz_energies = np.array(helmholtz_energies_list).T
        self.internal_energies = np.array(internal_energies_list).T
        self.entropies = np.array(entropies_list).T
        self.heat_capacities = np.array(heat_capacities_list).T

    def fit(
        self,
        volumes_fit: np.ndarray,
        order: int = 1,
    ) -> None:
        """
        Fits the Helmholtz free energy, entropy, and heat capacity as a function
        of volume for various fixed temperatures.

        Args:
            volumes_fit (np.ndarray):
                1D array of volumes used for fitting the properties, shape
                (n_volumes_fit,), in Å³.
            order (int):
                Order of the polynomial fit. Defaults to 1 (linear fit).

        Raises:
            ValueError:
                Thermodynamic properties have not been calculated.
        """

        # If helmholtz_energies is None, raise an error
        if self.helmholtz_energies is None:
            raise ValueError(
                "Thermodynamic properties not yet calculated. Please call process() first"
            )

        # If there is only one volume, raise an error
        if len(self.volumes) == 1:
            raise ValueError(
                "Only one volume found. Need at least two volumes to perform fitting."
            )

        # Initialize lists to store data
        volume_fit_list = []
        helmholtz_energy_fit_list = []
        entropy_fit_list = []
        heat_capacity_fit_list = []
        helmholtz_energies_poly_coeffs = []
        entropies_poly_coeffs = []
        heat_capacities_poly_coeffs = []

        # Fit the properties vs. volume for each temperature
        for i in range(len(self.temperatures)):

            # Fit Helmholtz free energy, entropy, and heat capacity vs. volume with a polynomial of the specified order
            helmholtz_energy_coefficients = np.polyfit(
                self.volumes, self.helmholtz_energies[i], order
            )
            entropy_coefficients = np.polyfit(self.volumes, self.entropies[i], order)
            heat_capacity_coefficients = np.polyfit(
                self.volumes, self.heat_capacities[i], order
            )

            helmholtz_energies_poly_coeffs.append(helmholtz_energy_coefficients)
            entropies_poly_coeffs.append(entropy_coefficients)
            heat_capacities_poly_coeffs.append(heat_capacity_coefficients)

            helmholtz_energy_polynomial = np.poly1d(helmholtz_energy_coefficients)
            entropy_polynomial = np.poly1d(entropy_coefficients)
            heat_capacity_polynomial = np.poly1d(heat_capacity_coefficients)

            # Evaluate the polynomials at the specified fit volumes
            helmholtz_energy_fit = helmholtz_energy_polynomial(volumes_fit)
            entropy_fit = entropy_polynomial(volumes_fit)
            heat_capacity_fit = heat_capacity_polynomial(volumes_fit)

            volume_fit_list.append(volumes_fit)
            helmholtz_energy_fit_list.append(helmholtz_energy_fit)
            entropy_fit_list.append(entropy_fit)
            heat_capacity_fit_list.append(heat_capacity_fit)

        # Convert lists to numpy arrays
        self.volumes_fit = np.array(volumes_fit)
        self.helmholtz_energies_fit = np.array(helmholtz_energy_fit_list)
        self.entropies_fit = np.array(entropy_fit_list)
        self.heat_capacities_fit = np.array(heat_capacity_fit_list)
        self.helmholtz_energies_poly_coeffs = np.array(helmholtz_energies_poly_coeffs)
        self.entropies_poly_coeffs = np.array(entropies_poly_coeffs)
        self.heat_capacities_poly_coeffs = np.array(heat_capacities_poly_coeffs)

    def plot_total_dos(self) -> go.Figure:
        """
        Plots the total electron DOS for different volumes.

        Raises:
            ValueError:
                DOS data not found. Please read or set the total electron DOS first
                using `read_total_electron_dos()` or `set_total_electron_dos()`.

        Returns:
            go.Figure:
                Plotly figure object containing the total electron DOS curves for
                the different volumes.
        """

        # If dos_list is None, raise an error
        if self.dos_list is None:
            raise ValueError(
                "DOS data not found. Please read or set the total electron DOS first "
                "using read_total_electron_dos() or set_total_electron_dos()."
            )

        fig = go.Figure()
        for i in range(len(self.volumes)):
            fig.add_trace(
                go.Scatter(
                    x=self.energies_list[i],
                    y=self.dos_list[i],
                    mode="lines",
                    name=f"{self.volumes[i]} Å<sup>3</sup>",
                    showlegend=True,
                )
            )
        plot_format(
            fig,
            xtitle="E - E<sub>F</sub> (eV)",
            ytitle=f"DOS (states/eV)",
        )

        return fig

    def plot_vt(self, type: str, selected_temperatures: np.ndarray = None) -> go.Figure:
        """
        Plots thermal electronic properties as a function of temperature or volume.

        Args:
            type (str):
                Must be one of the following values:
                ``'helmholtz_energy_vs_temperature'``, ``'entropy_vs_temperature'``,
                ``'heat_capacity_vs_temperature'``, ``'helmholtz_energy_vs_volume'``,
                ``'entropy_vs_volume'``, or ``'heat_capacity_vs_volume'``.
            selected_temperatures (np.ndarray, optional):
                Selected temperatures to use for volume plots, shape
                (n_selected_temperatures,). Defaults to None.

        Raises:
            ValueError:
                Thermodynamic properties have not been calculated.
            ValueError:
                The `type` argument is not one of the allowed values.

        Returns:
            go.Figure:
                Plotly figure object containing the requested thermal electronic
                property curves.
        """

        # If helmholtz_energies is None, raise an error
        if self.helmholtz_energies is None:
            raise ValueError(
                "Thermodynamic properties not yet calculated. "
                "Please call process() first."
            )

        type_map = {
            "helmholtz_energy_vs_temperature": (
                self.helmholtz_energies,
                "F<sub>el</sub> (eV)",
            ),
            "entropy_vs_temperature": (self.entropies, "S<sub>el</sub> (eV/K)"),
            "heat_capacity_vs_temperature": (
                self.heat_capacities,
                "C<sub>v,el</sub> (eV/K)",
            ),
            "helmholtz_energy_vs_volume": (
                self.helmholtz_energies,
                self.helmholtz_energies_fit,
                "F<sub>el</sub> (eV)",
            ),
            "entropy_vs_volume": (
                self.entropies,
                self.entropies_fit,
                "S<sub>el</sub> (eV/K)",
            ),
            "heat_capacity_vs_volume": (
                self.heat_capacities,
                self.heat_capacities_fit,
                "C<sub>v,el</sub> (eV/K)",
            ),
        }

        if type not in type_map:
            raise ValueError(
                "type must be one of "
                "'helmholtz_energy_vs_temperature', 'entropy_vs_temperature', "
                "'heat_capacity_vs_temperature', 'helmholtz_energy_vs_volume', "
                "'entropy_vs_volume', or 'heat_capacity_vs_volume'."
            )

        if "vs_temperature" in type:
            property_values, y_title = type_map[type]

            fig = go.Figure()
            for i, volume in enumerate(self.volumes):
                fig.add_trace(
                    go.Scatter(
                        x=self.temperatures,
                        y=property_values[:, i],
                        mode="lines",
                        name=f"{volume} Å³",
                        showlegend=True,
                    )
                )
            plot_format(fig, "Temperature (K)", y_title)

        elif "vs_volume" in type:
            property_values, property_values_fit, y_title = type_map[type]

            if selected_temperatures is None:
                if self.temperatures.size < 5:
                    selected_temperatures = self.temperatures
                else:
                    indices = np.linspace(0, len(self.temperatures) - 1, 5, dtype=int)
                    selected_temperatures = self.temperatures[indices]

            fig = go.Figure()
            colors = [
                "#636EFA",
                "#EF553B",
                "#00CC96",
                "#AB63FA",
                "#FFA15A",
                "#19D3F3",
                "#FF6692",
                "#B6E880",
                "#FF97FF",
                "#FECB52",
            ]
            colors = [
                f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},1)"
                for c in colors
            ]

            for i, temperature in enumerate(selected_temperatures):
                idx = np.where(self.temperatures == temperature)[0][0]
                color = colors[i % len(colors)]
                # Markers for calculated data
                fig.add_trace(
                    go.Scatter(
                        x=self.volumes,
                        y=property_values[idx],
                        mode="markers",
                        line=dict(color=color),
                        name=f"{temperature} K",
                        legendgroup=f"{temperature} K",
                        showlegend=True,
                    )
                )
                # Lines for fit, if available
                if self.volumes_fit is not None and property_values_fit is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=self.volumes_fit,
                            y=property_values_fit[idx],
                            mode="lines",
                            line=dict(color=color),
                            name=f"{temperature} K fit",
                            legendgroup=f"{temperature} K",
                            showlegend=False,
                        )
                    )

            plot_format(fig, "Volume (Å³)", y_title)
        return fig

    def calculate_chemical_potential(
        self,
        energies: np.ndarray,
        dos: np.ndarray,
        temperature: float,
        chemical_potential_range: np.ndarray = np.array([-0.1, 0.1]),
        electron_tol: float = 0.5,
    ) -> float:
        """
        Calculates the chemical potential at a given electronic DOS, temperature, and
        volume such that the number of electrons matches that at 0 K (within a
        specified tolerance). Note that this method currently assumes that the
        energies are given with respect to the Fermi energy.

        Args:
            energies (np.ndarray): Energy values for the electron DOS.
            dos (np.ndarray): Electron DOS values.
            temperature (float): Temperature in K.
            chemical_potential_range (np.ndarray, optional): Range to search for the
                chemical potential. Defaults to None.
            electron_tol (float, optional): Tolerance for electron number matching.
                Defaults to 0.5.

        Raises:
            ValueError: If `temperature < 0 K`.
            ValueError: If the chemical potential cannot be found within the specified
                range.

        Returns:
            float: Chemical potential at the given electronic DOS, temperature, and volume.
        """

        if temperature < 0:
            raise ValueError("Temperature cannot be less than 0 K")
        temperature = float(temperature)

        num_electrons_0K = ThermalElectronic.calculate_num_electrons(
            energies=energies, dos=dos, chemical_potential=0, temperature=0
        )

        if self.nelect is not None:
            if abs(num_electrons_0K - self.nelect) > electron_tol:
                warnings.warn(
                    f"Warning: The number of electrons at 0 K ({num_electrons_0K}) does not match the expected number of "
                    f"electrons ({self.nelect}) within the specified tolerance."
                    " Consider increasing NEDOS.",
                    UserWarning,
                )

        num_electrons_guess = ThermalElectronic.calculate_num_electrons(
            energies=energies,
            dos=dos,
            chemical_potential=0,
            temperature=temperature,
        )

        if abs(num_electrons_guess - num_electrons_0K) < electron_tol:
            return 0

        def electron_difference(chemical_potential):
            num_electrons = ThermalElectronic.calculate_num_electrons(
                energies=energies,
                dos=dos,
                chemical_potential=chemical_potential,
                temperature=temperature,
            )
            return num_electrons - num_electrons_0K

        try:
            chemical_potential = bisect(
                electron_difference,
                chemical_potential_range[0],
                chemical_potential_range[1],
            )
        except ValueError as e:
            print(
                f"Warning: The chemical potential could not be found within the range "
                f"{chemical_potential_range[0]} to {chemical_potential_range[1]} eV."
                " Consider increasing the chemical_potential_range."
            )
            chemical_potential = chemical_potential_range[1]

        return chemical_potential

    @staticmethod
    def fit_electron_dos(
        energies: np.ndarray,
        dos: np.ndarray,
        energy_range: np.ndarray,
        resolution: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fits the electron DOS with a spline.

        Args:
            energies (np.ndarray): Energy values for the electron DOS, in eV.
            dos (np.ndarray): Electron DOS values, in states/eV.
            energy_range (np.ndarray): Energy range to fit the electron DOS, in eV.
            resolution (float): Energy resolution for the spline, in eV.

        Returns:
            tuple[np.ndarray, np.ndarray]: Fitted energy and DOS values.
        """

        # Filter the energy and dos values within the energy range
        filtered_indices = (energies >= energy_range[0]) & (energies <= energy_range[1])
        filtered_energy = energies[filtered_indices]
        filtered_dos = dos[filtered_indices]

        # Fit the filtered energy and dos values with a spline
        spline = UnivariateSpline(filtered_energy, filtered_dos, s=0)
        energy_fit = np.arange(
            energy_range[0], energy_range[1] + resolution, resolution
        )
        dos_fit = spline(energy_fit)

        return energy_fit, dos_fit

    @staticmethod
    def fermi_dirac_distribution(
        energies: np.ndarray,
        chemical_potential: float,
        temperature: float,
        plot: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, go.Figure]:
        """
        Calculates the Fermi-Dirac distribution function.

        The function is given by:

        .. math::

            f(E, \\mu, T) = \\frac{1}{1 + \\exp\\left(\\frac{E - \\mu}{k_B T}\\right)}

        Args:
            energies (np.ndarray): Energy values for the electron DOS, in eV.
            chemical_potential (float): Chemical potential for a given volume and temperature, in eV.
            temperature (float): Temperature in K.
            plot (bool, optional): If True, plots the Fermi-Dirac distribution function
                vs. energy for the given temperature and chemical potential. Defaults to False.

        Raises:
            ValueError: If `temperature < 0 K`.

        Returns:
            np.ndarray or (np.ndarray, go.Figure):
                Fermi-Dirac distribution function values, and optionally the Plotly figure if `plot=True`.
        """

        chemical_potential = float(chemical_potential)
        temperature = float(temperature)

        if temperature < 0:
            raise ValueError("Temperature cannot be less than 0 K")

        if temperature == 0:
            fermi_dist = np.where(energies <= chemical_potential, 1, 0)
        else:
            # Note that expit(x) = 1/(1+exp(-x))
            fermi_dist = expit(
                -(energies - chemical_potential) / (BOLTZMANN_CONSTANT * temperature)
            )

        if plot:
            fig = ThermalElectronic.plot_fermi_dirac_distribution(
                energies, fermi_dist, chemical_potential, temperature
            )
            return fermi_dist, fig
        return fermi_dist

    @staticmethod
    def plot_fermi_dirac_distribution(
        energies: np.ndarray,
        fermi_dist: np.ndarray,
        chemical_potential: float,
        temperature: float,
    ) -> go.Figure:
        """
        Plots the Fermi-Dirac distribution function versus energy for a given
        temperature and chemical potential.

        Args:
            energy (np.ndarray): Energy values for the electron DOS, in eV.
            fermi_dist (np.ndarray): Fermi-Dirac distribution function values.
            chemical_potential (float): Chemical potential for a given volume and
                temperature, in eV.
            temperature (float): Temperature in K.

        Returns:
            go.Figure:
                Plotly figure object containing the Fermi-Dirac distribution function curve.
        """

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=energies, y=fermi_dist, mode="lines"))
        fig.update_layout(
            title=dict(
                text=f"T = {temperature} K, &mu; = {chemical_potential} eV",
                font=dict(size=20, color="rgb(0,0,0)"),
            ),
            margin=dict(t=130),
        )
        plot_format(fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="f (E, T, V) ")

        return fig

    @staticmethod
    def calculate_num_electrons(
        energies: np.ndarray,
        dos: np.ndarray,
        chemical_potential: float,
        temperature: float,
    ) -> float:
        """
        Calculates the number of electrons for a given electronic DOS, chemical potential,
        and temperature.

        .. math::

            N = \int_{-\infty}^{\infty} \mathrm{DOS}(E) \, f(E, \mu, T) \, dE

        Args:
            energies (np.ndarray): Energy values for the electron DOS, in eV.
            dos (np.ndarray): Electron DOS values, in states/eV.
            chemical_potential (float): Chemical potential for a given volume and temperature, in eV.
            temperature (float): Temperature in K.

        Raises:
            ValueError: If `temperature < 0 K`.

        Returns:
            float: Number of electrons.
        """

        chemical_potential = float(chemical_potential)
        temperature = float(temperature)

        if temperature < 0:
            raise ValueError("Temperature cannot be less than 0 K")

        fermi_dist = ThermalElectronic.fermi_dirac_distribution(
            energies, chemical_potential, temperature
        )
        integrand = dos * fermi_dist
        num_electrons = np.trapz(integrand, energies)

        return num_electrons

    def calculate_internal_energies(
        self,
        energies: np.ndarray,
        dos: np.ndarray,
        temperatures: np.ndarray,
        resolution: float = 0.001,
        plot: bool = False,
        plot_temperature: float = None,
    ) -> np.ndarray:
        """
        Calculates the thermal electronic contribution to the internal energy for a given volume.

        .. math::

            U_\mathrm{el}(T, V) =
            \int_{-\infty}^{\infty} \mathrm{DOS}(E) \, f(E, \mu, T) \, E \, dE
            - \int_{-\infty}^{E_F} \mathrm{DOS}(E) \, E \, dE

        Args:
            energies (np.ndarray): Energy values from the electron DOS, in eV.
            dos (np.ndarray): Electron DOS values, in states/eV.
            temperatures (np.ndarray): Temperatures in K.
            resolution (float, optional): Energy resolution for the spline, in eV. Defaults to 0.001.
            plot (bool, optional): If True, plots the integrand vs. energy. Defaults to False.
            plot_temperature (float, optional): Temperature to plot the integrand, in K.
                Required if `plot=True`. Defaults to None.

        Raises:
            ValueError: If any temperature is negative.
            ValueError: If `plot_temperature` is provided when `plot=False` or missing when `plot=True`.
            ValueError: If `plot_temperature` is not in `temperatures` when `plot=True`.

        Returns:
            np.ndarray: Internal energy values, in eV.
        """

        # If there are negative temperatures, raise an error
        if np.any(temperatures < 0):
            raise ValueError("Temperatures cannot be less than 0 K")

        # Ensure temperatures is a numpy array
        single_value = np.isscalar(temperatures)
        temperatures = np.atleast_1d(temperatures)

        # Fit the whole energy range of the electron DOS with a spline with the given resolution
        energies_fit, dos_fit = ThermalElectronic.fit_electron_dos(
            energies, dos, [np.min(energies), np.max(energies)], resolution
        )

        # Initialize lists to store data
        integrand_1_list = []
        filtered_energies_list = []
        integrand_2_list = []
        internal_energies_list = []

        # Calculate the internal energy for each temperature
        for i, temperature in enumerate(temperatures):

            # Calculate the Fermi-Dirac distribution function at the given temperature and chemical potential
            chemical_potential = self.calculate_chemical_potential(
                energies, dos, temperature
            )
            fermi_dist = ThermalElectronic.fermi_dirac_distribution(
                energies_fit, chemical_potential, temperature
            )

            # Evaluate the first integral over the entire energy range
            integrand_1 = dos_fit * fermi_dist * energies_fit
            integrand_1_list.append(integrand_1)
            integral_1 = np.trapz(integrand_1, energies_fit)

            # Evaluate the second integral from -infinity to the Fermi energy (shifted to 0 eV)
            mask = energies_fit < 0
            filtered_energies = energies_fit[mask]
            filtered_energies_list.append(filtered_energies)
            filtered_dos = dos_fit[mask]

            integrand_2 = filtered_dos * filtered_energies
            integrand_2_list.append(integrand_2)
            integral_2 = np.trapz(integrand_2, filtered_energies)

            internal_energies = integral_1 - integral_2
            internal_energies_list.append(internal_energies)

        # Assign to new variables for clarity
        integrand_1 = integrand_1_list
        filtered_energies = filtered_energies_list
        integrand_2 = integrand_2_list

        # Convert internal_energies_list to a numpy array
        internal_energies = np.array(internal_energies_list)

        # Plot the integrands if requested
        if (plot and plot_temperature is None) or (
            not plot and plot_temperature is not None
        ):
            raise ValueError(
                "plot_temperature must be provided if and only if plot is True."
            )

        if plot and plot_temperature is not None:
            if plot_temperature not in temperatures:
                raise ValueError(
                    "plot_temperature must be one of the temperatures provided."
                )
            index = np.where(temperatures == plot_temperature)[0][0]

            fig1, fig2 = ThermalElectronic.plot_internal_energy_integral(
                energies_fit,
                integrand_1[index],
                filtered_energies[index],
                integrand_2[index],
                plot_temperature,
            )
            if single_value:
                return internal_energies[0], fig1, fig2
            else:
                return internal_energies, fig1, fig2

        if single_value:
            return internal_energies[0]
        return internal_energies

    @staticmethod
    def plot_internal_energy_integral(
        energies: np.ndarray,
        integrand_1: np.ndarray,
        filtered_energies: np.ndarray,
        integrand_2: np.ndarray,
        plot_temperature: float,
    ) -> go.Figure:
        """
        Plots the integrands versus energy of the internal energy equation.

        Args:
            energies (np.ndarray): Energy values for the electron DOS, in eV.
            integrand_1 (np.ndarray): First integrand from the internal energy equation.
            filtered_energies (np.ndarray): Filtered energy values where E < mu, in eV.
            integrand_2 (np.ndarray): Second integrand from the internal energy equation.
            plot_temperature (float): Temperature at which the integrand is plotted, in K.

        Returns:
            go.Figure: Plotly figure object containing the integrand curves.
        """

        plot_temperature = float(plot_temperature)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=energies, y=integrand_1, mode="markers"))
        fig1.update_layout(
            title=dict(
                text=f"T = {plot_temperature} K",
                font=dict(size=20, color="rgb(0,0,0)"),
            ),
            margin=dict(t=130),
        )
        plot_format(
            fig1, xtitle="E - E<sub>F</sub> (eV)", ytitle="E<sub>el</sub> integrand 1"
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=filtered_energies, y=integrand_2, mode="markers"))
        fig2.update_layout(
            title=dict(
                text=f"T = {plot_temperature} K",
                font=dict(size=20, color="rgb(0,0,0)"),
            ),
            margin=dict(t=130),
        )
        plot_format(
            fig2, xtitle="E - E<sub>F</sub> (eV)", ytitle="E<sub>el</sub> integrand 2"
        )

        return fig1, fig2

    def calculate_entropies(
        self,
        energies: np.ndarray,
        dos: np.ndarray,
        temperatures: np.ndarray,
        energies_fit_range: np.ndarray = np.array([-2, 2]),
        resolution: float = 0.0001,
        plot: bool = False,
        plot_temperature: float = None,
    ) -> np.ndarray:
        #S_\mathrm{el}(T, V) = - k_B \int_{-\infty}^{\infty} \mathrm{DOS}(E) [f \ln f + (1 - f) \ln (1 - f)] dE
        """
        Calculates the thermal electronic contribution to the entropy for a given volume using the formula:

        .. math::

            S_\mathrm{el}(T, V) 

        Args:
            energies (np.ndarray): Energy values for the electron DOS, in eV.
            dos (np.ndarray): Electron DOS values, in states/eV.
            temperatures (np.ndarray): Temperatures in K.
            energies_fit_range (np.ndarray, optional): Energy range to fit the electron DOS, in eV. Defaults to np.array([-2, 2]).
            resolution (float, optional): Energy resolution for the spline, in eV. Defaults to 0.0001.
            plot (bool, optional): If True, plots the integrand vs. energy of the entropy equation. Defaults to False.
            plot_temperature (float, optional): Temperature to plot the integrand vs. energy. Defaults to None.

        Raises:
            ValueError: If there are negative temperatures.
            ValueError: If plot_temperature is provided when `plot=False` or not provided when `plot=True`.
            ValueError: If plot_temperature is not in `temperatures` when `plot=True`.

        Returns:
            np.ndarray: Entropy values as a function of temperature, in eV/K.
        """

        # If there are negative temperatures, raise an error
        if np.any(temperatures < 0):
            raise ValueError("Temperatures cannot be less than 0 K")

        # Ensure temperatures is a numpy array
        single_value = np.isscalar(temperatures)
        temperatures = np.atleast_1d(temperatures)

        # Fit the electron DOS within the specified energy range with a spline
        energies_fit, dos_fit = ThermalElectronic.fit_electron_dos(
            energies, dos, energies_fit_range, resolution
        )

        # Initialize lists to store data
        integrand_list = []
        entropies_list = []

        # Calculate the entropy for each temperature
        for i, temperature in enumerate(temperatures):
            # Calculate the chemical potential
            chemical_potential = self.calculate_chemical_potential(
                energies, dos, temperature
            )

            # Entropy is zero at 0 K
            if temperature == 0:
                entropy = 0
                entropies_list.append(entropy)

                integrand = np.zeros_like(energies_fit)
                integrand_list.append(integrand)

            # Calculate the entropy at finite temperatures
            elif temperature > 0:
                fermi_dist = ThermalElectronic.fermi_dirac_distribution(
                    energies_fit, chemical_potential, temperature
                )

                # At finite temperatures, f is never exactly 0 or 1, but due to lack of numerical precision, we may encounter these values.
                # The limit of f ln f + (1-f) ln (1-f) as f approaches 0 or 1 is 0
                # We use a mask to avoid log(0) issues
                mask = (fermi_dist == 0) | (fermi_dist == 1)
                integrand = np.zeros_like(fermi_dist)
                integrand[~mask] = dos_fit[~mask] * (
                    fermi_dist[~mask] * np.log(fermi_dist[~mask])
                    + (1 - fermi_dist[~mask]) * np.log(1 - fermi_dist[~mask])
                )
                integrand[mask] = 0
                integrand_list.append(integrand)

                entropy = -BOLTZMANN_CONSTANT * np.trapz(integrand, energies_fit)
                entropies_list.append(entropy)

            # Convert lists to numpy arrays
            integrand = np.array(integrand_list)
            entropies = np.array(entropies_list)

        # Plot the integrand if requested
        if (plot and plot_temperature is None) or (
            not plot and plot_temperature is not None
        ):
            raise ValueError(
                "plot_temperature must be provided if and only if plot is True."
            )

        if plot and plot_temperature is not None:
            if plot_temperature not in temperatures:
                raise ValueError(
                    "plot_temperature must be one of the temperatures provided."
                )
            index = np.where(temperatures == plot_temperature)[0][0]
            fig = ThermalElectronic.plot_entropy_integral(
                energies_fit, integrand[index], plot_temperature
            )
            if single_value:
                return entropies[0], fig
            else:
                return entropies, fig

        if single_value:
            return entropies[0]
        else:
            return entropies

    @staticmethod
    def plot_entropy_integral(
        energies: np.ndarray, integrand: np.ndarray, plot_temperature: float
    ) -> go.Figure:
        """
        Plots the integrand vs. energy of the entropy equation.

        Args:
            energies (np.ndarray): Energy values for the electron DOS, in eV.
            integrand (np.ndarray): Integrand from the entropy equation.
            plot_temperature (float): Temperature in K.

        Returns:
            go.Figure: Plotly figure object showing the integrand as a function of energy.
        """

        plot_temperature = float(plot_temperature)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=energies, y=-BOLTZMANN_CONSTANT * integrand, mode="markers")
        )
        fig.update_layout(
            title=dict(
                text=f"T = {plot_temperature} K",
                font=dict(size=20, color="rgb(0,0,0)"),
            ),
            margin=dict(t=130),
        )
        plot_format(
            fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="S<sub>el</sub> integrand"
        )

        return fig

    def calculate_heat_capacities(
        self,
        energies: np.ndarray,
        dos: np.ndarray,
        temperatures: np.ndarray,
        energies_fit_range: np.ndarray = np.array([-2, 2]),
        resolution: float = 0.0001,
        plot=False,
        plot_temperature: float = None,
    ) -> np.ndarray:

        """
        Calculates the thermal electronic contribution to the heat capacity for a given volume using the formula:

        .. math::

            C_{V,\mathrm{el}}(T, V) = \int_{-\infty}^{\infty} \mathrm{DOS}(E) \, f \,
                \bigl[ 1 - f \bigr] \frac{(E - \mu)^2}{k_B T^2} \, dE

        Args:
            energies (np.ndarray): Energy values for the electron DOS, in eV.
            dos (np.ndarray): Electron DOS values, in states/eV.
            temperatures (np.ndarray): Temperatures in K.
            energies_fit_range (np.ndarray, optional): Energy range to fit the electron DOS. Defaults to np.array([-2, 2]) eV.
            resolution (float, optional): Energy resolution for the spline. Defaults to 0.0001 eV.
            plot (bool, optional): If True, plots the integrand vs. energy of the heat capacity equation. Defaults to False.
            plot_temperature (float, optional): Temperature to plot the integrand. Defaults to None.

        Raises:
            ValueError: If there are negative temperatures.
            ValueError: If plot_temperature is provided when `plot=False` or not provided when `plot=True`.
            ValueError: If plot_temperature is not in `temperatures` when `plot=True`.

        Returns:
            np.ndarray: Heat capacity values in eV/K.
        """

        # If there are negative temperatures, raise an error
        if np.any(temperatures < 0):
            raise ValueError("Temperatures cannot be less than 0 K")

        # Ensure temperatures is a numpy array
        single_value = np.isscalar(temperatures)
        temperatures = np.atleast_1d(temperatures)

        # Fit the electron DOS within the specified energy range with a spline
        energies_fit, dos_fit = ThermalElectronic.fit_electron_dos(
            energies, dos, energies_fit_range, resolution
        )

        # Initialize lists to store data
        integrand_list = []
        heat_capacities_list = []

        # Calculate the heat capacity for each temperature
        for i, temperature in enumerate(temperatures):
            # Calculate the chemical potential
            chemical_potential = self.calculate_chemical_potential(
                energies, dos, temperature
            )

            # Heat capacity is zero at 0 K
            if temperature == 0:
                heat_capacity = 0
                heat_capacities_list.append(heat_capacity)

                integrand = np.zeros_like(energies_fit)
                integrand_list.append(integrand)

            # Calculate the heat capacity at finite temperatures
            elif temperature > 0:
                fermi_dist = ThermalElectronic.fermi_dirac_distribution(
                    energies_fit, chemical_potential, temperature
                )

                integrand = (
                    dos_fit
                    * fermi_dist
                    * (1 - fermi_dist)
                    * ((energies_fit - chemical_potential) / temperature) ** 2
                    / BOLTZMANN_CONSTANT
                )
                integrand_list.append(integrand)

                heat_capacities = np.trapz(integrand, energies_fit)
                heat_capacities_list.append(heat_capacities)

        # Convert lists to numpy arrays
        integrand = np.array(integrand_list)
        heat_capacities = np.array(heat_capacities_list)

        # Plot the integrand if requested
        if (plot and plot_temperature is None) or (
            not plot and plot_temperature is not None
        ):
            raise ValueError(
                "plot_temperature must be provided if and only if plot is True."
            )

        if plot and plot_temperature is not None:
            if plot_temperature not in temperatures:
                raise ValueError(
                    "plot_temperature must be one of the temperatures provided."
                )
            index = np.where(temperatures == plot_temperature)[0][0]
            fig = ThermalElectronic.plot_heat_capacity_integral(
                energies_fit, integrand[index], plot_temperature
            )
            if single_value:
                return heat_capacities[0], fig
            else:
                return heat_capacities, fig
        if single_value:
            return heat_capacities[0]
        else:
            return heat_capacities

    @staticmethod
    def plot_heat_capacity_integral(
        energies: np.ndarray, integrand: np.ndarray, plot_temperature: float
    ) -> go.Figure:
        """
        Plots the integrand vs. energy of the heat capacity equation.

        Args:
            energies (np.ndarray): Energy values for the electron DOS, in eV.
            integrand (np.ndarray): Integrand from the heat capacity equation.
            plot_temperature (float): Temperature in K.

        Returns:
            go.Figure: Plotly figure object containing the integrand vs. energy curve for the specified temperature.
        """

        plot_temperature = float(plot_temperature)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=energies, y=integrand, mode="markers"))
        fig.update_layout(
            title=dict(
                text=f"T = {plot_temperature} K",
                font=dict(size=20, color="rgb(0,0,0)"),
            ),
            margin=dict(t=130),
        )
        plot_format(
            fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="C<sub>v,el</sub> integrand"
        )

        return fig

    @staticmethod
    def calculate_helmholtz_energies(
        internal_energies: np.ndarray,
        entropies: np.ndarray,
        temperatures: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the thermal electronic contribution to the Helmholtz free energy for a given volume using the formula

        .. math::

            F_\mathrm{el}(T, V) = U_\mathrm{el}(T, V) - T \, S_\mathrm{el}(T, V)

        Args:
            internal_energies (np.ndarray): Internal energy values, in eV.
            entropies (np.ndarray): Entropy values, in eV/K.
            temperatures (np.ndarray): Temperatures in K.

        Returns:
            np.ndarray: Helmholtz free energy values, in eV.
        """

        helmholtz_energies = internal_energies - temperatures * entropies

        return helmholtz_energies
