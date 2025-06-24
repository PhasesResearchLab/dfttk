"""
DebyeGruneisen class to calculate the vibrational contribution to the Helmholtz energy, entropy, and heat capacity using the Debye-Grüneisen model.
"""

# Standard library imports
import numpy as np

# Third-party imports
import plotly.graph_objects as go
from scipy import constants
from scipy.integrate import quad

# DFTTK imports
from dfttk.plotly_format import plot_format

BOLTZMANN_CONSTANT = constants.physical_constants["Boltzmann constant in eV/K"][0]
HBAR = constants.physical_constants["Planck constant over 2 pi in eV s"][0]

class DebyeGruneisen:
    """Class to calculate the vibrational contribution to the Helmholtz energy, entropy, and heat capacity using the Debye-Grüneisen model."""

    def __init__(self):
        self.number_of_atoms: int = None
        self.volumes: np.ndarray = None
        self.temperatures: np.ndarray = None
        self.atomic_mass: float = None
        self.V0: float = None
        self.B: float = None
        self.BP: float = None
        self.scaling_factor: float = None
        self.gruneisen_x: float = None
        self.helmholtz_energies: np.ndarray = None
        self.entropies: np.ndarray = None
        self.heat_capacities: np.ndarray = None

    def calculate_gruneisen_parameter(self) -> float:
        """Calculates the Gruneisen parameter (gamma).

        Returns:
            float: Gruneisen parameter (gamma).
        """
        return (1 + self.BP) / 2 - self.gruneisen_x

    def calculate_debye_temperatures(self, gruneisen_parameter: float) -> np.ndarray:
        """Calculates the Debye temperatures in Kelvin within the Debye-Grüneisen model.

        Args:
            gruneisen_parameter (float): Gruneisen parameter.

        Returns:
            np.ndarray: Debye temperatures in Kelvin for each volume in the input array.
        """
        A = (6 * np.pi**2)**(1/3) * HBAR / BOLTZMANN_CONSTANT
        B = self.B * 1e12  # Convert GPa to g/(ms^2)
        V0 = self.V0 * 1e-30  # Convert Å^3 to m^3
        volumes = self.volumes * 1e-30  # Convert Å^3 to m^3
        atomic_mass = self.atomic_mass * 1.66054e-24  # Convert u to g

        debye_temperatures = (
            self.scaling_factor * A * V0 ** (1 / 6) *
            (B / atomic_mass) ** 0.5 *
            (V0 / volumes) ** gruneisen_parameter
        )
        return debye_temperatures

    def calculate_debye_integral_n3(self, x_array: np.ndarray) -> np.ndarray:
        """Calculate the Debye integral of order 3 for an array of upper limits, x_array.

        For each x in x_array, computes:
            D(x) = (3 / x^3) * ∫₀ˣ (t^3 / (exp(t) - 1)) dt.

        Args:
            x_array (np.ndarray): Array of upper integration limits, where each x is debye_temperature / temperature.
            The debye_temperature is fixed at a single volume and the temperature is varied.

        Returns:
            np.ndarray: Array of calculated Debye integrals of order 3 for each upper limit in x_array.
        """
        debye_integrals = np.zeros_like(x_array, dtype=float)
        for i, x in enumerate(x_array):
            factor = 3. / x ** 3
            integral, _ = quad(lambda t: t ** 3 / (np.exp(t) - 1.), 0, x)
            debye_integrals[i] = integral * factor
        return debye_integrals

    def calculate_entropies(self, debye_temperature: float) -> np.ndarray:
        """Calculate the vibrational entropy using the Debye model.

        Args:
            debye_temperature (float): Debye temperature in Kelvin for a given volume.

        Returns:
            np.ndarray: Array of vibrational entropy values in eV/K/number_of_atoms for each temperature.
        """
        zero_temp_mask = self.temperatures == 0
        non_zero_temp_mask = self.temperatures > 0

        entropies = np.zeros_like(self.temperatures)
        x_array = np.zeros_like(self.temperatures)
        debye_integrals = np.zeros_like(self.temperatures)

        # Entropy at T = 0 K
        entropies[zero_temp_mask] = 0

        x_array[non_zero_temp_mask] = debye_temperature / self.temperatures[non_zero_temp_mask]
        debye_integrals[non_zero_temp_mask] = self.calculate_debye_integral_n3(x_array[non_zero_temp_mask])

        entropies[non_zero_temp_mask] = (
            3 * self.number_of_atoms * BOLTZMANN_CONSTANT *
            (4 / 3 * debye_integrals[non_zero_temp_mask] -
             np.log(1 - np.exp(-x_array[non_zero_temp_mask])))
        )
        return entropies

    def calculate_helmholtz_energies(self, debye_temperature: float) -> np.ndarray:
        """Calculates the vibrational Helmholtz energy using the Debye model.

        Args:
            debye_temperature: Debye temperature in Kelvin for a given volume.

        Returns:
            np.ndarray: Array of vibrational Helmholtz energy values in eV/number_of_atoms for each temperature.
        """
        zero_temp_mask = self.temperatures == 0
        non_zero_temp_mask = self.temperatures > 0

        helmholtz_energies = np.zeros_like(self.temperatures)
        x_array = np.zeros_like(self.temperatures)
        debye_integrals = np.zeros_like(self.temperatures)

        # Zero point energy
        helmholtz_energies[zero_temp_mask] = self.number_of_atoms * (9 / 8 * BOLTZMANN_CONSTANT * debye_temperature)
        x_array[non_zero_temp_mask] = debye_temperature / self.temperatures[non_zero_temp_mask]
        debye_integrals[non_zero_temp_mask] = self.calculate_debye_integral_n3(x_array[non_zero_temp_mask])

        helmholtz_energies[non_zero_temp_mask] = self.number_of_atoms * (
            9 / 8 * BOLTZMANN_CONSTANT * debye_temperature +
            BOLTZMANN_CONSTANT * self.temperatures[non_zero_temp_mask] *
            (3 * np.log(1 - np.exp(-x_array[non_zero_temp_mask])) -
             debye_integrals[non_zero_temp_mask])
        )
        return helmholtz_energies

    def calculate_heat_capacities(self, debye_temperature: float) -> np.ndarray:
        """Calculates the vibrational heat capacity using the Debye model.
        
        The integral evaluated is:
        (3/x³) * ∫₀ˣ [(t⁴ * exp(t)) / (exp(t) - 1)²] dt.
            
        Args:
            debye_temperature (float): Debye temperature in Kelvin for a given volume.

        Returns:
            np.ndarray: Array of vibrational heat capacity values in eV/K/number_of_atoms for each temperature.
        """
        non_zero_temp_mask = self.temperatures > 0
        x_array = np.zeros_like(self.temperatures)
        debye_integrals = np.zeros_like(x_array, dtype=float)
        heat_capacities = np.zeros_like(self.temperatures, dtype=float)

        x_array[non_zero_temp_mask] = debye_temperature / self.temperatures[non_zero_temp_mask]
        for i, x in enumerate(x_array):
            if x == 0:
                debye_integrals[i] = 0
                heat_capacities[i] = 0 # Cv at T = 0 K
            else:
                factor = 3. / x ** 3
                integral, _ = quad(lambda t: (t ** 4 * np.exp(t)) / (np.exp(t) - 1.)**2, 0, x)
                debye_integrals[i] = integral * factor
                heat_capacities[i] = 3 * self.number_of_atoms * BOLTZMANN_CONSTANT * debye_integrals[i]
        return heat_capacities

    def process(
        self,
        number_of_atoms: int,
        volumes: np.ndarray,
        temperatures: np.ndarray,
        atomic_mass: float,
        V0: float,
        B: float,
        BP: float,
        scaling_factor: float = 0.617,
        gruneisen_x: float = 2/3,
    ) -> None:
        """
        Calculate and store the Helmholtz energy, entropy, and heat capacity using the Debye-Grüneisen model.

        This method computes the vibrational Helmholtz energy, entropy, and heat capacity for each combination
        of volume and temperature, and stores the results as attributes of the class instance.

        Args:
            number_of_atoms (int): Number of atoms in the supercell.
            volumes (np.ndarray): Array of input volumes in Å³.
            temperatures (np.ndarray): Array of input temperatures in Kelvin.
            atomic_mass (float): Atomic mass in atomic mass units (u).
            V0 (float): Equilibrium volume in Å³.
            B (float): Bulk modulus in GPa.
            BP (float): First derivative of the bulk modulus with respect to pressure.
            scaling_factor (float, optional): The scaling factor defaults to 0.617, as determined by Moruzzi et al. from their study on
            nonmagnetic cubic metals (https://doi.org/10.1103/PhysRevB.37.790).
            gruneisen_x (float, optional): x parameter for the Grüneisen parameter calculation. Defaults to 2/3.

        Returns:
            None. Results are stored in the instance attributes:
                2D arrays with shape (temperatures, volumes):
                - self.helmholtz_energies (np.ndarray) Helmholtz energies in eV/number_of_atoms.
                - self.entropies (np.ndarray) entropies in eV/K/number_of_atoms.
                - self.heat_capacity (np.ndarray) heat capacities in eV/K/number_of_atoms.
        """    
        self.number_of_atoms = number_of_atoms
        self.volumes = volumes
        self.temperatures = temperatures.astype(float)
        self.atomic_mass = atomic_mass
        self.V0 = V0
        self.B = B
        self.BP = BP
        self.scaling_factor = scaling_factor
        self.gruneisen_x = gruneisen_x

        gruneisen_parameter = self.calculate_gruneisen_parameter()
        debye_temperature = self.calculate_debye_temperatures(gruneisen_parameter)

        n_vol = len(volumes)
        n_temp = len(temperatures)
        entropies_v_t = np.zeros((n_vol, n_temp))
        helmholtz_energies_v_t = np.zeros((n_vol, n_temp))
        heat_capacities_v_t = np.zeros((n_vol, n_temp))

        for i in range(n_vol):
            entropies_v_t[i, :] = self.calculate_entropies(debye_temperature[i])
            helmholtz_energies_v_t[i, :] = self.calculate_helmholtz_energies(debye_temperature[i])
            heat_capacities_v_t[i, :] = self.calculate_heat_capacities(debye_temperature[i])

        self.helmholtz_energies = helmholtz_energies_v_t.T
        self.entropies = entropies_v_t.T
        self.heat_capacities = heat_capacities_v_t.T

    def plot(
        self,
        property: str,
        selected_temperatures: np.ndarray = None,
        selected_volumes: np.ndarray = None,
    ) -> tuple[go.Figure, go.Figure]:
        """Plot the Helmholtz energy, entropy, or heat capacity as a function of temperature and volume.

        Args:
            property (str): Property to plot. Must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'.
            selected_temperatures (np.ndarray, optional): Array of selected temperatures for plotting. Defaults to None.
            selected_volumes (np.ndarray, optional): Array of selected volumes for plotting. Defaults to None.

        Raises:
            ValueError: If the property is not one of 'helmholtz_energy', 'entropy', or 'heat_capacity'.

        Returns:
            tuple[go.Figure, go.Figure]: Plotly figures as a function of temperature and volume.
                1. fig_debye_t: Plot of the selected property as a function of temperature for selected volumes.
                2. fig_debye_v: Plot of the selected property as a function of volume for selected temperatures.
        """    
        properties = {
            'helmholtz_energy': (self.helmholtz_energies.T, f"F<sub>vib</sub> (eV/{self.number_of_atoms} atoms)"),
            'entropy': (self.entropies.T, f"S<sub>vib</sub> (eV/K/{self.number_of_atoms} atoms)"),
            'heat_capacity': (self.heat_capacities.T, f"C<sub>v,vib</sub> (eV/K/{self.number_of_atoms} atoms)")
        }

        if property not in properties:
            raise ValueError("property must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'")

        y, y_label = properties[property]

        fig_debye_t = go.Figure()
        if selected_volumes is None:
            indices = np.linspace(0, len(self.volumes) - 1, 5, dtype=int)
        else:
            indices = []
            for v in selected_volumes:
                try:
                    indices.append(np.where(self.volumes == v)[0][0])
                except IndexError:
                    nearest_volume = self.volumes[np.argmin(np.abs(self.volumes - v))]
                    indices.append(np.where(self.volumes == nearest_volume)[0][0])

        for i, volume in enumerate(self.volumes):
            if i in indices:
                fig_debye_t.add_trace(
                    go.Scatter(
                        x=self.temperatures,
                        y=y[i],
                        mode="lines",
                        name=f"{volume:.2f} \u212B<sup>3</sup>",
                    )
                )
        plot_format(fig_debye_t, "Temperature (K)", y_label)

        fig_debye_v = go.Figure()
        if selected_temperatures is None:
            indices = np.linspace(0, len(self.temperatures) - 1, 5, dtype=int)
            selected_temperatures = np.array([self.temperatures[j] for j in indices])
        else:
            indices = []
            for t in selected_temperatures:
                try:
                    indices.append(np.where(self.temperatures == t)[0][0])
                except IndexError:
                    nearest_temperature = self.temperatures[np.argmin(np.abs(self.temperatures - t))]
                    indices.append(np.where(self.temperatures == nearest_temperature)[0][0])
        for i in indices:
            fig_debye_v.add_trace(
                go.Scatter(
                    x=self.volumes,
                    y=y[:, i],
                    mode="lines",
                    name=f"{self.temperatures[i]} K",
                )
            )
        plot_format(fig_debye_v, "Volume (\u212B<sup>3</sup>)", y_label)

        return fig_debye_t, fig_debye_v