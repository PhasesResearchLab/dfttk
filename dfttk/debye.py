# Standard library imports
import numpy as np

# Third-party imports
import plotly.graph_objects as go
from scipy import constants
from scipy.integrate import quad

# DFTTK imports
from dfttk.plotly_format import plot_format

# Physical constants
BOLTZMANN_CONSTANT = constants.physical_constants["Boltzmann constant in eV/K"][0]
HBAR = constants.physical_constants["Planck constant over 2 pi in eV s"][0]


class DebyeGruneisen:
    """
    A class for computing vibrational contributions to the Helmholtz free energy, entropy,
    and heat capacity using the Debye-Grüneisen model, with built-in plotting utilities.
    """

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
        gruneisen_x: float = 2 / 3,
    ) -> None:
        """This method computes the vibrational Helmholtz energy, entropy, and heat capacity
        for each combination of volume and temperature, and stores the results as attributes.

        Args:
            number_of_atoms: Number of atoms in the supercell.
            volumes: Array of input volumes in Å³.
            temperatures: Array of input temperatures in K.
            atomic_mass: Atomic mass in atomic mass units (u).
            V0: Equilibrium volume in Å³.
            B: Bulk modulus in GPa.
            BP: First derivative of the bulk modulus with respect to pressure.
            scaling_factor (float, optional): The scaling factor defaults to 0.617,
                as determined by Moruzzi et al. from their study on
                nonmagnetic cubic metals (https://doi.org/10.1103/PhysRevB.37.790).
            gruneisen_x (float, optional): x parameter for the Grüneisen parameter calculation.
                Defaults to 2/3.
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

        # Calculate Debye model properties
        gruneisen_parameter = DebyeGruneisen.calculate_gruneisen_parameter(
            self.BP, self.gruneisen_x
        )
        debye_temperature = DebyeGruneisen.calculate_debye_temperatures(
            self.volumes,
            self.atomic_mass,
            self.V0,
            self.B,
            self.scaling_factor,
            gruneisen_parameter,
        )

        n_vol = len(volumes)
        n_temp = len(temperatures)
        entropies_v_t = np.zeros((n_vol, n_temp))
        helmholtz_energies_v_t = np.zeros((n_vol, n_temp))
        heat_capacities_v_t = np.zeros((n_vol, n_temp))

        # Loop over volumes, calculate properties for all temperatures at each volume
        for i in range(n_vol):
            helmholtz_energies_v_t[i, :] = self.calculate_helmholtz_energies(
                debye_temperature[i]
            )
            entropies_v_t[i, :] = self.calculate_entropies(debye_temperature[i])
            heat_capacities_v_t[i, :] = self.calculate_heat_capacities(
                debye_temperature[i]
            )

        # Store results as (temperature, volume) arrays
        self.helmholtz_energies = helmholtz_energies_v_t.T
        self.entropies = entropies_v_t.T
        self.heat_capacities = heat_capacities_v_t.T

    def plot_vt(
        self,
        type: str,
        selected_temperatures: np.ndarray = None,
        selected_volumes: np.ndarray = None,
    ) -> go.Figure:
        """Plots the vibrational Helmholtz energy, entropy, or heat capacity as a function
        of temperature or volume.

        Args:
            type:
                Must be one of the following values:
                ``'helmholtz_energy_vs_temperature'``, ``'entropy_vs_temperature'``,
                ``'heat_capacity_vs_temperature'``, ``'helmholtz_energy_vs_volume'``,
                ``'entropy_vs_volume'``, or ``'heat_capacity_vs_volume'``.
            selected_temperatures:
                Selected temperatures to use for volume plots, shape
                (n_selected_temperatures,). Defaults to None.
            selected_volumes:
                Selected volumes to use for temperature plots, shape
                (n_selected_volumes,). Defaults to None.

        Raises:
            RuntimeError: If process() has not been called before plot_vt().
            ValueError: The `type` argument is not one of the allowed values.

        Returns: Plotly figures as a function of temperature or volume.
        """

        def _nearest_indices(values: np.ndarray, selected: np.ndarray) -> list[int]:
            indices: list[int] = []
            for target in selected:
                matches = np.where(values == target)[0]
                if matches.size:
                    indices.append(int(matches[0]))
                else:
                    nearest_index = int(np.argmin(np.abs(values - target)))
                    indices.append(nearest_index)
            return indices

        # Raise an error if process has not been called yet
        if (
            self.helmholtz_energies is None
            or self.entropies is None
            or self.heat_capacities is None
        ):
            raise RuntimeError("process() must be called before plot().")

        type_map = {
            "helmholtz_energy_vs_temperature": (
                self.helmholtz_energies,
                "F<sub>vib</sub> (eV)",
            ),
            "entropy_vs_temperature": (self.entropies, "S<sub>vib</sub> (eV/K)"),
            "heat_capacity_vs_temperature": (
                self.heat_capacities,
                "C<sub>v,vib</sub> (eV/K)",
            ),
            "helmholtz_energy_vs_volume": (
                self.helmholtz_energies,
                "F<sub>vib</sub> (eV)",
            ),
            "entropy_vs_volume": (
                self.entropies,
                "S<sub>vib</sub> (eV/K)",
            ),
            "heat_capacity_vs_volume": (
                self.heat_capacities,
                "C<sub>v,vib</sub> (eV/K)",
            ),
        }

        if type not in type_map:
            raise ValueError(
                "type must be one of "
                "'helmholtz_energy_vs_temperature', 'entropy_vs_temperature', "
                "'heat_capacity_vs_temperature', 'helmholtz_energy_vs_volume', "
                "'entropy_vs_volume', or 'heat_capacity_vs_volume'."
            )

        property_values, y_title = type_map[type]

        if "vs_temperature" in type:
            # Plot property vs. temperature for selected volumes
            fig = go.Figure()
            if selected_volumes is None:
                # Default: plot for 5 evenly spaced volumes
                indices = np.linspace(0, len(self.volumes) - 1, 5, dtype=int)
            else:
                # Find indices for selected volumes (use nearest if not exact)
                indices = _nearest_indices(self.volumes, selected_volumes)

            for i in indices:
                volume = float(self.volumes[i])
                fig.add_trace(
                    go.Scatter(
                        x=self.temperatures,
                        y=property_values[:, i],
                        mode="lines",
                        name=f"{volume:.2f} \u212B<sup>3</sup>",
                    )
                )
            plot_format(fig, "Temperature (K)", y_title)

        elif "vs_volume" in type:
            # Plot property vs. volume for selected temperatures
            fig = go.Figure()
            if selected_temperatures is None:
                # Default: plot for 5 evenly spaced temperatures
                indices = np.linspace(0, len(self.temperatures) - 1, 5, dtype=int)
            else:
                # Find indices for selected temperatures (use nearest if not exact)
                indices = _nearest_indices(self.temperatures, selected_temperatures)
            for i in indices:
                temperature = float(self.temperatures[i])
                if np.isclose(temperature % 1, 0):
                    temperature_label = f"{int(round(temperature))} K"
                else:
                    temperature_label = f"{temperature} K"
                fig.add_trace(
                    go.Scatter(
                        x=self.volumes,
                        y=property_values[i],
                        mode="lines",
                        name=temperature_label,
                    )
                )
            plot_format(fig, f"Volume (\u212B<sup>3</sup>)", y_title)

        return fig

    @staticmethod
    def calculate_gruneisen_parameter(BP: float, gruneisen_x: float) -> float:
        """Calculates the Grüneisen parameter (gamma).
        Args:
            BP: First derivative of the bulk modulus with respect to pressure.
            gruneisen_x: x parameter for the Grüneisen parameter calculation.

        Returns: Grüneisen parameter (gamma).
        """
        return (1 + BP) / 2 - gruneisen_x

    @staticmethod
    def calculate_debye_temperatures(
        volumes: np.ndarray,
        atomic_mass: float,
        V0: float,
        B: float,
        scaling_factor: float,
        gruneisen_parameter: float,
    ) -> np.ndarray:
        """Compute the Debye temperature for each volume.

        Args:
            volumes: Array of input volumes in Å^3.
            atomic_mass: Atomic mass in atomic mass units (u).
            V0: Equilibrium volume in Å^3.
            B: Bulk modulus in GPa.
            scaling_factor: Scaling factor.
            gruneisen_parameter: Grüneisen parameter (gamma).

        Returns: Debye temperatures for each volume in K.
        """

        # Compute prefactor using physical constants
        A = (6 * np.pi**2) ** (1 / 3) * HBAR / BOLTZMANN_CONSTANT

        # Convert units for volume, bulk modulus, and atomic mass
        V0 = V0 * 1e-30  # Å^3 to m^3
        B = B * 1e12  # GPa to g/(ms^2)
        volumes = volumes * 1e-30  # Å^3 to m^3
        atomic_mass = atomic_mass * 1.66054e-24  # u to g

        # Debye temperature formula from the Debye-Grüneisen model
        debye_temperatures = (
            scaling_factor
            * A
            * V0 ** (1 / 6)
            * (B / atomic_mass) ** 0.5
            * (V0 / volumes) ** gruneisen_parameter
        )
        return debye_temperatures

    @staticmethod
    def calculate_debye_integral_n3(x_array: np.ndarray) -> np.ndarray:
        """Calculate the Debye integral of order 3 for an array of upper limits.

        For each value ``x`` in ``x_array``, this computes:

        .. math::

            D(x) = \\frac{3}{x^3} \\int_0^x \\frac{t^3}{e^t - 1} \\, dt

        Args:
            x_array: Array of upper integration limits, where each value is the Debye temperature
                divided by the temperature. The Debye temperature is fixed for a given volume,
                and the temperature is varied.

        ValueError: If any value in x_array is zero, since this would lead to division by
            zero in the formula.

        Returns:
            Array of Debye integrals of order 3 corresponding to each value in x_array.
        """

        # Raise an error in any elements of x_array has any 0
        if np.any(x_array == 0):
            raise ValueError(
                "x_array must not contain any zero values to avoid division by zero."
            )

        debye_integrals = np.zeros_like(x_array, dtype=float)
        for i, x in enumerate(x_array):
            factor = 3.0 / x**3
            integral, _ = quad(lambda t: t**3 / (np.exp(t) - 1.0), 0, x)
            debye_integrals[i] = integral * factor
        return debye_integrals

    def calculate_helmholtz_energies(self, debye_temperature: float) -> np.ndarray:
        """Calculates the vibrational Helmholtz energy.

        Args:
            debye_temperature: Debye temperature in K for a given volume.

        Returns: Array of vibrational Helmholtz energy values in eV for each temperature.
        """

        zero_temp_mask = self.temperatures == 0
        non_zero_temp_mask = self.temperatures > 0

        helmholtz_energies = np.zeros_like(self.temperatures)
        x_array = np.zeros_like(self.temperatures)
        debye_integrals = np.zeros_like(self.temperatures)

        # Zero point energy at T = 0 K
        helmholtz_energies[zero_temp_mask] = self.number_of_atoms * (
            9 / 8 * BOLTZMANN_CONSTANT * debye_temperature
        )
        x_array[non_zero_temp_mask] = (
            debye_temperature / self.temperatures[non_zero_temp_mask]
        )
        debye_integrals[non_zero_temp_mask] = (
            DebyeGruneisen.calculate_debye_integral_n3(x_array[non_zero_temp_mask])
        )

        # Debye Helmholtz energy formula
        helmholtz_energies[non_zero_temp_mask] = self.number_of_atoms * (
            9 / 8 * BOLTZMANN_CONSTANT * debye_temperature
            + BOLTZMANN_CONSTANT
            * self.temperatures[non_zero_temp_mask]
            * (
                3 * np.log(1 - np.exp(-x_array[non_zero_temp_mask]))
                - debye_integrals[non_zero_temp_mask]
            )
        )
        return helmholtz_energies

    def calculate_entropies(self, debye_temperature: float) -> np.ndarray:
        """Calculate the vibrational entropy.

        Args:
            debye_temperature: Debye temperature in K for a given volume.

        Returns: Array of vibrational entropy values in eV/K for each temperature.
        """

        # Masks for zero and nonzero temperatures
        zero_temp_mask = self.temperatures == 0
        non_zero_temp_mask = self.temperatures > 0

        entropies = np.zeros_like(self.temperatures)
        x_array = np.zeros_like(self.temperatures)
        debye_integrals = np.zeros_like(self.temperatures)

        # Entropy at T = 0 K is zero
        entropies[zero_temp_mask] = 0

        # Calculate x = theta_D / T for nonzero T
        x_array[non_zero_temp_mask] = (
            debye_temperature / self.temperatures[non_zero_temp_mask]
        )
        debye_integrals[non_zero_temp_mask] = (
            DebyeGruneisen.calculate_debye_integral_n3(x_array[non_zero_temp_mask])
        )

        # Debye entropy formula
        entropies[non_zero_temp_mask] = (
            3
            * self.number_of_atoms
            * BOLTZMANN_CONSTANT
            * (
                4 / 3 * debye_integrals[non_zero_temp_mask]
                - np.log(1 - np.exp(-x_array[non_zero_temp_mask]))
            )
        )
        return entropies

    def calculate_heat_capacities(self, debye_temperature: float) -> np.ndarray:
        """Calculates the vibrational heat capacity.

        The integral evaluated is:

        .. math::

        \\frac{3}{x^3} \\int_0^x \\frac{t^4 e^t}{(e^t - 1)^2} \\, dt

        Args:
            debye_temperature: Debye temperature in K for a given volume.

        Returns: Array of vibrational heat capacity values in eV/K for each temperature.
        """

        non_zero_temp_mask = self.temperatures > 0
        x_array = np.zeros_like(self.temperatures)
        debye_integrals = np.zeros_like(x_array, dtype=float)
        heat_capacities = np.zeros_like(self.temperatures, dtype=float)

        x_array[non_zero_temp_mask] = (
            debye_temperature / self.temperatures[non_zero_temp_mask]
        )
        for i, x in enumerate(x_array):
            if x == 0:
                debye_integrals[i] = 0
                heat_capacities[i] = 0  # Cv at T = 0 K
            else:
                factor = 3.0 / x**3
                integral, _ = quad(
                    lambda t: (t**4 * np.exp(t)) / (np.exp(t) - 1.0) ** 2, 0, x
                )
                debye_integrals[i] = integral * factor
                heat_capacities[i] = (
                    3 * self.number_of_atoms * BOLTZMANN_CONSTANT * debye_integrals[i]
                )
        return heat_capacities
