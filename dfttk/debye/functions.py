"""
Debye-Grüneisen module to calculate the vibrational contribution to the Helmholtz energy, entropy, and heat capacity.
"""

# Related third party imports
import numpy as np
import plotly.graph_objects as go
from scipy import constants
from scipy.integrate import quad

# DFTTK imports
from dfttk.plotly_format import plot_format

BOLTZMANN_CONSTANT = constants.physical_constants["Boltzmann constant in eV/K"][0]
HBAR = constants.physical_constants["Planck constant over 2 pi in eV s"][0]


def calculate_gruneisen_parameter(BP: float, gruneisen_x: float) -> float:
    """Calculates the Gruneisen parameter (gamma).

    Args:
        BP (float): First derivative of the bulk modulus with respect to pressure.
        gruneisen_x (float): Typical values are 2/3 (high temperature) and 1 (low temperature).

    Returns:
        float: Gruneisen parameter (gamma).
    """
    return (1 + BP) / 2 - gruneisen_x


def calculate_debye_temperatures(
    volumes: np.ndarray,
    V0: float,
    B: float,
    atomic_mass: float,
    gruneisen_parameter: float,
    scaling_factor: float = 0.617,
) -> np.ndarray:
    """Calculates the Debye temperatures in Kelvin within the Debye-Gruneisen model.

    Args:
        volumes (np.ndarray): Volumes of the supercell in Å^3.
        V0 (float): Equilibrium volume of the supercell in Å^3.
        B (float): Bulk modulus in GPa.
        atomic_mass (float): Atomic mass in u.
        gruneisen_parameter (float): Gruneisen parameter.
        scaling_factor (float, optional): The scaling factor defaults to 0.617, as determined by Moruzzi et al. from their study on
            nonmagnetic cubic metals (https://doi.org/10.1103/PhysRevB.37.790).

    Returns:
        np.ndarray: Debye temperatures in Kelvin for each volume in the input array.
    """
    A = (6 * np.pi**2)**(1/3) * HBAR / BOLTZMANN_CONSTANT # Constant in s.K
    B = B * 1e12 # Convert GPa to g/(ms^2)
    V0 = V0 * 1e-30 # Convert Å^3 to m^3
    volumes = volumes * 1e-30 # Convert Å^3 to m^3
    atomic_mass = atomic_mass * 1.66054e-24 # Convert u to g
    
    debye_temperatures = scaling_factor * A * V0 ** (1 / 6) * (
        B / atomic_mass
    ) ** (1 / 2) * (V0 / volumes) ** gruneisen_parameter
    
    return debye_temperatures


def calculate_debye_integral_n3(x_array: np.array) -> np.array:
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


def calculate_entropies(temperatures: np.ndarray, debye_temperature: float, number_of_atoms: int) -> np.ndarray:
    """Calculate the vibrational entropy using the Debye model.

    Args:
        temperatures (np.ndarray): Array of temperatures in Kelvin.
        debye_temperature (float): Debye temperature in Kelvin for a given volume.
        number_of_atoms (int): Number of atoms in the supercell.

    Returns:
        np.ndarray: Array of vibrational entropy values in eV/K/number_of_atoms for each temperature.
    """
    
    temperatures = temperatures.astype(float)
    zero_temp_mask = temperatures == 0
    non_zero_temp_mask = temperatures > 0

    entropies = np.zeros_like(temperatures)
    x_array = np.zeros_like(temperatures)
    debye_integrals = np.zeros_like(temperatures)

    # Entropy at T = 0 K
    entropies[zero_temp_mask] = 0

    x_array[non_zero_temp_mask] = debye_temperature / temperatures[non_zero_temp_mask]
    debye_integrals[non_zero_temp_mask] = calculate_debye_integral_n3(x_array[non_zero_temp_mask])

    entropies[non_zero_temp_mask] = (
        3
        * number_of_atoms
        * BOLTZMANN_CONSTANT
        * (
            4 / 3 * debye_integrals[non_zero_temp_mask]
            - np.log(1 - np.exp(-x_array[non_zero_temp_mask]))
        )
    )

    return entropies


def calculate_helmholtz_energies(
    temperatures: np.ndarray , debye_temperature: float, number_of_atoms: int
) -> np.ndarray:
    """Calculates the vibrational Helmholtz energy using the Debye model.

    Args:
        temperatures: Array of temperatures in Kelvin.
        debye_temperature: Debye temperature in Kelvin for a given volume.
        number_of_atoms: Number of atoms in the supercell.

    Returns:
        np.ndarray: Array of vibrational Helmholtz energy values in eV/number_of_atoms for each temperature.
    """
    
    temperatures = temperatures.astype(float)
    zero_temp_mask = temperatures == 0
    non_zero_temp_mask = temperatures > 0

    helmholtz_energies = np.zeros_like(temperatures)
    x_array = np.zeros_like(temperatures)
    debye_integrals= np.zeros_like(temperatures)

    # Zero point energy
    helmholtz_energies[zero_temp_mask] = number_of_atoms * (9 / 8 * BOLTZMANN_CONSTANT * debye_temperature)

    x_array[non_zero_temp_mask] = debye_temperature / temperatures[non_zero_temp_mask]
    debye_integrals[non_zero_temp_mask] = calculate_debye_integral_n3(x_array[non_zero_temp_mask])

    helmholtz_energies[non_zero_temp_mask] = number_of_atoms * (
        9 / 8 * BOLTZMANN_CONSTANT * debye_temperature
        + BOLTZMANN_CONSTANT
        * temperatures[non_zero_temp_mask]
        * (
            3 * np.log(1 - np.exp(-x_array[non_zero_temp_mask]))
            - debye_integrals[non_zero_temp_mask]
        )
    )

    return helmholtz_energies


def calculate_heat_capacities(
    temperatures: np.ndarray, debye_temperature: float, number_of_atoms: int
) -> np.ndarray:
    """Calculates the vibrational heat capacity using the Debye model.
    
    The integral evaluated is:
       (3/x³) * ∫₀ˣ [(t⁴ * exp(t)) / (exp(t) - 1)²] dt.
        
    Args:
        temperatures (np.ndarray): Array of temperatures in Kelvin.
        debye_temperature (float): Debye temperature in Kelvin for a given volume.
        number_of_atoms (int): Number of atoms in the supercell.

    Returns:
        np.ndarray: Array of vibrational heat capacity values in eV/K/number_of_atoms for each temperature.
    """
    
    temperatures = temperatures.astype(float)
    non_zero_temp_mask = temperatures > 0
    x_array = np.zeros_like(temperatures)
    debye_integrals = np.zeros_like(x_array, dtype=float)
    heat_capacities = np.zeros_like(temperatures, dtype=float)
    
    x_array[non_zero_temp_mask] = debye_temperature / temperatures[non_zero_temp_mask]
    
    for i, x in enumerate(x_array):
        if x == 0:
            debye_integrals[i] = 0
            heat_capacities[i] = 0 # Cv at T = 0 K
        else:
            factor = 3. / x ** 3
            integral, _ = quad(lambda t: (t ** 4 * np.exp(t)) / (np.exp(t) - 1.)**2, 0, x)
            debye_integrals[i] = integral * factor
            heat_capacities[i] = 3 * number_of_atoms * BOLTZMANN_CONSTANT * debye_integrals[i]  

    return heat_capacities


def process_debye_gruneisen(
    number_of_atoms: int,
    volumes: np.ndarray,
    temperatures: np.ndarray,
    atomic_mass: float,
    V0: float,
    B: float,
    BP: float,
    scaling_factor: float = 0.617,
    gruneisen_x: float = 2/3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience function to calculate the Helmholtz energy, entropy, and heat capacity using the Debye model.

    Args:
        number_of_atoms (int): Number of atoms in the supercell.
        volumes (np.ndarray): Array of input volumes in Å^3.
        temperatures (np.ndarray): Array of input temperatures in Kelvin.
        atomic_mass (float): Atomic mass in u.
        V0 (float): Equilibrium volume in Å^3.
        B (float): Bulk modulus in GPa.
        BP (float): First derivative of the bulk modulus with respect to pressure.
        scaling_factor (float, optional): Scaling factor for the Debye temperature calculation. Defaults to 0.617.
        gruneisen_x (float, optional): x parameter for the Gruneisen parameter calculation. Defaults to 2/3.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 
            Each 2D array has shape (volumes, temperatures):
            - Helmholtz energies (eV/number_of_atoms).
            - Entropies (eV/K/number_of_atoms).
            - Heat capacities (eV/K/number_of_atoms).
    """    

    temperatures = temperatures.astype(float)
    gruneisen_parameter = calculate_gruneisen_parameter(BP, gruneisen_x)
    debye_temperature = calculate_debye_temperatures(volumes, V0, B, atomic_mass, gruneisen_parameter, scaling_factor)

    entropies_v_t = np.zeros((len(volumes), len(temperatures)))
    helmholtz_energies_v_t = np.zeros((len(volumes), len(temperatures)))
    heat_capacities_v_t = np.zeros((len(volumes), len(temperatures)))

    for i, volume in enumerate(volumes):
        entropies_t = calculate_entropies(temperatures, debye_temperature[i], number_of_atoms)
        helmholtz_energies_t = calculate_helmholtz_energies(temperatures, debye_temperature[i], number_of_atoms)
        heat_capacities_t = calculate_heat_capacities(temperatures, debye_temperature[i], number_of_atoms)
        entropies_v_t[i, :] = entropies_t
        helmholtz_energies_v_t[i, :] = helmholtz_energies_t
        heat_capacities_v_t[i, :] = heat_capacities_t

    helmholtz_energies = helmholtz_energies_v_t.T
    entropies = entropies_v_t.T
    heat_capacities = heat_capacities_v_t.T
    print(heat_capacities)

    return helmholtz_energies, entropies, heat_capacities


def plot(
    property: str,
    number_of_atoms: int,
    temperatures: np.ndarray,
    volumes: np.ndarray,
    helmholtz_energies: np.ndarray,
    entropies: np.ndarray,
    heat_capacities: np.ndarray,
    selected_temperatures: np.ndarray = None,
    selected_volumes: np.ndarray = None,
) -> tuple[go.Figure, go.Figure]:
    """Plot the Helmholtz energy, entropy, or heat capacity as a function of temperature and volume.

    Args:
        property (str): Property to plot. Must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'.
        number_of_atoms (int): Number of atoms in the supercell.
        temperatures (np.ndarray): Array of temperatures in Kelvin.
        volumes (np.ndarray): Array of volumes in Å^3.
        helmholtz_energies (np.ndarray): 2D array of Helmholtz energies in eV/number_of_atoms.
        entropies (np.ndarray): 2D array of entropies in eV/K/number_of_atoms.
        heat_capacities (np.ndarray): 2D array of heat capacities in eV/K/number_of_atoms.
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
        'helmholtz_energy': (helmholtz_energies.T, f"F<sub>vib</sub> (eV/{number_of_atoms} atoms)"),
        'entropy': (entropies.T, f"S<sub>vib</sub> (eV/K/{number_of_atoms} atoms)"),
        'heat_capacity': (heat_capacities.T, f"C<sub>v,vib</sub> (eV/K/{number_of_atoms} atoms)")
    }

    if property not in properties:
        raise ValueError("property must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'")

    y, y_label = properties[property]

    fig_debye_t = go.Figure()
    if selected_volumes is None:
        indices = np.linspace(0, len(volumes) - 1, 5, dtype=int)
    else:
        indices = []
        for v in selected_volumes:
            try:
                indices.append(np.where(volumes == v)[0][0])
            except IndexError:
                nearest_volume = volumes[np.argmin(np.abs(volumes - v))]
                indices.append(np.where(volumes == nearest_volume)[0][0])

    for i, volume in enumerate(volumes):
        if i in indices:
            fig_debye_t.add_trace(
                go.Scatter(
                    x=temperatures,
                    y=y[i],
                    mode="lines",
                    name=f"{volume:.2f} \u212B<sup>3</sup>",
                )
            )
    plot_format(
        fig_debye_t,
        "Temperature (K)",
        y_label,
    )
    fig_debye_t.show()

    fig_debye_v = go.Figure()
    if selected_temperatures is None:
        indices = np.linspace(0, len(temperatures) - 1, 5, dtype=int)
        selected_temperatures = np.array([temperatures[j] for j in indices])
    else:
        indices = []
        for t in selected_temperatures:
            try:
                indices.append(np.where(temperatures == t)[0][0])
            except IndexError:
                nearest_temperature = temperatures[
                    np.argmin(np.abs(temperatures - t))
                ]
                indices.append(np.where(temperatures == nearest_temperature)[0][0])
    for i in indices:
        fig_debye_v.add_trace(
            go.Scatter(
                x=volumes,
                y=y[:, i],
                mode="lines",
                name=f"{temperatures[i]} K",
            )
        )
    plot_format(
        fig_debye_v,
        "Volume (\u212B<sup>3</sup>)",
        y_label,
    )
    fig_debye_v.show()

    return fig_debye_t, fig_debye_v