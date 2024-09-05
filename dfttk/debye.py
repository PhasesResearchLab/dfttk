import numpy as np
import plotly.graph_objects as go

from scipy import constants
from scipy.special import bernoulli
from scipy.special import gamma

from dfttk import eos_fit
from dfttk.aggregate_extraction import extract_configuration_data
from dfttk.qha_yphon import plot_format



# A is the ____ constant of th Debye-Gruneisen model in units of
    # A = (8 * constants.pi**2)**(1/3)*constants.hbar/constants.k
    

# The definition of the Debye temperature is hbar/k_B * (6 * pi^2)^1/3 * N/V)
# A = hbar/k_B * (6 * pi^2)^1/3
A = 231.04 # K/(A*GPa/amu)^1/2
BOLTZMANN = constants.physical_constants['Boltzmann constant in eV/K'][0]

# check docstring bulk modulus derivative with respect to ____
def gruneisen_parameter(
    bulk_modulus_prime: float,
    gruneisen_constant: float
) -> float:
    """Calculates the gruneisen parameter

    Args:
        bulk_modulus_prime: B_0', first derivative of the bulk modulus with respect to volume
        gruneisen_constant: x, Should be between 2/3 (high temperature) and 1 (low temperature)

    Returns:
        gamma, the gruneisen parameter
    """    
    return (1+bulk_modulus_prime)/2 - gruneisen_constant

def debye_temperature(
    volume: float,
    eos_parameters: tuple[float],
    mass: float,
    gru_param: float,
    scaling_factor: float = 0.617
) -> float:
    """Calculates the Debye temperature within the Debye-Gruneisen model.

    Args:
        volume: Volume of the cell
        eos_parameters: The parameters of the equation of state: (volume_0, energy_0, bulk_modulus, bulk_modulus_prime)
        mass: Total mass of the cell
        gru_param: gruneisen parameter
        scaling_factor: Scaling factor defualts to 0.617 obtained by Moruzzi et al from nonmagnetic cubic metals

    Returns:
        Debye temperature
    """    
    s = scaling_factor
    volume_0 = eos_parameters[0]
    energy_0 = eos_parameters[1]
    bulk_modulus = eos_parameters[2]
    bulk_modulus_prime = eos_parameters[3]
    bulk_modulus_2prime = eos_parameters[4]
    
    return s * A * volume_0**(1/6) * (bulk_modulus/mass)**(1/2) * (volume_0/volume)**gru_param

def debye_function(
    x_array: np.array,
    prec: float = 1e-12,
    nth_bernoulli: int = 100
) -> np.array:
    """Calculates the debye function with n=3 using one of two series expansions. Valid for |𝑋|<2𝜋 and 𝑁≥1,
    for -2pi < x < 0.7𝜋
    .. math::
       D(x) = 1 - 3/8x + 3 \sum{k=1} \frac{B_{2k}}{(2k+3) \Gamma(2k+1)} * x^{2k}
        
    for x >= 0.7𝜋 the series
    .. math::
       D(x) = \frac{\pi^4}{5x^3} - 3 \sum{k=1} \frac{1}{k} (1 + \frac{3}{kx} + \frac{6}{k^2x^2} + \frac{6}{k^3x^3}) * e^{-kx}
        
    See references,
    Gonzalez, I., Kondrashuk, I., Moll, V. H., & Vega, A. Analytic Expressions for Debye Functions and the Heat Capacity of a Solid. Mathematics, 10(10), 1745. https://doi.org/10.3390/math10101745
    Abramowitz, M. and Stegun, I.A. eds., 1968. Handbook of mathematical functions with formulas, graphs, and mathematical tables (Vol. 55). US Government printing office.
    Khishchenko, K., Analytic approximation of the Debye function, Mathematica Montisnigri (vol. 49), 2020.  https://doi.org/10.20948/MATHMONTIS-2020-49-8

    Args:
        x_array: array of input values for the debye function
        prec: Precission. Terminates the series expansion when the absolute value of the term is less than prec
        nth_bernoulli: Determines the nth Bernoulli number to calculate. A list of bernoulli numbers is generated prior calculating the series expansion. There should be no reason to change this value under normal circumstances.
        
    Raises:
        ValueError: If the precision is not between 0 and 1
        ValueError: If x < -2𝜋
        IndexError: If the bernoulli number at index 2k is not available. This indicates slow converges of the Debye function series expansion. If you wish to calculate values for x < -𝜋 convergence may be slow and you may need to increase nth_bernoulli.

    Returns:
        np.array: The value of the debye function evaluated at each x in x_array
    """
    if not 0 < prec < 1:
        raise ValueError("The precision must be between 0 and 1")
    result = np.zeros_like(x_array)
    bern_list = bernoulli(nth_bernoulli) # 2*k must be less than 100
    for i, x in enumerate(x_array):
        term = 1 # ensures the while loop runs at least once
        k = 1
        if x >= 0.7*np.pi:
            summation = np.pi**4/(5*x**3)
            while abs(term) > prec:
                term = -3*(1/k*(1 + 3/(k*x) + 6/(k**2*x**2) + 6/(k**3*x**3))*np.exp(-k*x))
                summation += term
                k += 1
            result[i] = summation
        elif -2*np.pi < x < 0.7*np.pi:
            summation = 1 - 3/8*x
            while abs(term) > prec:
                try:
                    term = 3 * (bern_list[2*k]/((2*k+3)*gamma(2*k+1)) * x**(2*k))
                except IndexError:
                    raise IndexError(f"IndexError: the bernoulli number at index {2*k} is not available. This indicates slow converges of the Debye function series expansion. If you wish to calculate values for x < -𝜋 convergence may be slow and you may need to increase nth_bernoulli.")
                summation += term
                k += 1
            result[i] = summation
        else:
            raise ValueError("The debye function series expansions used are only valid for x > -2𝜋")
    return result

        
def vibrational_energy(temperature: float, theta: float, number_of_atoms) -> float: 
    """Evaluates the debye function at x = theta/temperature then calculates the vibrational energy in eV.

    Args:
        temperature : Temperature in Kelvin
        theta: Debye temperature in Kelvin

    Returns:
        float: Vibrational energy in eV
    """    
    debye_value = debye_function(theta/temperature)
    return number_of_atoms * BOLTZMANN * (3 * temperature * debye_value + 9/8 * theta)

def vibrational_entropy(temperature: float, theta: float, number_of_atoms) -> float:
    """Evaluates the debye function at x = theta/temperature then
    calculates the vibrational entropy in eV/K.
    
    Args:
        temperature : Temperature in Kelvin
        theta: Debye temperature in Kelvin
    
    Returns:
        float: Vibrational entropy in eV/K
    
    """
    x = theta/temperature
    debye_value = debye_function(x)
    return 3* number_of_atoms *BOLTZMANN*(4/3*debye_value-np.log(1-np.exp(-x)))

def vibrational_helmholtz_energy(temperature: float, theta:float, number_of_atoms) -> float:
    """Evaluates the debye function at x = theta/temperature then
    calculates the vibrational Helmholtz energy in eV.
    
    Args:
        temperature : Temperature in Kelvin
        theta: Debye temperature in Kelvin
        
    Returns:
        float: Vibrational Helmholtz energy in eV
    """
    x = theta/temperature
    debye_value = debye_function(x)
    return number_of_atoms * (9/8*BOLTZMANN*theta + BOLTZMANN*temperature*(3*np.log(1-np.exp(-x)) - debye_value))

def vibrational_heat_capacity(temperature: float, theta: float, number_of_atoms) -> float:
    """Evaluates the debye function and its derivative at x = theta/temperature then
    calculates the vibrational heat capacity in eV/K.
    
    Args:
        temperature : Temperature in Kelvin
        theta: Debye temperature in Kelvin
        
    Returns:
        float: Vibrational heat capacity in eV/K
    """
    x = theta/temperature
    debye_value = debye_function(x)
    return  3*number_of_atoms*BOLTZMANN*(4*debye_value - 3*x/(np.exp(x)-1))

def plot_debye(
    temperatures: np.array,
    volumes: np.array,
    number_of_atoms: int,
    y: np.array,
    y_label: str,
    selected_temperatures: np.array = None,
    volume_decimals: int = 0,
    temperature_decimals: int = 0
) -> tuple[go.Figure, go.Figure]:
    """Plots the vibrational properties (S,F,C_V) as a function of temperature and volume
    
    Args:
        temperatures: Array of temperatures
        volumes: Array of volumes
        number_of_atoms: Number of atoms in the cell
        y: Array of vibrational properties (S,F,C_V)
        y_label: Label for the y-axis ('S', 'F', 'C_V')
        selected_temperatures: Array of selected temperatures curves to plot in the y vs volume plot. If None, 5 liniearly spaced temperatures are selected., 
        volume_decimals: Number of decimals to display for the volume in the plot
        temperature_decimals: Number of decimals to display for the temperature in the plot
        
    Returns:
        tuple[go.Figure, go.Figure]: Two plotly figures, one for the vibrational properties as a function of temperature and one for the vibrational properties as a function of volume
    """
    s_t_fig = go.Figure()
    for i, volume in enumerate(volumes):
        s_t_fig.add_trace(
            go.Scatter(
                x=temperatures,
                y=y[i],
                mode='lines',
                name=f'{volume:.{volume_decimals}f} \u212B<sup>3</sup>'))
    plot_format(s_t_fig,"Temperature (K)", f"{y_label}<sub>vib</sub> (eV/K/{number_of_atoms} atoms)")
    
    s_v_fig = go.Figure()
    if selected_temperatures is None:
        indices = np.linspace(0, len(temperatures) - 1, 5, dtype=int)
        selected_temperatures = np.array([temperatures[j] for j in indices])
    for i, temperature in enumerate(selected_temperatures):
        s_v_fig.add_trace(
            go.Scatter(
                x=volumes,
                y=y[:, i],
                mode='lines',
                name=f'{temperature:.{temperature_decimals}f} K'))
    plot_format(s_v_fig,"Volume (\u212B<sup>3</sup>)", f"{y_label}<sub>vib</sub> (eV/K/{number_of_atoms} atoms)")
    return s_t_fig, s_v_fig


def process_debye_gruneisen(
    config_path: str,
    scaling_factor: float = 0.617,
    gruneisen_constant: float = 1,
    volumes: np.array = None,
    temperatures: np.array = np.linspace(10, 1000, 100),
    outcar_name: str = "OUTCAR.3static",
    oszicar_name: str = "OSZICAR.3static",
    contcar_name: str = "CONTCAR.3static",
    collect_mag_data: bool = False,
    magmom_tolerance: float = 1e-12,
    total_magnetic_moment_tolerance: float = 1e-12,
    eos_fitting = eos_fit.BM4
) -> tuple[np.array, np.array, int, np.array, np.array, np.array]:
    """Applies the Debye-Gruneisen model to a given configuration for which E-V curve calculations have been performed.
    
    Args:
        config_path: Path to the config folder
        scaling_factor: s, Scaling factor for the Debye temperature
        gruneisen_constant: x, Gruneisen constant
        volumes: Array of volumes to evaluate the Debye thermal properties at
        temperatures: Array of temperatures to evaluate the Debye thermal properties at
        outcar_name: Name of the OUTCAR file
        oszicar_name: Name of the OSZICAR file
        contcar_name: Name of the CONTCAR file
        collect_mag_data: Weather or not to collect magnetic data
        magmom_tolerance: Magnetic moment tolerance for each atom
        total_magnetic_moment_tolerance: Total magnetic moment tolerance
        eos_fitting: Equation of state fitting function from the eos_fit module
        
        Returns:
            tuple[np.array, np.array, int, np.array, np.array, np.array]: temperatures, volumes, number of atoms, vibrational entropy, vibrational Helmholtz energy, vibrational heat capacity
    """
    # extract the volume and energy from the ev_curve_series
    df = extract_configuration_data(
        path = config_path,
        outcar_name = outcar_name,
        oszicar_name = oszicar_name,
        contcar_name = contcar_name,
        collect_mag_data = collect_mag_data,
        magmom_tolerance = magmom_tolerance,
        total_magnetic_moment_tolerance = total_magnetic_moment_tolerance
    )
    # fit the equation of state
    volume = df['volume']
    energy = df['energy']
    _, eos_parameters, _, _, _ = eos_fitting(volume, energy)
    volume_0, energy_0, bulk_modulus, bulk_modulus_prime, bulk_modulus_2prime = eos_parameters
    
    s = scaling_factor
    gru_const = gruneisen_constant
    gru_param = gruneisen_parameter(bulk_modulus_prime, gru_const)
    
    if volumes is None:
        volume_min = volume.min()
        volume_max = volume.max() 
        volumes = np.linspace(volume_min, volume_max, 10) # make volumes an input parameter
    
    total_mass = df['total_mass'][0]
    number_of_atoms = df['number_of_atoms'][0]
    atomic_mass = total_mass/number_of_atoms # this needs to be corrected. arithmetic mean is no good. need geometric or log
    
    theta = debye_temperature(volumes, eos_parameters, atomic_mass, gru_param, s)
    
    s_vib_v_t = np.zeros((len(volumes), len(temperatures)))
    f_vib_v_t = np.zeros((len(volumes), len(temperatures)))
    cv_vib_v_t = np.zeros((len(volumes), len(temperatures)))
    n = df['number_of_atoms'][0]

    for i, volume in enumerate(volumes):
        s_vib = vibrational_entropy(temperatures, theta[i], n)
        f_vib = vibrational_helmholtz_energy(temperatures, theta[i], n)
        cv_vib = vibrational_heat_capacity(temperatures, theta[i], n)
        s_vib_v_t[i, :] = s_vib
        f_vib_v_t[i, :] = f_vib 
        cv_vib_v_t[i, :] = cv_vib
    
    
    return temperatures, volumes, n, s_vib_v_t, f_vib_v_t, cv_vib_v_t
    