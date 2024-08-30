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
# here A has units of K*s/(A*GPa*amu)^1/2 ??? check this ???
A = 231.04
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
    
    return s * A * volume_0**(1/6) * (bulk_modulus/mass)**(1/2) * (volume_0/volume)**gru_param

def debye_function(x_array: np.array, prec = 1e-10, nth_bernoulli = 100):
    """Calculates the debye function with n=3 using one of two series expansions. Valid for |𝑋|<2𝜋 and 𝑁≥1,
    for -2pi < x < 0.7𝜋 the series
    .. math::
        1 - 3/8x + 3 \sum{k=1} \frac{B_{2k}}{(2k+3) \Gamma(2k+1)} * x^{2k}
        
    for x >= 0.7𝜋 the series
    .. math::
        \frac{\pi^4}{5x^3} - 3 \sum{k=1} \frac{1}{k} (1 + \frac{3}{kx} + \frac{6}{k^2x^2} + \frac{6}{k^3x^3}) * e^{-kx}
        
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

        
def debye_function_derivative(x_array, prec=1e-10, nth_bernoulli=100):
    """series expansion of the derivative of the debye function. valid for |𝑋|<2𝜋 and 𝑁≥1, comes from the expansion
    Gonzalez, I., Kondrashuk, I., Moll, V. H., & Vega, A. Analytic Expressions for Debye Functions and the Heat Capacity of a Solid. Mathematics, 10(10), 1745. https://doi.org/10.3390/math10101745
    and Abramowitz, M. and Stegun, I.A. eds., 1968. Handbook of mathematical functions with formulas, graphs, and mathematical tables (Vol. 55). US Government printing office.
    """
    if not 0 < prec < 1:
        raise ValueError("The precision must be between 0 and 1")
    result = np.zeros_like(x_array)
    bern_list = bernoulli(nth_bernoulli) # 2*k must be less than 100
    for i, x in enumerate(x_array):
        term = 1 # ensures the while loop runs at least once
        k = 1
        if x >= 0.7*np.pi:
            summation = -3*np.pi**4/(5*x**4)
            while abs(term) > prec:
                term = -3*(-np.exp(-k*x)*(1 + 3/(k*x) + 9/(k**2*x**2) + 18/(k**3*x**3) + 18/(k**4*x**4)))
                summation += term
                k += 1
            result[i] = summation
        elif -2*np.pi < x < 0.7*np.pi:
            summation = -3/8
            term = 1 # ensures the while loop runs at least once
            while abs(term) > prec:
                try:
                    term = 3*(bern_list[2*k]/((2*k+3)*gamma(2*k+1)) * 2*k*x**(2*k-1))
                except IndexError:
                    raise IndexError(f"IndexError: the bernoulli number at index {2*k} is not available. This indicates slow converges of the Debye function derivative series expansion. I hope you know what you are doing.")
                summation += term
                k += 1
            result[i] = summation
        else:
            raise ValueError("The debye function derivative series expansions used are only valid for x > -2𝜋")
    return result

def vibrational_energy(temperature, theta):
    debye_value = debye_function(theta/temperature)
    return 3 * BOLTZMANN * temperature * debye_value + 9/8 * BOLTZMANN * theta

def vibrational_entropy(temperature, theta):
    x = theta/temperature
    debye_value = debye_function(x)
    return 3*BOLTZMANN*(4/3*debye_value-np.log(1-np.exp(-x)))

def vibrational_helmholtz_energy(temperature, theta):
    x = theta/temperature
    debye_value = debye_function(x)
    return 9/8*BOLTZMANN*theta + BOLTZMANN*temperature*(3*np.log(1-np.exp(-x)) - debye_value)

def vibrational_heat_capacity(temperature, theta):
    x = theta/temperature
    return 3*BOLTZMANN*temperature*debye_function_derivative(x) + debye_function(x)

def plot_debye(
    temperatures,
    volumes,
    number_of_atoms,
    y,
    y_label,
    selected_temperatures = None,
    volume_decimals = 0,
    temperature_decimals = 0
    ):
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
    plot_format(s_v_fig,"Volume (\u212B<sup>3</sup>)", f"S<sub>vib</sub> (eV/K/{number_of_atoms} atoms)")
    return s_t_fig, s_v_fig


def process_debye_gruneisen(
    config_path,
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
):
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
    print(df)
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
    theta = debye_temperature(volumes, eos_parameters, total_mass, s, gru_param)
    
    s_vib_v_t = np.zeros((len(volumes), len(temperatures)))
    f_vib_v_t = np.zeros((len(volumes), len(temperatures)))
    cv_vib_v_t = np.zeros((len(volumes), len(temperatures)))
    
    for i, volume in enumerate(volumes):
        x = theta[i]/temperatures
        debye_value = debye_function(x)
        s_vib = vibrational_entropy(temperatures, theta[i])
        f_vib = vibrational_helmholtz_energy(temperatures, theta[i])
        cv_vib = vibrational_heat_capacity(temperatures, theta[i])
        # Compute the differences between successive elements
        d_s_vib = np.array([k - j for j, k in zip(s_vib[:-1], s_vib[1:])])
        dt = np.array([k - j for j, k in zip(temperatures[:-1], temperatures[1:])])
        c_p = temperatures[:-1] * d_s_vib / dt # this is incorrect because it is done at constant volume
        s_vib_v_t[i, :] = s_vib
        f_vib_v_t[i, :] = f_vib
        cv_vib_v_t[i, :] = cv_vib
    n = df['number_of_atoms'][0]
    
    return temperatures, volumes, n, s_vib_v_t, f_vib_v_t, cv_vib_v_t
    