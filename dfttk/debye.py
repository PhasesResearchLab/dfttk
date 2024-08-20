import os
import numpy as np
from scipy import constants
from scipy.special import bernoulli
from scipy.special import gamma
from pymatgen.io.vasp.outputs import Poscar
from dfttk import eos_fit
from dfttk.aggregate_extraction import extract_configuration_data
from dfttk.data_extraction import extract_atomic_masses



# A is the ____ constant of th Debye-Gruneisen model in units of
    # A = (8 * constants.pi**2)**(1/3)*constants.hbar/constants.k
A = 231.04
BOLTZMANN = constants.physical_constants['Boltzmann constant in eV/K'][0]

def scaling_factor():
    return 0.75

def gruneisen_constant():
    return 1.0

def gruneisen_parameter(bulk_modulus_prime, gruneisen_constant):
    return (1+bulk_modulus_prime)/2 - gruneisen_constant

def debye_temperature(
    volume,
    eos_parameters,
    mass,
    scaling_factor,
    gru_param
):
    s = scaling_factor
    volume_0 = eos_parameters[0]
    energy_0 = eos_parameters[1]
    bulk_modulus = eos_parameters[2]
    bulk_modulus_prime = eos_parameters[3]
    
    return s * A * volume_0**(1/6) * (bulk_modulus/mass)**(1/2) * (volume_0/volume)**gru_param

# TODO use a while loop to ensure convergence order=30 is plenty for x > -1.5𝜋
def debye_function(x_array: np.array, order: int = 30):
    """series expansion of the debye function. valid for |𝑋|<2𝜋 and 𝑁≥1,
    comes from the expansion
    Gonzalez, I., Kondrashuk, I., Moll, V. H., & Vega, A. Analytic Expressions for Debye Functions and the Heat Capacity of a Solid. Mathematics, 10(10), 1745. https://doi.org/10.3390/math10101745
    and Abramowitz, M. and Stegun, I.A. eds., 1968. Handbook of mathematical functions with formulas, graphs, and mathematical tables (Vol. 55). US Government printing office.

    Args:
        x: _description_
        n: _description_. Defaults to 3.
        order: the default is well within accuracy of floats and takes less than 0.0 seconds for n=3. Defaults to 30.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    order = int(order)
    if order < 2:
        raise ValueError("Order of the debye function series expansion must be greater than or equal to 2.")
    result = np.zeros_like(x_array)
    for i, x in enumerate(x_array):
        if x >= 0.7*np.pi:
            summation = sum(1/k*(1 + 3/(k*x) + 6/(k**2*x**2) + 6/(k**3*x**3))*np.exp(-k*x) for k in range(1, order-1))
            result[i] = np.pi**4/(5*x**3)-3*summation
        elif -2*np.pi < x < 0.7*np.pi:
            if order > 2:
                bern_list = bernoulli(2*(order-2))
                summation = sum(bern_list[2*k]/((2*k+3)*gamma(2*k+1)) * x**(2*k) for k in range(1, order-1))
                result[i] = 1 - 3/8*x + 3 * summation
            elif order == 2:
                result[i] = 1 - 3/8*x
            else:
                # value error
                raise ValueError("Order of the debye function series expansion must be greater than or equal to 2.")
        else:
            raise ValueError("The debye function series expansions used are only valid for x > -2𝜋")
    return result

        
# TODO use a while loop to ensure convergence. order=30 is plenty for x > -1.5𝜋
def debye_function_derivative(x, order=30):
    """series expansion of the derivative of the debye function. valid for |𝑋|<2𝜋 and 𝑁≥1, comes from the expansion
    Gonzalez, I., Kondrashuk, I., Moll, V. H., & Vega, A. Analytic Expressions for Debye Functions and the Heat Capacity of a Solid. Mathematics, 10(10), 1745. https://doi.org/10.3390/math10101745
    and Abramowitz, M. and Stegun, I.A. eds., 1968. Handbook of mathematical functions with formulas, graphs, and mathematical tables (Vol. 55). US Government printing office.
    """
    
    order = int(order)
    
    if x >= 0.7*np.pi:
        summation = sum(-np.exp(-k*x)*(1 + 3/(k*x) + 9/(k**2*x**2) + 18/(k**3*x**3) + 18/(k**4*x**4)) for k in range(1, order))
        return -3*np.pi**4/(5*x**4) - 3*summation
        
    elif -2*np.pi < x < 0.7*np.pi:
        if order > 1:
            bern_list = bernoulli(2*(order-1))
            summation = sum(bern_list[2*k]/((2*k+3)*gamma(2*k+1)) * 2*k*x**(2*k-1) for k in range(1, order))
            return -3/8 + 3*summation
        elif order == 1:
            return -3/8
        else:
            raise ValueError("Order of the debye function derivative series expansion must be greater than or equal to 1.")
    else:
        raise ValueError("The debye function derivative series expansions used are only valid for x > -2𝜋")

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

def process_debye_gruneisen(
    config_path,
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
    
    # fit the equation of state
    volume = df['volume']
    energy = df['energy']
    _, eos_parameters, _, _, _ = eos_fitting(volume, energy)
    volume_0, energy_0, bulk_modulus, bulk_modulus_prime, bulk_modulus_2prime = eos_parameters
    
    s = scaling_factor()
    gru_const = gruneisen_constant()
    gru_param = gruneisen_parameter(bulk_modulus_prime, gru_const)
    
    volume = np.array(1) # please finish this
    temperature =  np.array(1) # please finish this
    x=theta/temperature
    debye_value = debye_function(x)
    return
    