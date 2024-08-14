import os
import numpy as np
from scipy import constants
from scipy.special import bernoulli
from scipy.special import gamma
from dfttk import eos_fit
from dfttk.aggregate_extraction import extract_configuration_data
from dfttk.data_extraction import extract_mass



# A is the ____ constant of th Debye-Gruneisen model
A = (8 * constants.pi**2)**(1/3)*constants.hbar/constants.k

def scaling_factor():
    pass

def gruneisen_constant():
    pass

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

def debye_function(x: float, n: int = 3, order: int = 30):
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
    

    if x > 1.5*np.pi:
        return
        # return np.exp(-k*x)((x**n)/k + )
    if x <= 1.5*np.pi:
        if order > 2:
            bern_list = bernoulli(2*(order-2))
            summation = sum(bern_list[2*k]/((2*k+n)*gamma(2*k+1)) * x**(2*k) for k in range(1, order-1))
            return 1 - n/(2*(n+1))*x + n * summation
        elif order == 2:
            return 1 - n/(2*(n+1))*x
        elif order < 2:
            # value error
            raise ValueError("Order of the debye function series expansion must be greater than or equal to 2.")
    elif x == 0:
        return 1.0
        

def debye_function_derivative(x, n=3, order=5):
    """series expansion of the derivative of the debye function. valid for |𝑋|<2𝜋 and 𝑁≥1, comes from the expansion
    Gonzalez, I., Kondrashuk, I., Moll, V. H., & Vega, A. Analytic Expressions for Debye Functions and the Heat Capacity of a Solid. Mathematics, 10(10), 1745. https://doi.org/10.3390/math10101745
    and Abramowitz, M. and Stegun, I.A. eds., 1968. Handbook of mathematical functions with formulas, graphs, and mathematical tables (Vol. 55). US Government printing office.
    """
    
    summation = sum(bernoulli(2*k)/((2*k+n)*gamma(2*k+1)) * 2*k*x**(2*k-1) for k in range(1, order-2))
    return -n/(2*(n+1)) + n*summation
        
    return -n/((2*n+1)) + n*summation

def vibrational_energy(temperature, theta):
    debye_value = debye_function(theta/temperature)
    return 3 * constants.k * temperature * debye_value + 9/8 * constants.k * theta

def vibrational_entropy(temperature, theta):
    x = theta/temperature
    debye_value = debye_function(x)
    return 3*constants.k*(4/3*debye_value-np.log(1-np.exp(-x)))

def vibrational_helmholtz_energy(temperature, theta):
    x = theta/temperature
    debye_value = debye_function(x)
    return 9/8*constants.k*theta + constants.k*temperature*(3*np.log(1-np.exp(-x)) - debye_value)

def vibrational_heat_capacity(temperature, theta):
    x = theta/temperature
    return 3*constants.k*temperature*debye_function_derivative(x) + debye_function(x)

def debye_gruneisen(
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
    _, eos_parameters, _, _, _ = eos_fitting(df)
    volume_0, energy_0, bulk_modulus, bulk_modulus_prime, bulk_modulus_2prime = eos_parameters
    
    
    # extract the mass of each atom from the OUTCAR file
    mass = extract_mass(os.path.join(config_path, outcar_name))
    
    s = scaling_factor()
    gru_const = gruneisen_constant()
    gru_param = gruneisen_parameter(bulk_modulus_prime, gru_const)
    
    volume = np.array(1) # please finish this
    temperature =  np.array(1) # please finish this
    theta = debye_temperature(volume, eos_parameters, mass, s, gru_param)
    x=theta/temperature
    debye_value = debye_function(x)
    return
    