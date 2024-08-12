from scipy import constants
from scipy.special import bernoulli
from scipy.special import gamma
from dfttk import eos_fit


# A is the ____ constant of th Debye-Gruneisen model
A = (8 * constants.pi**2)**(1/3)*constants.hbar/constants.k)

def scaling_factor():
    pass

def gruneisen_constant():
    pass

def gruneisen_parameter(bulk_modulus_prime, gruneisen_constant):
    return (1+bulk_modulus_prime)/2 - gruneisen_constant

def debye_temperature(volume, eos_parameters,
    mass,
    scaling_factor,
    gruneisen_constant):
    s = scaling_factor
    volume_0 = eos_parameters[0]
    energy_0 = eos_parameters[1]
    bulk_modulus = eos_parameters[2]
    bulk_modulus_prime = eos_parameters[3]
    
    gru_param = gruneisen_parameter(bulk_modulus_prime, gruneisen_constant)
    
    return s * A * volume_0**(1/6) * (bulk_modulus/mass)**(1/2) * (volume_0/volume)**gru_param

def debye_function(x, n=3, order=5):
    """series expansion of the debye function. valid for |𝑋|<2𝜋 and 𝑁≥1, comes from the expansion
    Gonzalez, I., Kondrashuk, I., Moll, V. H., & Vega, A. Analytic Expressions for Debye Functions and the Heat Capacity of a Solid. Mathematics, 10(10), 1745. https://doi.org/10.3390/math10101745
    """
    
    for k in range(1, order-2):
        summation = bernoulli(2*k)/((2*k+n)*gamma(2*k+1)) * x**(2*k)
        
    return 1 - n/(2(n+1)*x) + n*summation

def debye_gruneisen_():
    pass