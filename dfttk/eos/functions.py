"""
The EOS functions are based on the following paper:

Shun-Li Shang et al., Computational Materials Science, 47, 4, (2010).
https://doi.org/10.1016/j.commatsci.2009.12.006

It includes the following equations of state: 4-parameter Teter-Shang modified Birch-Murnaghan (mBM4),
5-parameter Teter-Shang modified Birch-Murnaghan (mBM5),
4-parameter Birch-Murnaghan (BM4),
5-parameter Birch-Murnaghan (BM5),
4-parameter Natural (LOG4),
5-parameter Natural (LOG5),
4-parameter Murnaghan,
4-parameter Vinet,
and 4-parameter Morse.
"""

# Related third party imports
import numpy as np
from scipy.optimize import fsolve, curve_fit
from collections import namedtuple

# Conversion factor
EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3  = 160.21766208 GPa

TaylorCoefficients = namedtuple("TaylorCoefficients", ["a", "b", "c", "d", "e"])

def BM_parameters_to_taylor(
        V,
        E0,
        B,
        BP,
        B2P
    )->TaylorCoefficients:
    """
    Convert Birch-Murnaghan equation of state properties to Taylor 
    coefficients a, b, c, d, e (4th order/five parameters). Returns a named 
    tuple with the coefficients.
    """
    B = B/EV_PER_CUBIC_ANGSTROM_TO_GPA
    B2P = B2P*EV_PER_CUBIC_ANGSTROM_TO_GPA
    a = (128*E0 + 3*B*(287 + 9*B*B2P - 87*BP + 9*BP**2)*V)/128
    b = (-3*B*(239 + 9*B*B2P - 81*BP + 9*BP**2)*V**(5/3))/32
    c = (9*B*(199 + 9*B*B2P - 75*BP + 9*BP**2)*V**(7/3))/64
    d = (-3*B*(167 + 9*B*B2P - 69*BP + 9*BP**2)*V**3)/32
    e = (3*B*(143 + 9*B*B2P - 63*BP + 9*BP**2)*V**(11/3))/128
    return TaylorCoefficients(a, b, c, d, e)

def mBM_parameters_to_taylor(V, E0, B, BP, B2P):
    """
    Convert modified Birch-Murnaghan equation of state properties to Taylor
    coefficients a, b, c, d, e (4th order/five parameters). Returns a named
    tuple with the coefficients.
    """
    B = B/EV_PER_CUBIC_ANGSTROM_TO_GPA
    B2P = B2P*EV_PER_CUBIC_ANGSTROM_TO_GPA
    a=(8*E0 + 3*B*(122 + 9*B*B2P - 57*BP + 9*BP**2)*V)/8
    b=(-3*B*(107 + 9*B*B2P - 54*BP + 9*BP**2)*V**(4/3))/2
    c=(9*B*(94 + 9*B*B2P - 51*BP + 9*BP**2)*V**(5/3))/4
    d=(-3*B*(83 + 9*B*B2P - 48*BP + 9*BP**2)*V**2)/2
    e=(3*B*(74 + 9*B*B2P - 45*BP + 9*BP**2)*V**(7/3))/8
    return TaylorCoefficients(a, b, c, d, e)

def LOG_parameters_to_taylor(V, E0, B, BP, B2P):
    """
    Convert logarithmic equation of state properties to Taylor coefficients
    a, b, c, d, e (4th order/five parameters). Returns a named tuple with the
    coefficients.
    """
    B = B/EV_PER_CUBIC_ANGSTROM_TO_GPA
    B2P = B2P*EV_PER_CUBIC_ANGSTROM_TO_GPA
    a=(24*E0 + 
       12*B*V*np.np.log(V)**2 + 
       4*B*(-2 + BP)*V*np.log(V)**3 + 
       B*(3 + B*B2P - 3*BP + BP**2)*V*np.log(V)**4)/24
    b=-(B*V*np.log(V)*(6 + 3*(-2 + BP)*np.log(V) + 
            (3 + B*B2P - 3*BP + BP**2)*np.log(V)**2))/6
    c=(B*V*(2 + 2*(-2 + BP)*np.log(V) + 
            (3 + B*B2P - 3*BP + BP**2)*np.log(V)**2))/4
    d=-(B*V*(-2 + BP + (3 + B*B2P - 3*BP + BP**2)*np.log(V)))/6
    e=(B*(3 + B*B2P - 3*BP + BP**2)*V)/24
    if abs(e) < 1e-8: # avoid numerical errors
        e = 0
    return TaylorCoefficients(a, b, c, d, e) 

# mBM4 EOS Functions
def mBM4_equation(
    volume: float | np.ndarray, a: float, b: float, c: float, d: float
) -> float | np.ndarray:
    """mBM4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: energy
    """

    energy = (
        a + b * (volume) ** (-1 / 3) + c * (volume) ** (-2 / 3) + d * (volume) ** (-1)
    )
    return energy


def mBM4_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Derivative of mBM4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = (
        b * (-1 / 3) * (volume) ** (-4 / 3)
        + c * (-2 / 3) * (volume) ** (-5 / 3)
        + d * (-1) * (volume) ** (-2)
    )
    return energy_derivative


def mBM4_derivative2(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Second derivative of mBM4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: second derivative of energy
    """

    energy_derivative2 = (
        b * (4 / 9) * (volume) ** (-7 / 3)
        + c * (10 / 9) * (volume) ** (-8 / 3)
        + d * (2) * (volume) ** (-3)
    )
    return energy_derivative2


def mBM4_eos_parameters(
    a: float, b: float, c: float, d: float
) -> tuple[float, float, float, float, float]:
    """Calculate V0, E0, B, BP, and B2P from a, b, c, and d.

    Args:
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        tuple[float, float, float, float, float]: V0, E0, B, BP, B2P
    """

    V0 = (
        -(
            4 * c**3
            - 9 * b * c * d
            + np.sqrt((c**2 - 3 * b * d) * (4 * c**2 - 3 * b * d) ** 2)
        )
        / b**3
    )
    E0 = mBM4_equation(V0, a, b, c, d)
    B = (
        (2 * d) / V0**3 + (10 * c) / (9 * V0 ** (8 / 3)) + (4 * b) / (9 * V0 ** (7 / 3))
    ) * V0
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    BP = (54 * d * V0 ** (1 / 3) + 25 * c * V0 ** (2 / 3) + 8 * b * V0) / (
        27 * d * V0 ** (1 / 3) + 15 * c * V0 ** (2 / 3) + 6 * b * V0
    )
    B2P = (
        V0 ** (8 / 3)
        * (
            9 * d * (5 * c * V0 ** (2 / 3) + 8 * b * V0)
            + 2 * V0 ** (1 / 3) * (5 * c * (b * V0))
        )
    ) / (2 * (9 * d * V0 ** (1 / 3) + 5 * c * V0 ** (2 / 3) + 2 * b * V0) ** 3)
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA

    return V0, E0, B, BP, B2P


def mBM4(
    volume: float | np.ndarray,
    energy: float | np.ndarray,
    volume_min: float = None,
    volume_max: float = None,
    num_volumes: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the mBM4 EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    a, b, c, d = curve_fit(mBM4_equation, volume, energy, p0=[100, 100, 100, 100])[0]
    if volume_min is None:
        volume_min = min(volume)
    if volume_max is None:
        volume_max = max(volume)

    volume_range = np.linspace(volume_min, volume_max, num_volumes)

    energy_eos = mBM4_equation(volume_range, a, b, c, d)
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * mBM4_derivative(volume_range, b, c, d)
    )
    V0, E0, B, BP, B2P = mBM4_eos_parameters(a, b, c, d)
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([a, b, c, d, 0])

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# mBM5 EOS Functions
def mBM5_equation(
    volume: float | np.ndarray, a: float, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """mBM5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: energy
    """

    energy = (
        a
        + b * (volume) ** (-1 / 3)
        + c * (volume) ** (-2 / 3)
        + d * (volume) ** (-1)
        + e * (volume) ** (-4 / 3)
    )
    return energy


def mBM5_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """Derivative of mBM5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = (
        b * (-1 / 3) * (volume) ** (-4 / 3)
        + c * (-2 / 3) * (volume) ** (-5 / 3)
        + d * (-1) * (volume) ** (-2)
        + e * (-4 / 3) * (volume) ** (-7 / 3)
    )
    return energy_derivative


def mBM5_derivative2(
    volume: float | np.ndarray, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """Second derivative of mBM5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: second derivative of energy
    """

    energy_derivative2 = (
        b * (4 / 9) * (volume) ** (-7 / 3)
        + c * (10 / 9) * (volume) ** (-8 / 3)
        + d * (2) * (volume) ** (-3)
        + e * (28 / 9) * (volume) ** (-10 / 3)
    )
    return energy_derivative2


def mBM5_eos_parameters(
    volume_range: np.ndarray, a: float, b: float, c: float, d: float, e: float
) -> tuple[float, float, float, float, float]:
    """Calculate V0, E0, B, BP, and B2P from a, b, c, d, and e.

    Args:
        volume_range (np.ndarray): range of volumes
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        tuple[float, float, float, float, float]: V0, E0, B, BP, B2P
    """

    V0 = fsolve(mBM5_derivative, np.mean(volume_range), args=(b, c, d, e))[0]
    E0 = mBM5_equation(V0, a, b, c, d, e)
    B = (
        (28 * e) / (9 * V0 ** (10 / 3))
        + (2 * d) / V0**3
        + (10 * c) / (9 * V0 ** (8 / 3))
        + (4 * b) / (9 * V0 ** (7 / 3))
    ) * V0
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    BP = (98 * e + 54 * d * V0 ** (1 / 3) + 25 * c * V0 ** (2 / 3) + 8 * b * V0) / (
        42 * e + 27 * d * V0 ** (1 / 3) + 15 * c * V0 ** (2 / 3) + 6 * b * V0
    )
    B2P = (
        V0 ** (8 / 3)
        * (
            9 * d * (14 * e + 5 * c * V0 ** (2 / 3) + 8 * b * V0)
            + 2
            * V0 ** (1 / 3)
            * (126 * b * e * V0 ** (1 / 3) + 5 * c * (28 * e + b * V0))
        )
    ) / (2 * (14 * e + 9 * d * V0 ** (1 / 3) + 5 * c * V0 ** (2 / 3) + 2 * b * V0) ** 3)
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA

    return V0, E0, B, BP, B2P


def mBM5(
    volume: float | np.ndarray,
    energy: float | np.ndarray,
    volume_min: float = None,
    volume_max: float = None,
    num_volumes: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the mBM5 EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    a, b, c, d, e = curve_fit(
        mBM5_equation, volume, energy, p0=[100, 100, 100, 100, 100]
    )[0]

    if volume_min is None:
        volume_min = min(volume)
    if volume_max is None:
        volume_max = max(volume)

    volume_range = np.linspace(volume_min, volume_max, num_volumes)

    energy_eos = mBM5_equation(volume_range, a, b, c, d, e)
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * mBM5_derivative(volume_range, b, c, d, e)
    )
    V0, E0, B, BP, B2P = mBM5_eos_parameters(volume_range, a, b, c, d, e)
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([a, b, c, d, e])

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# BM4 EOS Functions
def BM4_equation(
    volume: float | np.ndarray, a: float, b: float, c: float, d: float
) -> float | np.ndarray:
    """BM4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: energy(s)
    """

    energy = (
        a + b * volume ** (-2 / 3) + c * (volume) ** (-4 / 3) + d * (volume) ** (-2)
    )
    return energy


def BM4_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Derivative of BM4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = (
        b * (-2 / 3) * volume ** (-5 / 3)
        + c * (-4 / 3) * (volume) ** (-7 / 3)
        + d * (-2) * (volume) ** (-3)
    )
    return energy_derivative


def BM4_derivative2(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Second derivative of BM4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: second derivative of energy
    """

    energy_derivative2 = (
        b * (10 / 9) * volume ** (-8 / 3)
        + c * (28 / 9) * (volume) ** (-10 / 3)
        + d * (6) * (volume) ** (-4)
    )
    return energy_derivative2


def BM4_eos_parameters(
    a: float, b: float, c: float, d: float
) -> tuple[float, float, float, float, float]:
    """Calculate V0, E0, B, BP, and B2P from a, b, c, and d.

    Args:
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        tuple[float, float, float, float, float]: V0, E0, B, BP, B2P
    """

    V0 = np.sqrt(
        -(
            (
                4 * c**3
                - 9 * b * c * d
                + np.sqrt((c**2 - 3 * b * d) * (4 * c**2 - 3 * b * d) ** 2)
            )
            / b**3
        )
    )
    E0 = BM4_equation(V0, a, b, c, d)
    B = (2 * (27 * d * V0 ** (2 / 3) + 14 * c * V0 ** (4 / 3) + 5 * b * V0**2)) / (
        9 * V0 ** (11 / 3)
    )
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    BP = (243 * d * V0 ** (2 / 3) + 98 * c * V0 ** (4 / 3) + 25 * b * V0**2) / (
        81 * d * V0 ** (2 / 3) + 42 * c * V0 ** (4 / 3) + 15 * b * V0**2
    )
    B2P = (
        4
        * V0 ** (13 / 3)
        * (
            27 * d * (7 * c * V0 ** (4 / 3) + 10 * b * V0**2)
            + V0 ** (2 / 3) * (7 * c * (5 * b * V0**2))
        )
    ) / (27 * d * V0 ** (2 / 3) + 14 * c * V0 ** (4 / 3) + 5 * b * V0**2) ** 3
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA

    return V0, E0, B, BP, B2P


def BM4(
    volume: float | np.ndarray,
    energy: float | np.ndarray,
    volume_min: float = None,
    volume_max: float = None,
    num_volumes: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the BM4 EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    a, b, c, d = curve_fit(BM4_equation, volume, energy, p0=[100, 100, 100, 100])[0]

    if volume_min is None:
        volume_min = min(volume)
    if volume_max is None:
        volume_max = max(volume)

    volume_range = np.linspace(volume_min, volume_max, num_volumes)

    energy_eos = BM4_equation(volume_range, a, b, c, d)
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * BM4_derivative(volume_range, b, c, d)
    )
    V0, E0, B, BP, B2P = BM4_eos_parameters(a, b, c, d)
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([a, b, c, d, 0])

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# BM5 EOS Functions
def BM5_equation(
    volume: float | np.ndarray, a: float, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """BM5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: energy
    """

    energy = (
        a
        + b * (volume) ** (-2 / 3)
        + c * (volume) ** (-4 / 3)
        + d * (volume) ** (-2)
        + e * (volume) ** (-8 / 3)
    )
    return energy


def BM5_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """Derivative of BM5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = (
        b * (-2 / 3) * (volume) ** (-5 / 3)
        + c * (-4 / 3) * (volume) ** (-7 / 3)
        + d * (-2) * (volume) ** (-3)
        + e * (-8 / 3) * (volume) ** (-11 / 3)
    )
    return energy_derivative


def BM5_derivative2(
    volume: float | np.ndarray, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """Second derivative of BM5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: second derivative of energy
    """

    energy_derivative2 = (
        b * (10 / 9) * volume ** (-8 / 3)
        + c * (28 / 9) * (volume) ** (-10 / 3)
        + d * (6) * (volume) ** (-4)
        + e * (88 / 9) * (volume) ** (-14 / 3)
    )
    return energy_derivative2


def BM5_eos_parameters(
    volume_range: np.ndarray, a: float, b: float, c: float, d: float, e: float
) -> tuple[float, float, float, float, float]:
    """Calculate V0, E0, B, BP, and B2P from a, b, c, d, and e.

    Args:
        volume_range (np.ndarray): range of volumes
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        tuple[float, float, float, float, float]: V0, E0, B, BP, B2P
    """

    V0 = fsolve(BM5_derivative, np.mean(volume_range), args=(b, c, d, e))[0]
    E0 = BM5_equation(V0, a, b, c, d, e)
    B = (
        2 * (44 * e + 27 * d * V0 ** (2 / 3) + 14 * c * V0 ** (4 / 3) + 5 * b * V0**2)
    ) / (9 * V0 ** (11 / 3))
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    BP = (
        484 * e + 243 * d * V0 ** (2 / 3) + 98 * c * V0 ** (4 / 3) + 25 * b * V0**2
    ) / (132 * e + 81 * d * V0 ** (2 / 3) + 42 * c * V0 ** (4 / 3) + 15 * b * V0**2)
    B2P = (
        4
        * V0 ** (13 / 3)
        * (
            27 * d * (22 * e + 7 * c * V0 ** (4 / 3) + 10 * b * V0**2)
            + V0 ** (2 / 3)
            * (990 * b * e * V0 ** (2 / 3) + 7 * c * (176 * e + 5 * b * V0**2))
        )
    ) / (
        44 * e + 27 * d * V0 ** (2 / 3) + 14 * c * V0 ** (4 / 3) + 5 * b * V0**2
    ) ** 3
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA

    return V0, E0, B, BP, B2P


def BM5(
    volume: float | np.ndarray,
    energy: float | np.ndarray,
    volume_min: float = None,
    volume_max: float = None,
    num_volumes: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the BM5 EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    a, b, c, d, e = curve_fit(
        BM5_equation, volume, energy, p0=[100, 100, 100, 100, 100]
    )[0]

    if volume_min is None:
        volume_min = min(volume)
    if volume_max is None:
        volume_max = max(volume)

    volume_range = np.linspace(volume_min, volume_max, num_volumes)

    energy_eos = BM5_equation(volume_range, a, b, c, d, e)
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * BM5_derivative(volume_range, b, c, d, e)
    )
    V0, E0, B, BP, B2P = BM5_eos_parameters(volume_range, a, b, c, d, e)
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([a, b, c, d, e])

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# LOG4 EOS Functions
def LOG4_equation(
    volume: float | np.ndarray, a: float, b: float, c: float, d: float
) -> float | np.ndarray:
    """LOG4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: energy
    """

    energy = a + b * np.log(volume) + c * np.log(volume) ** 2 + d * np.log(volume) ** 3
    return energy


def LOG4_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Derivative of LOG4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = (
        b + 2 * c * np.log(volume) + 3 * d * np.log(volume) ** 2
    ) / volume
    return energy_derivative


def LOG4_derivative2(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Second derivative of LOG4 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        float | np.ndarray: second derivative of energy
    """

    energy_derivative2 = (
        -b / (volume**2)
        - 2 * c * (np.log(volume) - 1) / (volume**2)
        - (3 * d * (np.log(volume) - 2) * np.log(volume)) / (volume**2)
    )
    return energy_derivative2


def LOG4_eos_parameters(
    volume_range: np.ndarray, a: float, b: float, c: float, d: float
) -> tuple[float, float, float, float, float]:
    """Calculate V0, E0, B, BP, and B2P from a, b, c, and d.

    Args:
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter

    Returns:
        tuple[float, float, float, float, float]: V0, E0, B, BP, B2P
    """

    V0 = fsolve(LOG4_derivative, np.mean(volume_range), args=(b, c, d))[0]
    E0 = LOG4_equation(V0, a, b, c, d)
    B = -((b - 2 * c + 2 * (c - 3 * d) * np.log(V0) + 3 * d * np.log(V0) ** 2) / V0)
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    BP = (
        b - 4 * c + 6 * d + 2 * (c - 6 * d) * np.log(V0) + 3 * d * np.log(V0) ** 2
    ) / (b - 2 * c + 2 * (c - 3 * d) * np.log(V0) + 3 * d * np.log(V0) ** 2)
    B2P = (
        2
        * V0
        * (
            2 * c**2
            - 3 * b * d
            + 18 * d**2
            - 6 * c * d
            + 6 * (c * d - 3 * d**2) * np.log(V0)
            + 9 * d**2 * np.log(V0) ** 2
        )
    ) / (b - 2 * c + 2 * (c - 3 * d) * np.log(V0) + 3 * d * np.log(V0) ** 2) ** 3
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA
    return V0, E0, B, BP, B2P


def LOG4(
    volume: float | np.ndarray,
    energy: float | np.ndarray,
    volume_min: float = None,
    volume_max: float = None,
    num_volumes: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the LOG4 EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    a, b, c, d = curve_fit(LOG4_equation, volume, energy, p0=[100, 100, 100, 100])[0]

    if volume_min is None:
        volume_min = min(volume)
    if volume_max is None:
        volume_max = max(volume)

    volume_range = np.linspace(volume_min, volume_max, num_volumes)

    energy_eos = LOG4_equation(volume_range, a, b, c, d)
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * LOG4_derivative(volume_range, b, c, d)
    )
    V0, E0, B, BP, B2P = LOG4_eos_parameters(volume_range, a, b, c, d)
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([a, b, c, d, 0])

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# LOG5 EOS Functions
def LOG5_equation(
    volume: float | np.ndarray, a: float, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """LOG5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: energy
    """

    energy = (
        a
        + b * np.log(volume)
        + c * np.log(volume) ** 2
        + d * np.log(volume) ** 3
        + e * np.log(volume) ** 4
    )
    return energy


def LOG5_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """Derivative of LOG5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy = (
        b
        + 2 * c * np.log(volume)
        + 3 * d * np.log(volume) ** 2
        + 4 * e * np.log(volume) ** 3
    ) / volume
    return energy


def LOG5_derivative2(
    volume: float | np.ndarray, b: float, c: float, d: float, e: float
) -> float | np.ndarray:
    """Second derivative of LOG5 EOS.

    Args:
        volume (float | np.ndarray): input volume
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        float | np.ndarray: second derivative of energy
    """

    energy_derivative2 = (
        -b / (volume**2)
        - 2 * c * (np.log(volume) - 1) / (volume**2)
        - (3 * d * (np.log(volume) - 2) * np.log(volume)) / (volume**2)
        - (4 * e * (np.log(volume) - 3) * np.log(volume) ** 2) / (volume**2)
    )
    return energy_derivative2


def LOG5_eos_parameters(
    volume_range: np.ndarray, a: float, b: float, c: float, d: float, e: float
) -> tuple[float, float, float, float, float]:
    """Calculate V0, E0, B, BP, and B2P from a, b, c, d, and e.

    Args:
        volume_range (np.ndarray): range of volumes
        a (float): a-parameter
        b (float): b-parameter
        c (float): c-parameter
        d (float): d-parameter
        e (float): e-parameter

    Returns:
        tuple[float, float, float, float, float]: V0, E0, B, BP, B2P
    """

    V0 = fsolve(LOG5_derivative, np.mean(volume_range), args=(b, c, d, e))[0]
    E0 = LOG5_equation(V0, a, b, c, d, e)

    B = -(
        (
            b
            - 2 * c
            + 2 * (c - 3 * d) * np.log(V0)
            + 3 * (d - 4 * e) * np.log(V0) ** 2
            + 4 * e * np.log(V0) ** 3
        )
        / V0
    )
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    BP = (
        b
        - 4 * c
        + 6 * d
        + 2 * (c - 6 * d + 12 * e) * np.log(V0)
        + 3 * (d - 8 * e) * np.log(V0) ** 2
        + 4 * e * np.log(V0) ** 3
    ) / (
        b
        - 2 * c
        + 2 * (c - 3 * d) * np.log(V0)
        + 3 * (d - 4 * e) * np.log(V0) ** 2
        + 4 * e * np.log(V0) ** 3
    )
    B2P = (
        2
        * V0
        * (
            2 * c**2
            - 3 * b * d
            + 18 * d**2
            + 12 * b * e
            - 6 * c * (d + 4 * e)
            + 6 * (c * d - 3 * d**2 - 2 * b * e + 12 * d * e) * np.log(V0)
            + 9 * (d - 4 * e) ** 2 * np.log(V0) ** 2
            + 24 * (d - 4 * e) * e * np.log(V0) ** 3
            + 24 * e**2 * np.log(V0) ** 4
        )
    ) / (
        b
        - 2 * c
        + 2 * (c - 3 * d) * np.log(V0)
        + 3 * (d - 4 * e) * np.log(V0) ** 2
        + 4 * e * np.log(V0) ** 3
    ) ** 3
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA
    return V0, E0, B, BP, B2P


def LOG5(
    volume: float | np.ndarray,
    energy: float | np.ndarray,
    volume_min: float = None,
    volume_max: float = None,
    num_volumes: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the LOG5 EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    a, b, c, d, e = curve_fit(LOG5_equation, volume, energy, p0=[1, 1, 1, 1, 1])[0]

    if volume_min is None:
        volume_min = min(volume)
    if volume_max is None:
        volume_max = max(volume)

    volume_range = np.linspace(volume_min, volume_max, num_volumes)

    energy_eos = LOG5_equation(volume_range, a, b, c, d, e)
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * LOG5_derivative(volume_range, b, c, d, e)
    )
    V0, E0, B, BP, B2P = LOG5_eos_parameters(volume_range, a, b, c, d, e)
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([a, b, c, d, e])

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# Murnaghan EOS Functions
def murnaghan_equation(
    volume: float | np.ndarray, V0: float, E0: float, B: float, BP: float
) -> float | np.ndarray:
    """Murnaghan EOS

    Args:
        volume (float | np.ndarray): input volume
        V0 (float): equilibrium volume
        E0 (float): equilibrium energy
        B (float): bulk modulus
        BP (float): derivative of bulk modulus with respect to pressure

    Returns:
        float | np.ndarray: energy
    """

    energy = (
        E0
        - (B * V0) / (BP - 1)
        + (B * volume / BP) * (1 + (V0 / volume) ** BP / (BP - 1))
    )
    return energy


def murnaghan_derivative(
    volume: float | np.ndarray, V0: float, B: float, BP: float
) -> float | np.ndarray:
    """Derivative of Murnaghan EOS

    Args:
        volume (float | np.ndarray): input volume
        V0 (float): equilibrium volume
        B (float): bulk modulus
        BP (float): derivative of bulk modulus with respect to pressure

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = (B / BP) * (1 - (V0 / volume) ** BP)
    return energy_derivative


def murnaghan(
    volume: float | np.ndarray,
    energy: float | np.ndarray,
    volume_min: float = None,
    volume_max: float = None,
    num_volumes: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the Murnaghan EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    if volume_min is None:
        volume_min = min(volume)
    if volume_max is None:
        volume_max = max(volume)

    volume_range = np.linspace(volume_min, volume_max, num_volumes)

    [eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos] = mBM4(
        volume, energy
    )
    initial_guess = [
        eos_parameters[0],
        eos_parameters[1],
        eos_parameters[2] / EV_PER_CUBIC_ANGSTROM_TO_GPA,
        eos_parameters[3],
    ]

    V0, E0, B, BP = curve_fit(murnaghan_equation, volume, energy, p0=initial_guess)[0]

    energy_eos = murnaghan_equation(volume_range, V0, E0, B, BP)
    eos_parameters = np.array([V0, E0, B * EV_PER_CUBIC_ANGSTROM_TO_GPA, BP, 0])
    eos_constants = np.array([0, 0, 0, 0, 0])
    pressure_eos = (
        -1
        * EV_PER_CUBIC_ANGSTROM_TO_GPA
        * murnaghan_derivative(volume_range, V0, B, BP)
    )

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# Vinet EOS Functions
def vinet_equation(
    volume: float | np.ndarray, V0: float, E0: float, B: float, BP: float
) -> float | np.ndarray:
    """Vinet EOS

    Args:
        volume (float | np.ndarray): input volume
        V0 (float): equilibrium volume
        E0 (float): equilibrium energy
        B (float): bulk modulus
        BP (float): derivative of bulk modulus with respect to pressure

    Returns:
        float | np.ndarray: energy
    """

    energy = (
        E0
        + (4 * B * V0) / (BP - 1) ** 2
        - (4 * B * V0)
        / (BP - 1) ** 2
        * (1 - 3 / 2 * (BP - 1) * (1 - (volume / V0) ** (1 / 3)))
        * (np.exp(3 / 2 * (BP - 1) * (1 - (volume / V0) ** (1 / 3))))
    )
    return energy


def vinet_derivative(volume: float | np.ndarray, V0, B, BP) -> float | np.ndarray:
    """Derivative of Vinet EOS

    Args:
        volume (float | np.ndarray): input volume
        V0 (float): equilibrium volume
        B (float): bulk modulus
        BP (float): derivative of bulk modulus with respect to pressure

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = -(
        3
        * B
        * (1 - (volume / V0) ** (1 / 3))
        / ((volume / V0) ** (2 / 3))
        * np.exp(3 / 2 * (BP - 1) * (1 - (volume / V0) ** (1 / 3)))
    )
    return energy_derivative


def vinet(
    volume: float | np.ndarray,
    energy: float | np.ndarray,
    volume_min: float = None,
    volume_max: float = None,
    num_volumes: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the Vinet EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    if volume_min is None:
        volume_min = min(volume)
    if volume_max is None:
        volume_max = max(volume)

    volume_range = np.linspace(volume_min, volume_max, num_volumes)

    [eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos] = mBM4(
        volume, energy
    )
    initial_guess = [
        eos_parameters[0],
        eos_parameters[1],
        eos_parameters[2] / EV_PER_CUBIC_ANGSTROM_TO_GPA,
        eos_parameters[3],
    ]

    V0, E0, B, BP = curve_fit(vinet_equation, volume, energy, p0=initial_guess)[0]

    energy_eos = vinet_equation(volume_range, V0, E0, B, BP)

    B2P = (19 - 18 * BP - 9 * BP**2) / (36 * B)
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    eos_parameters = np.array([V0, E0, B, BP, B2P])
    eos_constants = np.array([0, 0, 0, 0, 0])
    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * vinet_derivative(volume_range, V0, B, BP)
    )
    
    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


# Morse EOS Functions
def morse_equation(
    volume: float | np.ndarray, V0: float, E0: float, B: float, BP: float
) -> float | np.ndarray:
    """Morse EOS

    Args:
        volume (float | np.ndarray): input volume
        V0 (float): equilibrium volume
        E0 (float): equilibrium energy
        B (float): bulk modulus
        BP (float): derivative of bulk modulus with respect to pressure

    Returns:
        float | np.ndarray: energy
    """

    a = E0 + (9 * B * V0) / (2 * (BP - 1) ** 2)
    b = (-9 * B * V0 * np.exp(BP - 1)) / (BP - 1) ** 2
    c = (9 * B * V0 * np.exp(2 * BP - 2)) / (2 * (BP - 1) ** 2)
    d = (1 - BP) / V0 ** (1 / 3)
    energy = (
        a + b * np.exp(d * volume ** (1 / 3)) + c * np.exp(2 * d * volume ** (1 / 3))
    )
    return energy


def morse_derivative(
    volume: float | np.ndarray, b: float, c: float, d: float
) -> float | np.ndarray:
    """Derivative of Morse EOS

    Args:
        volume (float | np.ndarray): input volume
        V0 (float): equilibrium volume
        B (float): bulk modulus
        BP (float): derivative of bulk modulus with respect to pressure

    Returns:
        float | np.ndarray: derivative of energy
    """

    energy_derivative = b * d * np.exp(d * volume ** (1 / 3)) / (
        3 * volume ** (2 / 3)
    ) + 2 * c * d * np.exp(2 * d * volume ** (1 / 3)) / (3 * volume ** (2 / 3))
    return energy_derivative


def morse(
    volume: float | np.ndarray,
    energy: float | np.ndarray,
    volume_min: float = None,
    volume_max: float = None,
    num_volumes: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the Morse EOS to the input volume and energy data.

    Args:
        volume (float | np.ndarray): volume data
        energy (float | np.ndarray): energy data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: EOS parameters and the corresponding volume, energy, and pressure
    """

    if volume_min is None:
        volume_min = min(volume)
    if volume_max is None:
        volume_max = max(volume)

    volume_range = np.linspace(volume_min, volume_max, num_volumes)

    [eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos] = mBM4(
        volume, energy
    )
    initial_guess = [
        eos_parameters[0],
        eos_parameters[1],
        eos_parameters[2] / EV_PER_CUBIC_ANGSTROM_TO_GPA,
        eos_parameters[3],
    ]

    V0, E0, B, BP = curve_fit(morse_equation, volume, energy, p0=initial_guess)[0]

    energy_eos = morse_equation(volume_range, V0, E0, B, BP)

    B2P = (5 - 5 * BP - 2 * BP**2) / (9 * B)
    B2P = B2P / EV_PER_CUBIC_ANGSTROM_TO_GPA
    B = B * EV_PER_CUBIC_ANGSTROM_TO_GPA
    eos_parameters = np.array([V0, E0, B, BP, B2P])

    a = E0 + (9 * B * V0) / (2 * (BP - 1) ** 2)
    b = (-9 * B * V0 * np.exp(BP - 1)) / (BP - 1) ** 2
    c = (9 * B * V0 * np.exp(2 * BP - 2)) / (2 * (BP - 1) ** 2)
    d = (1 - BP) / V0 ** (1 / 3)
    eos_constants = np.array([a, b, c, d, 0])

    pressure_eos = (
        -1 * EV_PER_CUBIC_ANGSTROM_TO_GPA * morse_derivative(volume_range, b, c, d)
    )
    
    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos
