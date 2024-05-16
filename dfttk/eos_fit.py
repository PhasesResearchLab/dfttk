"""
This EOS fitting code is based on the following paper:
Shun-Li Shang et al., Computational Materials Science, 47, 4, (2010).
https://doi.org/10.1016/j.commatsci.2009.12.006

It includes the following equations of state:
1:  4-parameter (Teter-Shang) mBM4   1
2:  5-parameter (Teter-Shang) mBM5   2
3:  4-parameter               BM4    3
4:  5-parameter               BM5    4
5:  4-parameter Natural       Log4   5
6:  5-parameter Natural       Log5   6
7:  4-parameter Murnaghan     Mur    7
8:  4-parameter Vinet         Vinet  8
9:  4-parameter Morse         Morse  9
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from distinctipy import get_colors
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.optimize import leastsq


"""The EOS functions mBM4, mBM5, BM4, BM5, Log4, Log5, Mur, Vinet, and Morse are used to fit the energy-volume data 
and return the EOS parameters. 

    Args:
        volume (numpy.array): volumes to be fitted
        energy (numpy.array): energies to be fitted

    Returns:
        results (numpy.array): EOS parameters
        volume_range (numpy.array): The range of volumes used for the fitting
        energy_eos (numpy.array): The fitted energies of the equation of state.
        pressure_eos (numpy.array): The resulting pressures from the fitted equation of state.
    """


def mBM4_equation(volume, a, b, c, d):
    energy = (
        a + b * (volume) ** (-1 / 3) + c * (volume) ** (-2 / 3) + d * (volume) ** (-1)
    )
    return energy


def mBM4_derivative(volume, b, c, d):
    energy = (
        b * (-1 / 3) * (volume) ** (-4 / 3)
        + c * (-2 / 3) * (volume) ** (-5 / 3)
        + d * (-1) * (volume) ** (-2)
    )
    return energy


def mBM4_eos_parameters(a, b, c, d):
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
        (2 * d) / V0 ** 3 + (10 * c) / (9 * V0 ** (8 / 3)) + (4 * b) / (9 * V0 ** (7 / 3))
    ) * V0
    B = B * 160.2176621
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
    B2P = B2P / 160.2176621

    return V0, E0, B, BP, B2P


def mBM4(volume, energy):
    a, b, c, d = curve_fit(mBM4_equation, volume, energy, p0 = [100, 100, 100, 100])[0]
    volume_range = np.linspace(min(volume), max(volume), 1000)

    energy_eos = mBM4_equation(volume_range, a, b, c, d)
    pressure_eos = -1 * 160.2176621 * mBM4_derivative(volume_range, b, c, d)
    V0, E0, B, BP, B2P = mBM4_eos_parameters(a, b, c, d)
    eos_parameters = np.array([V0, E0, B, BP, B2P])

    return eos_parameters, volume_range, energy_eos, pressure_eos


def mBM5_equation(volume, a, b, c, d, e):
    energy = (
        a
        + b * (volume) ** (-1 / 3)
        + c * (volume) ** (-2 / 3)
        + d * (volume) ** (-1)
        + e * (volume) ** (-4 / 3)
    )
    return energy


def mBM5_derivative(volume, b, c, d, e):
    energy = (
        b * (-1 / 3) * (volume) ** (-4 / 3)
        + c * (-2 / 3) * (volume) ** (-5 / 3)
        + d * (-1) * (volume) ** (-2)
        + e * (-4 / 3) * (volume) ** (-7 / 3)
    )
    return energy


def mBM5_eos_parameters(volume_range, a, b, c, d, e):
    function = (
        lambda volume_range: (
            (4 * e) / (3 * volume_range ** (7 / 3))
            + d / volume_range**2
            + (2 * c) / (3 * volume_range ** (5 / 3))
            + b / (3 * volume_range ** (4 / 3))
        )
        * 160.2176621
    )
    V0 = fsolve(function, np.mean(volume_range))[0]
    E0 = mBM5_equation(V0, a, b, c, d, e)
    B = (
        (28 * e) / (9 * V0 ** (10 / 3))
        + (2 * d) / V0**3
        + (10 * c) / (9 * V0 ** (8 / 3))
        + (4 * b) / (9 * V0 ** (7 / 3))
    ) * V0
    B = B * 160.2176621
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
    B2P = B2P / 160.2176621

    return V0, E0, B, BP, B2P


def mBM5(volume, energy):
    a, b, c, d, e = curve_fit(mBM5_equation, volume, energy, p0 = [100, 100, 100, 100, 100])[0]
    volume_range = np.linspace(min(volume), max(volume), 1000)

    energy_eos = mBM5_equation(volume_range, a, b, c, d, e)
    pressure_eos = -1 * 160.2176621 * mBM5_derivative(volume_range, b, c, d, e)
    V0, E0, B, BP, B2P = mBM5_eos_parameters(volume_range, a, b, c, d, e)
    eos_parameters = np.array([V0, E0, B, BP, B2P])

    return eos_parameters, volume_range, energy_eos, pressure_eos


def BM4_equation(volume, a, b, c, d):
    energy = (
        a + b * volume ** (-2 / 3) + c * (volume) ** (-4 / 3) + d * (volume) ** (-2)
    )
    return energy


def BM4_derivative(volume, b, c, d):
    energy = (
        b * (-2 / 3) * volume ** (-5 / 3)
        + c * (-4 / 3) * (volume) ** (-7 / 3)
        + d * (-2) * (volume) ** (-3)
    )
    return energy


def BM4_eos_parameters(a, b, c, d):
    V0 = math.sqrt(
        -(
            (
                4 * c ** 3
                - 9 * b * c * d
                + math.sqrt((c ** 2 - 3 * b * d) * (4 * c ** 2 - 3 * b * d) ** 2)
            )
            / b ** 3
        )
    )
    E0 = BM4_equation(V0, a, b, c, d)
    B = (2 * (27 * d * V0 ** (2 / 3) + 14 * c * V0 ** (4 / 3) + 5 * b * V0 ** 2)) / (
        9 * V0 ** (11 / 3)
    )
    B = B * 160.2176621
    BP = (243 * d * V0 ** (2 / 3) + 98 * c * V0 ** (4 / 3) + 25 * b * V0 ** 2) / (
        81 * d * V0 ** (2 / 3) + 42 * c * V0 ** (4 / 3) + 15 * b * V0 ** 2
    )
    B2P = (
        4
        * V0 ** (13 / 3)
        * (
            27 * d * (7 * c * V0 ** (4 / 3) + 10 * b * V0 ** 2)
            + V0 ** (2 / 3) * (7 * c * (5 * b * V0 ** 2))
        )
    ) / (27 * d * V0 ** (2 / 3) + 14 * c * V0 ** (4 / 3) + 5 * b * V0 ** 2) ** 3
    B2P = B2P / 160.2176621

    return V0, E0, B, BP, B2P


def BM4(volume, energy):
    a, b, c, d = curve_fit(BM4_equation, volume, energy, p0 = [100, 100, 100, 100])[0]
    volume_range = np.linspace(min(volume), max(volume), 1000)

    energy_eos = BM4_equation(volume_range, a, b, c, d)
    pressure_eos = -1 * 160.2176621 * BM4_derivative(volume_range, b, c, d)
    V0, E0, B, BP, B2P = BM4_eos_parameters(a, b, c, d)
    eos_parameters = np.array([V0, E0, B, BP, B2P])

    return eos_parameters, volume_range, energy_eos, pressure_eos


def BM5_equation(volume, a, b, c, d, e):
    energy = (
        a
        + b * (volume) ** (-2 / 3)
        + c * (volume) ** (-4 / 3)
        + d * (volume) ** (-2)
        + e * (volume) ** (-8 / 3)
    )
    return energy


def BM5_derivative(volume, b, c, d, e):
    energy = (
        b * (-2 / 3) * (volume) ** (-5 / 3)
        + c * (-4 / 3) * (volume) ** (-7 / 3)
        + d * (-2) * (volume) ** (-3)
        + e * (-8 / 3) * (volume) ** (-11 / 3)
    )
    return energy


def BM5_eos_parameters(volume_range, a, b, c, d, e):
    function = (
        lambda volume_range: (
            (8 * e) / (3 * volume_range ** (11 / 3))
            + (2 * d) / volume_range**3
            + (4 * c) / (3 * volume_range ** (7 / 3))
            + (2 * b) / (3 * volume_range ** (5 / 3))
        )
        * 160.2176621
    )

    V0 = fsolve(function, np.mean(volume_range))[0]
    E0 = BM5_equation(V0, a, b, c, d, e)
    B = (
        2 * (44 * e + 27 * d * V0 ** (2 / 3) + 14 * c * V0 ** (4 / 3) + 5 * b * V0 ** 2)
    ) / (9 * V0 ** (11 / 3))
    B = B * 160.2176621
    BP = (
        484 * e + 243 * d * V0 ** (2 / 3) + 98 * c * V0 ** (4 / 3) + 25 * b * V0 ** 2
    ) / (132 * e + 81 * d * V0 ** (2 / 3) + 42 * c * V0 ** (4 / 3) + 15 * b * V0 ** 2)
    B2P = (
        4
        * V0 ** (13 / 3)
        * (
            27 * d * (22 * e + 7 * c * V0 ** (4 / 3) + 10 * b * V0 ** 2)
            + V0 ** (2 / 3)
            * (990 * b * e * V0 ** (2 / 3) + 7 * c * (176 * e + 5 * b * V0 ** 2))
        )
    ) / (
        44 * e + 27 * d * V0 ** (2 / 3) + 14 * c * V0 ** (4 / 3) + 5 * b * V0 ** 2
    ) ** 3
    B2P = B2P / 160.2176621

    return V0, E0, B, BP, B2P


def BM5(volume, energy):
    a, b, c, d, e = curve_fit(BM5_equation, volume, energy, p0 = [100, 100, 100, 100, 100])[0]
    volume_range = np.linspace(min(volume), max(volume), 1000)

    energy_eos = BM5_equation(volume_range, a, b, c, d, e)
    pressure_eos = -1 * 160.2176621 * BM5_derivative(volume_range, b, c, d, e)
    V0, E0, B, BP, B2P = BM5_eos_parameters(volume_range, a, b, c, d, e)
    eos_parameters = np.array([V0, E0, B, BP, B2P])

    return eos_parameters, volume_range, energy_eos, pressure_eos


def LOG4_equation(volume, a, b, c, d):
    energy = a + b * np.log(volume) + c * np.log(volume) ** 2 + d * np.log(volume) ** 3
    return energy


def LOG4_derivative(volume, b, c, d):
    energy = (b + 2 * c * np.log(volume) + 3 * d * np.log(volume) ** 2) / volume
    return energy


def LOG4_eos_parameters(volume_range, a, b, c, d):
    function = (
        lambda volume_range: (
            -(
                (
                    b
                    + 2 * c * math.log(volume_range)
                    + 3 * d * math.log(volume_range) ** 2
                )
                / volume_range
            )
        )
        * 160.2176621
    )

    V0 = fsolve(function, np.mean(volume_range))[0]
    E0 = LOG4_equation(V0, a, b, c, d)
    B = -(
        (b - 2 * c + 2 * (c - 3 * d) * math.log(V0) + 3 * d * math.log(V0) ** 2) / V0
    )
    B = B * 160.2176621
    BP = (
        b - 4 * c + 6 * d + 2 * (c - 6 * d) * math.log(V0) + 3 * d * math.log(V0) ** 2
    ) / (b - 2 * c + 2 * (c - 3 * d) * math.log(V0) + 3 * d * math.log(V0) ** 2)
    B2P = (
        2
        * V0
        * (
            2 * c ** 2
            - 3 * b * d
            + 18 * d ** 2
            - 6 * c * d
            + 6 * (c * d - 3 * d ** 2) * math.log(V0)
            + 9 * d ** 2 * math.log(V0) ** 2
        )
    ) / (b - 2 * c + 2 * (c - 3 * d) * math.log(V0) + 3 * d * math.log(V0) ** 2) ** 3
    B2P = B2P / 160.2176621
    return V0, E0, B, BP, B2P


def LOG4(volume, energy):
    a, b, c, d = curve_fit(LOG4_equation, volume, energy, p0 = [100, 100, 100, 100])[0]
    volume_range = np.linspace(min(volume), max(volume), 1000)

    energy_eos = LOG4_equation(volume_range, a, b, c, d)
    pressure_eos = -1 * 160.2176621 * LOG4_derivative(volume_range, b, c, d)
    V0, E0, B, BP, B2P = LOG4_eos_parameters(volume_range, a, b, c, d)
    eos_parameters = np.array([V0, E0, B, BP, B2P])

    return eos_parameters, volume_range, energy_eos, pressure_eos


def LOG5_equation(volume, a, b, c, d, e):
    energy = (
        a
        + b * np.log(volume)
        + c * np.log(volume) ** 2
        + d * np.log(volume) ** 3
        + e * np.log(volume) ** 4
    )
    return energy


def LOG5_derivative(volume, b, c, d, e):
    energy = (
        b
        + 2 * c * np.log(volume)
        + 3 * d * np.log(volume) ** 2
        + 4 * e * np.log(volume) ** 3
    ) / volume
    return energy


def LOG5_eos_parameters(volume_range, a, b, c, d, e):
    function = (
        lambda volume_range: (
            -(
                (
                    b
                    + 2 * c * math.log(volume_range)
                    + 3 * d * math.log(volume_range) ** 2
                    + 4 * e * math.log(volume_range) ** 3
                )
                / volume_range
            )
        )
        * 160.2176621
    )

    V0 = fsolve(function, np.mean(volume_range))[0]
    E0 = LOG5_equation(V0, a, b, c, d, e)

    B = -(
        (
            b
            - 2 * c
            + 2 * (c - 3 * d) * math.log(V0)
            + 3 * (d - 4 * e) * math.log(V0) ** 2
            + 4 * e * math.log(V0) ** 3
        )
        / V0
    )
    B = B * 160.2176621
    BP = (
        b
        - 4 * c
        + 6 * d
        + 2 * (c - 6 * d + 12 * e) * math.log(V0)
        + 3 * (d - 8 * e) * math.log(V0) ** 2
        + 4 * e * math.log(V0) ** 3
    ) / (
        b
        - 2 * c
        + 2 * (c - 3 * d) * math.log(V0)
        + 3 * (d - 4 * e) * math.log(V0) ** 2
        + 4 * e * math.log(V0) ** 3
    )
    B2P = (
        2
        * V0
        * (
            2 * c ** 2
            - 3 * b * d
            + 18 * d ** 2
            + 12 * b * e
            - 6 * c * (d + 4 * e)
            + 6 * (c * d - 3 * d ** 2 - 2 * b * e + 12 * d * e) * math.log(V0)
            + 9 * (d - 4 * e) ** 2 * math.log(V0) ** 2
            + 24 * (d - 4 * e) * e * math.log(V0) ** 3
            + 24 * e ** 2 * math.log(V0) ** 4
        )
    ) / (
        b
        - 2 * c
        + 2 * (c - 3 * d) * math.log(V0)
        + 3 * (d - 4 * e) * math.log(V0) ** 2
        + 4 * e * math.log(V0) ** 3
    ) ** 3
    B2P = B2P / 160.2176621
    return V0, E0, B, BP, B2P


def LOG5(volume, energy):
    a, b, c, d, e = curve_fit(LOG5_equation, volume, energy, , p0 = [100, 100, 100, 100, 100])[0]
    volume_range = np.linspace(min(volume), max(volume), 1000)

    energy_eos = LOG5_equation(volume_range, a, b, c, d, e)
    pressure_eos = -1 * 160.2176621 * LOG5_derivative(volume_range, b, c, d, e)
    V0, E0, B, BP, B2P = LOG5_eos_parameters(volume_range, a, b, c, d, e)
    eos_parameters = np.array([V0, E0, B, BP, B2P])

    return eos_parameters, volume_range, energy_eos, pressure_eos


def murnaghan_eq(xini, Data):
    V = xini[0]
    E0 = xini[1]
    B = xini[2]
    bp = xini[3]
    volume_range = Data[:, 0]
    y = Data[:, 1]
    eng = (
        E0
        - (B * V) / (-1 + bp)
        + (B * (1 + (V / volume_range) ** bp / (-1 + bp)) * volume_range) / bp
    )
    return eng - y


def murnaghan(volume, energy):
    volume_range = np.linspace(min(volume), max(volume), 1000)

    Data = np.vstack((volume, energy))
    Data = Data.T

    [eos_parameters, volume_range, energy_eos, pressure_eos] = mBM4(volume, energy)
    xini = [
        eos_parameters[0],
        eos_parameters[1],
        eos_parameters[2] / 160.2176621,
        eos_parameters[3],
    ]
    [xout, resnorm] = leastsq(murnaghan_eq, xini, Data)

    V = xout[0]
    E0 = xout[1]
    B = xout[2]
    bp = xout[3]

    energy_eos = (
        E0
        - (B * V) / (-1 + bp)
        + (B * (1 + (V / volume_range) ** bp / (-1 + bp)) * volume_range) / bp
    )

    eos_parameters = np.array([V, E0, B * 160.2176621, bp, 0])

    pressure_eos = 160.2176621 * (B * (-1 + (V / volume_range) ** bp)) / bp

    return eos_parameters, volume_range, energy_eos, pressure_eos


def vinet_eq(xini, Data):
    V = xini[0]
    E0 = xini[1]
    B = xini[2]
    bp = xini[3]
    volume_range = Data[:, 0]
    y = Data[:, 1]
    eng = (
        E0
        + (4 * B * V) / (-1 + bp) ** 2
        - (4 * B * V * (1 + (3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2))
        / (
            (-1 + bp) ** 2
            * np.exp((3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2)
        )
    )
    return eng - y


def vinet(volume, energy):
    volume_range = np.linspace(min(volume), max(volume), 1000)

    Data = np.vstack((volume, energy))
    Data = Data.T

    [eos_parameters, volume_range, energy_eos, pressure_eos] = mBM4(volume, energy)
    xini = [
        eos_parameters[0],
        eos_parameters[1],
        eos_parameters[2] / 160.2176621,
        eos_parameters[3],
    ]
    [xout, resnorm] = leastsq(vinet_eq, xini, Data)
    V = xout[0]
    E0 = xout[1]
    B = xout[2]
    bp = xout[3]

    energy_eos = (
        E0
        + (4 * B * V) / (-1 + bp) ** 2
        - (4 * B * V * (1 + (3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2))
        / (
            (-1 + bp) ** 2
            * np.exp((3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2)
        )
    )

    b2p = (19 - 18 * bp - 9 * bp**2) / (36 * B)
    eos_parameters = np.array([V, E0, B * 160.2176621, bp, b2p / 160.2176621])

    pressure_eos = (
        160.2176621
        * (-3 * B * (-1 + (volume_range / V) ** (1 / 3)))
        / (
            np.exp((3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2)
            * (volume_range / V) ** (2 / 3)
        )
    )

    return eos_parameters, volume_range, energy_eos, pressure_eos


def morse_eq(xini, Data):
    V = xini[0]
    E0 = xini[1]
    B = xini[2]
    bp = xini[3]

    a = E0 + (9 * B * V) / (2 * (-1 + bp) ** 2)
    b = (-9 * B * np.exp(-1 + bp) * V) / (-1 + bp) ** 2
    c = (9 * B * np.exp(-2 + 2 * bp) * V) / (2 * (-1 + bp) ** 2)
    d = (1 - bp) / V ** (1 / 3)
    volume_range = Data[:, 0]
    y = Data[:, 1]
    eng = (
        a
        + b * np.exp(d * volume_range ** (1 / 3))
        + c * np.exp(2 * d * volume_range ** (1 / 3))
    )
    return eng - y


def morse(volume, energy):
    volume_range = np.linspace(min(volume), max(volume), 1000)

    Data = np.vstack((volume, energy))
    Data = Data.T

    [eos_parameters, volume_range, energy_eos, pressure_eos] = mBM4(volume, energy)
    xini = [
        eos_parameters[0],
        eos_parameters[1],
        eos_parameters[2] / 160.2176621,
        eos_parameters[3],
    ]
    [xout, resnorm] = leastsq(morse_eq, xini, Data)
    V = xout[0]
    E0 = xout[1]
    B = xout[2]
    bp = xout[3]

    a = E0 + (9 * B * V) / (2 * (-1 + bp) ** 2)
    b = (-9 * B * np.exp(-1 + bp) * V) / (-1 + bp) ** 2
    c = (9 * B * np.exp(-2 + 2 * bp) * V) / (2 * (-1 + bp) ** 2)
    d = (1 - bp) / V ** (1 / 3)

    energy_eos = (
        a
        + b * np.exp(d * volume_range ** (1 / 3))
        + c * np.exp(2 * d * volume_range ** (1 / 3))
    )

    b2p = (5 - 5 * bp - 2 * bp**2) / (9 * B)
    eos_parameters = np.array([V, E0, B * 160.2176621, bp, b2p / 160.2176621])

    pressure_eos = (
        -160.2176621
        * (
            d
            * np.exp(d * volume_range ** (1 / 3))
            * (b + 2 * c * np.exp(d * volume_range ** (1 / 3)))
        )
        / (3 * volume_range ** (2 / 3))
    )

    return eos_parameters, volume_range, energy_eos, pressure_eos


def fit_to_all_eos(df):
    """Fits the volume and energies of configurations to all EOS functions and returns the results in a dataframe.

    Args:
        df (pandas.DataFrame): Dataframe frome workflows.extract_configuration_data that contains the volumes,
        energies, and number of atoms of each configuration.

    Returns:
        eos_df (pandas.DataFrame): contains all columns.
        eos_parameters_df (pandas.DataFrame): only contains the EOS parameters.
    """

    eos_df = pd.DataFrame(
        columns=[
            "config",
            "EOS",
            "V0",
            "E0",
            "B",
            "BP",
            "volumes",
            "energies",
            "pressures",
            "number_of_atoms",
        ]
    )
    eos_functions = [mBM4, mBM5, BM4, BM5, LOG4, LOG5, murnaghan, vinet, morse]

    for config in df["config"].unique():
        config_df = df[df["config"] == config]
        volumes = config_df["volume"].values
        energies = config_df["energy"].values
        number_of_atoms = config_df["number_of_atoms"].values

        for eos_function in eos_functions:
            eos_parameters, volume_range, energy_eos, pressure_eos = eos_function(
                volumes, energies
            )
            eos_name = eos_function.__name__

            eos_df = pd.concat(
                [
                    eos_df,
                    pd.DataFrame(
                        [
                            [
                                config,
                                eos_name,
                                eos_parameters[0],
                                eos_parameters[1],
                                eos_parameters[2],
                                eos_parameters[3],
                                volume_range,
                                energy_eos,
                                pressure_eos,
                                number_of_atoms,
                            ]
                        ],
                        columns=[
                            "config",
                            "EOS",
                            "V0",
                            "E0",
                            "B",
                            "BP",
                            "volumes",
                            "energies",
                            "pressures",
                            "number_of_atoms",
                        ],
                    ),
                ],
                ignore_index=True,
            )

    eos_parameters_df = eos_df.drop(
        columns=["volumes", "energies", "pressures", "number_of_atoms"]
    )

    return eos_df, eos_parameters_df


# TODO: review
def convert_input_files_to_df(input_files, left_col, right_col):
    """
    The input_files should be a list of strings containing the file names of the input files

    the name of each input files should end in'_x' where x is the config_name ex:
        str_0, str_1, str_2, ...
        volume_energy_0, volume_energy_1, volume_energy_2, ...

    The contents of the input files should be organized as two columns separated by a space
    and not contain any headers ex:
        1.0 2.0
        2.0 3.0
        3.0 4.0
        ...
    the data reader is np.loadtxt()

    left_col is the type of data in the left column ex: 'volume'
    right_col is the type of data in the right column ex: 'energy'
    """
    df = pd.DataFrame(columns=["config", left_col, right_col])
    for input_file in input_files:
        config = os.path.splitext(os.path.basename(input_file))[0]
        if "_" in config:
            config = config.split("_")[-1]
        data = np.loadtxt(input_file)
        left_data = data[:, 0]
        right_data = data[:, 1]
        for i in range(len(left_data)):
            df.loc[len(df)] = {
                "config": config,
                left_col: left_data[i],
                right_col: right_data[i],
            }
    return df


# TODO: review
def select_data(df, selection_dict):
    """
    the selection_dict should be a dictionary with the following format:
    selection_dict = {10: [0,1,2,3],
                        11: [0,1,2,3],
                        12: [0,1,2,3],
                        ...}
    where the keys are the config # and the values are the volumes 0=lowest
    volume, 1=second lowest volume, etc.
    """
    # first rank the volumes from lowest to highest
    df["volume_rank"] = df.groupby("config")["volume"].rank(method="dense").astype(int)
    print(df)
    selected_data = pd.DataFrame()  # Initialize selected_data variable
    for config in selection_dict.keys():
        for volume_rank in selection_dict[config]:
            selected_data = pd.concat(
                [
                    selected_data,
                    df[(df["config"] == config) & (df["volume_rank"] == volume_rank)],
                ],
                ignore_index=True,
            )
    return selected_data


# TODO: review
def plot_mv(df, show_fig=True):
    """
    data may be a single pandas data frame or a list of pandas data frames
    data may also be a list of input_file names as strings ex:
        ['str_0', 'str_1', 'str_2', ...]

    Not sure if a list of dataframe will actually work. the function simply
    concats them all together # TODO: test this

    df is a data frame with columns ['config', '# of ion', 'volume', 'tot']
    breaks if missing these columns
    """

    fig = px.line(
        df,
        x="volume",
        y="tot",
        color="# of ion",
        symbol="# of ion",
        hover_data=["config", "# of ion", "volume", "tot"],
        template="plotly_white",
    )
    fig.update_layout(
        title="Mag-V", xaxis_title="Volume [A^3]", yaxis_title="Magnetic Moment [mu_B]"
    )

    fig.update_yaxes(nticks=10)
    fig.update_xaxes(nticks=10)

    # Loop over each trace and update dash length
    for i, trace in enumerate(fig.data):
        dash_length = (
            f"{2+(i+1)}px,{2+2*(i+1)}px"  # Dash length changes with each iteration
        )
        fig.data[-i - 1].update(
            mode="markers+lines",
            marker=dict(size=8, line=dict(width=1), opacity=0.5),
            line=dict(width=3, dash=dash_length),
        )

    if show_fig:
        fig.show()
    return fig
    # determine the type of data and how to handle it.
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, list) and all(
        type(elem) == type(data[0]) for elem in data
    ):  # check if each elem of the list is the same type as the zeroth element
        if isinstance(data[0], pd.DataFrame):
            df = pd.concat(data, ignore_index=True)
        elif isinstance(data[0], str):
            df = convert_input_files_to_df(data, left_col, right_col)
            print(df)
    else:
        raise ValueError(
            "data must be a pandas DataFrame or a list of pandas DataFrames or a list of input_file names as strings"
        )

    # create a data frame with the eos fits for each config
    if eos_fitting != None:
        eos_df = fit_to_all_eos(df)

    # plot the data
    unique_configs = df["config"].unique()
    fig = go.Figure()
    for config in df["config"].unique():
        config_df = df[df["config"] == config]
        if per_atom == False:
            x = config_df["volume"]
            y = config_df["energy"]
        elif per_atom == True:
            x = config_df["volume"] / config_df["number_of_atoms"]
            y = config_df["energy"] / config_df["number_of_atoms"]
        else:
            print("per_atom must be True or False")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    colorscale="Viridis",
                    symbol="circle",  # Use the symbol from the dictionary
                    opacity=0.5,
                    size=7,
                    line=dict(width=1, color="DarkSlateGrey"),  # Add an outline
                ),
                legendgroup="EOS",
                name=f"Config {config}",
            )
        )
    if per_atom == False:
        fig.update_layout(
            title="E-V",
            xaxis_title="Volume [Å^3]",
            yaxis_title="Energy (eV)",
            template="plotly_white",
        )
    elif per_atom == True:
        fig.update_layout(
            title="E-V",
            xaxis_title="Volume [Å^3/atom]",
            yaxis_title="Energy (eV/atom)",
            template="plotly_white",
        )
    else:
        print("per_atom must be True or False")

    # loop over configs in the eos data frame and plot the eos fits
    if eos_fitting != None:
        for config in eos_df["config"].unique():
            eos_config_df = eos_df[eos_df["config"] == config]
            if eos_fitting in eos_config_df["eos_name"].unique():
                eos_name_df = eos_config_df[eos_config_df["eos_name"] == eos_fitting]
                if per_atom == False:
                    fig.add_trace(
                        go.Scatter(
                            x=eos_name_df["volumes"].values[0],
                            y=eos_name_df["energies"].values[0],
                            mode="lines",
                            name=f"{eos_fitting} fit",
                            line=dict(width=1),
                            legendgroup="data",
                        )
                    )
                elif per_atom == True:
                    fig.add_trace(
                        go.Scatter(
                            x=eos_name_df["volumes"].values[0]
                            / eos_name_df["number_of_atoms"].values[0][0],
                            y=eos_name_df["energies"].values[0]
                            / eos_name_df["number_of_atoms"].values[0][0],
                            mode="lines",
                            name=f"{eos_fitting} fit",
                            line=dict(width=1),
                            legendgroup="data",
                        )
                    )
                # plot the minimum energy data point for each config from the fitting equation
                if highlight_minimum == True:
                    min_energy = min(eos_name_df["energies"].values[0])
                    volume_at_min_energy = eos_name_df["volumes"].values[0][
                        np.where(eos_name_df["energies"].values[0] == min_energy)[0][0]
                    ]
                    if per_atom == False:
                        fig.add_trace(
                            go.Scatter(
                                x=[volume_at_min_energy],
                                y=[min_energy],
                                mode="markers",
                                name=f"{eos_fitting} min energy",
                                marker=dict(color="black", size=10, symbol="cross"),
                                legendgroup="minimum",
                            )
                        )
                    elif per_atom == True:
                        fig.add_trace(
                            go.Scatter(
                                x=[
                                    volume_at_min_energy
                                    / eos_name_df["number_of_atoms"].values[0][0]
                                ],
                                y=[
                                    min_energy
                                    / eos_name_df["number_of_atoms"].values[0][0]
                                ],
                                mode="markers",
                                name=f"{eos_fitting} min energy",
                                marker=dict(color="black", size=10, symbol="cross"),
                                legendgroup="minimum",
                            )
                        )
                    else:
                        print("per_atom must be True or False")
                elif highlight_minimum == False:
                    pass
                else:
                    print("highlight_minimum must be True or False")
            elif eos_fitting == "all":
                for eos_name in eos_config_df["eos_name"].unique():
                    eos_name_df = eos_config_df[eos_config_df["eos_name"] == eos_name]
                    if per_atom == False:
                        fig.add_trace(
                            go.Scatter(
                                x=eos_name_df["volumes"].values[0],
                                y=eos_name_df["energies"].values[0],
                                mode="lines",
                                name=f"{eos_name} fit",
                                line=dict(width=1),
                                legendgroup="eos",
                            )
                        )
                    elif per_atom == True:
                        fig.add_trace(
                            go.Scatter(
                                x=eos_name_df["volumes"].values[0]
                                / eos_name_df["number_of_atoms"].values[0][0],
                                y=eos_name_df["energies"].values[0]
                                / eos_name_df["number_of_atoms"].values[0][0],
                                mode="lines",
                                name=f"{eos_name} fit",
                                line=dict(width=1),
                                legendgroup="eos",
                            )
                        )
                    else:
                        print("per_atom must be True or False")
                    # plot the minimum energy data point for each config from the fitting equation
                    if highlight_minimum == True:
                        min_energy = min(eos_name_df["energies"].values[0])
                        volume_at_min_energy = eos_name_df["volumes"].values[0][
                            np.where(eos_name_df["energies"].values[0] == min_energy)[
                                0
                            ][0]
                        ]
                        if per_atom == False:
                            fig.add_trace(
                                go.Scatter(
                                    x=[volume_at_min_energy],
                                    y=[min_energy],
                                    mode="markers",
                                    name=f"{eos_name} min energy",
                                    marker=dict(color="black", size=8, symbol="cross"),
                                    legendgroup="minimum",
                                )
                            )
                        elif per_atom == True:
                            fig.add_trace(
                                go.Scatter(
                                    x=[
                                        volume_at_min_energy
                                        / eos_name_df["number_of_atoms"].values[0][0]
                                    ],
                                    y=[
                                        min_energy
                                        / eos_name_df["number_of_atoms"].values[0][0]
                                    ],
                                    mode="markers",
                                    name=f"{eos_name} min energy",
                                    marker=dict(color="black", size=8, symbol="cross"),
                                    legendgroup="minimum",
                                )
                            )
                        else:
                            print("per_atom must be True or False")
            elif eos_fitting == None:
                pass
            else:
                print(
                    f"Warning: eos_fitting '{eos_fitting}' not found in eos_df. Skipping."
                )
    fig.update_layout(
        plot_bgcolor="white", width=800, height=600, margin=dict(l=80, r=30, t=80, b=80)
    )
    fig.update_yaxes(
        showline=True,  # add line at x=0
        linecolor="black",
        linewidth=2.4,
        ticks="inside",
        mirror="allticks",  # add ticks to top/right axes
        tickwidth=2.4,
        tickcolor="black",
        showgrid=False,
    )
    fig.update_xaxes(
        showline=True,
        showticklabels=True,
        linecolor="black",
        linewidth=2.4,
        ticks="inside",
        mirror="allticks",
        tickwidth=2.4,
        tickcolor="black",
        showgrid=False,
    )
    if show_fig:
        fig.show()
    return fig


def assign_colors_to_configs(df, alpha=1, cmap="plotly"):
    unique_configs = df["config"].unique()

    if cmap == "plotly":
        colors = px.colors.qualitative.Plotly
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
            f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {alpha})"
            for color in colors
        ]

    elif cmap == "distinctipy":
        colors = get_colors(len(unique_configs))
        colors = [
            f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha})" for color in colors
        ]
    else:
        print("cmap must be 'plotly' or 'distinctipy'")

    config_colors = {
        config: colors[i % len(colors)] for i, config in enumerate(unique_configs)
    }
    return config_colors


def assign_marker_symbols_to_configs(df):
    unique_configs = df["config"].unique()
    symbols = [
        "circle",
        "square",
        "diamond",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "pentagon",
        "hexagon",
        "octagon",
        "star",
        "hexagram",
        "star-triangle-up",
        "star-triangle-down",
        "star-square",
        "star-diamond",
        "diamond-tall",
        "diamond-wide",
        "hourglass",
        "bowtie",
        "circle-cross",
        "circle-x",
        "square-cross",
        "square-x",
        "diamond-cross",
        "diamond-x",
        "cross-thin",
        "x-thin",
        "asterisk",
        "hash",
        "y-up",
        "y-down",
        "y-left",
        "y-right",
        "line-ew",
        "line-ns",
        "line-ne",
        "line-nw",
    ]
    config_symbols = {
        config: symbols[i % len(symbols)] for i, config in enumerate(unique_configs)
    }
    return config_symbols


def plot_ev(
    data,
    eos_fitting="BM4",
    highlight_minimum=True,
    per_atom=False,
    title=None,
    show_fig=True,
    left_col="volume",
    right_col="energy",
    cmap="plotly",
    marker_alpha=1,
    marker_size=10,
):
    """_summary_

    Args:
        data (pandas.DataFrame, list of pandas.DataFrame, or list of str): Data must be a pandas
        DataFrame, list of pandas DataFrames, or a list of input_file names as strings containing the
        volumes, energies, and number of atoms of each configuration.
        eos_fitting (str, optional): EOS name. Defaults to "BM4".
        highlight_minimum (bool, optional): Defaults to True.
        per_atom (bool, optional):Defaults to False.
        title (_type_, optional): Defaults to None.
        show_fig (bool, optional): Defaults to True.
        left_col (str, optional): Defaults to "volume".
        right_col (str, optional): Defaults to "energy".
        cmap (str, optional): Defaults to 'plotly'.
        marker_alpha (int, optional): Defaults to 1.
        marker_size (int, optional): Defaults to 10.

    Returns:
        fig (plotly.graph_objs._figure.Figure): A Plotly figure.
    """

    # Check if data is a pandas DataFrame or a list of pandas DataFrames
    if isinstance(data, pd.DataFrame):
        df = data

    # Check if each element of the list is the same type as the zeroth element
    elif isinstance(data, list) and all(
        type(element) == type(data[0]) for element in data
    ):
        if isinstance(data[0], pd.DataFrame):
            df = pd.concat(data, ignore_index=True)
        elif isinstance(data[0], str):
            df = convert_input_files_to_df(data, left_col, right_col)

    else:
        raise ValueError(
            "data must be a pandas DataFrame, list of pandas DataFrames, or a list of input_file names as strings"
        )

    # Create a data frame with the eos fits for each config
    if eos_fitting != None:
        eos_df, _ = fit_to_all_eos(df)

    # Assign colors and symbols
    config_colors = assign_colors_to_configs(df, alpha=marker_alpha, cmap=cmap)
    config_symbols = assign_marker_symbols_to_configs(df)

    # Plot the data
    fig = go.Figure()
    fig.update_layout(
        font=dict(
            family="Devaju Sans",
        )
    )

    for config in df["config"].unique():
        config_df = df[df["config"] == config]

        if isinstance(per_atom, bool):
            x = config_df["volume"]
            y = config_df["energy"]

            if per_atom:
                x = x / config_df["number_of_atoms"]
                y = y / config_df["number_of_atoms"]
        else:
            raise ValueError("per_atom must be True or False")

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=config_colors[config],
                    symbol=config_symbols[config],
                ),
                legendgroup="EOS",
                name=f"Config {config}",
            )
        )

    if isinstance(per_atom, bool):
        atom_suffix = "/atom" if per_atom else ""
        fig.update_xaxes(
            title=dict(
                text=f"Volume (Å<sup>3</sup>{atom_suffix})",
                font=dict(size=22, color="rgb(0,0,0)"),
            )
        )
        fig.update_yaxes(
            title=dict(
                text=f"Energy (eV{atom_suffix})", font=dict(size=22, color="rgb(0,0,0)")
            )
        )

    # Loop over configs in the eos data frame and plot the eos fits
    if eos_fitting != None:
        for config in eos_df["config"].unique():
            eos_config_df = eos_df[eos_df["config"] == config]
            if eos_fitting in eos_config_df["EOS"].unique():
                eos_name_df = eos_config_df[eos_config_df["EOS"] == eos_fitting]

                x = eos_name_df["volumes"].values[0]
                y = eos_name_df["energies"].values[0]

                if per_atom:
                    num_atoms = eos_name_df["number_of_atoms"].values[0][0]
                    x = x / num_atoms
                    y = y / num_atoms

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"{eos_fitting} fit",
                        line=dict(width=1.75, color=config_colors[config]),
                        legendgroup="data",
                        showlegend=False,
                    )
                )

                # Plot the equilibrium energy and volume for each config
                if highlight_minimum == True:
                    min_energy = min(eos_name_df["energies"].values[0])
                    volume_at_min_energy = eos_name_df["volumes"].values[0][
                        np.where(eos_name_df["energies"].values[0] == min_energy)[0][0]
                    ]

                    x = volume_at_min_energy
                    y = min_energy

                    if per_atom:
                        num_atoms = eos_name_df["number_of_atoms"].values[0][0]
                        x = x / num_atoms
                        y = y / num_atoms

                    fig.add_trace(
                        go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers",
                            name=f"{eos_fitting} min energy",
                            marker=dict(
                                color="black", size=marker_size, symbol="cross"
                            ),
                            legendgroup="minimum",
                            showlegend=False,
                        )
                    )

                elif highlight_minimum == False:
                    pass

                else:
                    raise ValueError("highlight_minimum must be True or False")

            # TODO: Do we really need all?
            elif eos_fitting == "all":
                for eos_name in eos_config_df["EOS"].unique():
                    eos_name_df = eos_config_df[eos_config_df["EOS"] == eos_name]

                    if per_atom == False:
                        fig.add_trace(
                            go.Scatter(
                                x=eos_name_df["volumes"].values[0],
                                y=eos_name_df["energies"].values[0],
                                mode="lines",
                                name=f"{eos_name} fit",
                                line=dict(width=1),
                                legendgroup="eos",
                                showlegend=False,
                            )
                        )

                    elif per_atom == True:
                        fig.add_trace(
                            go.Scatter(
                                x=eos_name_df["volumes"].values[0]
                                / eos_name_df["number_of_atoms"].values[0][0],
                                y=eos_name_df["energies"].values[0]
                                / eos_name_df["number_of_atoms"].values[0][0],
                                mode="lines",
                                name=f"{eos_name} fit",
                                line=dict(width=1),
                                legendgroup="eos",
                                showlegend=False,
                            )
                        )

                    # Plot the minimum energy data point for each config from the fitting equation
                    if highlight_minimum == True:
                        min_energy = min(eos_name_df["energies"].values[0])
                        volume_at_min_energy = eos_name_df["volumes"].values[0][
                            np.where(eos_name_df["energies"].values[0] == min_energy)[
                                0
                            ][0]
                        ]
                        if per_atom == False:
                            fig.add_trace(
                                go.Scatter(
                                    x=[volume_at_min_energy],
                                    y=[min_energy],
                                    mode="markers",
                                    name=f"{eos_name} min energy",
                                    marker=dict(
                                        color="black", size=marker_size, symbol="cross"
                                    ),
                                    legendgroup="minimum",
                                    showlegend=False,
                                )
                            )
                        elif per_atom == True:
                            fig.add_trace(
                                go.Scatter(
                                    x=[
                                        volume_at_min_energy
                                        / eos_name_df["number_of_atoms"].values[0][0]
                                    ],
                                    y=[
                                        min_energy
                                        / eos_name_df["number_of_atoms"].values[0][0]
                                    ],
                                    mode="markers",
                                    name=f"{eos_name} min energy",
                                    marker=dict(
                                        color="black", size=marker_size, symbol="cross"
                                    ),
                                    legendgroup="minimum",
                                    showlegend=False,
                                )
                            )

            elif eos_fitting == None:
                pass

            else:
                print(
                    f"Warning: eos_fitting '{eos_fitting}' not found in eos_df. Skipping."
                )
                sys.exit(1)

    axis_params = dict(
        showline=True,
        linecolor="black",
        linewidth=1,
        ticks="outside",
        mirror="allticks",
        tickwidth=1,
        tickcolor="black",
        showgrid=False,
        tickfont=dict(color="rgb(0,0,0)", size=20),
    )

    fig.update_layout(
        plot_bgcolor="white",
        width=840,
        height=600,
        legend=dict(font=dict(size=20, color="black")),
        xaxis=axis_params,
        yaxis=axis_params,
    )

    if title != None:
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(color="rgb(0,0,0)", size=30))
        )

    if show_fig:
        fig.show()

    return fig


def plot_energy_difference(
    df,
    reference_config,
    per_atom=False,
    show_fig=True,
    convert_to_mev=False,
    volume_precision=0.0001,
    title=None,
    marker_alpha=1,
    cmap="plotly",
    marker_size=10,
):
    """Takes a dataframe and plots the energy difference with respect to a reference configuration
    as a function of volume. Utilizes plot_ev() for the actual plotting.

    Args:
        df (pandas.DataFrame): dataframe containing the volumes, energies, and number of atoms of each
        configuration.
        reference_config (str): name of the configuration to be used as the reference state
        per_atom (bool, optional): Defaults to False.
        show_fig (bool, optional): Defaults to True.
        convert_to_mev (bool, optional): Defaults to False.
        title (_type_, optional): Defaults to None.
        marker_alpha (int, optional): Defaults to 1.
        cmap (str, optional): Defaults to 'plotly'.
        marker_size (int, optional): Defaults to 10.

    Returns:
        fig (plotly.graph_objs._figure.Figure): A Plotly figure.
    """

    df_list = []
    for config in df["config"].unique():
        df_list.append(df[df["config"] == config].reset_index(drop=True))
        if config == reference_config:
            reference_df = df[df["config"] == config].reset_index(drop=True)

    # Subtract reference energies
    missing_volumes = []
    for df_el in df_list:
        for i, row in df_el.iterrows():
            try:
                reference_energy = reference_df[np.isclose(reference_df['volume'], row['volume'], atol=volume_precision)]['energy'].values[0]
                df_el.at[i, 'energy'] = row['energy'] - reference_energy
            except Exception as e:
                missing_volumes.append((row["config"], row["volume"]))
    if missing_volumes:
        print(f"Warning: Missing volumes for configurations: {missing_volumes}")

    energy_difference_df = pd.concat(df_list)

    if convert_to_mev == True:
        energy_difference_df["energy"] *= 1000

    # plot energy difference vs volume
    fig = plot_ev(
        energy_difference_df,
        eos_fitting=None,
        per_atom=per_atom,
        show_fig=False,
        title=title,
        cmap=cmap,
        marker_alpha=marker_alpha,
        marker_size=marker_size,
    )

    fig.update_layout(
        font=dict(
            family="Devaju Sans",
        )
    )

    unit = "meV" if convert_to_mev else "eV"
    per_atom_suffix = "/atom" if per_atom else ""
    title_text = f"ΔEnergy ({unit}{per_atom_suffix})"

    fig.update_yaxes(
        title=dict(text=title_text, font=dict(size=22, color="rgb(0,0,0)"))
    )

    if show_fig:
        fig.show()

    # Plot a horizontal line a y=0
    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=min(fig.data[0].x * 0.95),
        x1=max(fig.data[0].x * 1.05),
        y0=0,
        y1=0,
        line=dict(
            color="black",
            width=2,
            dash="dash",
        ),
    )

    return fig


# TODO: review
def plot_config_energy(
    df, number_of_lowest_configs=5, show_fig=True, xmax=None, ymax=None
):
    new_df = df
    try:
        new_df = df.drop("# of ion", axis=1)
        new_df = new_df.drop("tot", axis=1)
        new_df = new_df.drop_duplicates()
        new_df["energy_per_atom"] = new_df["energy"] / new_df["number_of_atoms"]
        new_df = new_df.nsmallest(number_of_lowest_configs, "energy_per_atom").copy()
    except Exception as e:
        print("possible error. Could not strip magnetic data: ", e)
        new_df["energy_per_atom"] = new_df["energy"] / new_df["number_of_atoms"]
        new_df = df.nsmallest(number_of_lowest_configs, "energy_per_atom").copy()
    new_df["energy_difference"] = (
        new_df["energy_per_atom"] - new_df["energy_per_atom"].min()
    ) * 1000
    new_df = new_df.reset_index(drop=True)
    new_df["rank"] = new_df["energy_difference"].rank(method="min") - 1
    if xmax == None:
        xmax = new_df["rank"].max()
    if ymax == None:
        max_energy_difference = new_df["energy_difference"].max()
        # Get the order of magnitude of the max_energy_difference
        rounding_order_of_magnitude = 10 ** (len(str(int(max_energy_difference))) - 2)

        # Round up to the nearest order of magnitude
        ymax = (
            math.ceil(max_energy_difference / rounding_order_of_magnitude)
            * rounding_order_of_magnitude
        )

        # Get the next multiple of order of magnitude with the second digit being 0
        ymax = ((ymax // rounding_order_of_magnitude) + 1) * rounding_order_of_magnitude
    fig = px.scatter(
        new_df, x="rank", y="energy_difference", color="config", template="plotly_white"
    )
    fig.update_traces(
        marker=dict(size=5, symbol="cross-thin-open", color="blue"),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        title="Configuration Energy",
        xaxis_title="Configuration",
        yaxis_title="Energy difference (meV/atom)",
    )
    fig.update_layout(showlegend=False)
    fig.update_layout(
        title_text="Configuration Energy",
        plot_bgcolor="white",
        width=600,
        height=600,
        margin=dict(l=80, r=30, t=80, b=80),
    )
    fig.update_yaxes(
        range=[0, ymax],
        showline=True,  # add line at x=0
        linecolor="black",
        linewidth=2.4,
        ticks="inside",
        mirror="allticks",  # add ticks to top/right axes
        tickwidth=2.4,
        tickcolor="black",
        showgrid=False,
    )
    fig.update_xaxes(
        range=[0, xmax],
        showline=True,
        showticklabels=True,
        linecolor="black",
        linewidth=2.4,
        ticks="inside",
        mirror="allticks",
        tickwidth=2.4,
        tickcolor="black",
        showgrid=False,
    )
    if show_fig:
        fig.show()
    return fig


# TODO: review
def plot_energy_histogram(df, nbins=None, show_fig=True):
    try:
        new_df = df.drop("# of ion", axis=1)
        new_df = new_df.drop("tot", axis=1)
        new_df = new_df.drop_duplicates()
        new_df["energy_per_atom"] = new_df["energy"] / new_df["number_of_atoms"]
    except Exception as e:
        print("possible error. Could not strip magnetic data: ", e)
        new_df["energy_per_atom"] = new_df["energy"] / new_df["number_of_atoms"]
    new_df["relative_energy"] = (
        new_df["energy_per_atom"] - new_df["energy_per_atom"].min()
    ) * 1000
    fig = px.histogram(
        new_df, x="relative_energy", nbins=nbins, template="plotly_white"
    )
    print(new_df)
    if show_fig:
        fig.show()
