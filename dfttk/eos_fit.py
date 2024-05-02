"""
This EOS fitting code is based on the following paper:
Shun-Li Shang et al., Computational Materials Science, 47, 4, (2010).
https://doi.org/10.1016/j.commatsci.2009.12.006

Equations of State:
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
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from distinctipy import get_colors
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import fsolve
from scipy.optimize import leastsq


def mBM4(volume, energy):
    eos_index = 1
    volume_range = np.linspace(min(volume), max(volume), 1000)

    AA = np.vstack(
        (
            np.ones(np.shape(volume)),
            volume ** (-1 / 3),
            volume ** (-2 / 3),
            volume ** (-1),
        )
    )
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(energy)
    a = xx[0]
    b = xx[1]
    c = xx[2]
    d = xx[3]
    e = 0.0

    energy_eos = (
        a
        + b * (volume_range) ** (-1 / 3)
        + c * (volume_range) ** (-2 / 3)
        + d * (volume_range) ** (-1)
        + e * (volume_range) ** (-4 / 3)
    )
    energy_eos_points = (
        a
        + b * (volume) ** (-1 / 3)
        + c * (volume) ** (-2 / 3)
        + d * (volume) ** (-1)
        + e * (volume) ** (-4 / 3)
    )
    energy_difference = energy_eos_points - energy

    V = (
        4 * c**3
        - 9 * b * c * d
        + np.sqrt((c**2 - 3 * b * d) * (4 * c**2 - 3 * b * d) ** 2)
    )
    V = -V / b**3

    P = (
        (4 * e) / (3 * V ** (7 / 3))
        + d / V**2
        + (2 * c) / (3 * V ** (5 / 3))
        + b / (3 * V ** (4 / 3))
    )
    P = P * 160.2176621
    B = (
        (28 * e) / (9 * V ** (10 / 3))
        + (2 * d) / V**3
        + (10 * c) / (9 * V ** (8 / 3))
        + (4 * b) / (9 * V ** (7 / 3))
    ) * V
    B = B * 160.2176621
    BP = (98 * e + 54 * d * V ** (1 / 3) + 25 * c * V ** (2 / 3) + 8 * b * V) / (
        42 * e + 27 * d * V ** (1 / 3) + 15 * c * V ** (2 / 3) + 6 * b * V
    )
    B2P = (
        V ** (8 / 3)
        * (
            9 * d * (14 * e + 5 * c * V ** (2 / 3) + 8 * b * V)
            + 2 * V ** (1 / 3) * (126 * b * e * V ** (1 / 3) + 5 * c * (28 * e + b * V))
        )
    ) / (2 * (14 * e + 9 * d * V ** (1 / 3) + 5 * c * V ** (2 / 3) + 2 * b * V) ** 3)
    B2P = B2P / 160.2176621
    E0 = a + b * V ** (-1 / 3) + c * V ** (-2 / 3) + d * V ** (-1) + e * V ** (-4 / 3)
    eos_parameters = [V, E0, P, B, BP, B2P]

    fitting_error = np.array(
        [math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))]
    )
    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10**4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = (
        (4 * e) / (3 * volume_range ** (7 / 3))
        + d / volume_range**2
        + (2 * c) / (3 * volume_range ** (5 / 3))
        + b / (3 * volume_range ** (4 / 3))
    ) * 160.2176621
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    # Used for eos_index = 7 and 8
    xini = [
        eos_parameters[0],
        eos_parameters[1],
        eos_parameters[3] / 160.2176621,
        eos_parameters[4],
    ]
    return results, volume_range, energy_eos, pressure_eos


def mBM5(volume, energy):
    eos_index = 2
    volume_range = np.linspace(min(volume), max(volume), 1000)

    AA = np.vstack(
        (
            np.ones(np.shape(volume)),
            volume ** (-1 / 3),
            volume ** (-2 / 3),
            volume ** (-1),
            volume ** (-4 / 3),
        )
    )
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(energy)
    a = xx[0]
    b = xx[1]
    c = xx[2]
    d = xx[3]
    e = xx[4]

    energy_eos = (
        a
        + b * (volume_range) ** (-1 / 3)
        + c * (volume_range) ** (-2 / 3)
        + d * (volume_range) ** (-1)
        + e * (volume_range) ** (-4 / 3)
    )
    energy_eos_points = (
        a
        + b * (volume) ** (-1 / 3)
        + c * (volume) ** (-2 / 3)
        + d * (volume) ** (-1)
        + e * (volume) ** (-4 / 3)
    )
    energy_difference = energy_eos_points - energy

    func = (
        lambda volume_range: (
            (4 * e) / (3 * volume_range ** (7 / 3))
            + d / volume_range**2
            + (2 * c) / (3 * volume_range ** (5 / 3))
            + b / (3 * volume_range ** (4 / 3))
        )
        * 160.2176621
    )
    V = fsolve(func, np.mean(volume))
    P = (
        (4 * e) / (3 * V ** (7 / 3))
        + d / V**2
        + (2 * c) / (3 * V ** (5 / 3))
        + b / (3 * V ** (4 / 3))
    )
    P = P * 160.2176621
    B = (
        (28 * e) / (9 * V ** (10 / 3))
        + (2 * d) / V**3
        + (10 * c) / (9 * V ** (8 / 3))
        + (4 * b) / (9 * V ** (7 / 3))
    ) * V
    B = B * 160.2176621
    BP = (98 * e + 54 * d * V ** (1 / 3) + 25 * c * V ** (2 / 3) + 8 * b * V) / (
        42 * e + 27 * d * V ** (1 / 3) + 15 * c * V ** (2 / 3) + 6 * b * V
    )
    B2P = (
        V ** (8 / 3)
        * (
            9 * d * (14 * e + 5 * c * V ** (2 / 3) + 8 * b * V)
            + 2 * V ** (1 / 3) * (126 * b * e * V ** (1 / 3) + 5 * c * (28 * e + b * V))
        )
    ) / (2 * (14 * e + 9 * d * V ** (1 / 3) + 5 * c * V ** (2 / 3) + 2 * b * V) ** 3)
    B2P = B2P / 160.2176621
    E0 = a + b * V ** (-1 / 3) + c * V ** (-2 / 3) + d * V ** (-1) + e * V ** (-4 / 3)
    eos_parameters = [V, E0, P, B, BP, B2P]

    fitting_error = np.array(
        [math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))]
    )
    results = np.concatenate(([eos_index], *eos_parameters, fitting_error * (10**4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = (
        (4 * e) / (3 * volume_range ** (7 / 3))
        + d / volume_range**2
        + (2 * c) / (3 * volume_range ** (5 / 3))
        + b / (3 * volume_range ** (4 / 3))
    ) * 160.2176621
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos


def BM4(volume, energy):
    eos_index = 3
    volume_range = np.linspace(min(volume), max(volume), 1000)

    AA = np.vstack(
        (
            np.ones(np.shape(volume)),
            volume ** (-2 / 3),
            volume ** (-4 / 3),
            volume ** (-2),
        )
    )
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(energy)
    a = xx[0]
    b = xx[1]
    c = xx[2]
    d = xx[3]
    e = 0.0

    energy_eos = (
        a
        + b * (volume_range) ** (-2 / 3)
        + c * (volume_range) ** (-4 / 3)
        + d * (volume_range) ** (-2)
        + e * (volume_range) ** (-8 / 3)
    )
    energy_eos_points = (
        a
        + b * (volume) ** (-2 / 3)
        + c * (volume) ** (-4 / 3)
        + d * (volume) ** (-2)
        + e * (volume) ** (-8 / 3)
    )
    energy_difference = energy_eos_points - energy

    V = math.sqrt(
        -(
            (
                4 * c**3
                - 9 * b * c * d
                + math.sqrt((c**2 - 3 * b * d) * (4 * c**2 - 3 * b * d) ** 2)
            )
            / b**3
        )
    )
    P = (
        (8 * e) / (3 * V ** (11 / 3))
        + (2 * d) / V**3
        + (4 * c) / (3 * V ** (7 / 3))
        + (2 * b) / (3 * V ** (5 / 3))
    )
    P = P * 160.2176621
    B = (
        2 * (44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V**2)
    ) / (9 * V ** (11 / 3))
    B = B * 160.2176621
    BP = (
        484 * e + 243 * d * V ** (2 / 3) + 98 * c * V ** (4 / 3) + 25 * b * V**2
    ) / (132 * e + 81 * d * V ** (2 / 3) + 42 * c * V ** (4 / 3) + 15 * b * V**2)
    B2P = (
        4
        * V ** (13 / 3)
        * (
            27 * d * (22 * e + 7 * c * V ** (4 / 3) + 10 * b * V**2)
            + V ** (2 / 3)
            * (990 * b * e * V ** (2 / 3) + 7 * c * (176 * e + 5 * b * V**2))
        )
    ) / (44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V**2) ** 3
    B2P = B2P / 160.2176621
    E0 = a + e / V ** (8 / 3) + d / V**2 + c / V ** (4 / 3) + b / V ** (2 / 3)
    eos_parameters = [V, E0, P, B, BP, B2P]

    fitting_error = np.array(
        [math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))]
    )

    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10**4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = (
        (8 * e) / (3 * volume_range ** (11 / 3))
        + (2 * d) / volume_range**3
        + (4 * c) / (3 * volume_range ** (7 / 3))
        + (2 * b) / (3 * volume_range ** (5 / 3))
    ) * 160.2176621
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos


def BM5(volume, energy):
    eos_index = 4
    volume_range = np.linspace(min(volume), max(volume), 1000)

    AA = np.vstack(
        (
            np.ones(np.shape(volume)),
            volume ** (-2 / 3),
            volume ** (-4 / 3),
            volume ** (-2),
            volume ** (-8 / 3),
        )
    )
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(energy)
    a = xx[0]
    b = xx[1]
    c = xx[2]
    d = xx[3]
    e = xx[4]

    energy_eos = (
        a
        + b * (volume_range) ** (-2 / 3)
        + c * (volume_range) ** (-4 / 3)
        + d * (volume_range) ** (-2)
        + e * (volume_range) ** (-8 / 3)
    )
    energy_eos_points = (
        a
        + b * (volume) ** (-2 / 3)
        + c * (volume) ** (-4 / 3)
        + d * (volume) ** (-2)
        + e * (volume) ** (-8 / 3)
    )
    energy_difference = energy_eos_points - energy

    func = (
        lambda volume_range: (
            (8 * e) / (3 * volume_range ** (11 / 3))
            + (2 * d) / volume_range**3
            + (4 * c) / (3 * volume_range ** (7 / 3))
            + (2 * b) / (3 * volume_range ** (5 / 3))
        )
        * 160.2176621
    )
    V = fsolve(func, np.mean(volume))
    P = (
        (8 * e) / (3 * V ** (11 / 3))
        + (2 * d) / V**3
        + (4 * c) / (3 * V ** (7 / 3))
        + (2 * b) / (3 * V ** (5 / 3))
    )
    P = P * 160.2176621
    B = (
        2 * (44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V**2)
    ) / (9 * V ** (11 / 3))
    B = B * 160.2176621
    BP = (
        484 * e + 243 * d * V ** (2 / 3) + 98 * c * V ** (4 / 3) + 25 * b * V**2
    ) / (132 * e + 81 * d * V ** (2 / 3) + 42 * c * V ** (4 / 3) + 15 * b * V**2)
    B2P = (
        4
        * V ** (13 / 3)
        * (
            27 * d * (22 * e + 7 * c * V ** (4 / 3) + 10 * b * V**2)
            + V ** (2 / 3)
            * (990 * b * e * V ** (2 / 3) + 7 * c * (176 * e + 5 * b * V**2))
        )
    ) / (44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V**2) ** 3
    B2P = B2P / 160.2176621
    E0 = a + e / V ** (8 / 3) + d / V**2 + c / V ** (4 / 3) + b / V ** (2 / 3)
    eos_parameters = [V, E0, P, B, BP, B2P]

    fitting_error = np.array(
        [math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))]
    )
    results = np.concatenate(([eos_index], *eos_parameters, fitting_error * (10**4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = (
        (8 * e) / (3 * volume_range ** (11 / 3))
        + (2 * d) / volume_range**3
        + (4 * c) / (3 * volume_range ** (7 / 3))
        + (2 * b) / (3 * volume_range ** (5 / 3))
    ) * 160.2176621
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos


def LOG4(volume, energy):
    eos_index = 5
    volume_range = np.linspace(min(volume), max(volume), 1000)

    AA = np.vstack(
        (
            np.ones(np.shape(volume)),
            np.log(volume),
            np.log(volume) ** 2,
            np.log(volume) ** 3,
        )
    )
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(energy)
    a = xx[0]
    b = xx[1]
    c = xx[2]
    d = xx[3]
    e = 0.0

    energy_eos = (
        a
        + b * np.log(volume_range)
        + c * np.log(volume_range) ** 2
        + d * np.log(volume_range) ** 3
        + e * np.log(volume_range) ** 4
    )
    energy_eos_points = (
        a
        + b * np.log(volume)
        + c * np.log(volume) ** 2
        + d * np.log(volume) ** 3
        + e * np.log(volume) ** 4
    )
    energy_difference = energy_eos_points - energy

    func = (
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
    V = fsolve(func, np.mean(volume))
    V = np.mean(V)
    P = -(
        (b + 2 * c * math.log(V) + 3 * d * math.log(V) ** 2 + 4 * e * math.log(V) ** 3)
        / V
    )
    P = np.mean(P)
    P = P * 160.2176621
    B = -(
        (
            b
            - 2 * c
            + 2 * (c - 3 * d) * math.log(V)
            + 3 * (d - 4 * e) * math.log(V) ** 2
            + 4 * e * math.log(V) ** 3
        )
        / V
    )
    B = np.mean(B)
    B = B * 160.2176621
    BP = (
        b
        - 4 * c
        + 6 * d
        + 2 * (c - 6 * d + 12 * e) * math.log(V)
        + 3 * (d - 8 * e) * math.log(V) ** 2
        + 4 * e * math.log(V) ** 3
    ) / (
        b
        - 2 * c
        + 2 * (c - 3 * d) * math.log(V)
        + 3 * (d - 4 * e) * math.log(V) ** 2
        + 4 * e * math.log(V) ** 3
    )
    B2P = (
        2
        * V
        * (
            2 * c**2
            - 3 * b * d
            + 18 * d**2
            + 12 * b * e
            - 6 * c * (d + 4 * e)
            + 6 * (c * d - 3 * d**2 - 2 * b * e + 12 * d * e) * math.log(V)
            + 9 * (d - 4 * e) ** 2 * math.log(V) ** 2
            + 24 * (d - 4 * e) * e * math.log(V) ** 3
            + 24 * e**2 * math.log(V) ** 4
        )
    ) / (
        b
        - 2 * c
        + 2 * (c - 3 * d) * math.log(V)
        + 3 * (d - 4 * e) * math.log(V) ** 2
        + 4 * e * math.log(V) ** 3
    ) ** 3
    B2P = np.mean(B2P)
    B2P = B2P / 160.2176621
    E0 = (
        a
        + b * math.log(V)
        + c * math.log(V) ** 2
        + d * math.log(V) ** 3
        + e * math.log(V) ** 4
    )
    eos_parameters = [V, E0, P, B, BP, B2P]

    fitting_error = np.array(
        [math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))]
    )
    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10**4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = (
        -(
            (
                b
                + 2 * c * np.log(volume_range)
                + 3 * d * np.log(volume_range) ** 2
                + 4 * e * np.log(volume_range) ** 3
            )
            / volume_range
        )
    ) * 160.2176621
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos


def LOG5(volume, energy):
    eos_index = 6
    volume_range = np.linspace(min(volume), max(volume), 1000)

    AA = np.vstack(
        (
            np.ones(np.shape(volume)),
            np.log(volume),
            np.log(volume) ** 2,
            np.log(volume) ** 3,
            np.log(volume) ** 4,
        )
    )
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(energy)
    a = xx[0]
    b = xx[1]
    c = xx[2]
    d = xx[3]
    e = xx[4]

    energy_eos = (
        a
        + b * np.log(volume_range)
        + c * np.log(volume_range) ** 2
        + d * np.log(volume_range) ** 3
        + e * np.log(volume_range) ** 4
    )
    energy_eos_points = (
        a
        + b * np.log(volume)
        + c * np.log(volume) ** 2
        + d * np.log(volume) ** 3
        + e * np.log(volume) ** 4
    )
    energy_difference = energy_eos_points - energy

    func = (
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
    V = fsolve(func, np.mean(volume))
    V = np.mean(V)
    P = -(
        (b + 2 * c * math.log(V) + 3 * d * math.log(V) ** 2 + 4 * e * math.log(V) ** 3)
        / V
    )
    P = np.mean(P)
    P = P * 160.2176621
    B = -(
        (
            b
            - 2 * c
            + 2 * (c - 3 * d) * math.log(V)
            + 3 * (d - 4 * e) * math.log(V) ** 2
            + 4 * e * math.log(V) ** 3
        )
        / V
    )
    B = np.mean(B)
    B = B * 160.2176621
    BP = (
        b
        - 4 * c
        + 6 * d
        + 2 * (c - 6 * d + 12 * e) * math.log(V)
        + 3 * (d - 8 * e) * math.log(V) ** 2
        + 4 * e * math.log(V) ** 3
    ) / (
        b
        - 2 * c
        + 2 * (c - 3 * d) * math.log(V)
        + 3 * (d - 4 * e) * math.log(V) ** 2
        + 4 * e * math.log(V) ** 3
    )
    B2P = (
        2
        * V
        * (
            2 * c**2
            - 3 * b * d
            + 18 * d**2
            + 12 * b * e
            - 6 * c * (d + 4 * e)
            + 6 * (c * d - 3 * d**2 - 2 * b * e + 12 * d * e) * math.log(V)
            + 9 * (d - 4 * e) ** 2 * math.log(V) ** 2
            + 24 * (d - 4 * e) * e * math.log(V) ** 3
            + 24 * e**2 * math.log(V) ** 4
        )
    ) / (
        b
        - 2 * c
        + 2 * (c - 3 * d) * math.log(V)
        + 3 * (d - 4 * e) * math.log(V) ** 2
        + 4 * e * math.log(V) ** 3
    ) ** 3
    B2P = np.mean(B2P)
    B2P = B2P / 160.2176621
    E0 = (
        a
        + b * math.log(V)
        + c * math.log(V) ** 2
        + d * math.log(V) ** 3
        + e * math.log(V) ** 4
    )
    eos_parameters = [V, E0, P, B, BP, B2P]

    fitting_error = np.array(
        [math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))]
    )
    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10**4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = (
        -(
            (
                b
                + 2 * c * np.log(volume_range)
                + 3 * d * np.log(volume_range) ** 2
                + 4 * e * np.log(volume_range) ** 3
            )
            / volume_range
        )
    ) * 160.2176621
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos


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
    eos_index = 7
    volume_range = np.linspace(min(volume), max(volume), 1000)

    volume = volume
    Data = np.vstack((volume, energy))
    Data = Data.T

    [results, volume_range, energy_eos, pressure_eos] = mBM4(volume, energy)
    xini = [results[1], results[2], results[4] / 160.2176621, results[5]]
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
    energy_eos_points = (
        E0
        - (B * V) / (-1 + bp)
        + (B * (1 + (V / volume) ** bp / (-1 + bp)) * volume) / bp
    )
    energy_difference = energy_eos_points - energy
    eos_parameters = [V, E0, 0, B * 160.2176621, bp, 0]

    fitting_error = np.array(
        [math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))]
    )
    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10**4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = 160.2176621 * (B * (-1 + (V / volume_range) ** bp)) / bp
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos


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
    eos_index = 8
    volume_range = np.linspace(min(volume), max(volume), 1000)

    Data = np.vstack((volume, energy))
    Data = Data.T

    [results, volume_range, energy_eos, pressure_eos] = mBM4(volume, energy)
    xini = [results[1], results[2], results[4] / 160.2176621, results[5]]
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
    energy_eos_points = (
        E0
        + (4 * B * V) / (-1 + bp) ** 2
        - (4 * B * V * (1 + (3 * (-1 + bp) * (-1 + (volume / V) ** (1 / 3))) / 2))
        / (
            (-1 + bp) ** 2
            * np.exp((3 * (-1 + bp) * (-1 + (volume / V) ** (1 / 3))) / 2)
        )
    )
    energy_difference = energy_eos_points - energy

    b2p = (19 - 18 * bp - 9 * bp**2) / (36 * B)
    eos_parameters = [V, E0, 0, B * 160.2176621, bp, b2p / 160.2176621]

    fitting_error = np.array(
        [math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))]
    )
    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10**4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = (
        160.2176621
        * (-3 * B * (-1 + (volume_range / V) ** (1 / 3)))
        / (
            np.exp((3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2)
            * (volume_range / V) ** (2 / 3)
        )
    )
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos


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
    eos_index = 9
    volume_range = np.linspace(min(volume), max(volume), 1000)

    Data = np.vstack((volume, energy))
    Data = Data.T

    [results, volume_range, energy_eos, pressure_eos] = mBM4(volume, energy)
    xini = [results[1], results[2], results[4] / 160.2176621, results[5]]
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
    energy_eos_points = (
        a + b * np.exp(d * volume ** (1 / 3)) + c * np.exp(2 * d * volume ** (1 / 3))
    )
    energy_difference = energy_eos_points - energy

    b2p = (5 - 5 * bp - 2 * bp**2) / (9 * B)
    eos_parameters = [V, E0, 0, B * 160.2176621, bp, b2p / 160.2176621]

    fitting_error = np.array(
        [math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))]
    )
    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10**4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = (
        -160.2176621
        * (
            d
            * np.exp(d * volume_range ** (1 / 3))
            * (b + 2 * c * np.exp(d * volume_range ** (1 / 3)))
        )
        / (3 * volume_range ** (2 / 3))
    )
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos


def fit_to_all_eos(df):
    eos_df = pd.DataFrame(
        columns=[
            "config",
            "eos_name",
            "results",
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

        for eos_func in eos_functions:
            results, volume_range, energy_eos, pressure_eos = eos_func(
                volumes, energies
            )
            energy_eos = energy_eos[1:]
            pressure_eos = pressure_eos[1:]
            eos_name = eos_func.__name__
            eos_df = pd.concat(
                [
                    eos_df,
                    pd.DataFrame(
                        [
                            [
                                config,
                                eos_name,
                                results,
                                volume_range,
                                energy_eos,
                                pressure_eos,
                                number_of_atoms,
                            ]
                        ],
                        columns=[
                            "config",
                            "eos_name",
                            "results",
                            "volumes",
                            "energies",
                            "pressures",
                            "number_of_atoms",
                        ],
                    ),
                ],
                ignore_index=True,
            )
    return eos_df


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


def convert_input_files_to_df(input_files, left_col, right_col):
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


"""
the selection_dict should be a dictionary with the following format:
    selection_dict = {10: [0,1,2,3],
                        11: [0,1,2,3],
                        12: [0,1,2,3],
                        ...}
where the keys are the config # and the values are the volumes 0=lowest
volume, 1=second lowest volume, etc.
"""


def select_data(df, selection_dict):
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


"""
data may be a single pandas data frame or a list of pandas data frames
data may also be a list of input_file names as strings ex:
    ['str_0', 'str_1', 'str_2', ...]

Not sure if a list of dataframe will actually work. the function simply
concats them all together
"""

"""
df is a data frame with columns ['config', '# of ion', 'volume', 'tot']
breaks if missing these columns
"""


def plot_mv(df, show_fig=True):
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
    elif isinstance(data, list) and all(type(elem) == type(data[0]) for elem in data): #check if each elem of the list is the same type as the zeroth element
        if isinstance(data[0], pd.DataFrame):
            df = pd.concat(data, ignore_index=True)
        elif isinstance(data[0], str):
            df = convert_input_files_to_df(data, left_col, right_col)
            print(df)
    else:
        raise ValueError("data must be a pandas DataFrame or a list of pandas DataFrames or a list of input_file names as strings")
    
    # create a data frame with the eos fits for each config
    if eos_fitting != None:
        eos_df = fit_to_all_eos(df)

    # plot the data
    unique_configs = df['config'].unique()
    fig = go.Figure()
    for config in df['config'].unique():
        config_df = df[df['config'] == config]
        if per_atom == False:
            x = config_df['volume']
            y = config_df['energy']
        elif per_atom == True:
            x = config_df['volume'] / config_df['number_of_atoms']
            y = config_df['energy'] / config_df['number_of_atoms']
        else:
            print('per_atom must be True or False')
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                colorscale='Viridis',
                symbol='circle',  # Use the symbol from the dictionary
                opacity=0.5, 
                size=7,  
                line=dict(width=1, color='DarkSlateGrey')  # Add an outline
            ),
            legendgroup='EOS',
            name=f'Config {config}'
        ))
    if per_atom == False:
        fig.update_layout(title='E-V', xaxis_title='Volume [Å^3]', yaxis_title='Energy (eV)', template='plotly_white')
    elif per_atom == True:
        fig.update_layout(title='E-V', xaxis_title='Volume [Å^3/atom]', yaxis_title='Energy (eV/atom)', template='plotly_white')
    else:
        print('per_atom must be True or False')
    
    # loop over configs in the eos data frame and plot the eos fits
    if eos_fitting != None:
        for config in eos_df['config'].unique():
            eos_config_df = eos_df[eos_df['config'] == config]
            if eos_fitting in eos_config_df['eos_name'].unique():
                eos_name_df = eos_config_df[eos_config_df['eos_name'] == eos_fitting]
                if per_atom == False:
                    fig.add_trace(go.Scatter(x=eos_name_df['volumes'].values[0], y=eos_name_df['energies'].values[0],
                                            mode='lines', name=f'{eos_fitting} fit', line=dict(width=1), legendgroup='data'))
                elif per_atom == True:
                    fig.add_trace(go.Scatter(x=eos_name_df['volumes'].values[0] / eos_name_df['number_of_atoms'].values[0][0], y=eos_name_df['energies'].values[0] / eos_name_df['number_of_atoms'].values[0][0],
                                            mode='lines', name=f'{eos_fitting} fit', line=dict(width=1), legendgroup='data'))
                # plot the minimum energy data point for each config from the fitting equation
                if highlight_minimum == True:
                    min_energy = min(eos_name_df['energies'].values[0])
                    volume_at_min_energy = eos_name_df['volumes'].values[0][np.where(eos_name_df['energies'].values[0] == min_energy)[0][0]]
                    if per_atom == False:
                        fig.add_trace(go.Scatter(x=[volume_at_min_energy], y=[min_energy], mode='markers',
                                                name=f'{eos_fitting} min energy', marker=dict(color='black',
                                                size=10, symbol='cross'), legendgroup='minimum'))
                    elif per_atom == True:
                        fig.add_trace(go.Scatter(x=[volume_at_min_energy / eos_name_df['number_of_atoms'].values[0][0]], y=[min_energy / eos_name_df['number_of_atoms'].values[0][0]], mode='markers',
                                                name=f'{eos_fitting} min energy', marker=dict(color='black',
                                                size=10, symbol='cross'), legendgroup='minimum'))
                    else:
                        print('per_atom must be True or False')
                elif highlight_minimum == False:
                    pass
                else:
                    print('highlight_minimum must be True or False')
            elif eos_fitting == 'all':
                for eos_name in eos_config_df['eos_name'].unique():
                    eos_name_df = eos_config_df[eos_config_df['eos_name'] == eos_name]
                    if per_atom == False:
                        fig.add_trace(go.Scatter(x=eos_name_df['volumes'].values[0], y=eos_name_df['energies'].values[0],
                                                mode='lines', name=f'{eos_name} fit', line=dict(width=1), legendgroup='eos'))
                    elif per_atom == True:
                        fig.add_trace(go.Scatter(x=eos_name_df['volumes'].values[0] / eos_name_df['number_of_atoms'].values[0][0], y=eos_name_df['energies'].values[0] / eos_name_df['number_of_atoms'].values[0][0],
                                                mode='lines', name=f'{eos_name} fit', line=dict(width=1), legendgroup='eos'))
                    else:
                        print('per_atom must be True or False')
                    # plot the minimum energy data point for each config from the fitting equation
                    if highlight_minimum == True:
                        min_energy = min(eos_name_df['energies'].values[0])
                        volume_at_min_energy = eos_name_df['volumes'].values[0][np.where(eos_name_df['energies'].values[0] == min_energy)[0][0]]
                        if per_atom == False:
                            fig.add_trace(go.Scatter(x=[volume_at_min_energy], y=[min_energy], mode='markers',
                                                    name=f'{eos_name} min energy', marker=dict(color='black',
                                                    size=8, symbol='cross'), legendgroup='minimum'))
                        elif per_atom == True:
                            fig.add_trace(go.Scatter(x=[volume_at_min_energy / eos_name_df['number_of_atoms'].values[0][0]], y=[min_energy / eos_name_df['number_of_atoms'].values[0][0]], mode='markers',
                                                    name=f'{eos_name} min energy', marker=dict(color='black',
                                                    size=8, symbol='cross'), legendgroup='minimum'))
                        else:
                            print('per_atom must be True or False')
            elif eos_fitting == None:
                pass
            else:
                print(f"Warning: eos_fitting '{eos_fitting}' not found in eos_df. Skipping.")
    fig.update_layout(plot_bgcolor='white',
                          width=800,
                          height=600,
                          margin=dict(l=80, r=30, t=80, b=80)
                          )
    fig.update_yaxes(showline=True,  # add line at x=0
                            linecolor='black',
                            linewidth=2.4,
                            ticks='inside',
                            mirror='allticks',  # add ticks to top/right axes
                            tickwidth=2.4,
                            tickcolor='black',
                            showgrid=False
                            )
    fig.update_xaxes(showline=True,
                            showticklabels=True,
                            linecolor='black',
                            linewidth=2.4,
                            ticks='inside',
                            mirror='allticks',
                            tickwidth=2.4,
                            tickcolor='black',
                            showgrid=False
                            )
    if show_fig:
        fig.show()
    return fig


def assign_colors_to_configs(df, alpha=1):
    
    unique_configs = df["config"].unique()
    colors = get_colors(len(unique_configs))
    colors = [f'rgba({color[0]}, {color[1]}, {color[2]}, {alpha})' for color in colors]
    config_colors = {config: colors[i % len(colors)] for i, config in enumerate(unique_configs)}
    return config_colors

def assign_marker_symbols_to_configs(df):
    unique_configs = df["config"].unique()
    symbols = ['circle', 'square', 'diamond', 'x',
               'triangle-up', 'triangle-down', 'triangle-left',
               'triangle-right', 'pentagon', 'hexagon', 'octagon',
               'star', 'hexagram', 'star-triangle-up',
               'star-triangle-down', 'star-square', 'star-diamond',
               'diamond-tall', 'diamond-wide', 'hourglass',
               'bowtie', 'circle-cross', 'circle-x',
               'square-cross', 'square-x', 'diamond-cross',
               'diamond-x', 'cross-thin', 'x-thin', 'asterisk',
               'hash', 'y-up', 'y-down', 'y-left', 'y-right',
               'line-ew', 'line-ns', 'line-ne', 'line-nw']
    config_symbols = {config: symbols[i % len(symbols)] for i, config in enumerate(unique_configs)}
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
    marker_alpha=1
):
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

    # assign colors and symbols
    config_colors = assign_colors_to_configs(df, alpha=marker_alpha)
    config_symbols = assign_marker_symbols_to_configs(df)
        
    # plot the data
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
                    size=10,
                    color=config_colors[config],
                    symbol=config_symbols[config]
                ),
                legendgroup="EOS",
                name=f"Config {config}",
            )
        )
    if per_atom == False:
        fig.update_layout(
            xaxis_title=dict(text=r"$\text{Volume } (\text{Å}^3)$", font=dict(color= 'rgb(0,0,0)')),
            yaxis_title=dict(text=r"$\text{Energy (eV)}$", font=dict(color= 'rgb(0,0,0)')),
            template="plotly_white",
        )
    elif per_atom == True:
        fig.update_layout(
            xaxis_title=dict(text=r"$\text{Volume } (\text{Å}^3 \text{/atom)}$", font=dict(color= 'rgb(0,0,0)')),
            yaxis_title=dict(text=r"$\text{Energy (eV/atom)}$", font=dict(color= 'rgb(0,0,0)')),
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
                            line=dict(width=1.75, color= config_colors[config]),
                            legendgroup="data",
                            showlegend=False
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
                            line=dict(width=1.75, color= config_colors[config]),
                            legendgroup="data",
                            showlegend=False
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
                                showlegend=False
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
                                showlegend=False
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
                                showlegend=False
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
                                showlegend=False
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
                                    showlegend=False
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
                                    showlegend=False
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
    fig.update_layout(plot_bgcolor='white',
                            width=660,
                            height=600,
                            margin=dict(l=80, r=30, t=80, b=80)
                            )
    fig.update_yaxes(showline=True,  # add line at x=0
                            linecolor='black',
                            linewidth=2.4,
                            ticks='inside',
                            mirror='allticks',  # add ticks to top/right axes
                            tickwidth=2.4,
                            tickcolor='black',
                            showgrid=False,
                            tickfont=dict(color='rgb(0,0,0)', size=16)
                            )
    fig.update_xaxes(showline=True,
                            showticklabels=True,
                            linecolor='black',
                            linewidth=2.4,
                            ticks='inside',
                            mirror='allticks',
                            tickwidth=2.4,
                            tickcolor='black',
                            showgrid=False,
                            tickfont=dict(color='rgb(0,0,0)', size=16)
                            )
    if title != None:
        fig.update_layout(title=dict(text=title, font=dict(color='rgb(0,0,0)', size=24)))
    if show_fig:
        fig.show()
    return fig

def plot_energy_difference(
    df,
    reference_config,
    per_atom=False,
    show_fig=True,
    convert_to_mev=False,
    title=None,
    marker_alpha=1):
    """
    Takes a dataframe and plots the energy difference from a 
    reference configuration within the dataframe vs volume.
    
    Utilizes plot_ev() for the actual plotting.
    """
    df_list = []
    for config in df['config'].unique():
        df_list.append(df[df['config'] == config].reset_index(drop=True))
        if config == reference_config:
            reference_df = df[df['config'] == config].reset_index(drop=True)

    # Subtract reference energies
    missing_volumes = []
    for df_el in df_list:
        for i, row in df_el.iterrows():
            try:
                reference_energy = reference_df[reference_df['volume'] == row['volume']]['energy'].values[0]
                df_el.at[i, 'energy'] = row['energy'] - reference_energy
            except Exception as e:
                missing_volumes.append((row['config'],row['volume']))
    if missing_volumes:
        print(f"Warning: Missing volumes for configurations: {missing_volumes}")
        
    energy_difference_df = pd.concat(df_list)
    
    # convert to meV
    if convert_to_mev == True:
        energy_difference_df['energy'] *= 1000
    
    # plot energy difference vs volume
    fig = plot_ev(energy_difference_df,
                  eos_fitting=None,
                  per_atom=per_atom,
                  show_fig=False,
                  marker_alpha=marker_alpha)
    if convert_to_mev and not per_atom:
        fig.update_layout(yaxis_title=dict(text=r"$\Delta \text{E (meV)}$", font=dict(color='rgb(0,0,0)')))
    elif not convert_to_mev and not per_atom:
        fig.update_layout(yaxis_title=dict(text=r"$\Delta \text{E (eV)}$", font=dict(color='rgb(0,0,0)')))
    elif convert_to_mev and per_atom:
        fig.update_layout(yaxis_title=dict(text=r"$\Delta \text{E (meV/atom)}$", font=dict(color='rgb(0,0,0)')))
    elif not convert_to_mev and per_atom:
        fig.update_layout(yaxis_title=dict(text=r"$\Delta \text{E (eV/atom)}$"), font=dict(color='rgb(0,0,0)'))
        
    if title != None:
        fig.update_layout(title=dict(text=title, font=dict(color='rgb(0,0,0)', size=24)))

    if show_fig:
        fig.show()
    return fig

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
