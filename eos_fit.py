"""
Shun-Li Shang wrote the original MATLAB code for EOS fitting.
Hui Sun converted the MATLAB code to python code.
Nigel Hew modified the python code to make it more user-friendly and added more functions.
Luke Myers performed additional modifications to simplify the plotting functions, and
include the ability to use dataframe inputs and plotly.
"""

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
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import leastsq
import os
import glob
import shutil
import pandas as pd
import numpy as np
import plotly.express as px
import json
import sys
import os
import plotly.graph_objects as go

def mBM4(volume, energy):
    eos_index = 1
    volume_range = np.linspace(min(volume), max(volume), 1000)

    AA = np.vstack((np.ones(np.shape(volume)), volume ** (-1 / 3), volume ** (-2 / 3), volume ** (-1)))  # (nx4)
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(energy)  # (4x1) = (4xn)*(nx1), solve by pseudo-inversion: Ax=b
    a = xx[0]
    b = xx[1]
    c = xx[2]
    d = xx[3]
    e = 0.0

    energy_eos = a + b * (volume_range) ** (-1 / 3) + c * (volume_range) ** (-2 / 3) + d * (volume_range) ** (
        -1) + e * (volume_range) ** (-4 / 3)
    energy_eos_points = a + b * (volume) ** (-1 / 3) + c * (volume) ** (-2 / 3) + d * (volume) ** (-1) + e * (
        volume) ** (-4 / 3)
    energy_difference = energy_eos_points - energy

    V = 4 * c ** 3 - 9 * b * c * d + np.sqrt((c ** 2 - 3 * b * d) * (4 * c ** 2 - 3 * b * d) ** 2)
    V = -V / b ** 3

    P = (4 * e) / (3 * V ** (7 / 3)) + d / V ** 2 + (2 * c) / (3 * V ** (5 / 3)) + b / (3 * V ** (4 / 3))
    P = P * 160.2176621
    B = ((28 * e) / (9 * V ** (10 / 3)) + (2 * d) / V ** 3 + (10 * c) / (9 * V ** (8 / 3)) + (4 * b) / (
            9 * V ** (7 / 3))) * V
    B = B * 160.2176621
    BP = (98 * e + 54 * d * V ** (1 / 3) + 25 * c * V ** (2 / 3) + 8 * b * V) / (
            42 * e + 27 * d * V ** (1 / 3) + 15 * c * V ** (2 / 3) + 6 * b * V)
    B2P = (V ** (8 / 3) * (9 * d * (14 * e + 5 * c * V ** (2 / 3) + 8 * b * V) + 2 * V ** (1 / 3) * (
            126 * b * e * V ** (1 / 3) + 5 * c * (28 * e + b * V)))) / (
                    2 * (14 * e + 9 * d * V ** (1 / 3) + 5 * c * V ** (2 / 3) + 2 * b * V) ** 3)
    B2P = B2P / 160.2176621
    E0 = a + b * V ** (-1 / 3) + c * V ** (-2 / 3) + d * V ** (-1) + e * V ** (-4 / 3)
    eos_parameters = [V, E0, P, B, BP, B2P]

    fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = ((4 * e) / (3 * volume_range ** (7 / 3)) + d / volume_range ** 2 + (2 * c) / (
            3 * volume_range ** (5 / 3)) + b / (3 * volume_range ** (4 / 3))) * 160.2176621
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    xini = [eos_parameters[0], eos_parameters[1], eos_parameters[3] / 160.2176621,
            eos_parameters[4]]  # used for eos_index = 7 and 8
    return results, volume_range, energy_eos, pressure_eos

def mBM5(volume, energy):
    eos_index = 2
    volume_range = np.linspace(min(volume), max(volume), 1000)

    AA = np.vstack(
        (np.ones(np.shape(volume)), volume ** (-1 / 3), volume ** (-2 / 3), volume ** (-1), volume ** (-4 / 3)))
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
    a = xx[0]
    b = xx[1]
    c = xx[2]
    d = xx[3]
    e = xx[4]

    energy_eos = a + b * (volume_range) ** (-1 / 3) + c * (volume_range) ** (-2 / 3) + d * (volume_range) ** (
        -1) + e * (volume_range) ** (-4 / 3)
    energy_eos_points = a + b * (volume) ** (-1 / 3) + c * (volume) ** (-2 / 3) + d * (volume) ** (-1) + e * (
        volume) ** (-4 / 3)
    energy_difference = energy_eos_points - energy

    func = lambda volume_range: ((4 * e) / (3 * volume_range ** (7 / 3)) + d / volume_range ** 2 + (2 * c) / (
            3 * volume_range ** (5 / 3)) + b / (
                                            3 * volume_range ** (4 / 3))) * 160.2176621
    V = fsolve(func, np.mean(volume))
    P = (4 * e) / (3 * V ** (7 / 3)) + d / V ** 2 + (2 * c) / (3 * V ** (5 / 3)) + b / (3 * V ** (4 / 3))
    P = P * 160.2176621
    B = ((28 * e) / (9 * V ** (10 / 3)) + (2 * d) / V ** 3 + (10 * c) / (9 * V ** (8 / 3)) + (4 * b) / (
            9 * V ** (7 / 3))) * V
    B = B * 160.2176621
    BP = (98 * e + 54 * d * V ** (1 / 3) + 25 * c * V ** (2 / 3) + 8 * b * V) / (
            42 * e + 27 * d * V ** (1 / 3) + 15 * c * V ** (2 / 3) + 6 * b * V)
    B2P = (V ** (8 / 3) * (9 * d * (14 * e + 5 * c * V ** (2 / 3) + 8 * b * V) + 2 * V ** (1 / 3) * (
            126 * b * e * V ** (1 / 3) + 5 * c * (28 * e + b * V)))) / (
                    2 * (14 * e + 9 * d * V ** (1 / 3) + 5 * c * V ** (2 / 3) + 2 * b * V) ** 3)
    B2P = B2P / 160.2176621
    E0 = a + b * V ** (-1 / 3) + c * V ** (-2 / 3) + d * V ** (-1) + e * V ** (-4 / 3)
    eos_parameters = [V, E0, P, B, BP, B2P]

    fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
    results = np.concatenate(([eos_index], *eos_parameters, fitting_error * (10 ** 4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = ((4 * e) / (3 * volume_range ** (7 / 3)) + d / volume_range ** 2 + (2 * c) / (
            3 * volume_range ** (5 / 3)) + b / (
                            3 * volume_range ** (4 / 3))) * 160.2176621
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos

def BM4(volume, energy):
    eos_index = 3
    volume_range = np.linspace(min(volume), max(volume), 1000)

    AA = np.vstack((np.ones(np.shape(volume)), volume ** (-2 / 3), volume ** (-4 / 3), volume ** (-2)))
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
    a = xx[0]
    b = xx[1]
    c = xx[2]
    d = xx[3]
    e = 0.0

    energy_eos = a + b * (volume_range) ** (-2 / 3) + c * (volume_range) ** (-4 / 3) + d * (volume_range) ** (
        -2) + e * (volume_range) ** (-8 / 3)
    energy_eos_points = a + b * (volume) ** (-2 / 3) + c * (volume) ** (-4 / 3) + d * (volume) ** (-2) + e * (
        volume) ** (-8 / 3)
    energy_difference = energy_eos_points - energy

    V = math.sqrt(-((4 * c ** 3 - 9 * b * c * d + math.sqrt(
        (c ** 2 - 3 * b * d) * (4 * c ** 2 - 3 * b * d) ** 2)) / b ** 3))
    P = (8 * e) / (3 * V ** (11 / 3)) + (2 * d) / V ** 3 + (4 * c) / (3 * V ** (7 / 3)) + (2 * b) / (
            3 * V ** (5 / 3))
    P = P * 160.2176621
    B = (2 * (44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2)) / (9 * V ** (11 / 3))
    B = B * 160.2176621
    BP = (484 * e + 243 * d * V ** (2 / 3) + 98 * c * V ** (4 / 3) + 25 * b * V ** 2) / (
            132 * e + 81 * d * V ** (2 / 3) + 42 * c * V ** (4 / 3) + 15 * b * V ** 2);
    B2P = (4 * V ** (13 / 3) * (27 * d * (22 * e + 7 * c * V ** (4 / 3) + 10 * b * V ** 2) + V ** (2 / 3) * (
            990 * b * e * V ** (2 / 3) + 7 * c * (176 * e + 5 * b * V ** 2)))) / (
                    44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2) ** 3
    B2P = B2P / 160.2176621
    E0 = a + e / V ** (8 / 3) + d / V ** 2 + c / V ** (4 / 3) + b / V ** (2 / 3)
    eos_parameters = [V, E0, P, B, BP, B2P]

    fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])

    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = ((8 * e) / (3 * volume_range ** (11 / 3)) + (2 * d) / volume_range ** 3 + (4 * c) / (
            3 * volume_range ** (7 / 3)) + (2 * b) / (
                            3 * volume_range ** (5 / 3))) * 160.2176621
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos

def BM5(volume, energy):
    eos_index = 4
    volume_range = np.linspace(min(volume), max(volume), 1000)

    AA = np.vstack(
        (np.ones(np.shape(volume)), volume ** (-2 / 3), volume ** (-4 / 3), volume ** (-2), volume ** (-8 / 3)))
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
    a = xx[0]
    b = xx[1]
    c = xx[2]
    d = xx[3]
    e = xx[4]

    energy_eos = a + b * (volume_range) ** (-2 / 3) + c * (volume_range) ** (-4 / 3) + d * (volume_range) ** (
        -2) + e * (volume_range) ** (-8 / 3)
    energy_eos_points = a + b * (volume) ** (-2 / 3) + c * (volume) ** (-4 / 3) + d * (volume) ** (-2) + e * (
        volume) ** (-8 / 3)
    energy_difference = energy_eos_points - energy

    func = lambda volume_range: ((8 * e) / (3 * volume_range ** (11 / 3)) + (2 * d) / volume_range ** 3 + (
            4 * c) / (3 * volume_range ** (7 / 3)) + (2 * b) / (3 * volume_range ** (5 / 3))) * 160.2176621
    V = fsolve(func, np.mean(volume))
    P = (8 * e) / (3 * V ** (11 / 3)) + (2 * d) / V ** 3 + (4 * c) / (3 * V ** (7 / 3)) + (2 * b) / (
            3 * V ** (5 / 3))
    P = P * 160.2176621
    B = (2 * (44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2)) / (9 * V ** (11 / 3))
    B = B * 160.2176621
    BP = (484 * e + 243 * d * V ** (2 / 3) + 98 * c * V ** (4 / 3) + 25 * b * V ** 2) / (
            132 * e + 81 * d * V ** (2 / 3) + 42 * c * V ** (4 / 3) + 15 * b * V ** 2);
    B2P = (4 * V ** (13 / 3) * (27 * d * (22 * e + 7 * c * V ** (4 / 3) + 10 * b * V ** 2) + V ** (2 / 3) * (
            990 * b * e * V ** (2 / 3) + 7 * c * (176 * e + 5 * b * V ** 2)))) / (
                    44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2) ** 3
    B2P = B2P / 160.2176621
    E0 = a + e / V ** (8 / 3) + d / V ** 2 + c / V ** (4 / 3) + b / V ** (2 / 3)
    eos_parameters = [V, E0, P, B, BP, B2P]

    fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
    results = np.concatenate(([eos_index], *eos_parameters, fitting_error * (10 ** 4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = ((8 * e) / (3 * volume_range ** (11 / 3)) + (2 * d) / volume_range ** 3 + (4 * c) / (
            3 * volume_range ** (7 / 3)) + (2 * b) / (3 * volume_range ** (5 / 3))) * 160.2176621
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos

def LOG4(volume, energy):
    eos_index = 5
    volume_range = np.linspace(min(volume), max(volume), 1000)

    AA = np.vstack((np.ones(np.shape(volume)), np.log(volume), np.log(volume) ** 2, np.log(volume) ** 3))
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
    a = xx[0]
    b = xx[1]
    c = xx[2]
    d = xx[3]
    e = 0.0

    energy_eos = a + b * np.log(volume_range) + c * np.log(volume_range) ** 2 + d * np.log(
        volume_range) ** 3 + e * np.log(volume_range) ** 4
    energy_eos_points = a + b * np.log(volume) + c * np.log(volume) ** 2 + d * np.log(volume) ** 3 + e * np.log(
        volume) ** 4
    energy_difference = energy_eos_points - energy

    func = lambda volume_range: (-(
            (b + 2 * c * math.log(volume_range) + 3 * d * math.log(volume_range) ** 2 + 4 * e * math.log(
                volume_range) ** 3) / volume_range)) * 160.2176621
    V = fsolve(func, np.mean(volume))
    V = np.mean(V)
    P = -((b + 2 * c * math.log(V) + 3 * d * math.log(V) ** 2 + 4 * e * math.log(V) ** 3) / V)
    P = np.mean(P)
    P = P * 160.2176621
    B = -((b - 2 * c + 2 * (c - 3 * d) * math.log(V) + 3 * (d - 4 * e) * math.log(V) ** 2 + 4 * e * math.log(
        V) ** 3) / V)
    B = np.mean(B)
    B = B * 160.2176621
    BP = (b - 4 * c + 6 * d + 2 * (c - 6 * d + 12 * e) * math.log(V) + 3 * (d - 8 * e) * math.log(
        V) ** 2 + 4 * e * math.log(V) ** 3) / (
                    b - 2 * c + 2 * (c - 3 * d) * math.log(V) + 3 * (d - 4 * e) * math.log(
                V) ** 2 + 4 * e * math.log(V) ** 3)
    B2P = (2 * V * (2 * c ** 2 - 3 * b * d + 18 * d ** 2 + 12 * b * e - 6 * c * (d + 4 * e) + 6 * (
            c * d - 3 * d ** 2 - 2 * b * e + 12 * d * e) * math.log(V) + 9 * (d - 4 * e) ** 2 * math.log(
        V) ** 2 + 24 * (d - 4 * e) * e * math.log(V) ** 3 + 24 * e ** 2 * math.log(V) ** 4)) / (
                    b - 2 * c + 2 * (c - 3 * d) * math.log(V) + 3 * (d - 4 * e) * math.log(
                V) ** 2 + 4 * e * math.log(V) ** 3) ** 3
    B2P = np.mean(B2P)
    B2P = B2P / 160.2176621
    E0 = a + b * math.log(V) + c * math.log(V) ** 2 + d * math.log(V) ** 3 + e * math.log(V) ** 4
    eos_parameters = [V, E0, P, B, BP, B2P]

    fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = (-((b + 2 * c * np.log(volume_range) + 3 * d * np.log(volume_range) ** 2 + 4 * e * np.log(
        volume_range) ** 3) / volume_range)) * 160.2176621
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos

def LOG5(volume, energy):
    eos_index = 6
    volume_range = np.linspace(min(volume), max(volume), 1000)

    AA = np.vstack(
        (np.ones(np.shape(volume)), np.log(volume), np.log(volume) ** 2, np.log(volume) ** 3, np.log(volume) ** 4))
    AA = AA.T
    xx1 = np.linalg.pinv(AA)
    xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
    a = xx[0]
    b = xx[1]
    c = xx[2]
    d = xx[3]
    e = xx[4]

    energy_eos = a + b * np.log(volume_range) + c * np.log(volume_range) ** 2 + d * np.log(
        volume_range) ** 3 + e * np.log(volume_range) ** 4
    energy_eos_points = a + b * np.log(volume) + c * np.log(volume) ** 2 + d * np.log(volume) ** 3 + e * np.log(
        volume) ** 4
    energy_difference = energy_eos_points - energy

    func = lambda volume_range: (-(
            (b + 2 * c * math.log(volume_range) + 3 * d * math.log(volume_range) ** 2 + 4 * e * math.log(
                volume_range) ** 3) / volume_range)) * 160.2176621
    V = fsolve(func, np.mean(volume))
    V = np.mean(V)
    P = -((b + 2 * c * math.log(V) + 3 * d * math.log(V) ** 2 + 4 * e * math.log(V) ** 3) / V)
    P = np.mean(P)
    P = P * 160.2176621
    B = -((b - 2 * c + 2 * (c - 3 * d) * math.log(V) + 3 * (d - 4 * e) * math.log(V) ** 2 + 4 * e * math.log(
        V) ** 3) / V)
    B = np.mean(B)
    B = B * 160.2176621
    BP = (b - 4 * c + 6 * d + 2 * (c - 6 * d + 12 * e) * math.log(V) + 3 * (d - 8 * e) * math.log(
        V) ** 2 + 4 * e * math.log(V) ** 3) / (
                    b - 2 * c + 2 * (c - 3 * d) * math.log(V) + 3 * (d - 4 * e) * math.log(
                V) ** 2 + 4 * e * math.log(V) ** 3)
    B2P = (2 * V * (2 * c ** 2 - 3 * b * d + 18 * d ** 2 + 12 * b * e - 6 * c * (d + 4 * e) + 6 * (
            c * d - 3 * d ** 2 - 2 * b * e + 12 * d * e) * math.log(V) + 9 * (d - 4 * e) ** 2 * math.log(
        V) ** 2 + 24 * (d - 4 * e) * e * math.log(V) ** 3 + 24 * e ** 2 * math.log(V) ** 4)) / (
                    b - 2 * c + 2 * (c - 3 * d) * math.log(V) + 3 * (d - 4 * e) * math.log(
                V) ** 2 + 4 * e * math.log(V) ** 3) ** 3
    B2P = np.mean(B2P)
    B2P = B2P / 160.2176621
    E0 = a + b * math.log(V) + c * math.log(V) ** 2 + d * math.log(V) ** 3 + e * math.log(V) ** 4
    eos_parameters = [V, E0, P, B, BP, B2P]

    fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = (-((b + 2 * c * np.log(volume_range) + 3 * d * np.log(volume_range) ** 2 + 4 * e * np.log(
        volume_range) ** 3) / volume_range)) * 160.2176621
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
    eng = E0 - (B * V) / (-1 + bp) + (B * (1 + (V / volume_range) ** bp / (-1 + bp)) * volume_range) / bp
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

    energy_eos = E0 - (B * V) / (-1 + bp) + (B * (1 + (V / volume_range) ** bp / (-1 + bp)) * volume_range) / bp
    energy_eos_points = (E0 - (B * V) / (-1 + bp) + (B * (1 + (V / volume) ** bp / (-1 + bp)) * volume) / bp)
    energy_difference = energy_eos_points - energy
    eos_parameters = [V, E0, 0, B * 160.2176621, bp, 0]

    fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
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
    eng = E0 + (4 * B * V) / (-1 + bp) ** 2 - (
            4 * B * V * (1 + (3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2)) / (
                    (-1 + bp) ** 2 * np.exp((3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2))
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

    energy_eos = E0 + (4 * B * V) / (-1 + bp) ** 2 - (
            4 * B * V * (1 + (3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2)) / (
                            (-1 + bp) ** 2 * np.exp((3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2))
    energy_eos_points = E0 + (4 * B * V) / (-1 + bp) ** 2 - (
            4 * B * V * (1 + (3 * (-1 + bp) * (-1 + (volume / V) ** (1 / 3))) / 2)) / (
                                (-1 + bp) ** 2 * np.exp((3 * (-1 + bp) * (-1 + (volume / V) ** (1 / 3))) / 2))
    energy_difference = energy_eos_points - energy

    b2p = (19 - 18 * bp - 9 * bp ** 2) / (36 * B)
    eos_parameters = [V, E0, 0, B * 160.2176621, bp, b2p / 160.2176621]

    fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = 160.2176621 * (-3 * B * (-1 + (volume_range / V) ** (1 / 3))) / (
            np.exp((3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2) * (volume_range / V) ** (2 / 3))
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
    eng = a + b * np.exp(d * volume_range ** (1 / 3)) + c * np.exp(2 * d * volume_range ** (1 / 3))
    return eng - y

def morse(volume, energy):
    eos_index = 9
    volume_range = np.linspace(min(volume), max(volume), 1000)

    Data = np.vstack((volume, energy))
    Data = Data.T

    [results, volume_range, energy_eos, pressure_eos] = mBM4(volume, energy)
    xini = [results[1], results[2], results[4] / 160.2176621,
            results[5]]
    [xout, resnorm] = leastsq(morse_eq, xini, Data)
    V = xout[0]
    E0 = xout[1]
    B = xout[2]
    bp = xout[3]

    a = E0 + (9 * B * V) / (2 * (-1 + bp) ** 2)
    b = (-9 * B * np.exp(-1 + bp) * V) / (-1 + bp) ** 2
    c = (9 * B * np.exp(-2 + 2 * bp) * V) / (2 * (-1 + bp) ** 2)
    d = (1 - bp) / V ** (1 / 3)

    energy_eos = a + b * np.exp(d * volume_range ** (1 / 3)) + c * np.exp(2 * d * volume_range ** (1 / 3))
    energy_eos_points = a + b * np.exp(d * volume ** (1 / 3)) + c * np.exp(2 * d * volume ** (1 / 3))
    energy_difference = energy_eos_points - energy

    b2p = (5 - 5 * bp - 2 * bp ** 2) / (9 * B)
    eos_parameters = [V, E0, 0, B * 160.2176621, bp, b2p / 160.2176621]

    fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
    results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
    np.set_printoptions(precision=4, suppress=True)

    pressure_eos = -160.2176621 * (
            d * np.exp(d * volume_range ** (1 / 3)) * (b + 2 * c * np.exp(d * volume_range ** (1 / 3)))) / (
                            3 * volume_range ** (2 / 3))
    energy_eos = np.concatenate(([eos_index], energy_eos))
    pressure_eos = np.concatenate(([eos_index], pressure_eos))

    return results, volume_range, energy_eos, pressure_eos

def fit_to_all_eos(df):
    eos_df = pd.DataFrame(columns=['config', 'eos_name', 'results', 'volumes', 'energies', 'pressures'])
    eos_functions = [mBM4, mBM5, BM4, BM5, LOG4, LOG5, murnaghan, vinet, morse]  # Add more EOS functions here
    
    for config in df['config'].unique():
        config_df = df[df['config'] == config]
        volumes = config_df['volume'].values
        energies = config_df['energy'].values
        
        for eos_func in eos_functions:
            results, volume_range, energy_eos, pressure_eos = eos_func(volumes, energies)
            energy_eos = energy_eos[1:]
            pressure_eos = pressure_eos[1:]
            eos_name = eos_func.__name__
            eos_df = pd.concat([eos_df, pd.DataFrame([[config, eos_name, results, volume_range, energy_eos, pressure_eos]], columns=['config', 'eos_name', 'results', 'volumes', 'energies', 'pressures'])], ignore_index=True)    
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
    df = pd.DataFrame(columns=['config', left_col, right_col])
    for input_file in input_files:
        config = os.path.splitext(os.path.basename(input_file))[0]
        if '_' in config:
            config = config.split('_')[-1]
        data = np.loadtxt(input_file)
        left_data = data[:, 0]
        right_data = data[:, 1]
        for i in range(len(left_data)):
            df.loc[len(df)] = {'config': config, left_col: left_data[i], right_col: right_data[i]}
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
    df['volume_rank'] = df.groupby('config')['volume'].rank(method='dense').astype(int)
    print(df)
    selected_data = pd.DataFrame()  # Initialize selected_data variable
    for config in selection_dict.keys():
        for volume_rank in selection_dict[config]:
            selected_data = pd.concat([selected_data, df[(df['config'] == config) & (df['volume_rank'] == volume_rank)]], ignore_index=True)
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
    fig = px.line(df,
                    x='volume',
                    y='tot',
                    color='# of ion',symbol='# of ion',
                    hover_data=['config', '# of ion', 'volume', 'tot'],
                    template='plotly_white')
    fig.update_layout(title='Mag-V',
                        xaxis_title='Volume [A^3]',
                        yaxis_title='Magnetic Moment [mu_B]')
    
    fig.update_yaxes(nticks=10)
    fig.update_xaxes(nticks=10)
    
    # Loop over each trace and update dash length
    for i, trace in enumerate(fig.data):
        dash_length = f"{2+(i+1)}px,{2+2*(i+1)}px"  # Dash length changes with each iteration
        fig.data[-i-1].update(mode='markers+lines',
                            marker=dict(size=8, line=dict(width=1), opacity=0.5),
                            line=dict(width=3, dash=dash_length))


    if show_fig:
        fig.show()
    return fig

def plot_ev(data, eos_fitting='mBM4', highlight_minimum=True ,show_fig=True, left_col='volume', right_col='energy'):
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
    eos_df = fit_to_all_eos(df)

    # plot the fitted equations
    fig = px.scatter(df, x='volume', y='energy', color='config', template='plotly_white')
    fig.update_layout(title='E-V', xaxis_title='Volume [Å^3]', yaxis_title='Energy (eV)')
    
    # loop over configs in the data frame
    for config in eos_df['config'].unique():
        eos_config_df = eos_df[eos_df['config'] == config]
        if eos_fitting in eos_config_df['eos_name'].unique():
            eos_name_df = eos_config_df[eos_config_df['eos_name'] == eos_fitting]
            fig.add_trace(go.Scatter(x=eos_name_df['volumes'].values[0], y=eos_name_df['energies'].values[0], mode='lines', name=f'{eos_fitting} fit', line=dict(width=1)))
            # plot the minimum energy data point for each config from the fitting equation
            if highlight_minimum == True:
                min_energy = min(eos_name_df['energies'].values[0])
                volume_at_min_energy = eos_name_df['volumes'].values[0][np.where(eos_name_df['energies'].values[0] == min_energy)[0][0]] #check this
                fig.add_trace(go.Scatter(x=[volume_at_min_energy], y=[min_energy], mode='markers', name=f'{eos_fitting} min energy', marker=dict(color='black', size=10, symbol='cross')))
            elif highlight_minimum == False:
                pass
            else:
                print('highlight_minimum must be True or False')
        elif eos_fitting == 'all':
            for eos_name in eos_config_df['eos_name'].unique():
                eos_name_df = eos_config_df[eos_config_df['eos_name'] == eos_name]
                fig.add_trace(go.Scatter(x=eos_name_df['volumes'].values[0], y=eos_name_df['energies'].values[0], mode='lines', name=f'{eos_name} fit', line=dict(width=1)))
                # plot the minimum energy data point for each config from the fitting equation
                if highlight_minimum == True:
                    min_energy = min(eos_name_df['energies'].values[0])
                    volume_at_min_energy = eos_name_df['volumes'].values[0][np.where(eos_name_df['energies'].values[0] == min_energy)[0][0]] #check this
                    fig.add_trace(go.Scatter(x=[volume_at_min_energy], y=[min_energy], mode='markers', name=f'{eos_name} min energy', marker=dict(color='black', size=8, symbol='cross')))
        elif eos_fitting == None:
            pass
        else:
            print(f"Warning: eos_fitting '{eos_fitting}' not found in eos_df. Skipping.")
    if show_fig:
        fig.show()
    return fig

def plot_pv():
    pass


def ev_fit(num_structures, *input_files, eos_index, plot_ev=True, plot_pv=False):
    # 1. Load the energy-volume data
    if num_structures == 'single':
        data = np.loadtxt(input_files[0])
        volume = data[:, 0]
        energy = data[:, 1]
        volume_range = np.linspace(min(volume), max(volume), 1000)
    if num_structures == 'multiple':
        volume = {}
        energy = {}
        volume_range_list = {}
        for input_file in input_files:
            volume[input_file] = np.loadtxt(input_file, usecols=(0,))
            energy[input_file] = np.loadtxt(input_file, usecols=(1,))
            volume_range_list[input_file] = np.linspace(min(volume[input_file]), max(volume[input_file]), 1000)

        # EOS fitting for multiple structures only supports one EOS at a time
        if len(eos_index) > 1:
            raise ValueError("eos_index for multiple structures should only have one value")

    # Remove any duplicates
    eos_index = list(set(eos_index))
    eos_index.sort()

    # Conversion factor: 1 eV/Å^3 = 160.2176621 GPa
    conv_factor = 160.2176621

    # 2. Use the selected EOS to fit the energy-volume data
    def mBM4(volume, energy):
        eos_index = 1

        AA = np.vstack((np.ones(np.shape(volume)), volume ** (-1 / 3), volume ** (-2 / 3), volume ** (-1)))  # (nx4)
        AA = AA.T
        xx1 = np.linalg.pinv(AA)
        xx = xx1.dot(energy)  # (4x1) = (4xn)*(nx1), solve by pseudo-inversion: Ax=b
        a = xx[0]
        b = xx[1]
        c = xx[2]
        d = xx[3]
        e = 0.0

        energy_eos = a + b * (volume_range) ** (-1 / 3) + c * (volume_range) ** (-2 / 3) + d * (volume_range) ** (
            -1) + e * (volume_range) ** (-4 / 3)
        energy_eos_points = a + b * (volume) ** (-1 / 3) + c * (volume) ** (-2 / 3) + d * (volume) ** (-1) + e * (
            volume) ** (-4 / 3)
        energy_difference = energy_eos_points - energy

        V = 4 * c ** 3 - 9 * b * c * d + np.sqrt((c ** 2 - 3 * b * d) * (4 * c ** 2 - 3 * b * d) ** 2)
        V = -V / b ** 3

        P = (4 * e) / (3 * V ** (7 / 3)) + d / V ** 2 + (2 * c) / (3 * V ** (5 / 3)) + b / (3 * V ** (4 / 3))
        P = P * conv_factor
        B = ((28 * e) / (9 * V ** (10 / 3)) + (2 * d) / V ** 3 + (10 * c) / (9 * V ** (8 / 3)) + (4 * b) / (
                9 * V ** (7 / 3))) * V
        B = B * conv_factor
        BP = (98 * e + 54 * d * V ** (1 / 3) + 25 * c * V ** (2 / 3) + 8 * b * V) / (
                42 * e + 27 * d * V ** (1 / 3) + 15 * c * V ** (2 / 3) + 6 * b * V)
        B2P = (V ** (8 / 3) * (9 * d * (14 * e + 5 * c * V ** (2 / 3) + 8 * b * V) + 2 * V ** (1 / 3) * (
                126 * b * e * V ** (1 / 3) + 5 * c * (28 * e + b * V)))) / (
                      2 * (14 * e + 9 * d * V ** (1 / 3) + 5 * c * V ** (2 / 3) + 2 * b * V) ** 3)
        B2P = B2P / conv_factor
        E0 = a + b * V ** (-1 / 3) + c * V ** (-2 / 3) + d * V ** (-1) + e * V ** (-4 / 3)
        eos_parameters = [V, E0, P, B, BP, B2P]

        fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
        results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
        np.set_printoptions(precision=4, suppress=True)

        pressure_eos = ((4 * e) / (3 * volume_range ** (7 / 3)) + d / volume_range ** 2 + (2 * c) / (
                3 * volume_range ** (5 / 3)) + b / (3 * volume_range ** (4 / 3))) * conv_factor
        energy_eos = np.concatenate(([eos_index], energy_eos))
        pressure_eos = np.concatenate(([eos_index], pressure_eos))

        xini = [eos_parameters[0], eos_parameters[1], eos_parameters[3] / conv_factor,
                eos_parameters[4]]  # used for eos_index = 7 and 8
        return results, energy_eos, pressure_eos

    def mBM5(volume, energy):
        eos_index = 2

        AA = np.vstack(
            (np.ones(np.shape(volume)), volume ** (-1 / 3), volume ** (-2 / 3), volume ** (-1), volume ** (-4 / 3)))
        AA = AA.T
        xx1 = np.linalg.pinv(AA)
        xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
        a = xx[0]
        b = xx[1]
        c = xx[2]
        d = xx[3]
        e = xx[4]

        energy_eos = a + b * (volume_range) ** (-1 / 3) + c * (volume_range) ** (-2 / 3) + d * (volume_range) ** (
            -1) + e * (volume_range) ** (-4 / 3)
        energy_eos_points = a + b * (volume) ** (-1 / 3) + c * (volume) ** (-2 / 3) + d * (volume) ** (-1) + e * (
            volume) ** (-4 / 3)
        energy_difference = energy_eos_points - energy

        func = lambda volume_range: ((4 * e) / (3 * volume_range ** (7 / 3)) + d / volume_range ** 2 + (2 * c) / (
                3 * volume_range ** (5 / 3)) + b / (
                                             3 * volume_range ** (4 / 3))) * conv_factor
        V = fsolve(func, np.mean(volume))
        P = (4 * e) / (3 * V ** (7 / 3)) + d / V ** 2 + (2 * c) / (3 * V ** (5 / 3)) + b / (3 * V ** (4 / 3))
        P = P * conv_factor
        B = ((28 * e) / (9 * V ** (10 / 3)) + (2 * d) / V ** 3 + (10 * c) / (9 * V ** (8 / 3)) + (4 * b) / (
                9 * V ** (7 / 3))) * V
        B = B * conv_factor
        BP = (98 * e + 54 * d * V ** (1 / 3) + 25 * c * V ** (2 / 3) + 8 * b * V) / (
                42 * e + 27 * d * V ** (1 / 3) + 15 * c * V ** (2 / 3) + 6 * b * V)
        B2P = (V ** (8 / 3) * (9 * d * (14 * e + 5 * c * V ** (2 / 3) + 8 * b * V) + 2 * V ** (1 / 3) * (
                126 * b * e * V ** (1 / 3) + 5 * c * (28 * e + b * V)))) / (
                      2 * (14 * e + 9 * d * V ** (1 / 3) + 5 * c * V ** (2 / 3) + 2 * b * V) ** 3)
        B2P = B2P / conv_factor
        E0 = a + b * V ** (-1 / 3) + c * V ** (-2 / 3) + d * V ** (-1) + e * V ** (-4 / 3)
        eos_parameters = [V, E0, P, B, BP, B2P]

        fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
        results = np.concatenate(([eos_index], *eos_parameters, fitting_error * (10 ** 4)))
        np.set_printoptions(precision=4, suppress=True)

        pressure_eos = ((4 * e) / (3 * volume_range ** (7 / 3)) + d / volume_range ** 2 + (2 * c) / (
                3 * volume_range ** (5 / 3)) + b / (
                                3 * volume_range ** (4 / 3))) * conv_factor
        energy_eos = np.concatenate(([eos_index], energy_eos))
        pressure_eos = np.concatenate(([eos_index], pressure_eos))

        return results, energy_eos, pressure_eos

    def BM4(volume, energy):
        eos_index = 3
        AA = np.vstack((np.ones(np.shape(volume)), volume ** (-2 / 3), volume ** (-4 / 3), volume ** (-2)))
        AA = AA.T
        xx1 = np.linalg.pinv(AA)
        xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
        a = xx[0]
        b = xx[1]
        c = xx[2]
        d = xx[3]
        e = 0.0

        energy_eos = a + b * (volume_range) ** (-2 / 3) + c * (volume_range) ** (-4 / 3) + d * (volume_range) ** (
            -2) + e * (volume_range) ** (-8 / 3)
        energy_eos_points = a + b * (volume) ** (-2 / 3) + c * (volume) ** (-4 / 3) + d * (volume) ** (-2) + e * (
            volume) ** (-8 / 3)
        energy_difference = energy_eos_points - energy

        V = math.sqrt(-((4 * c ** 3 - 9 * b * c * d + math.sqrt(
            (c ** 2 - 3 * b * d) * (4 * c ** 2 - 3 * b * d) ** 2)) / b ** 3))
        P = (8 * e) / (3 * V ** (11 / 3)) + (2 * d) / V ** 3 + (4 * c) / (3 * V ** (7 / 3)) + (2 * b) / (
                3 * V ** (5 / 3))
        P = P * conv_factor
        B = (2 * (44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2)) / (9 * V ** (11 / 3))
        B = B * conv_factor
        BP = (484 * e + 243 * d * V ** (2 / 3) + 98 * c * V ** (4 / 3) + 25 * b * V ** 2) / (
                132 * e + 81 * d * V ** (2 / 3) + 42 * c * V ** (4 / 3) + 15 * b * V ** 2);
        B2P = (4 * V ** (13 / 3) * (27 * d * (22 * e + 7 * c * V ** (4 / 3) + 10 * b * V ** 2) + V ** (2 / 3) * (
                990 * b * e * V ** (2 / 3) + 7 * c * (176 * e + 5 * b * V ** 2)))) / (
                      44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2) ** 3
        B2P = B2P / conv_factor
        E0 = a + e / V ** (8 / 3) + d / V ** 2 + c / V ** (4 / 3) + b / V ** (2 / 3)
        eos_parameters = [V, E0, P, B, BP, B2P]

        fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])

        results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
        np.set_printoptions(precision=4, suppress=True)

        pressure_eos = ((8 * e) / (3 * volume_range ** (11 / 3)) + (2 * d) / volume_range ** 3 + (4 * c) / (
                3 * volume_range ** (7 / 3)) + (2 * b) / (
                                3 * volume_range ** (5 / 3))) * conv_factor
        energy_eos = np.concatenate(([eos_index], energy_eos))
        pressure_eos = np.concatenate(([eos_index], pressure_eos))

        return results, energy_eos, pressure_eos

    def BM5(volume, energy):
        eos_index = 4
        AA = np.vstack(
            (np.ones(np.shape(volume)), volume ** (-2 / 3), volume ** (-4 / 3), volume ** (-2), volume ** (-8 / 3)))
        AA = AA.T
        xx1 = np.linalg.pinv(AA)
        xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
        a = xx[0]
        b = xx[1]
        c = xx[2]
        d = xx[3]
        e = xx[4]

        energy_eos = a + b * (volume_range) ** (-2 / 3) + c * (volume_range) ** (-4 / 3) + d * (volume_range) ** (
            -2) + e * (volume_range) ** (-8 / 3)
        energy_eos_points = a + b * (volume) ** (-2 / 3) + c * (volume) ** (-4 / 3) + d * (volume) ** (-2) + e * (
            volume) ** (-8 / 3)
        energy_difference = energy_eos_points - energy

        func = lambda volume_range: ((8 * e) / (3 * volume_range ** (11 / 3)) + (2 * d) / volume_range ** 3 + (
                4 * c) / (3 * volume_range ** (7 / 3)) + (2 * b) / (3 * volume_range ** (5 / 3))) * conv_factor
        V = fsolve(func, np.mean(volume))
        P = (8 * e) / (3 * V ** (11 / 3)) + (2 * d) / V ** 3 + (4 * c) / (3 * V ** (7 / 3)) + (2 * b) / (
                3 * V ** (5 / 3))
        P = P * conv_factor
        B = (2 * (44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2)) / (9 * V ** (11 / 3))
        B = B * conv_factor
        BP = (484 * e + 243 * d * V ** (2 / 3) + 98 * c * V ** (4 / 3) + 25 * b * V ** 2) / (
                132 * e + 81 * d * V ** (2 / 3) + 42 * c * V ** (4 / 3) + 15 * b * V ** 2);
        B2P = (4 * V ** (13 / 3) * (27 * d * (22 * e + 7 * c * V ** (4 / 3) + 10 * b * V ** 2) + V ** (2 / 3) * (
                990 * b * e * V ** (2 / 3) + 7 * c * (176 * e + 5 * b * V ** 2)))) / (
                      44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2) ** 3
        B2P = B2P / conv_factor
        E0 = a + e / V ** (8 / 3) + d / V ** 2 + c / V ** (4 / 3) + b / V ** (2 / 3)
        eos_parameters = [V, E0, P, B, BP, B2P]

        fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
        results = np.concatenate(([eos_index], *eos_parameters, fitting_error * (10 ** 4)))
        np.set_printoptions(precision=4, suppress=True)

        pressure_eos = ((8 * e) / (3 * volume_range ** (11 / 3)) + (2 * d) / volume_range ** 3 + (4 * c) / (
                3 * volume_range ** (7 / 3)) + (2 * b) / (3 * volume_range ** (5 / 3))) * conv_factor
        energy_eos = np.concatenate(([eos_index], energy_eos))
        pressure_eos = np.concatenate(([eos_index], pressure_eos))

        return results, energy_eos, pressure_eos

    def LOG4(volume, energy):
        eos_index = 5
        AA = np.vstack((np.ones(np.shape(volume)), np.log(volume), np.log(volume) ** 2, np.log(volume) ** 3))
        AA = AA.T
        xx1 = np.linalg.pinv(AA)
        xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
        a = xx[0]
        b = xx[1]
        c = xx[2]
        d = xx[3]
        e = 0.0

        energy_eos = a + b * np.log(volume_range) + c * np.log(volume_range) ** 2 + d * np.log(
            volume_range) ** 3 + e * np.log(volume_range) ** 4
        energy_eos_points = a + b * np.log(volume) + c * np.log(volume) ** 2 + d * np.log(volume) ** 3 + e * np.log(
            volume) ** 4
        energy_difference = energy_eos_points - energy

        func = lambda volume_range: (-(
                (b + 2 * c * math.log(volume_range) + 3 * d * math.log(volume_range) ** 2 + 4 * e * math.log(
                    volume_range) ** 3) / volume_range)) * conv_factor
        V = fsolve(func, np.mean(volume))
        V = np.mean(V)
        P = -((b + 2 * c * math.log(V) + 3 * d * math.log(V) ** 2 + 4 * e * math.log(V) ** 3) / V)
        P = np.mean(P)
        P = P * conv_factor
        B = -((b - 2 * c + 2 * (c - 3 * d) * math.log(V) + 3 * (d - 4 * e) * math.log(V) ** 2 + 4 * e * math.log(
            V) ** 3) / V)
        B = np.mean(B)
        B = B * conv_factor
        BP = (b - 4 * c + 6 * d + 2 * (c - 6 * d + 12 * e) * math.log(V) + 3 * (d - 8 * e) * math.log(
            V) ** 2 + 4 * e * math.log(V) ** 3) / (
                     b - 2 * c + 2 * (c - 3 * d) * math.log(V) + 3 * (d - 4 * e) * math.log(
                 V) ** 2 + 4 * e * math.log(V) ** 3)
        B2P = (2 * V * (2 * c ** 2 - 3 * b * d + 18 * d ** 2 + 12 * b * e - 6 * c * (d + 4 * e) + 6 * (
                c * d - 3 * d ** 2 - 2 * b * e + 12 * d * e) * math.log(V) + 9 * (d - 4 * e) ** 2 * math.log(
            V) ** 2 + 24 * (d - 4 * e) * e * math.log(V) ** 3 + 24 * e ** 2 * math.log(V) ** 4)) / (
                      b - 2 * c + 2 * (c - 3 * d) * math.log(V) + 3 * (d - 4 * e) * math.log(
                  V) ** 2 + 4 * e * math.log(V) ** 3) ** 3
        B2P = np.mean(B2P)
        B2P = B2P / conv_factor
        E0 = a + b * math.log(V) + c * math.log(V) ** 2 + d * math.log(V) ** 3 + e * math.log(V) ** 4
        eos_parameters = [V, E0, P, B, BP, B2P]

        fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
        results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
        np.set_printoptions(precision=4, suppress=True)

        pressure_eos = (-((b + 2 * c * np.log(volume_range) + 3 * d * np.log(volume_range) ** 2 + 4 * e * np.log(
            volume_range) ** 3) / volume_range)) * conv_factor
        energy_eos = np.concatenate(([eos_index], energy_eos))
        pressure_eos = np.concatenate(([eos_index], pressure_eos))

        return results, energy_eos, pressure_eos

    def LOG5(volume, energy):
        eos_index = 6
        AA = np.vstack(
            (np.ones(np.shape(volume)), np.log(volume), np.log(volume) ** 2, np.log(volume) ** 3, np.log(volume) ** 4))
        AA = AA.T
        xx1 = np.linalg.pinv(AA)
        xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
        a = xx[0]
        b = xx[1]
        c = xx[2]
        d = xx[3]
        e = xx[4]

        energy_eos = a + b * np.log(volume_range) + c * np.log(volume_range) ** 2 + d * np.log(
            volume_range) ** 3 + e * np.log(volume_range) ** 4
        energy_eos_points = a + b * np.log(volume) + c * np.log(volume) ** 2 + d * np.log(volume) ** 3 + e * np.log(
            volume) ** 4
        energy_difference = energy_eos_points - energy

        func = lambda volume_range: (-(
                (b + 2 * c * math.log(volume_range) + 3 * d * math.log(volume_range) ** 2 + 4 * e * math.log(
                    volume_range) ** 3) / volume_range)) * conv_factor
        V = fsolve(func, np.mean(volume))
        V = np.mean(V)
        P = -((b + 2 * c * math.log(V) + 3 * d * math.log(V) ** 2 + 4 * e * math.log(V) ** 3) / V)
        P = np.mean(P)
        P = P * conv_factor
        B = -((b - 2 * c + 2 * (c - 3 * d) * math.log(V) + 3 * (d - 4 * e) * math.log(V) ** 2 + 4 * e * math.log(
            V) ** 3) / V)
        B = np.mean(B)
        B = B * conv_factor
        BP = (b - 4 * c + 6 * d + 2 * (c - 6 * d + 12 * e) * math.log(V) + 3 * (d - 8 * e) * math.log(
            V) ** 2 + 4 * e * math.log(V) ** 3) / (
                     b - 2 * c + 2 * (c - 3 * d) * math.log(V) + 3 * (d - 4 * e) * math.log(
                 V) ** 2 + 4 * e * math.log(V) ** 3)
        B2P = (2 * V * (2 * c ** 2 - 3 * b * d + 18 * d ** 2 + 12 * b * e - 6 * c * (d + 4 * e) + 6 * (
                c * d - 3 * d ** 2 - 2 * b * e + 12 * d * e) * math.log(V) + 9 * (d - 4 * e) ** 2 * math.log(
            V) ** 2 + 24 * (d - 4 * e) * e * math.log(V) ** 3 + 24 * e ** 2 * math.log(V) ** 4)) / (
                      b - 2 * c + 2 * (c - 3 * d) * math.log(V) + 3 * (d - 4 * e) * math.log(
                  V) ** 2 + 4 * e * math.log(V) ** 3) ** 3
        B2P = np.mean(B2P)
        B2P = B2P / conv_factor
        E0 = a + b * math.log(V) + c * math.log(V) ** 2 + d * math.log(V) ** 3 + e * math.log(V) ** 4
        eos_parameters = [V, E0, P, B, BP, B2P]

        fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
        results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
        np.set_printoptions(precision=4, suppress=True)

        pressure_eos = (-((b + 2 * c * np.log(volume_range) + 3 * d * np.log(volume_range) ** 2 + 4 * e * np.log(
            volume_range) ** 3) / volume_range)) * conv_factor
        energy_eos = np.concatenate(([eos_index], energy_eos))
        pressure_eos = np.concatenate(([eos_index], pressure_eos))

        return results, energy_eos, pressure_eos

    def murnaghan_eq(xini, Data):
        V = xini[0]
        E0 = xini[1]
        B = xini[2]
        bp = xini[3]
        volume_range = Data[:, 0]
        y = Data[:, 1]
        eng = E0 - (B * V) / (-1 + bp) + (B * (1 + (V / volume_range) ** bp / (-1 + bp)) * volume_range) / bp
        return eng - y

    def murnaghan(volume, energy):
        eos_index = 7
        volume = volume
        Data = np.vstack((volume, energy))
        Data = Data.T

        [results, energy_eos, pressure_eos] = mBM4(volume, energy)
        xini = [results[1], results[2], results[4] / conv_factor, results[5]]
        [xout, resnorm] = leastsq(murnaghan_eq, xini, Data)

        V = xout[0]
        E0 = xout[1]
        B = xout[2]
        bp = xout[3]

        energy_eos = E0 - (B * V) / (-1 + bp) + (B * (1 + (V / volume_range) ** bp / (-1 + bp)) * volume_range) / bp
        energy_eos_points = (E0 - (B * V) / (-1 + bp) + (B * (1 + (V / volume) ** bp / (-1 + bp)) * volume) / bp)
        energy_difference = energy_eos_points - energy
        eos_parameters = [V, E0, 0, B * conv_factor, bp, 0]

        fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
        results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
        np.set_printoptions(precision=4, suppress=True)

        pressure_eos = conv_factor * (B * (-1 + (V / volume_range) ** bp)) / bp
        energy_eos = np.concatenate(([eos_index], energy_eos))
        pressure_eos = np.concatenate(([eos_index], pressure_eos))

        return results, energy_eos, pressure_eos

    def vinet_eq(xini, Data):
        V = xini[0]
        E0 = xini[1]
        B = xini[2]
        bp = xini[3]
        volume_range = Data[:, 0]
        y = Data[:, 1]
        eng = E0 + (4 * B * V) / (-1 + bp) ** 2 - (
                4 * B * V * (1 + (3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2)) / (
                      (-1 + bp) ** 2 * np.exp((3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2))
        return eng - y

    def vinet(volume, energy):
        eos_index = 8
        Data = np.vstack((volume, energy))
        Data = Data.T

        [results, energy_eos, pressure_eos] = mBM4(volume, energy)
        xini = [results[1], results[2], results[4] / conv_factor, results[5]]
        [xout, resnorm] = leastsq(vinet_eq, xini, Data)
        V = xout[0]
        E0 = xout[1]
        B = xout[2]
        bp = xout[3]

        energy_eos = E0 + (4 * B * V) / (-1 + bp) ** 2 - (
                4 * B * V * (1 + (3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2)) / (
                             (-1 + bp) ** 2 * np.exp((3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2))
        energy_eos_points = E0 + (4 * B * V) / (-1 + bp) ** 2 - (
                4 * B * V * (1 + (3 * (-1 + bp) * (-1 + (volume / V) ** (1 / 3))) / 2)) / (
                                    (-1 + bp) ** 2 * np.exp((3 * (-1 + bp) * (-1 + (volume / V) ** (1 / 3))) / 2))
        energy_difference = energy_eos_points - energy

        b2p = (19 - 18 * bp - 9 * bp ** 2) / (36 * B)
        eos_parameters = [V, E0, 0, B * conv_factor, bp, b2p / conv_factor]

        fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
        results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
        np.set_printoptions(precision=4, suppress=True)

        pressure_eos = conv_factor * (-3 * B * (-1 + (volume_range / V) ** (1 / 3))) / (
                np.exp((3 * (-1 + bp) * (-1 + (volume_range / V) ** (1 / 3))) / 2) * (volume_range / V) ** (2 / 3))
        energy_eos = np.concatenate(([eos_index], energy_eos))
        pressure_eos = np.concatenate(([eos_index], pressure_eos))

        return results, energy_eos, pressure_eos

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
        eng = a + b * np.exp(d * volume_range ** (1 / 3)) + c * np.exp(2 * d * volume_range ** (1 / 3))
        return eng - y

    def morse(volume, energy):
        eos_index = 9
        Data = np.vstack((volume, energy))
        Data = Data.T

        [results, energy_eos, pressure_eos] = mBM4(volume, energy)
        xini = [results[1], results[2], results[4] / conv_factor,
                results[5]]
        [xout, resnorm] = leastsq(morse_eq, xini, Data)
        V = xout[0]
        E0 = xout[1]
        B = xout[2]
        bp = xout[3]

        a = E0 + (9 * B * V) / (2 * (-1 + bp) ** 2)
        b = (-9 * B * np.exp(-1 + bp) * V) / (-1 + bp) ** 2
        c = (9 * B * np.exp(-2 + 2 * bp) * V) / (2 * (-1 + bp) ** 2)
        d = (1 - bp) / V ** (1 / 3)

        energy_eos = a + b * np.exp(d * volume_range ** (1 / 3)) + c * np.exp(2 * d * volume_range ** (1 / 3))
        energy_eos_points = a + b * np.exp(d * volume ** (1 / 3)) + c * np.exp(2 * d * volume ** (1 / 3))
        energy_difference = energy_eos_points - energy

        b2p = (5 - 5 * bp - 2 * bp ** 2) / (9 * B)
        eos_parameters = [V, E0, 0, B * conv_factor, bp, b2p / conv_factor]

        fitting_error = np.array([math.sqrt(sum((energy_difference / energy) ** 2 / len(energy)))])
        results = np.concatenate(([eos_index], eos_parameters, fitting_error * (10 ** 4)))
        np.set_printoptions(precision=4, suppress=True)

        pressure_eos = -conv_factor * (
                d * np.exp(d * volume_range ** (1 / 3)) * (b + 2 * c * np.exp(d * volume_range ** (1 / 3)))) / (
                               3 * volume_range ** (2 / 3))
        energy_eos = np.concatenate(([eos_index], energy_eos))
        pressure_eos = np.concatenate(([eos_index], pressure_eos))

        return results, energy_eos, pressure_eos

    # 3. Plot the results
    results = []
    energy_eos = []
    pressure_eos = []
    if num_structures == 'single':
        for i in eos_index:
            if i == 1:
                [results1, energy_eos1, pressure_eos1] = mBM4(volume, energy)
                results.append(results1)
                energy_eos.append(energy_eos1)
                pressure_eos.append(pressure_eos1)
            if i == 2:
                [results2, energy_eos2, pressure_eos2] = mBM5(volume, energy)
                results.append(results2)
                energy_eos.append(energy_eos2)
                pressure_eos.append(pressure_eos2)
            if i == 3:
                [results3, energy_eos3, pressure_eos3] = BM4(volume, energy)
                results.append(results3)
                energy_eos.append(energy_eos3)
                pressure_eos.append(pressure_eos3)
            if i == 4:
                [results4, energy_eos4, pressure_eos4] = BM5(volume, energy)
                results.append(results4)
                energy_eos.append(energy_eos4)
                pressure_eos.append(pressure_eos4)
            if i == 5:
                [results5, energy_eos5, pressure_eos5] = LOG4(volume, energy)
                results.append(results5)
                energy_eos.append(energy_eos5)
                pressure_eos.append(pressure_eos5)
            if i == 6:
                [results6, energy_eos6, pressure_eos6] = LOG5(volume, energy)
                results.append(results6)
                energy_eos.append(energy_eos6)
                pressure_eos.append(pressure_eos6)
            if i == 7:
                [results7, energy_eos7, pressure_eos7] = murnaghan(volume, energy)
                results.append(results7)
                energy_eos.append(energy_eos7)
                pressure_eos.append(pressure_eos7)
            if i == 8:
                [results8, energy_eos8, pressure_eos8] = vinet(volume, energy)
                results.append(results8)
                energy_eos.append(energy_eos8)
                pressure_eos.append(pressure_eos8)
            if i == 9:
                [results9, energy_eos9, pressure_eos9] = morse(volume, energy)
                results.append(results9)
                energy_eos.append(energy_eos9)
                pressure_eos.append(pressure_eos9)

        if plot_ev and not plot_pv:
            plt.figure(figsize=(10, 5))
            plt.scatter(volume, energy, label='DFT')
            for i in eos_index:
                if i == 1:
                    plt.plot(volume_range, energy_eos1[1:], label='mBM4')
                if i == 2:
                    plt.plot(volume_range, energy_eos2[1:], label='mBM5')
                if i == 3:
                    plt.plot(volume_range, energy_eos3[1:], label='BM4')
                if i == 4:
                    plt.plot(volume_range, energy_eos4[1:], label='BM5')
                if i == 5:
                    plt.plot(volume_range, energy_eos5[1:], label='LOG4')
                if i == 6:
                    plt.plot(volume_range, energy_eos6[1:], label='LOG5')
                if i == 7:
                    plt.plot(volume_range, energy_eos7[1:], label='Murnaghan')
                if i == 8:
                    plt.plot(volume_range, energy_eos8[1:], label='Vinet')
                if i == 9:
                    plt.plot(volume_range, energy_eos9[1:], label='Morse')
            plt.xlabel(r'Volume ($\mathrm{\AA}^3$)')
            plt.ylabel('Energy (eV)')
            plt.legend()
            plt.show()

        if plot_pv and not plot_ev:
            plt.figure(figsize=(10, 5))
            for i in eos_index:
                if i == 1:
                    plt.plot(volume_range, pressure_eos1[1:], label='mBM4')
                if i == 2:
                    plt.plot(volume_range, pressure_eos2[1:], label='mBM5')
                if i == 3:
                    plt.plot(volume_range, pressure_eos3[1:], label='BM4')
                if i == 4:
                    plt.plot(volume_range, pressure_eos4[1:], label='BM5')
                if i == 5:
                    plt.plot(volume_range, pressure_eos5[1:], label='LOG4')
                if i == 6:
                    plt.plot(volume_range, pressure_eos6[1:], label='LOG5')
                if i == 7:
                    plt.plot(volume_range, pressure_eos7[1:], label='Murnaghan')
                if i == 8:
                    plt.plot(volume_range, pressure_eos8[1:], label='Vinet')
                if i == 9:
                    plt.plot(volume_range, pressure_eos9[1:], label='Morse')
            plt.xlabel(r'Volume ($\mathrm{\AA}^3$)')
            plt.ylabel('Pressure (GPa)')
            plt.legend()
            plt.show()

        if plot_ev and plot_pv:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
            axes[0].scatter(volume, energy, label='DFT')
            for i in eos_index:
                if i == 1:
                    axes[0].plot(volume_range, energy_eos1[1:], label='mBM4')
                if i == 2:
                    axes[0].plot(volume_range, energy_eos2[1:], label='mBM5')
                if i == 3:
                    axes[0].plot(volume_range, energy_eos3[1:], label='BM4')
                if i == 4:
                    axes[0].plot(volume_range, energy_eos4[1:], label='BM5')
                if i == 5:
                    axes[0].plot(volume_range, energy_eos5[1:], label='LOG4')
                if i == 6:
                    axes[0].plot(volume_range, energy_eos6[1:], label='LOG5')
                if i == 7:
                    axes[0].plot(volume_range, energy_eos7[1:], label='Murnaghan')
                if i == 8:
                    axes[0].plot(volume_range, energy_eos8[1:], label='Vinet')
                if i == 9:
                    axes[0].plot(volume_range, energy_eos9[1:], label='Morse')
            axes[0].set_xlabel(r'Volume ($\mathrm{\AA}^3$)')
            axes[0].set_ylabel('Energy (eV)')
            axes[0].legend()

            for i in eos_index:
                if i == 1:
                    axes[1].plot(volume_range, pressure_eos1[1:], label='mBM4')
                if i == 2:
                    axes[1].plot(volume_range, pressure_eos2[1:], label='mBM5')
                if i == 3:
                    axes[1].plot(volume_range, pressure_eos3[1:], label='BM4')
                if i == 4:
                    axes[1].plot(volume_range, pressure_eos4[1:], label='BM5')
                if i == 5:
                    axes[1].plot(volume_range, pressure_eos5[1:], label='LOG4')
                if i == 6:
                    axes[1].plot(volume_range, pressure_eos6[1:], label='LOG5')
                if i == 7:
                    axes[1].plot(volume_range, pressure_eos7[1:], label='Murnaghan')
                if i == 8:
                    axes[1].plot(volume_range, pressure_eos8[1:], label='Vinet')
                if i == 9:
                    axes[1].plot(volume_range, pressure_eos9[1:], label='Morse')
            axes[1].set_xlabel(r'Volume ($\mathrm{\AA}^3$)')
            axes[1].set_ylabel('Pressure (GPa)')
            axes[1].legend()

            plt.tight_layout()  # Automatically adjust subplots to fit in the figure
            plt.show()

        results = np.vstack(results)
        sorted_indices = np.argsort(results[:, -1])
        results = results[sorted_indices]
        results = np.delete(results, 3, axis=1)

        energy_eos = np.vstack(energy_eos)
        pressure_eos = np.vstack(pressure_eos)

        headers = ["EOS", 'Volume (Å³)', "Energy (eV)", "B (GPa)", "B'", "B'' (1/GPa)", "Fitting error x 10⁻⁴"]
        format_string = "{:^15}" * len(headers)
        print(format_string.format(*headers))
        for i in range(len(results)):
            formatted_results = ["{:.4f}".format(result) if isinstance(result, np.float64) else result for result in
                                 results[i]]
            formatted_results[0] = str(int(float(formatted_results[0])))
            print(format_string.format(*formatted_results))
        return results, energy_eos, pressure_eos, volume_range

    if num_structures == 'multiple':
        for key in volume:
            volume_range = volume_range_list[key]
            for i in eos_index:
                if i == 1:
                    [results1, energy_eos1, pressure_eos1] = mBM4(volume[key], energy[key])
                    results.append(results1)
                    energy_eos.append(energy_eos1)
                    pressure_eos.append(pressure_eos1)
                if i == 2:
                    [results2, energy_eos2, pressure_eos2] = mBM5(volume[key], energy[key])
                    results.append(results2)
                    energy_eos.append(energy_eos2)
                    pressure_eos.append(pressure_eos2)
                if i == 3:
                    [results3, energy_eos3, pressure_eos3] = BM4(volume[key], energy[key])
                    results.append(results3)
                    energy_eos.append(energy_eos3)
                    pressure_eos.append(pressure_eos3)
                if i == 4:
                    [results4, energy_eos4, pressure_eos4] = BM5(volume[key], energy[key])
                    results.append(results4)
                    energy_eos.append(energy_eos4)
                    pressure_eos.append(pressure_eos4)
                if i == 5:
                    [results5, energy_eos5, pressure_eos5] = LOG4(volume[key], energy[key])
                    results.append(results5)
                    energy_eos.append(energy_eos5)
                    pressure_eos.append(pressure_eos5)
                if i == 6:
                    [results6, energy_eos6, pressure_eos6] = LOG5(volume[key], energy[key])
                    results.append(results6)
                    energy_eos.append(energy_eos6)
                    pressure_eos.append(pressure_eos6)
                if i == 7:
                    [results7, energy_eos7, pressure_eos7] = murnaghan(volume[key], energy[key])
                    results.append(results7)
                    energy_eos.append(energy_eos7)
                    pressure_eos.append(pressure_eos7)
                if i == 8:
                    [results8, energy_eos8, pressure_eos8] = vinet(volume[key], energy[key])
                    results.append(results8)
                    energy_eos.append(energy_eos8)
                    pressure_eos.append(pressure_eos8)
                if i == 9:
                    [results9, energy_eos9, pressure_eos9] = morse(volume[key], energy[key])
                    results.append(results9)
                    energy_eos.append(energy_eos9)
                    pressure_eos.append(pressure_eos9)

        results = np.vstack(results)
        results = np.delete(results, 3, axis=1)

        energy_eos = np.vstack(energy_eos)
        pressure_eos = np.vstack(pressure_eos)

        structure_names = []
        for input_file in input_files:
            structure_names.append(input_file)
        sorted_indices = np.argsort(results[:, 2])[::-1]  # Sort the results by minimum energy in descending order
        results = results[sorted_indices]
        structure_names = [structure_names[i] for i in sorted_indices]

        if plot_ev and not plot_pv:
            j = 0
            plt.figure(figsize=(10, 8))
            for key in volume:
                volume_range = volume_range_list[key]
                plt.scatter(volume[key], energy[key])
                plt.scatter(results[j][1], results[j][2], facecolors='none',
                            edgecolors='k')  # Plot the minimum energy point
                for i in eos_index:
                    if i == 1:
                        plt.plot(volume_range, energy_eos[j][1:], label=key)
                        plt.title('mBM4')
                    if i == 2:
                        plt.plot(volume_range, energy_eos[j][1:], label=key)
                        plt.title('mBM5')
                    if i == 3:
                        plt.plot(volume_range, energy_eos[j][1:], label=key)
                        plt.title('BM4')
                    if i == 4:
                        plt.plot(volume_range, energy_eos[j][1:], label=key)
                        plt.title('BM5')
                    if i == 5:
                        plt.plot(volume_range, energy_eos[j][1:], label=key)
                        plt.title('LOG4')
                    if i == 6:
                        plt.plot(volume_range, energy_eos[j][1:], label=key)
                        plt.title('LOG5')
                    if i == 7:
                        plt.plot(volume_range, energy_eos[j][1:], label=key)
                        plt.title('Murnaghan')
                    if i == 8:
                        plt.plot(volume_range, energy_eos[j][1:], label=key)
                        plt.title('Vinet')
                    if i == 9:
                        plt.plot(volume_range, energy_eos[j][1:], label=key)
                        plt.title('Morse')
                j += 1

            plt.xlabel(r'Volume ($\mathrm{\AA}^3$)')
            plt.ylabel('Energy (eV)')
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend([handles[idx] for idx in sorted_indices], [labels[idx] for idx in sorted_indices],
                       loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
            plt.show()

        if plot_pv and not plot_ev:
            j = 0
            plt.figure(figsize=(10, 8))
            for key in volume:
                volume_range = volume_range_list[key]
                for i in eos_index:
                    if i == 1:
                        plt.plot(volume_range, pressure_eos[j][1:], label=key)
                        plt.title('mBM4')
                    if i == 2:
                        plt.plot(volume_range, pressure_eos[j][1:], label=key)
                        plt.title('mBM5')
                    if i == 3:
                        plt.plot(volume_range, pressure_eos[j][1:], label=key)
                        plt.title('BM4')
                    if i == 4:
                        plt.plot(volume_range, pressure_eos[j][1:], label=key)
                        plt.title('BM5')
                    if i == 5:
                        plt.plot(volume_range, pressure_eos[j][1:], label=key)
                        plt.title('LOG4')
                    if i == 6:
                        plt.plot(volume_range, pressure_eos[j][1:], label=key)
                        plt.title('LOG5')
                    if i == 7:
                        plt.plot(volume_range, pressure_eos[j][1:], label=key)
                        plt.title('Murnaghan')
                    if i == 8:
                        plt.plot(volume_range, pressure_eos[j][1:], label=key)
                        plt.title('Vinet')
                    if i == 9:
                        plt.plot(volume_range, pressure_eos[j][1:], label=key)
                        plt.title('Morse')
                j += 1
            plt.xlabel(r'Volume ($\mathrm{\AA}^3$)')
            plt.ylabel('Pressure (GPa)')
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend([handles[idx] for idx in sorted_indices], [labels[idx] for idx in sorted_indices],
                       loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
            plt.show()

        if plot_ev and plot_pv:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
            j = 0
            for key in volume:
                volume_range = volume_range_list[key]
                axes[0].scatter(volume[key], energy[key])
                for i in eos_index:
                    if i == 1:
                        axes[0].plot(volume_range, energy_eos[j][1:], label=key)
                        axes[0].set_title('mBM4')
                    if i == 2:
                        axes[0].plot(volume_range, energy_eos[j][1:], label=key)
                        axes[0].set_title('mBM5')
                    if i == 3:
                        axes[0].plot(volume_range, energy_eos[j][1:], label=key)
                        axes[0].set_title('BM4')
                    if i == 4:
                        axes[0].plot(volume_range, energy_eos[j][1:], label=key)
                        axes[0].set_title('BM5')
                    if i == 5:
                        axes[0].plot(volume_range, energy_eos[j][1:], label=key)
                        axes[0].set_title('LOG4')
                    if i == 6:
                        axes[0].plot(volume_range, energy_eos[j][1:], label=key)
                        axes[0].set_title('LOG5')
                    if i == 7:
                        axes[0].plot(volume_range, energy_eos[j][1:], label=key)
                        axes[0].set_title('Murnaghan')
                    if i == 8:
                        axes[0].plot(volume_range, energy_eos[j][1:], label=key)
                        axes[0].set_title('Vinet')
                    if i == 9:
                        axes[0].plot(volume_range, energy_eos[j][1:], label=key)
                        axes[0].set_title('Morse')
                j += 1
            axes[0].set_xlabel(r'Volume ($\mathrm{\AA}^3$)')
            axes[0].set_ylabel('Energy (eV)')
            axes[0].legend()

            j = 0
            for key in volume:
                volume_range = volume_range_list[key]
                for i in eos_index:
                    if i == 1:
                        axes[1].plot(volume_range, pressure_eos[j][1:], label=key)
                        axes[1].set_title('mBM4')
                    if i == 2:
                        axes[1].plot(volume_range, pressure_eos[j][1:], label=key)
                        axes[1].set_title('mBM5')
                    if i == 3:
                        axes[1].plot(volume_range, pressure_eos[j][1:], label=key)
                        axes[1].set_title('BM4')
                    if i == 4:
                        axes[1].plot(volume_range, pressure_eos[j][1:], label=key)
                        axes[1].set_title('BM5')
                    if i == 5:
                        axes[1].plot(volume_range, pressure_eos[j][1:], label=key)
                        axes[1].set_title('LOG4')
                    if i == 6:
                        axes[1].plot(volume_range, pressure_eos[j][1:], label=key)
                        axes[1].set_title('LOG5')
                    if i == 7:
                        axes[1].plot(volume_range, pressure_eos[j][1:], label=key)
                        axes[1].set_title('Murnaghan')
                    if i == 8:
                        axes[1].plot(volume_range, pressure_eos[j][1:], label=key)
                        axes[1].set_title('Vinet')
                    if i == 9:
                        axes[1].plot(volume_range, pressure_eos[j][1:], label=key)
                        axes[1].set_title('Morse')
                j += 1
            axes[1].set_xlabel(r'Volume ($\mathrm{\AA}^3$)')
            axes[1].set_ylabel('Pressure (GPa)')
            axes[1].legend()
            plt.tight_layout()  # Automatically adjust subplots to fit in the figure
            plt.show()

        headers = ["Structure", "EOS", 'Volume (Å³)', "Energy (eV)", "B (GPa)", "B'", "B'' (1/GPa)",
                   "Fitting error x 10⁻⁴"]
        format_string = "{:^15}" * len(headers)
        print(format_string.format(*headers))

        for i in range(len(results)):
            formatted_results = ["{:.4f}".format(result) if isinstance(result, np.float64) else result for result in
                                 results[i]]
            formatted_results[0] = str(int(float(formatted_results[0])))
            formatted_results.insert(0, structure_names[i])
            print(format_string.format(*formatted_results))
        return results, energy_eos, pressure_eos


def pv_fit(num_structures, *input_files, eos_index, plot_pv=True):
    # 1. Load the pressure-volume data
    if num_structures == 'single':
        data = np.loadtxt(input_files[0])
        volume = data[:, 0]
        pressure = data[:, 1] / 10  # Convert kB to GPa
        volume_range = np.linspace(min(volume), max(volume), 1000)
    if num_structures == 'multiple':
        volume = {}
        pressure = {}
        volume_range_list = {}
        for input_file in input_files:
            volume[input_file] = np.loadtxt(input_file, usecols=(0,))
            pressure[input_file] = np.loadtxt(input_file, usecols=(1,)) / 10  # GPa
            volume_range_list[input_file] = np.linspace(min(volume[input_file]), max(volume[input_file]), 1000)

        # EOS fitting for multiple structures only supports 1 EOS at a time
        if len(eos_index) > 1:
            raise ValueError("eos_index for multiple structures should only have one value")

    # Remove any duplicates
    eos_index = list(set(eos_index))
    eos_index.sort()

    # Conversion factor: 1 eV/Å^3 = 160.2176621 GPa
    conv_factor = 160.2176621

    # 2. Fit the pressure-volume data to the selected EOS
    def mBM4_pv(volume, pressure):
        eos_index = 1
        AA = [volume ** (-4 / 3) * (1 / 3), volume ** (-5 / 3) * (2 / 3), volume ** (-2)]
        AA = np.array(AA)
        AA = AA.T
        xx1 = np.linalg.pinv(AA)
        xx = xx1.dot(pressure)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
        b = xx[0]
        c = xx[1]
        d = xx[2]
        e = 0.0

        pressure_eos = (4 * e) / (3 * volume_range ** (7 / 3)) + d / volume_range ** 2 + (2 * c) / (
                3 * volume_range ** (5 / 3)) + b / (3 * volume_range ** (4 / 3))
        pressure_eos_points = (4 * e) / (3 * volume ** (7 / 3)) + d / volume ** 2 + (2 * c) / (
                3 * volume ** (5 / 3)) + b / (
                                      3 * volume ** (4 / 3))

        V = 4 * c ** 3 - 9 * b * c * d + np.sqrt((c ** 2 - 3 * b * d) * (4 * c ** 2 - 3 * b * d) ** 2)
        V = -V / b ** 3
        P = (4 * e) / (3 * V ** (7 / 3)) + d / V ** 2 + (2 * c) / (3 * V ** (5 / 3)) + b / (3 * V ** (4 / 3))
        B = ((28 * e) / (9 * V ** (10 / 3)) + (2 * d) / V ** 3 + (10 * c) / (9 * V ** (8 / 3)) + (4 * b) / (
                9 * V ** (7 / 3))) * V
        BP = (98 * e + 54 * d * V ** (1 / 3) + 25 * c * V ** (2 / 3) + 8 * b * V) / (
                42 * e + 27 * d * V ** (1 / 3) + 15 * c * V ** (2 / 3) + 6 * b * V)
        B2P = (V ** (8 / 3) * (9 * d * (14 * e + 5 * c * V ** (2 / 3) + 8 * b * V) + 2 * V ** (1 / 3) * (
                126 * b * e * V ** (1 / 3) + 5 * c * (28 * e + b * V)))) / (
                      2 * (14 * e + 9 * d * V ** (1 / 3) + 5 * c * V ** (2 / 3) + 2 * b * V) ** 3)

        eos_parameters = [V, P, B, BP, B2P]
        pressure_difference = pressure_eos_points - pressure

        fitting_error = math.sqrt(sum((pressure_difference ** 2) / len(pressure)))
        results = np.hstack((np.insert(eos_parameters, 0, eos_index), fitting_error))

        return results, pressure_eos

    def mBM5_pv(volume, pressure):
        eos_index = 2
        AA = [volume ** (-4 / 3) * (1 / 3), volume ** (-5 / 3) * (2 / 3), volume ** (-2), volume ** (-7 / 3) * (4 / 3)]
        AA = np.array(AA)
        AA = AA.T
        xx1 = np.linalg.pinv(AA)
        xx = xx1.dot(pressure)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
        b = xx[0]
        c = xx[1]
        d = xx[2]
        e = xx[3]

        pressure_eos = (4 * e) / (3 * volume_range ** (7 / 3)) + d / volume_range ** 2 + (2 * c) / (
                3 * volume_range ** (5 / 3)) + b / (3 * volume_range ** (4 / 3))
        pressure_eos_points = (4 * e) / (3 * volume ** (7 / 3)) + d / volume ** 2 + (2 * c) / (
                3 * volume ** (5 / 3)) + b / (
                                      3 * volume ** (4 / 3))
        pressure_difference = pressure_eos_points - pressure

        fun = lambda x: (
                (4 * e) / (3 * x ** (7 / 3)) + d / x ** 2 + (2 * c) / (3 * x ** (5 / 3)) + b / (3 * x ** (4 / 3)))
        V = fsolve(fun, np.mean(volume))
        P = (4 * e) / (3 * V ** (7 / 3)) + d / V ** 2 + (2 * c) / (3 * V ** (5 / 3)) + b / (3 * V ** (4 / 3))
        B = ((28 * e) / (9 * V ** (10 / 3)) + (2 * d) / V ** 3 + (10 * c) / (9 * V ** (8 / 3)) + (4 * b) / (
                9 * V ** (7 / 3))) * V
        BP = (98 * e + 54 * d * V ** (1 / 3) + 25 * c * V ** (2 / 3) + 8 * b * V) / (
                42 * e + 27 * d * V ** (1 / 3) + 15 * c * V ** (2 / 3) + 6 * b * V)
        B2P = (V ** (8 / 3) * (9 * d * (14 * e + 5 * c * V ** (2 / 3) + 8 * b * V) + 2 * V ** (1 / 3) * (
                126 * b * e * V ** (1 / 3) + 5 * c * (28 * e + b * V)))) / (
                      2 * (14 * e + 9 * d * V ** (1 / 3) + 5 * c * V ** (2 / 3) + 2 * b * V) ** 3)
        eos_parameters = [V, P, B, BP, B2P]

        fitting_error = math.sqrt(sum((pressure_difference ** 2) / len(pressure)))
        results = np.hstack((np.insert(eos_parameters, 0, eos_index), fitting_error))

        return results, pressure_eos

    def BM4_pv(volume, pressure):
        eos_index = 3
        AA = [volume ** (-5 / 3) * (2 / 3), volume ** (-7 / 3) * (4 / 3), volume ** (-3) * 2]
        AA = np.array(AA)
        AA = AA.T
        xx1 = np.linalg.pinv(AA)
        xx = xx1.dot(pressure)  # (4x1)=(4xn)*(nx1), solve by pseudo-inversion: Ax=b
        b = xx[0]
        c = xx[1]
        d = xx[2]
        e = 0.0

        pressure_eos = (8 * e) / (3 * volume_range ** (11 / 3)) + (2 * d) / volume_range ** 3 + (4 * c) / (
                3 * volume_range ** (7 / 3)) + (2 * b) / (
                               3 * volume_range ** (5 / 3))
        pressure_eos_points = (8 * e) / (3 * volume ** (11 / 3)) + (2 * d) / volume ** 3 + (4 * c) / (
                3 * volume ** (7 / 3)) + (2 * b) / (
                                      3 * volume ** (5 / 3))
        pressure_difference = pressure_eos_points - pressure

        V = np.sqrt(
            -((4 * c ** 3 - 9 * b * c * d + np.sqrt((c ** 2 - 3 * b * d) * (4 * c ** 2 - 3 * b * d) ** 2)) / b ** 3))
        P = (8 * e) / (3 * V ** (11 / 3)) + (2 * d) / V ** 3 + (4 * c) / (3 * V ** (7 / 3)) + (2 * b) / (
                3 * V ** (5 / 3))
        B = (2 * (44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2)) / (9 * V ** (11 / 3))
        BP = (484 * e + 243 * d * V ** (2 / 3) + 98 * c * V ** (4 / 3) + 25 * b * V ** 2) / (
                132 * e + 81 * d * V ** (2 / 3) + 42 * c * V ** (4 / 3) + 15 * b * V ** 2)
        B2P = (4 * V ** (13 / 3) * (27 * d * (22 * e + 7 * c * V ** (4 / 3) + 10 * b * V ** 2) + V ** (2 / 3) * (
                990 * b * e * V ** (2 / 3) + 7 * c * (176 * e + 5 * b * V ** 2)))) / (
                      44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2) ** 3
        eos_parameters = [V, P, B, BP, B2P]

        fitting_error = math.sqrt(sum((pressure_difference ** 2) / len(pressure)))
        results = np.hstack((np.insert(eos_parameters, 0, eos_index), fitting_error))

        return results, pressure_eos

    def BM5_pv(volume, pressure):
        eos_index = 4
        AA = [volume ** (-5 / 3) * (2 / 3), volume ** (-7 / 3) * (4 / 3), volume ** (-3) * 2,
              volume ** (-11 / 3) * (8 / 3)]
        AA = np.array(AA)
        AA = AA.T
        xx1 = np.linalg.pinv(AA)
        xx = xx1.dot(pressure)  # (4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        b = xx[0]
        c = xx[1]
        d = xx[2]
        e = xx[3]

        pressure_eos = (8 * e) / (3 * volume_range ** (11 / 3)) + (2 * d) / volume_range ** 3 + (4 * c) / (
                3 * volume_range ** (7 / 3)) + (2 * b) / (
                               3 * volume_range ** (5 / 3))
        pressure_eos_points = (8 * e) / (3 * volume ** (11 / 3)) + (2 * d) / volume ** 3 + (4 * c) / (
                3 * volume ** (7 / 3)) + (2 * b) / (
                                      3 * volume ** (5 / 3))
        pressure_difference = pressure_eos_points - pressure

        fun = lambda x: ((8 * e) / (3 * x ** (11 / 3)) + (2 * d) / x ** 3 + (4 * c) / (3 * x ** (7 / 3)) + (2 * b) / (
                3 * x ** (5 / 3))) * conv_factor
        V = fsolve(fun, np.mean(volume))

        P = (8 * e) / (3 * V ** (11 / 3)) + (2 * d) / V ** 3 + (4 * c) / (3 * V ** (7 / 3)) + (2 * b) / (
                3 * V ** (5 / 3))
        B = (2 * (44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2)) / (9 * V ** (11 / 3))
        BP = (484 * e + 243 * d * V ** (2 / 3) + 98 * c * V ** (4 / 3) + 25 * b * V ** 2) / (
                132 * e + 81 * d * V ** (2 / 3) + 42 * c * V ** (4 / 3) + 15 * b * V ** 2)
        B2P = (4 * V ** (13 / 3) * (27 * d * (22 * e + 7 * c * V ** (4 / 3) + 10 * b * V ** 2) + V ** (2 / 3) * (
                990 * b * e * V ** (2 / 3) + 7 * c * (176 * e + 5 * b * V ** 2)))) / (
                      44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2) ** 3
        eos_parameters = [V, P, B, BP, B2P]

        fitting_error = math.sqrt(sum((pressure_difference ** 2) / len(pressure)))
        results = np.hstack((np.insert(eos_parameters, 0, eos_index), fitting_error))

        return results, pressure_eos

    # 3. Plot the EOS fitting results
    results = []
    pressure_eos = []
    if num_structures == 'single':
        for i in eos_index:
            if i == 1:
                [results1, pressure_eos1] = mBM4_pv(volume, pressure)
                results.append(results1)
                pressure_eos.append(pressure_eos1)
            if i == 2:
                [results2, pressure_eos2] = mBM5_pv(volume, pressure)
                results.append(results2)
                pressure_eos.append(pressure_eos2)
            if i == 3:
                [results3, pressure_eos3] = BM4_pv(volume, pressure)
                results.append(results3)
                pressure_eos.append(pressure_eos3)
            if i == 4:
                [results4, pressure_eos4] = BM5_pv(volume, pressure)
                results.append(results4)
                pressure_eos.append(pressure_eos4)

        if plot_pv:
            plt.figure(figsize=(10, 5))
            plt.scatter(volume, pressure, label='DFT')
            for i in eos_index:
                if i == 1:
                    plt.plot(volume_range, pressure_eos1, label='mBM4')
                if i == 2:
                    plt.plot(volume_range, pressure_eos2, label='mBM5')
                if i == 3:
                    plt.plot(volume_range, pressure_eos3, label='BM4')
                if i == 4:
                    plt.plot(volume_range, pressure_eos4, label='BM5')

            plt.xlabel(r'Volume ($\mathrm{\AA}^3$)')
            plt.ylabel('Pressure (GPa)')
            plt.legend()
            plt.show()

        results = np.vstack(results)
        sorted_indices = np.argsort(results[:, -1])
        results = results[sorted_indices]
        pressure_eos = np.vstack(pressure_eos)

        headers = ["EOS", 'Volume (Å³)', "Energy (eV)", "B (GPa)", "B'", "B'' (1/GPa)", "Fitting error x 10⁻⁴"]
        format_string = "{:^15}" * len(headers)
        print(format_string.format(*headers))
        for i in range(len(results)):
            formatted_results = ["{:.4f}".format(result) if isinstance(result, np.float64) else result for result in
                                 results[i]]
            formatted_results[0] = str(int(float(formatted_results[0])))
            print(format_string.format(*formatted_results))
        return results, pressure_eos

    if num_structures == 'multiple':
        for key in volume:
            volume_range = volume_range_list[key]
            for i in eos_index:
                if i == 1:
                    [results1, pressure_eos1] = mBM4_pv(volume[key], pressure[key])
                    results.append(results1)
                    pressure_eos.append(pressure_eos1)
                if i == 2:
                    [results2, pressure_eos2] = mBM5_pv(volume[key], pressure[key])
                    results.append(results2)
                    pressure_eos.append(pressure_eos2)
                if i == 3:
                    [results3, pressure_eos3] = BM4_pv(volume[key], pressure[key])
                    results.append(results3)
                    pressure_eos.append(pressure_eos3)
                if i == 4:
                    [results4, pressure_eos4] = BM5_pv(volume[key], pressure[key])
                    results.append(results4)
                    pressure_eos.append(pressure_eos4)

        if plot_pv:
            j = 0
            plt.figure(figsize=(10, 5))
            for key in volume:
                volume_range = volume_range_list[key]
                plt.scatter(volume[key], pressure[key])
                for i in eos_index:
                    if i == 1:
                        plt.plot(volume_range, pressure_eos[j][:], label=key)
                        plt.title('mBM4')
                    if i == 2:
                        plt.plot(volume_range, pressure_eos[j][:], label=key)
                        plt.title('mBM5')
                    if i == 3:
                        plt.plot(volume_range, pressure_eos[j][:], label=key)
                        plt.title('BM4')
                    if i == 4:
                        plt.plot(volume_range, pressure_eos[j][:], label=key)
                        plt.title('BM5')
                j += 1
            plt.xlabel(r'Volume ($\mathrm{\AA}^3$)')
            plt.ylabel('Pressure (GPa)')
            plt.legend()
            plt.show()

        results = np.vstack(results)
        pressure_eos = np.vstack(pressure_eos)
        structure_names = []

        for input_file in input_files:
            structure_names.append(input_file)
        headers = ["Structure", "EOS", 'Volume (Å³)', "Energy (eV)", "B (GPa)", "B'", "B'' (1/GPa)",
                   "Fitting error x 10⁻⁴"]
        format_string = "{:^15}" * len(headers)
        print(format_string.format(*headers))
        for i in range(len(results)):
            formatted_results = ["{:.4f}".format(result) if isinstance(result, np.float64) else result for result in
                                 results[i]]
            formatted_results[0] = str(int(float(formatted_results[0])))
            formatted_results.insert(0, structure_names[i])
            print(format_string.format(*formatted_results))
        return results, pressure_eos

# Done: double checked the values of eos_index=1:9 with MATLAB already. Agrees well!
# Done: append all of the results
# Done: append the eos_index to energy_eos and pressure_eos
# Done: sort out the plotting options
# Done: multiple structures
# Done: Pressure is still in the results. Get rid of it.
# Done: Add structure for the multiple structures
# Done: Incorporate eosparameter45 into each eos function
# Done: EOS multiple output an error if more than 1 eos_index is selected
# Done: Something wrong with multiple structures. Mixes up the structure order!
# Done: Modify the volume range for the multiple and single structures.
# Done: Add PV fitting.
# Done: Rank multiple configurations by decreasing volume. See if I can alter the arrangement of the legend as well.
# TODO: Check PV fitting with MATLAB.
# Done: Clean up the code.
