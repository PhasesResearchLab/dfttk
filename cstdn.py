
import math
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
import plotly.graph_objects as go

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler
from custodian.vasp.jobs import VaspJob
from pymatgen.core import structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun

"""
eos fuctions
Shun-Li Shang wrote the original MATLAB code for EOS fitting.
Hui Sun converted the MATLAB code to python code.
Nigel Hew modified the python code to make it more user-friendly and added more functions.
Luke Myers performed some additional modifications to the python code.
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
"""
end eos functions
"""


# Function to extract the last occurrence of volume from OUTCAR files
def extract_volume(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'volume' in line:
                volume = float(line.split()[-1])
                break  # Stop searching after finding the last occurrence
    return volume


# Function to extract the last occurrence of pressure from OUTCAR files
def extract_pressure(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'pressure' in line:
                pressure = float(line.split()[3])
                break  # Stop searching after finding the last occurrence
    return pressure


# Function to extract energy from OSZICAR files
def extract_energy(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'F=' in line:
                energy = float(line.split()[4])
                break  # Stop searching after finding the last occurrence
    return energy


def extract_mag_data(outcar_path='OUTCAR'):
    if not os.path.isfile(outcar_path):
        print(f"Warning: File {outcar_path} does not exist. Skipping.")
        return None
    with open(outcar_path, 'r') as file:
        data = []
        step = 0
        found_mag_data = False
        data_start = False
        lines = file.readlines()
        for line in lines:
            if 'magnetization (x)' in line:
                found_mag_data = True
                step += 1
            elif found_mag_data and not data_start and '----' in line:
                data_start = True
            elif data_start and '----' not in line:
                ion = int(line.split()[0])
                s = float(line.split()[1])
                p = float(line.split()[2])
                d = float(line.split()[3])
                tot = float(line.split()[4])
                data.append((step, ion, s, p, d, tot))
            elif data_start and '----' in line:
                data_start = False
                found_mag_data = False
        df = pd.DataFrame(data, columns=['step', '# of ion', 's', 'p', 'd', 'tot'])
        return df


"""
Returns only the 'tot' magnetization of the last step for each specified ion

ion_list should be a list of integers ex: [1, 2, 3, 4]
"""
def extract_simple_mag_data(ion_list, outcar_path='OUTCAR'):
    all_mag_data = extract_mag_data(outcar_path)
    last_step_data = all_mag_data[all_mag_data['step'] == all_mag_data['step'].max()]
    simple_data = last_step_data[last_step_data['# of ion'].isin(ion_list)][['# of ion', 'tot']]
    simple_data.reset_index(drop=True, inplace=True)
    return simple_data



def plot_ev(df, eos_fitting='mBM4' ,show_fig=True):
    eos_df = fit_to_all_eos(df)
    fig = px.scatter(df, x='volume', y='energy', color='config', template='plotly_white')
    fig.update_layout(title='E-V', xaxis_title='Volume [A^3]', yaxis_title='Energy (eV)')
    
    for config in eos_df['config'].unique():
        eos_config_df = eos_df[eos_df['config'] == config]
        if eos_fitting in eos_config_df['eos_name'].unique():
            eos_name_df = eos_config_df[eos_config_df['eos_name'] == eos_fitting]
            fig.add_trace(go.Scatter(x=eos_name_df['volumes'].values[0], y=eos_name_df['energies'].values[0], mode='lines', name=f'{eos_fitting} fit', line=dict(width=1)))
        elif eos_fitting == 'all':
            for eos_name in eos_config_df['eos_name'].unique():
                eos_name_df = eos_config_df[eos_config_df['eos_name'] == eos_name]
                fig.add_trace(go.Scatter(x=eos_name_df['volumes'].values[0], y=eos_name_df['energies'].values[0], mode='lines', name=f'{eos_name} fit', line=dict(width=1)))
        elif eos_fitting == None:
            pass
        else:
            print(f"Warning: eos_fitting '{eos_fitting}' not found in eos_df. Skipping.")
    if show_fig:
        fig.show()
    return fig

"""
~~~WARNING~~~ The currect intent is to replace this function with extract_config_data()
This function grabs the necessary magnetic and volume data from the OUTCAR
for each volume and returns a data frame.

Within the path, there should be folders named vol_0, vol_1, etc.

There should be no other files or directories in the path with 
names starting with 'vol_'.

outcar_name and oszicar_name must be the same in each volume folder.

Consider adding config_name column to the data frame
"""
def extract_config_mv_data(path, ion_list, outcar_name='OUTCAR'):
    dfs_list = []
    start = path.find('config_') + len('config_') # Find the index where "config_" starts and add its length
    config = path[start:] #get the string following "config_"
    for vol_dir in glob.glob(os.path.join(path, 'vol_*')):
        outcar_path = os.path.join(vol_dir, outcar_name)
        if not os.path.isfile(outcar_path):
            print(f"Warning: File {outcar_path} does not exist. Skipping.")
            continue
        vol = extract_volume(outcar_path)
        mag_data = extract_simple_mag_data(ion_list, outcar_path)
        mag_data['volume'] = vol
        mag_data['config'] = config
        dfs_list.append(mag_data)
    df = pd.concat(dfs_list, ignore_index=True).sort_values(by=['volume', '# of ion']).reset_index(drop=True)
    return df

"""
This function grabs all necessary data from the OUTCAR
for each volume and returns a data frame in the tidy data format.

Todo: extract the pressure data,
Todo: extract any other data that might be useful

Within the path, there should be folders named vol_0, vol_1, etc.

There should be no other files or directories in the path with 
names starting with 'vol_'.

outcar_name and oszicar_name must be the same in each volume folder.

Consider adding config_name column to the data frame
"""

def extract_config_data(path, ion_list, outcar_name='OUTCAR', oszicar_name='OSZICAR'):
    dfs_list = []
    start = path.find('config_') + len('config_') # Find the index where "config_" starts and add its length
    config = path[start:] #get the string following "config_"
    for vol_dir in glob.glob(os.path.join(path, 'vol_*')):
        
        outcar_path = os.path.join(vol_dir, outcar_name)
        if not os.path.isfile(outcar_path):
            print(f"Warning: File {outcar_path} does not exist. Skipping.")
            continue

        oszicar_path = os.path.join(vol_dir, oszicar_name)
        if not os.path.isfile(oszicar_path):
            print(f"Warning: File {oszicar_path} does not exist. Skipping.")
            continue

        vol = extract_volume(outcar_path)
        energy = extract_energy(oszicar_path)
        data_collection = extract_simple_mag_data(ion_list, outcar_path)
        data_collection['volume'] = vol
        data_collection['config'] = config
        data_collection['energy'] = energy
        dfs_list.append(data_collection)
    df = pd.concat(dfs_list, ignore_index=True).sort_values(by=['volume', '# of ion']).reset_index(drop=True)
    return df

def three_step_relaxation(path, vasp_cmd, handlers, backup=True):  # Path should contain necessary VASP config files
    original_dir = os.getcwd()
    os.chdir(path)
    step1 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=True,
        final=False,
        suffix='.1relax',
        backup=backup,
    )

    step2 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=True,
        final=False,
        suffix='.2relax',
        backup=backup,
        settings_override=[
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
        ]
    )

    step3 = VaspJob(
        vasp_cmd=vasp_cmd,
        copy_magmom=True,
        final=True,
        suffix='.3static',
        backup=backup,
        settings_override=[
            {"dict": "INCAR", "action": {"_set": {
                "IBRION": -1,
                "NSW": 0,
                "ISMEAR": -5
            }}},
            {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
        ]
    )

    jobs = [step1, step2, step3]
    c = Custodian(handlers, jobs, max_errors=3)
    c.run()
    os.chdir(original_dir)


"""
!!!WARNING!!! You probably want to have volumes in decreasing order eg:
volumes = []
for vol in range(300, 370, 10):
    volumes.append(vol)
volumes.reverse()

or
volumes = list(np.linspace(340, 270, 11))

Path should contain starting POSCAR, POTCAR, INCAR, KPOINTS

When restarting, the last volume folder will be deleted and
the second to last volume folder will be used as the starting point.

"""
def vol_series(path, volumes, vasp_cmd, handlers, restarting=False):  
    #write a params.json file to keep track of the parameters used
    #unfortunately, handlers is not json serializable, so the value is replace by a useless string
    params = {'path': path,
              'volumes': volumes,
              'vasp_cmd': vasp_cmd,
              'handlers': 'handlers is not json serializable',
              'restarting': restarting}
    params_json_path = os.path.join(path, 'params.json')
    n = 0
    while os.path.isfile(params_json_path):
        n += 1
        params_json_path = os.path.join(path, 'params_' + str(n) + '.json')
    with open(params_json_path, 'w') as file:
        json.dump(params, file)
    
    # If restarting, find the last volume folder and delete it
    if restarting:
        for j in range(len(volumes)):
            vol_folder_name = 'vol_' + str(j)
            vol_folder_path = os.path.join(path, vol_folder_name)
            if not os.path.exists(vol_folder_path):
                last_vol_folder_name = 'vol_' + str(j - 1)
                last_vol_folder_path = os.path.join(path, last_vol_folder_name)
                break
        if j == 0:
            print("No volumes to restart from. You might want to set restarting=False (which is the default) or check if 'vol_0' exists inside the path")
            return
        
        # Delete the last volume folder
        last_vol_index = j-1
        shutil.rmtree(last_vol_folder_path)

    for i, vol in enumerate(volumes):
        # if restarting, skip volumes that have already been run
        if restarting and i < last_vol_index:
            continue

        # Create vol folder
        vol_folder_name = 'vol_' + str(i)
        vol_folder_path = os.path.join(path, vol_folder_name)
        os.makedirs(vol_folder_path)

        if i == 0:  # Copy from path
            files_to_copy = ['INCAR', 'KPOINTS', 'POSCAR', 'POTCAR']
            for file_name in files_to_copy:
                if os.path.isfile(os.path.join(path, file_name)):
                    shutil.copy2(os.path.join(path, file_name), os.path.join(vol_folder_path, file_name))
        else:  # Copy from previous folder and delete WAVECARs, CHGCARs, CHGs, PROCARs from previous volume folder
            previous_vol_folder_path = os.path.join(path, 'vol_' + str(i - 1))
            source_name_dest_name = [('CONTCAR.3static', 'POSCAR'),
                                    ('INCAR.2relax', 'INCAR'),
                                    ('KPOINTS.1relax', 'KPOINTS'),
                                    ('POTCAR', 'POTCAR'),
                                    ('WAVECAR.3static', 'WAVECAR'),
                                    ('CHGCAR.3static', 'CHGCAR')]
            for file_name in source_name_dest_name:
                file_source = os.path.join(previous_vol_folder_path, file_name[0])
                file_dest = os.path.join(vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)
            # After copying, it is safe to delete some of the WAVECARS, CHGCARS, CHG and PROCARS from the previous volume folder to save space
            # Keeps WAVECAR.3static and CHGCAR.3static
            files_to_delete = ['WAVECAR.1relax', 'WAVECAR.2relax',
                            'CHGCAR.1relax', 'CHGCAR.2relax',
                            'CHG.1relax', 'CHG.2relax', 'CHG.3static',
                            'PROCAR.1relax', 'PROCAR.2relax', 'PROCAR.3static']
            paths_to_delete = []
            for file_name in files_to_delete:
                file_path = os.path.join(previous_vol_folder_path, file_name)
                paths_to_delete.append(file_path)

            for file_path in paths_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                elif restarting and i == last_vol_index:
                    pass
                else:
                    print(f"The file {file_path} does not exist.")

        # Change the volume of the POSCAR
        poscar = os.path.join(vol_folder_path, 'POSCAR')
        struct = structure.Structure.from_file(poscar)
        struct.scale_lattice(vol)
        struct.to_file(poscar, "POSCAR")

        # Run VASP
        print('Running three step relaxation for volume ' + str(vol))
        three_step_relaxation(vol_folder_path, vasp_cmd, handlers, backup=False)


"""
kpoints_list should be a list of strings ex:
    ['1 1 1', '2 2 2', '3 3 3']
incar_tags should be a dictionary ex:
    {'encut' 'ISMEAR': -5, 'IBRION': 2}
only edits the forth line of the KPOINTS file

Todo: use pymatgen to edit the KPOINTS file, not koints_list
"""
def kpoints_conv_test(path, kpoints_list, vasp_cmd, handlers,
                      backup=False):  # Path should contain starting POSCAR, POTCAR, INCAR, KPOINTS
    original_dir = os.getcwd()
    kpoints_conv_dir = os.path.join(path, 'kpoints_conv')
    os.makedirs(kpoints_conv_dir)
    shutil.copy2(os.path.join(path, 'POSCAR'), os.path.join(kpoints_conv_dir, 'POSCAR'))
    shutil.copy2(os.path.join(path, 'POTCAR'), os.path.join(kpoints_conv_dir, 'POTCAR'))
    shutil.copy2(os.path.join(path, 'INCAR'), os.path.join(kpoints_conv_dir, 'INCAR'))
    shutil.copy2(os.path.join(path, 'KPOINTS'), os.path.join(kpoints_conv_dir, 'KPOINTS'))
    os.chdir(kpoints_conv_dir)
    for i, el in enumerate(kpoints_list):
        # Change the kpoints file
        with open('KPOINTS', 'r') as file:
            lines = file.readlines()
            lines[3] = el + '\n'

        with open('KPOINTS', 'w') as file:
            file.writelines(lines)

        # Run the VASP job
        if i == len(kpoints_list) - 1:
            final = True
        else:
            final = False

        job = VaspJob(
            vasp_cmd=vasp_cmd,
            final=False,
            suffix=f'.{i}',
            backup=backup
        )
        c = Custodian(handlers, [job], max_errors=3)
        c.run()
        if os.path.isfile(f'WAVECAR.[i-1]'):
            os.remove(f'WAVECAR.[i-1]')
        if os.path.isfile(f'CHGCAR.[i-1]'):
            os.remove(f'CHGCAR.[i-1]')
        if os.path.isfile(f'CHG.[i-1]'):
            os.remove(f'CHG.[i-1]')
        if os.path.isfile(f'PROCAR.[i-1]'):
            os.remove(f'PROCAR.[i-1]')
    os.chdir(original_dir)


# TODO: Good idea for the below. Maybe we can combine the convergence and plot in the above functions?
def calculate_kpoint_convergence():
    pass


def plot_kpoint_convergence():
    pass


def calculate_encut_convergence():
    pass


def encut_convergence_test():
    pass


def plot_encut_convergence():
    pass


if __name__ == "__main__":
    print("This is a module for importing. It is not meant to be run directly.")

    # # Specify custodian handlers
    # subset = list(VaspErrorHandler.error_msgs.keys())
    # subset.remove("algo_tet")
    # handlers = [VaspErrorHandler(errors_subset_to_catch=subset)]

    # # Specify VASP command
    # vasp_cmd = ["srun", "vasp_std"]

    # # three_step_relaxation('', vasp_cmd, handlers)
    
    # volumes = list(np.linspace(370, 270, 15))

    # vol_series(os.getcwd(), volumes, vasp_cmd, handlers, restarting=True)

    # # kpoints_list = ['4 4 5', '5 5 6', '6 6 7', '7 7 8', '7 7 9', '8 8 10', '12 12 15']
    # # kpoints_conv_test(os.getcwd(), kpoints_list, vasp_cmd, handlers, backup=False)
