"""
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

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from scipy.optimize import fsolve
from scipy.optimize import leastsq
import argparse

def eos_fit(num_structures, *input_files, eos_index, plot_ev=True, plot_pv=False):
    # 1. Load the data
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

        if len(eos_index) > 1:
            raise ValueError("eos_index for multiple structures should only have one value")

    # Remove any duplicates
    eos_index = list(set(eos_index))
    eos_index.sort()

    # Conversion factor: 1 eV/Å^3 = 160.2176621 GPa
    conv_factor = 160.2176621

    # 2. Fit to equations of state
    # Equations of state for E-V fitting
    def mBM4(volume, energy):
        eos_index = 1

        AA = np.vstack((np.ones(np.shape(volume)), volume ** (-1 / 3), volume ** (-2 / 3), volume ** (-1)))  # (nx4)
        AA = AA.T
        xx1 = np.linalg.pinv(AA)
        xx = xx1.dot(energy)  # (4x1) = (4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        a = xx[0]
        b = xx[1]
        c = xx[2]
        d = xx[3]
        e = 0.0
        xx = [a, b, c, d, e]

        energy_eos = a + b * (volume_range) ** (-1 / 3) + c * (volume_range) ** (-2 / 3) + d * (volume_range) ** (
            -1) + e * (volume_range) ** (-4 / 3)
        energy_eos_points = a + b * (volume) ** (-1 / 3) + c * (volume) ** (-2 / 3) + d * (volume) ** (-1) + e * (
            volume) ** (
                                    -4 / 3)
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
        xx1 = np.linalg.pinv(AA);
        xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        a = xx[0]
        b = xx[1]
        c = xx[2]
        d = xx[3]
        e = xx[4]
        xx = [a, b, c, d, e]

        energy_eos = a + b * (volume_range) ** (-1 / 3) + c * (volume_range) ** (-2 / 3) + d * (volume_range) ** (
            -1) + e * (volume_range) ** (-4 / 3)
        energy_eos_points = a + b * (volume) ** (-1 / 3) + c * (volume) ** (-2 / 3) + d * (volume) ** (-1) + e * (
            volume) ** (
                                    -4 / 3)
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
        xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        a = xx[0]
        b = xx[1]
        c = xx[2]
        d = xx[3]
        e = 0.0
        xx = [a, b, c, d, e]

        energy_eos = a + b * (volume_range) ** (-2 / 3) + c * (volume_range) ** (-4 / 3) + d * (volume_range) ** (
            -2) + e * (
                         volume_range) ** (-8 / 3)
        energy_eos_points = a + b * (volume) ** (-2 / 3) + c * (volume) ** (-4 / 3) + d * (volume) ** (-2) + e * (
            volume) ** (
                                    -8 / 3)
        energy_difference = energy_eos_points - energy

        V = math.sqrt(
            -((4 * c ** 3 - 9 * b * c * d + math.sqrt(
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
        xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        a = xx[0]
        b = xx[1]
        c = xx[2]
        d = xx[3]
        e = xx[4]
        xx = [a, b, c, d, e]

        energy_eos = a + b * (volume_range) ** (-2 / 3) + c * (volume_range) ** (-4 / 3) + d * (volume_range) ** (
            -2) + e * (volume_range) ** (-8 / 3)
        energy_eos_points = a + b * (volume) ** (-2 / 3) + c * (volume) ** (-4 / 3) + d * (volume) ** (-2) + e * (
            volume) ** (
                                    -8 / 3)
        energy_difference = energy_eos_points - energy

        func = lambda volume_range: ((8 * e) / (3 * volume_range ** (11 / 3)) + (2 * d) / volume_range ** 3 + (
                4 * c) / (3 * volume_range ** (7 / 3)) + (2 * b) / (
                                             3 * volume_range ** (5 / 3))) * conv_factor
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
                3 * volume_range ** (7 / 3)) + (2 * b) / (
                                3 * volume_range ** (5 / 3))) * conv_factor
        energy_eos = np.concatenate(([eos_index], energy_eos))
        pressure_eos = np.concatenate(([eos_index], pressure_eos))

        return results, energy_eos, pressure_eos

    def LOG4(volume, energy):
        eos_index = 5
        AA = np.vstack((np.ones(np.shape(volume)), np.log(volume), np.log(volume) ** 2, np.log(volume) ** 3))
        AA = AA.T
        xx1 = np.linalg.pinv(AA)
        xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        a = xx[0]
        b = xx[1]
        c = xx[2]
        d = xx[3]
        e = 0.0
        xx = [a, b, c, d, e, ]

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
        xx = xx1.dot(energy)  # (4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
        a = xx[0]
        b = xx[1]
        c = xx[2]
        d = xx[3]
        e = xx[4]
        xx = [a, b, c, d, e]

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
        # V-1      E0-2       B-3        bp-4
        V = xini[0]
        E0 = xini[1]
        B = xini[2]
        bp = xini[3]
        volume_range = Data[:, 0]
        y = Data[:, 1]
        eng = E0 - (B * V) / (-1 + bp) + (B * (1 + (V / volume_range) ** bp / (-1 + bp)) * volume_range) / bp
        return (eng - y)

    def murnaghan(volume, energy):
        eos_index = 7
        volume = volume
        Data = np.vstack((volume, energy))
        Data = Data.T

        [results, energy_eos, pressure_eos] = mBM4(volume, energy)
        xini = [results[1], results[2], results[4] / conv_factor,
                results[5]]

        # % V-1      E-2       B-3           bp-4
        # %xini = [eos_parameters(1), eos_parameters(2), eos_parameters(3)/conv_factor, eos_parameters(4)]; % FEG_vdos_original as ieos==0
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
        # V-1      E0-2       B-3        bp-4
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
        xini = [results[1], results[2], results[4] / conv_factor,
                results[5]]
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

    # 3. Plot and output the results
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
        if plot_ev and not plot_pv:
            j = 0
            plt.figure(figsize=(8, 5))
            for key in volume:
                volume_range = volume_range_list[key]
                plt.scatter(volume[key], energy[key])
                plt.scatter(results[j][1], results[j][2], facecolors='none', edgecolors='k')
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
            plt.legend()
            plt.show()
        if plot_pv and not plot_ev:
            j = 0
            plt.figure(figsize=(8, 5))
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
            plt.legend()
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

        results = np.vstack(results)
        results = np.delete(results, 3, axis=1)

        energy_eos = np.vstack(energy_eos)
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
        return results, energy_eos, pressure_eos

os.chdir('PbTiO3/EV_curves/r2SCAN')
#[results, energy_eos, pressure_eos, volume_range] = eos_fit('single', '90DW', eos_index=[1,2], plot_ev=True, plot_pv=False)


input_files = ['FEG', '90DW', '180DW']
[results, energy_eos, pressure_eos] = eos_fit('multiple', *input_files, eos_index=[4], plot_ev=True, plot_pv=False)

ipress = -1
if ipress > 0:
    press0 = np.loadtxt('pressure_FEG') # kBar
    pressure = press0 / 10 # GPa


def pvbuildeos(prop, L):
    # %  V0  P0  B  BP  B2P  av_diff   max_diff
    # %  1   2   3   4  5     6           7
    V = prop[0];  # %  V0[A]3/atom)
    P0 = prop[1];  # %  P0[G]a)
    B = prop[2];  # %  B  GPa
    bp = prop[3];  # %  BP
    b2p = prop[4];  # %  b2p  (1/GPa)

    if L == 1 or L == 2:
        e = (3 * B * (74 + 9 * B * b2p - 45 * bp + 9 * bp ** 2) * V ** (7 / 3)) / 8;
        d = (-3 * B * (83 + 9 * B * b2p - 48 * bp + 9 * bp ** 2) * V ** 2) / 2;
        c = (9 * B * (94 + 9 * B * b2p - 51 * bp + 9 * bp ** 2) * V ** (5 / 3)) / 4;
        b = (-3 * B * (107 + 9 * B * b2p - 54 * bp + 9 * bp ** 2) * V ** (4 / 3)) / 2;
        if abs(e) < 1e-8:
            e = 0;
        res1 = [b, c, d, e];

    if L == 3 or L == 4:
        e = (3 * B * (143 + 9 * B * b2p - 63 * bp + 9 * bp ** 2) * V ** (11 / 3)) / 128;
        d = (-3 * B * (167 + 9 * B * b2p - 69 * bp + 9 * bp ** 2) * V ** 3) / 32;
        c = (9 * B * (199 + 9 * B * b2p - 75 * bp + 9 * bp ** 2) * V ** (7 / 3)) / 64;
        b = (-3 * B * (239 + 9 * B * b2p - 81 * bp + 9 * bp ** 2) * V ** (5 / 3)) / 32;
        if abs(e) < 1e-8:
            e = 0;
        res1 = [b, c, d, e];
    return res1


def pv2prop(xx, vzero, icase):
    chunit = 160.2189
    b = xx[0]
    c = xx[1]
    d = xx[2]
    e = xx[3]

    if icase == 1:
        V = 4 * c ** 3 - 9 * b * c * d + np.sqrt((c ** 2 - 3 * b * d) * (4 * c ** 2 - 3 * b * d) ** 2)
        V = -V / b ** 3

    if icase == 2:
        fun = lambda x: (
                (4 * e) / (3 * x ** (7 / 3)) + d / x ** 2 + (2 * c) / (3 * x ** (5 / 3)) + b / (3 * x ** (4 / 3)))
        V = fsolve(fun, vzero)

    if icase == 1 or icase == 2:
        P = (4 * e) / (3 * V ** (7 / 3)) + d / V ** 2 + (2 * c) / (3 * V ** (5 / 3)) + b / (3 * V ** (4 / 3))
        B = ((28 * e) / (9 * V ** (10 / 3)) + (2 * d) / V ** 3 + (10 * c) / (9 * V ** (8 / 3)) + (4 * b) / (
                9 * V ** (7 / 3))) * V
        BP = (98 * e + 54 * d * V ** (1 / 3) + 25 * c * V ** (2 / 3) + 8 * b * V) / (
                42 * e + 27 * d * V ** (1 / 3) + 15 * c * V ** (2 / 3) + 6 * b * V)
        B2P = (V ** (8 / 3) * (9 * d * (14 * e + 5 * c * V ** (2 / 3) + 8 * b * V) + 2 * V ** (1 / 3) * (
                126 * b * e * V ** (1 / 3) + 5 * c * (28 * e + b * V)))) / (
                      2 * (14 * e + 9 * d * V ** (1 / 3) + 5 * c * V ** (2 / 3) + 2 * b * V) ** 3)
        res1 = [V, P, B, BP, B2P]

    if icase == 3:
        V = np.sqrt(
            -((4 * c ** 3 - 9 * b * c * d + np.sqrt((c ** 2 - 3 * b * d) * (4 * c ** 2 - 3 * b * d) ** 2)) / b ** 3))

    if icase == 4:
        fun = lambda x: ((8 * e) / (3 * x ** (11 / 3)) + (2 * d) / x ** 3 + (4 * c) / (3 * x ** (7 / 3)) + (2 * b) / (
                3 * x ** (5 / 3))) * chunit
        V = fsolve(fun, vzero)

    if icase == 3 or icase == 4:
        P = (8 * e) / (3 * V ** (11 / 3)) + (2 * d) / V ** 3 + (4 * c) / (3 * V ** (7 / 3)) + (2 * b) / (
                3 * V ** (5 / 3))
        B = (2 * (44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2)) / (9 * V ** (11 / 3));
        BP = (484 * e + 243 * d * V ** (2 / 3) + 98 * c * V ** (4 / 3) + 25 * b * V ** 2) / (
                132 * e + 81 * d * V ** (2 / 3) + 42 * c * V ** (4 / 3) + 15 * b * V ** 2)
        B2P = (4 * V ** (13 / 3) * (27 * d * (22 * e + 7 * c * V ** (4 / 3) + 10 * b * V ** 2) + V ** (2 / 3) * (
                990 * b * e * V ** (2 / 3) + 7 * c * (176 * e + 5 * b * V ** 2)))) / (
                      44 * e + 27 * d * V ** (2 / 3) + 14 * c * V ** (4 / 3) + 5 * b * V ** 2) ** 3
        res1 = [V, P, B, BP, B2P]

    return res1


def pveosfit(volume, pressure, volume_range, isave, ifigure, kkfig):
    numbereos = 4
    ieos = [1, 2, 3, 4]
    chunit = 160.2189
    res = []
    resdiff = []
    resxx = []
    respp = []

    for L in ieos:

        if L == 1:  # mBM4
            AA = [volume ** (-4 / 3) * (1 / 3), volume ** (-5 / 3) * (2 / 3), volume ** (-2)]
            AA = np.array(AA)
            AA = AA.T
            xx1 = np.linalg.pinv(AA)
            xx = xx1.dot(pressure)  # (4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
            b = xx[0]
            c = xx[1]
            d = xx[2]
            e = 0.0
            xx = [b, c, d, e]

            pressure_eos = (4 * e) / (3 * volume_range ** (7 / 3)) + d / volume_range ** 2 + (2 * c) / (3 * volume_range ** (5 / 3)) + b / (3 * volume_range ** (4 / 3))
            pressure_eos_points = (4 * e) / (3 * volume ** (7 / 3)) + d / volume ** 2 + (2 * c) / (3 * volume ** (5 / 3)) + b / (
                    3 * volume ** (4 / 3))
            plt.plot(volume_range, pressure_eos)
            plt.scatter(volume, pressure)
            plt.title('mBM4')
            plt.xlabel(r'Volume ($\mathrm{\AA}^3$)')
            plt.ylabel('Pressure (GPa)')
            plt.show()
            # Continue here!
            prop = pv2prop(xx, np.mean(volume), L)  # [V0, P, B, BP, B2P]
            newxx = pvbuildeos(prop, L)
            xxnp = np.array(xx)
            resxx1 = np.insert(xxnp, 0, L)
            resxx2 = np.insert(newxx, 0, L)
            resxx_2 = np.vstack((resxx1, resxx2))
            diffp = pressure_eos_points - pressure

            resxx = resxx_2
            nnn = len(pressure)
            qwe = diffp ** 2
            asd = math.sqrt(sum(qwe / nnn))
            respre = np.insert(prop, 0, L)
            res_2 = np.hstack((respre, asd))
            res = res_2
            resee = res
            respp = pressure_eos
            resdiff = diffp

        if L == 2:  # mBM5
            AA = [volume ** (-4 / 3) * (1 / 3), volume ** (-5 / 3) * (2 / 3), volume ** (-2), volume ** (-7 / 3) * (4 / 3)]
            AA = np.array(AA)
            AA = AA.T
            xx1 = np.linalg.pinv(AA)
            xx = xx1.dot(pressure)  # (4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
            b = xx[0]
            c = xx[1]
            d = xx[2]
            e = xx[3]
            xx = [b, c, d, e]

            pressure_eos = (4 * e) / (3 * volume_range ** (7 / 3)) + d / volume_range ** 2 + (2 * c) / (3 * volume_range ** (5 / 3)) + b / (3 * volume_range ** (4 / 3))
            pressure_eos_points = (4 * e) / (3 * volume ** (7 / 3)) + d / volume ** 2 + (2 * c) / (3 * volume ** (5 / 3)) + b / (
                    3 * volume ** (4 / 3))

            plt.plot(volume_range, pressure_eos, volume, pressure, 'o')
            plt.title('P-V FITTED curve, mBM5, No. 2, GPa')
            diffp = pressure_eos_points - pressure
            prop = pv2prop(xx, np.mean(volume), L)  # [V0, P, B, BP, B2P]
            newxx = pvbuildeos(prop, L)
            xxnp = np.array(xx)
            resxx1 = np.insert(xxnp, 0, L)
            resxx2 = np.insert(newxx, 0, L)
            resxx_2 = np.vstack((resxx1, resxx2))
            resxx = np.hstack((resxx, resxx_2))
            nnn = len(pressure)
            qwe = diffp ** 2
            asd = math.sqrt(sum(qwe / nnn))
            respre = np.insert(prop, 0, L)
            res_2 = np.hstack((respre, asd))
            res = np.vstack((res, res_2))
            resee = np.vstack((resee, res))
            respp = np.vstack((respp, pressure_eos))
            resdiff = np.vstack((resdiff, diffp))

        if L == 3:  # BM4
            AA = [volume ** (-5 / 3) * (2 / 3), volume ** (-7 / 3) * (4 / 3), volume ** (-3) * 2]
            AA = np.array(AA)
            AA = AA.T
            xx1 = np.linalg.pinv(AA)
            xx = xx1.dot(pressure)  # (4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
            b = xx[0]
            c = xx[1]
            d = xx[2]
            e = 0.0
            xx = [b, c, d, e]
            x = volume_range
            pressure_eos = (8 * e) / (3 * x ** (11 / 3)) + (2 * d) / x ** 3 + (4 * c) / (3 * x ** (7 / 3)) + (2 * b) / (
                    3 * x ** (5 / 3))
            x = volume
            pressure_eos_points = (8 * e) / (3 * x ** (11 / 3)) + (2 * d) / x ** 3 + (4 * c) / (3 * x ** (7 / 3)) + (2 * b) / (
                    3 * x ** (5 / 3))

            plt.plot(volume_range, pressure_eos, volume, pressure, 'o')
            plt.title('P-V FITTED curve, BM4, No. 3, GPa')
            diffp = pressure_eos_points - pressure
            prop = pv2prop(xx, np.mean(volume), L)  # [V0, P, B, BP, B2P]
            newxx = pvbuildeos(prop, L)

            xxnp = np.array(xx)
            resxx1 = np.insert(xxnp, 0, L)
            resxx2 = np.insert(newxx, 0, L)
            resxx_2 = np.vstack((resxx1, resxx2))
            resxx = np.hstack((resxx, resxx_2))
            nnn = len(pressure)
            qwe = diffp ** 2
            asd = math.sqrt(sum(qwe / nnn))
            respre = np.insert(prop, 0, L)
            res_2 = np.hstack((respre, asd))
            res = np.vstack((res, res_2))
            resee = np.vstack((resee, res))
            respp = np.vstack((respp, pressure_eos))
            resdiff = np.vstack((resdiff, diffp))

        if L == 4:  # BM5
            AA = [volume ** (-5 / 3) * (2 / 3), volume ** (-7 / 3) * (4 / 3), volume ** (-3) * 2, volume ** (-11 / 3) * (8 / 3)]
            AA = np.array(AA)
            AA = AA.T
            xx1 = np.linalg.pinv(AA)
            xx = xx1.dot(pressure)  # (4x1)=(4xn)*(nx1), solve it by pseudo-inversion: Ax=b
            b = xx[0]
            c = xx[1]
            d = xx[2]
            e = xx[3]
            xx = [b, c, d, e]

            x = volume_range
            pressure_eos = (8 * e) / (3 * x ** (11 / 3)) + (2 * d) / x ** 3 + (4 * c) / (3 * x ** (7 / 3)) + (2 * b) / (
                    3 * x ** (5 / 3))
            x = volume
            pressure_eos_points = (8 * e) / (3 * x ** (11 / 3)) + (2 * d) / x ** 3 + (4 * c) / (3 * x ** (7 / 3)) + (2 * b) / (
                    3 * x ** (5 / 3))

            plt.plot(volume_range, pressure_eos, volume, pressure, 'o')
            plt.title('P-V curve, BM5, No. 4, GPa')
            diffp = pressure_eos_points - pressure
            prop = pv2prop(xx, np.mean(volume), L)  # [V0, P, B, BP, B2P]

            newxx = pvbuildeos(prop, L)
            xxnp = np.array(xx)
            resxx1 = np.insert(xxnp, 0, L)
            resxx2 = np.insert(newxx, 0, L)
            resxx_2 = np.vstack((resxx1, resxx2))
            resxx = np.hstack((resxx, resxx_2))
            nnn = len(pressure)
            qwe = diffp ** 2
            asd = math.sqrt(sum(qwe / nnn))
            respre = np.insert(prop, 0, L)
            res_2 = np.hstack((respre, asd))
            res = np.vstack((res, res_2))
            resee = np.vstack((resee, res))
            respp = np.vstack((respp, pressure_eos))
            resdiff = np.vstack((resdiff, diffp))

    if numbereos == 4:
        pp4 = np.vstack((respp[0, ...], respp[2, ...]))
        pp4 = pp4.T
        pp5 = np.vstack((respp[1, ...], respp[3, ...]))
        pp5 = pp5.T
        if kkfig > 0:
            figpv4 = plt.figure('fig-pv4')
            plt.plot(volume_range, pp4, volume_range, pp5, '--', volume, pressure, 'o')
            plt.title('P-V Fitted of 4 curves')
            plt.savefig('fig2_pv_fitted.png')

    if numbereos == 2:
        pp4 = respp[0, ...]
        pp4 = pp4.T
        pp5 = respp[1, ...]
        pp5 = pp5.T
        if kkfig > 0:
            figpv2 = plt.figure('fig-pv2')
            plt.plot(volume_range, pp4, volume_range, pp5, '--', volume, pressure, 'o')
            plt.title('P-V Fitted of 2 curves'),

    resdiff = resdiff.T
    max_diff = []
    n = 0
    max_diff1 = np.fabs(resdiff)
    max_diff2 = max_diff1.argmax(axis=0)
    for i in max_diff2:
        max_diff = np.append(max_diff, max_diff1[i, n])
        n = n + 1

    av_diff = np.mean(max_diff1, axis=0)
    dpp_av_max = np.vstack((av_diff, max_diff))

    pvfit_res = res  # [res(:,1:2), res(:,4:end)]
    np.savetxt("pvfit_res.txt", pvfit_res, fmt='%.4f')

    if isave > 0:
        resvp = np.hstack((volume_range.T, respp.T))
        np.savetxt("out_fit_VP.txt", resvp, fmt='%.4f')
    return res


if ipress > 0:
    data = np.loadtxt('FEG')
    volume = data[:, 0]
    energy = data[:, 1]
    volume_range = np.arange(min(volume) - 1, max(volume) + 2, 0.05)
    pvfit_res = pveosfit(volume, pressure, volume_range, -9, -9, 9)
    # compare_res = np.vstack((fitted_res[0, ...], pvfit_res[0, ...], fitted_res[1, ...], pvfit_res[1, ...],
    #                         fitted_res[2, ...], pvfit_res[2, ...], fitted_res[3, ...], pvfit_res[3, ...]))
    # final_res = np.vstack((fitted_res, pvfit_res))


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
# TODO: Add PV fitting

