import os
import pytest
import numpy as np
from dfttk.config import Configuration


def test_analyze_encut_conv():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "tests_data/Al/conv_test")
    config_Al = Configuration(path, "config_Al")
    encut_conv_df, fig = config_Al.analyze_encut_conv(plot=False)

    encut = encut_conv_df["encut"].values
    kpoints_grid = encut_conv_df["kpoint_grid"].values[0]
    kppa = encut_conv_df["kppa"].values[0]
    energy = encut_conv_df["energy"].values
    number_of_atoms = encut_conv_df["number_of_atoms"].values[0]
    energy_per_atom = encut_conv_df["energy_per_atom"].values
    difference_mev_per_atom = encut_conv_df["difference_mev_per_atom"].values

    expected_encut = np.array(
        [270, 320, 370, 420, 470, 520, 570, 620, 670, 720, 770, 820]
    )
    expected_kpoints_grid = [9, 9, 9]
    expected_kppa = 2916
    expected_energy = np.array(
        [
            -14.952094,
            -14.966816,
            -14.973924,
            -14.975857,
            -14.976588,
            -14.97668,
            -14.976812,
            -14.977072,
            -14.977261,
            -14.977352,
            -14.977392,
            -14.977423,
        ]
    )
    expected_number_of_atoms = 4
    expected_energy_per_atom = np.array(
        [
            -3.7380235,
            -3.741704,
            -3.743481,
            -3.74396425,
            -3.744147,
            -3.74417,
            -3.744203,
            -3.744268,
            -3.74431525,
            -3.744338,
            -3.744348,
            -3.74435575,
        ]
    )
    expected_difference_mev_per_atom = np.array(
        [
            np.nan,
            -3.6805,
            -1.777,
            -0.48325,
            -0.18275,
            -0.023,
            -0.033,
            -0.065,
            -0.04725,
            -0.02275,
            -0.01,
            -0.00775,
        ]
    )

    assert np.array_equal(
        encut, expected_encut
    ), f"Expected {expected_encut}, but got {encut}"
    assert np.array_equal(
        kpoints_grid, expected_kpoints_grid
    ), f"Expected {expected_kpoints_grid}, but got {kpoints_grid}"
    assert kppa == expected_kppa, f"Expected {expected_kppa}, but got {kppa}"
    assert np.array_equal(
        energy, expected_energy
    ), f"Expected {expected_energy}, but got {energy}"
    assert (
        number_of_atoms == expected_number_of_atoms
    ), f"Expected {expected_number_of_atoms}, but got {number_of_atoms}"
    assert np.array_equal(
        energy_per_atom, expected_energy_per_atom
    ), f"Expected {expected_energy_per_atom}, but got {energy_per_atom}"
    assert np.allclose(
        difference_mev_per_atom, expected_difference_mev_per_atom, equal_nan=True
    ), f"Expected {expected_difference_mev_per_atom}, but got {difference_mev_per_atom}"


def test_analyze_kpoints_conv():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "tests_data/Al/conv_test")
    config_Al = Configuration(path, "config_Al")
    kpoints_conv_df, fig = config_Al.analyze_kpoints_conv(plot=False)

    encut = kpoints_conv_df["encut"].values[0]
    kpoints_grid = np.array([list(x) for x in kpoints_conv_df["kpoint_grid"].values])
    kppa = kpoints_conv_df["kppa"].values
    energy = kpoints_conv_df["energy"].values
    number_of_atoms = kpoints_conv_df["number_of_atoms"].values[0]
    energy_per_atom = kpoints_conv_df["energy_per_atom"].values
    difference_mev_per_atom = kpoints_conv_df["difference_mev_per_atom"].values

    expected_encut = 520
    expected_kpoints_grid = np.array(
        [
            [6, 6, 6],
            [7, 7, 7],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
        ]
    )
    expected_kppa = np.array([864, 1372, 2916, 4000, 5324, 6912, 8788])
    expected_energy = np.array(
        [
            -15.058461,
            -14.990809,
            -14.97668,
            -14.973059,
            -14.98156,
            -14.980429,
            -14.987405,
        ]
    )
    expected_number_of_atoms = 4
    expected_energy_per_atom = np.array(
        [
            -3.76461525,
            -3.74770225,
            -3.74417,
            -3.74326475,
            -3.74539,
            -3.74510725,
            -3.74685125,
        ]
    )
    expected_difference_mev_per_atom = np.array(
        [np.nan, 16.913, 3.53225, 0.90525, -2.12525, 0.28275, -1.744]
    )

    assert encut == expected_encut, f"Expected {expected_encut}, but got {encut}"
    assert np.array_equal(
        kpoints_grid, expected_kpoints_grid
    ), f"Expected {expected_kpoints_grid}, but got {kpoints_grid}"
    assert np.array_equal(
        kppa, expected_kppa
    ), f"Expected {expected_kppa}, but got {kppa}"
    assert np.array_equal(
        energy, expected_energy
    ), f"Expected {expected_energy}, but got {energy}"
    assert (
        number_of_atoms == expected_number_of_atoms
    ), f"Expected {expected_number_of_atoms}, but got {number_of_atoms}"
    assert np.array_equal(
        energy_per_atom, expected_energy_per_atom
    ), f"Expected {expected_energy_per_atom}, but got {energy_per_atom}"
    assert np.allclose(
        difference_mev_per_atom, expected_difference_mev_per_atom, equal_nan=True
    ), f"Expected {expected_difference_mev_per_atom}, but got {difference_mev_per_atom}"


if __name__ == "__main__":
    pytest.main()
