# Standard library imports
import os
import math
from math import isclose
import json

# Third-party library imports
import numpy as np
import pytest

# DFTTK imports
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


def _convert_pbc_lists_to_tuples(data):
    data["lattice"]["pbc"] = tuple(data["lattice"]["pbc"])
    return data


def _assert_selected_keys_almost_equal(dict1, dict2, keys, tol=1e-4):
    for key in keys:
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], float) and isinstance(dict2[key], float):
                assert math.isclose(
                    dict1[key], dict2[key], rel_tol=tol
                ), f"Expected {dict2[key]} for key '{key}', but got {dict1[key]}"
            else:
                assert (
                    dict1[key] == dict2[key]
                ), f"Expected {dict2[key]} for key '{key}', but got {dict1[key]}"


def test_process_ev_curves():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "tests_data/Al/config_Al")
    config_Al = Configuration(path, "config_Al")

    config_Al.process_ev_curves()

    with open(os.path.join(current_dir, "expected_ev_curves_incars.json"), "r") as f:
        expected_incars = json.load(f)

    for actual_incar, expected_incar in zip(
        config_Al.ev_curves.incars, expected_incars
    ):
        assert (
            actual_incar == expected_incar
        ), f"Expected {expected_incar}, but got {actual_incar}"

    with open(os.path.join(current_dir, "expected_ev_curves_kpoints.json"), "r") as f:
        expected_kpoints = json.load(f)

    for actual_kpoint, expected_kpoint in zip(
        config_Al.ev_curves.kpoints.as_dict(), expected_kpoints
    ):
        assert (
            actual_kpoint == expected_kpoint
        ), f"Expected {expected_kpoint}, but got {actual_kpoint}"

    assert (
        config_Al.ev_curves.number_of_atoms == 4
    ), f"Expected 4, but got {config_Al.ev_curves.number_of_atoms}"
    assert config_Al.ev_curves.volumes == [
        74.0,
        72.0,
        70.0,
        68.0,
        66.0,
        64.0,
        62.0,
        60.0,
    ], f"Expected [74.0, 72.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0], but got {config_Al.ev_curves.volumes}"
    assert config_Al.ev_curves.energies == [
        -14.787067,
        -14.863567,
        -14.92244,
        -14.960229,
        -14.973035,
        -14.955434,
        -14.902786,
        -14.808673,
    ], f"Expected [-14.787067, -14.863567, -14.92244, -14.960229, -14.973035, -14.955434, -14.902786, -14.808673], but got {config_Al.ev_curves.energies}"

    assert config_Al.ev_curves.atomic_masses == {
        "Al": 26.981
    }, f"Expected {'Al': 26.981}, but got {config_Al.ev_curves.atomic_masses}"
    assert (
        config_Al.ev_curves.average_mass == 26.981
    ), f"Expected 26.981, but got {config_Al.ev_curves.average_mass}"
    assert (
        config_Al.ev_curves.total_magnetic_moment == None
    ), f"Expected None, but got {config_Al.ev_curves.total_magnetic_moment}"
    assert (
        config_Al.ev_curves.magnetic_ordering == None
    ), f"Expected None, but got {config_Al.ev_curves.magnetic_ordering}"
    assert (
        config_Al.ev_curves.mag_data == []
    ), f"Expected [], but got {config_Al.ev_curves.mag_data}"

    expected_eos_parameters = {
        "V0": 66.10191547034127,
        "E0": -14.972775074363833,
        "B": 77.92792067011315,
        "BP": 4.612739661291564,
        "B2P": -0.06258448064264342,
    }
    keys_to_compare = ["V0", "E0", "B", "BP", "B2P"]
    _assert_selected_keys_almost_equal(
        config_Al.ev_curves.eos_parameters, expected_eos_parameters, keys_to_compare
    )

    actual_relaxed_structures = [
        structure.as_dict() for structure in config_Al.ev_curves.relaxed_structures
    ]

    with open(
        os.path.join(current_dir, "expected_ev_curves_relaxed_structures.json"), "r"
    ) as f:
        expected_phonon_structures = json.load(f)

    for i, expected_relaxed_structure in enumerate(expected_phonon_structures):
        expected_phonon_structures[i] = _convert_pbc_lists_to_tuples(
            expected_relaxed_structure
        )

    assert (
        actual_relaxed_structures == expected_phonon_structures
    ), f"Expected {expected_phonon_structures}, but got {actual_relaxed_structures}"


def test_process_phonons():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "tests_data/Al/config_Al")
    config_Al = Configuration(path, "config_Al")

    num_atoms = 4
    temp_range = list(range(0, 1010, 10))
    config_Al.process_phonons(num_atoms, temp_range)

    with open(os.path.join(current_dir, "expected_phonons_incars.json"), "r") as f:
        expected_incars = json.load(f)

    for actual_incar, expected_incar in zip(config_Al.phonons.incars, expected_incars):
        assert (
            actual_incar == expected_incar
        ), f"Expected {expected_incar}, but got {actual_incar}"

    with open(os.path.join(current_dir, "expected_ev_curves_kpoints.json"), "r") as f:
        expected_kpoints = json.load(f)

    for actual_kpoint, expected_kpoint in zip(
        config_Al.phonons.kpoints.as_dict(), expected_kpoints
    ):
        assert (
            actual_kpoint == expected_kpoint
        ), f"Expected {expected_kpoint}, but got {actual_kpoint}"

    actual_phonon_structures = [
        structure.as_dict() for structure in config_Al.phonons.phonon_structures
    ]

    with open(
        os.path.join(current_dir, "expected_phonons_phonon_structures.json"), "r"
    ) as f:
        expected_phonon_structures = json.load(f)

    for i, expected_relaxed_structure in enumerate(expected_phonon_structures):
        expected_phonon_structures[i] = _convert_pbc_lists_to_tuples(
            expected_relaxed_structure
        )

    assert (
        actual_phonon_structures == expected_phonon_structures
    ), f"Expected {expected_phonon_structures}, but got {actual_phonon_structures}"

    expected_number_of_atoms = 4
    assert (
        config_Al.phonons.number_of_atoms == expected_number_of_atoms
    ), f"Expected 4, but got {config_Al.phonons.number_of_atoms}"

    expected_temperatures = list(range(0, 1010, 10))
    assert (
        config_Al.phonons.temperatures == expected_temperatures
    ), f"Expected {temp_range}, but got {config_Al.phonons.temperatures}"

    expected_volumes = [60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0]
    assert (
        config_Al.phonons.volumes == expected_volumes
    ), f"Expected {expected_volumes}, but got {config_Al.phonons.volumes}"

    tolerance = 1e-4
    with open(
        os.path.join(current_dir, "expected_phonons_helmholtz_energy.json"), "r"
    ) as f:
        expected_helmholtz_energy = json.load(f)
    for temp, expected_values in expected_helmholtz_energy.items():
        actual_values = config_Al.phonons.helmholtz_energy[temp]
        for expected, actual in zip(expected_values, actual_values):
            assert isclose(
                expected, actual, rel_tol=tolerance
            ), f"Expected {expected}, but got {actual} with tolerance {tolerance}"

    with open(
        os.path.join(current_dir, "expected_phonons_internal_energy.json"), "r"
    ) as f:
        expected_internal_energy = json.load(f)
    for temp, expected_values in expected_internal_energy.items():
        actual_values = config_Al.phonons.internal_energy[temp]
        for expected, actual in zip(expected_values, actual_values):
            assert isclose(
                expected, actual, rel_tol=tolerance
            ), f"Expected {expected}, but got {actual} with tolerance {tolerance}"

    with open(os.path.join(current_dir, "expected_phonons_entropy.json"), "r") as f:
        expected_entropy = json.load(f)
    for temp, expected_values in expected_entropy.items():
        actual_values = config_Al.phonons.entropy[temp]
        for expected, actual in zip(expected_values, actual_values):
            assert isclose(
                expected, actual, rel_tol=tolerance
            ), f"Expected {expected}, but got {actual} with tolerance {tolerance}"

    with open(
        os.path.join(current_dir, "expected_phonons_heat_capacity.json"), "r"
    ) as f:
        expected_heat_capacity = json.load(f)
    for temp, expected_values in expected_heat_capacity.items():
        actual_values = config_Al.phonons.heat_capacity[temp]
        for expected, actual in zip(expected_values, actual_values):
            assert isclose(
                expected, actual, rel_tol=tolerance
            ), f"Expected {expected}, but got {actual} with tolerance {tolerance}"

    with open(
        os.path.join(current_dir, "expected_phonons_helmholtz_energy_fit.json"), "r"
    ) as f:
        expected_helmholtz_energy_fit = json.load(f)
    for temp, expected_values in expected_helmholtz_energy_fit[
        "polynomial_coefficients"
    ].items():
        actual_values = config_Al.phonons.helmholtz_energy_fit[
            "polynomial_coefficients"
        ][temp]
        for expected, actual in zip(expected_values, actual_values):
            assert isclose(
                expected, actual, rel_tol=tolerance
            ), f"Expected {expected}, but got {actual} with tolerance {tolerance}"

    with open(os.path.join(current_dir, "expected_phonons_entropy_fit.json"), "r") as f:
        expected_entropy_fit = json.load(f)
    for temp, expected_values in expected_entropy_fit[
        "polynomial_coefficients"
    ].items():
        actual_values = config_Al.phonons.entropy_fit["polynomial_coefficients"][temp]
        for expected, actual in zip(expected_values, actual_values):
            assert isclose(
                expected, actual, rel_tol=tolerance
            ), f"Expected {expected}, but got {actual} with tolerance {tolerance}"

    with open(
        os.path.join(current_dir, "expected_phonons_heat_capacity_fit.json"), "r"
    ) as f:
        expected_heat_capacity_fit = json.load(f)
    for temp, expected_values in expected_heat_capacity_fit[
        "polynomial_coefficients"
    ].items():
        actual_values = config_Al.phonons.heat_capacity_fit["polynomial_coefficients"][
            temp
        ]
        for expected, actual in zip(expected_values, actual_values):
            assert isclose(
                expected, actual, rel_tol=tolerance
            ), f"Expected {expected}, but got {actual} with tolerance {tolerance}"


def test_process_debye():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "tests_data/Al/config_Al")
    config_Al = Configuration(path, "config_Al")

    config_Al.process_ev_curves()
    config_Al.process_debye(scaling_factor=0.617, gruneisen_x=2 / 3)

    expected_number_of_atoms = 4
    assert (
        config_Al.debye.number_of_atoms == expected_number_of_atoms
    ), f"Expected 4, but got {config_Al.debye.number_of_atoms}"

    expected_scaling_factor = 0.617
    assert (
        config_Al.debye.scaling_factor == expected_scaling_factor
    ), f"Expected 0.617, but got {config_Al.debye.scaling_factor}"

    expected_gruneisen_x = 2 / 3
    assert (
        config_Al.debye.gruneisen_x == expected_gruneisen_x
    ), f"Expected 2/3, but got {config_Al.debye.gruneisen_x}"

    expected_temperatures = list(range(0, 1010, 10))
    assert (
        config_Al.debye.temperatures == expected_temperatures
    ), f"Expected {expected_temperatures}, but got {config_Al.debye.temperatures}"

    expected_volumes = np.linspace(0.98 * 60.0, 1.02 * 74.0, 1000)
    assert np.allclose(
        config_Al.debye.volumes, expected_volumes, rtol=1e-4
    ), f"Expected {expected_volumes}, but got {config_Al.debye.volumes}"

    with open(os.path.join(current_dir, "expected_debye_free_energy.json"), "r") as f:
        expected_free_energy = json.load(f)

    for i, (expected_list, actual_list) in enumerate(zip(expected_free_energy, config_Al.debye.free_energy)):
        if not np.allclose(actual_list, expected_list, rtol=1e-4):
            max_diff = np.max(np.abs(np.array(actual_list) - np.array(expected_list)))
            print(f"Mismatch at index {i}: Expected {expected_list}, but got {actual_list}. Max difference: {max_diff}")
        assert np.allclose(
            actual_list, expected_list, rtol=1e-4
        ), f"Expected {expected_list}, but got {actual_list}. Max difference: {max_diff}"

    with open(os.path.join(current_dir, "expected_debye_entropy.json"), "r") as f:
        expected_entropy = json.load(f)
    assert np.allclose(config_Al.debye.entropy, expected_entropy, rtol=1e-4), (
        f"Expected {expected_entropy}, " f"but got {config_Al.debye.entropy}"
    )

    with open(os.path.join(current_dir, "expected_debye_heat_capacity.json"), "r") as f:
        expected_heat_capacity = json.load(f)
    assert np.allclose(
        config_Al.debye.heat_capacity, expected_heat_capacity, rtol=1e-4
    ), (
        f"Expected {expected_heat_capacity}, "
        f"but got {config_Al.debye.heat_capacity}"
    )


if __name__ == "__main__":
    pytest.main()
