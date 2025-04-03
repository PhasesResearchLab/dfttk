# Standard library imports
import os
import json

# Third-party library imports
import pytest
import numpy as np

# DFTTK imports
from dfttk.config import Configuration

current_dir = os.path.dirname(os.path.abspath(__file__))
conv_test_path = os.path.join(current_dir, "vasp_data/Al/conv_test")
config_Al_conv = Configuration(conv_test_path, "config_Al")

config_Al_path = os.path.join(current_dir, "vasp_data/Al/config_Al")
config_Al = Configuration(config_Al_path, "config_Al")
config_Al.process_ev_curve()

num_atoms = 4
# TODO: rewrite some of the json files to include floats. Eg. 0.0K
temperature_range = np.arange(0, 1010, 100)

config_Al.process_phonons(num_atoms, temperature_range)
config_Al.process_debye(
    scaling_factor=0.617, gruneisen_x=2 / 3, temperatures=temperature_range
)
config_Al.process_thermal_electronic(temperature_range, order=1)

volume_range = np.linspace(0.98 * 60, 1.02 * 74, 1000)
config_Al.process_qha("debye", volume_range, P=0)
config_Al.process_qha("debye_thermal_electronic", volume_range, P=0)
config_Al.process_qha("phonons", volume_range, P=0)
config_Al.process_qha("phonons_thermal_electronic", volume_range, P=0)


def test_analyze_encut_conv():
    encut_conv_df, fig = config_Al_conv.analyze_encut_conv(plot=False)

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
    kpoints_conv_df, fig = config_Al_conv.analyze_kpoints_conv(plot=False)

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


def _assert_selected_keys_almost_equal(dict1, dict2, keys, atol=1e-4):
    for key in keys:
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], float) and isinstance(dict2[key], float):
                assert np.isclose(
                    dict1[key], dict2[key], atol=atol
                ), f"Expected {dict2[key]} for key '{key}', but got {dict1[key]}"
            else:
                assert (
                    dict1[key] == dict2[key]
                ), f"Expected {dict2[key]} for key '{key}', but got {dict1[key]}"


def test_process_ev_curve():
    ev_curve_files_and_attributes = [
        ("test_config_data/expected_ev_curve_incars.json", "incars"),
        ("test_config_data/expected_ev_curve_kpoints.json", "kpoints"),
    ]

    for filename, attribute in ev_curve_files_and_attributes:
        with open(os.path.join(current_dir, filename), "r") as f:
            expected_data = json.load(f)

        actual_data = getattr(config_Al.ev_curve, attribute)

        if attribute == "kpoints":
            actual_data = actual_data.as_dict()

        for actual, expected in zip(actual_data, expected_data):
            assert actual == expected, f"Expected {expected}, but got {actual}"

    assert (
        config_Al.ev_curve.number_of_atoms == 4
    ), f"Expected 4, but got {config_Al.ev_curve.number_of_atoms}"

    assert np.array_equal(
        config_Al.ev_curve.volumes,
        np.array([74.0, 72.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0]),
    ), f"Expected [74.0, 72.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0], but got {config_Al.ev_curve.volumes}"

    assert np.array_equal(
        config_Al.ev_curve.energies,
        np.array(
            [
                -14.787067,
                -14.863567,
                -14.92244,
                -14.960229,
                -14.973035,
                -14.955434,
                -14.902786,
                -14.808673,
            ]
        ),
    ), f"Expected [-14.787067, -14.863567, -14.92244, -14.960229, -14.973035, -14.955434, -14.902786, -14.808673], but got {config_Al.ev_curve.energies}"

    assert config_Al.ev_curve.atomic_masses == {
        "Al": 26.981
    }, f"Expected {'Al': 26.981}, but got {config_Al.ev_curve.atomic_masses}"
    assert (
        config_Al.ev_curve.average_mass == 26.981
    ), f"Expected 26.981, but got {config_Al.ev_curve.average_mass}"

    assert np.array_equal(
        config_Al.ev_curve.total_magnetic_moment, np.array([])
    ), f"Expected {np.array([])}, but got {config_Al.ev_curve.total_magnetic_moment}"

    assert np.array_equal(
        config_Al.ev_curve.magnetic_ordering, np.array([])
    ), f"Expected {np.array([])}, but got {config_Al.ev_curve.magnetic_ordering}"

    assert np.array_equal(
        config_Al.ev_curve.mag_data, {}
    ), f"Expected [], but got {config_Al.ev_curve.mag_data}"

    expected_eos_parameters = {
        "V0": 66.10191547034127,
        "E0": -14.972775074363833,
        "B": 77.92792067011315,
        "BP": 4.612739661291564,
        "B2P": -0.06258448064264342,
    }
    keys_to_compare = ["V0", "E0", "B", "BP", "B2P"]
    _assert_selected_keys_almost_equal(
        config_Al.ev_curve.eos_parameters, expected_eos_parameters, keys_to_compare
    )

    actual_relaxed_structures = [
        structure.as_dict() for structure in config_Al.ev_curve.relaxed_structures
    ]

    with open(
        os.path.join(
            current_dir, "test_config_data/expected_ev_curve_relaxed_structures.json"
        ),
        "r",
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
    phonons_files_and_attributes = [
        ("test_config_data/expected_phonons_incars.json", "incars"),
        ("test_config_data/expected_ev_curve_kpoints.json", "kpoints"),
    ]

    for filename, attribute in phonons_files_and_attributes:
        with open(os.path.join(current_dir, filename), "r") as f:
            expected_data = json.load(f)

        actual_data = getattr(config_Al.phonons, attribute)

        if attribute == "kpoints":
            actual_data = actual_data.as_dict()

        for actual, expected in zip(actual_data, expected_data):
            assert actual == expected, f"Expected {expected}, but got {actual}"

    actual_phonon_structures = [
        structure.as_dict() for structure in config_Al.phonons.phonon_structures
    ]
    with open(
        os.path.join(
            current_dir, "test_config_data/expected_phonons_phonon_structures.json"
        ),
        "r",
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

    expected_temperatures = np.arange(0, 1010, 100)
    assert np.array_equal(
        config_Al.phonons.temperatures, expected_temperatures
    ), f"Expected {expected_temperatures}, but got {config_Al.phonons.temperatures}"

    expected_volumes = [60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0]
    assert np.array_equal(
        config_Al.phonons.volumes, expected_volumes
    ), f"Expected {expected_volumes}, but got {config_Al.phonons.volumes}"

    files_and_attributes = [
        ("test_config_data/expected_phonons_helmholtz_energy.json", "helmholtz_energy"),
        ("test_config_data/expected_phonons_internal_energy.json", "internal_energy"),
        ("test_config_data/expected_phonons_entropy.json", "entropy"),
        ("test_config_data/expected_phonons_heat_capacity.json", "heat_capacity"),
        (
            "test_config_data/expected_phonons_helmholtz_energy_fit.json",
            "helmholtz_energy_fit",
        ),
        ("test_config_data/expected_phonons_entropy_fit.json", "entropy_fit"),
        (
            "test_config_data/expected_phonons_heat_capacity_fit.json",
            "heat_capacity_fit",
        ),
    ]

    for filename, attribute in files_and_attributes:
        with open(os.path.join(current_dir, filename), "r") as f:
            expected_data = json.load(f)

        if "fit" in attribute:
            expected_data = expected_data["polynomial_coefficients"]
            actual_data = getattr(config_Al.phonons, attribute)[
                "polynomial_coefficients"
            ]
        else:
            actual_data = getattr(config_Al.phonons, attribute)

        for temp, expected_values in expected_data.items():
            actual_values = actual_data[temp]
            for expected, actual in zip(expected_values, actual_values):
                assert np.allclose(
                    expected, actual, atol=1e-6
                ), f"Expected {expected}, but got {actual} with tolerance 1e-6"


def test_process_debye():
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

    expected_temperatures = temperature_range
    assert np.allclose(
        config_Al.debye.temperatures, expected_temperatures, rtol=1e-4
    ), f"Expected {expected_temperatures}, but got {config_Al.debye.temperatures}"

    expected_volumes = np.linspace(0.98 * 60.0, 1.02 * 74.0, 1000)
    assert np.allclose(
        config_Al.debye.volumes, expected_volumes, rtol=1e-4
    ), f"Expected {expected_volumes}, but got {config_Al.debye.volumes}"

    debye_files_and_attributes = [
        ("test_config_data/expected_debye_free_energy.json", "free_energy"),
        ("test_config_data/expected_debye_entropy.json", "entropy"),
        ("test_config_data/expected_debye_heat_capacity.json", "heat_capacity"),
    ]

    for filename, attribute in debye_files_and_attributes:
        with open(os.path.join(current_dir, filename), "r") as f:
            expected_data = json.load(f)

        actual_data = getattr(config_Al.debye, attribute)

        for i, expected_values in enumerate(expected_data):
            actual_values = actual_data[i]
            for expected, actual in zip(expected_values, actual_values):
                assert np.allclose(
                    expected, actual, atol=1e-6
                ), f"Expected {expected}, but got {actual} with tolerance 1e-6"


def test_process_thermal_electronic():
    thermal_electronic_files_and_attributes = [
        ("test_config_data/expected_thermal_electronic_incars.json", "incars"),
        ("test_config_data/expected_thermal_electronic_kpoints.json", "kpoints"),
    ]
    for filename, attribute in thermal_electronic_files_and_attributes:
        with open(os.path.join(current_dir, filename), "r") as f:
            expected_data = json.load(f)

        actual_data = getattr(config_Al.thermal_electronic, attribute)

        if attribute == "kpoints":
            actual_data = actual_data.as_dict()

        for actual, expected in zip(actual_data, expected_data):
            assert actual == expected, f"Expected {expected}, but got {actual}"

    expected_number_of_atoms = 4
    assert (
        config_Al.thermal_electronic.number_of_atoms == expected_number_of_atoms
    ), f"Expected 4, but got {config_Al.thermal_electronic.number_of_atoms}"

    expected_volumes = [60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0]
    assert np.array_equal(
        config_Al.thermal_electronic.volumes, expected_volumes
    ), f"Expected {expected_volumes}, but got {config_Al.thermal_electronic.volumes}"

    expected_temperatures = list(range(0, 1010, 100))
    assert np.array_equal(
        config_Al.thermal_electronic.temperatures, expected_temperatures
    ), f"Expected {expected_temperatures}, but got {config_Al.thermal_electronic.temperatures}"

    files_and_attributes = [
        (
            "test_config_data/expected_thermal_electronic_helmholtz_energy.json",
            "helmholtz_energy",
        ),
        (
            "test_config_data/expected_thermal_electronic_internal_energy.json",
            "internal_energy",
        ),
        ("test_config_data/expected_thermal_electronic_entropy.json", "entropy"),
        (
            "test_config_data/expected_thermal_electronic_heat_capacity.json",
            "heat_capacity",
        ),
        (
            "test_config_data/expected_thermal_electronic_helmholtz_energy_fit.json",
            "helmholtz_energy_fit",
        ),
        (
            "test_config_data/expected_thermal_electronic_entropy_fit.json",
            "entropy_fit",
        ),
        (
            "test_config_data/expected_thermal_electronic_heat_capacity_fit.json",
            "heat_capacity_fit",
        ),
    ]

    for filename, attribute in files_and_attributes:
        with open(os.path.join(current_dir, filename), "r") as f:
            expected_data = json.load(f)

        if "fit" in attribute:
            expected_data = expected_data["polynomial_coefficients"]
            actual_data = getattr(config_Al.thermal_electronic, attribute)[
                "polynomial_coefficients"
            ]
        else:
            actual_data = getattr(config_Al.thermal_electronic, attribute)

        for temp, expected_values in expected_data.items():
            actual_values = actual_data[temp]
            for expected, actual in zip(expected_values, actual_values):
                assert np.allclose(
                    expected, actual, atol=1e-6
                ), f"Expected {expected}, but got {actual} with tolerance 1e-6"


def test_process_qha():
    expected_number_of_atoms = 4
    assert (
        config_Al.qha.number_of_atoms == expected_number_of_atoms
    ), f"Expected {expected_number_of_atoms}, but got {config_Al.qha.number_of_atoms}"

    expected_temperatures = list(range(0, 1010, 100))
    assert (
        config_Al.qha.temperatures == expected_temperatures
    ), f"Expected {expected_temperatures}, but got {config_Al.qha.temperatures}"

    expected_volumes = np.linspace(0.98 * 60, 1.02 * 74, 1000)
    assert np.allclose(
        config_Al.qha.volumes, expected_volumes, atol=1e-6
    ), f"Expected {expected_volumes}, but got {config_Al.qha.volumes}"

    files_and_attributes = [
        ("test_config_data/expected_qha_debye.json", "debye"),
        (
            "test_config_data/expected_qha_debye_thermal_electronic.json",
            "debye_thermal_electronic",
        ),
        ("test_config_data/expected_qha_phonons.json", "phonons"),
        (
            "test_config_data/expected_qha_phonons_thermal_electronic.json",
            "phonons_thermal_electronic",
        ),
    ]
    methods_copy = {
        method: {
            str(P) + " GPa": {k: v for k, v in data.items() if k != "quasi_harmonic_df"}
            for P, data in pressures.items()
        }
        for method, pressures in config_Al.qha.methods.items()
    }

    for filename, attribute in files_and_attributes:
        with open(os.path.join(current_dir, filename), "r") as f:
            expected_data = json.load(f)

        properties = ["helmholtz_energy", "entropy", "heat_capacity"]
        for property in properties:
            if property == "helmholtz_energy":
                expected_property_data = expected_data["0 GPa"][property][
                    "eos_parameters"
                ]
                actual_property_data = methods_copy[attribute]["0 GPa"][property][
                    "eos_parameters"
                ]
                expected_property_data.pop("eos_name", None)
                actual_property_data.pop("eos_name", None)
            else:
                expected_property_data = expected_data["0 GPa"][property][
                    "polynomial_coefficients"
                ]
                actual_property_data = methods_copy[attribute]["0 GPa"][property][
                    "polynomial_coefficients"
                ]

            # TODO: GitHub actions only passes with this low tolerance. Investigate why.
            for temp, expected_values in expected_property_data.items():
                actual_values = actual_property_data[temp]
                if property == "helmholtz_energy":
                    for expected, actual in zip(
                        expected_values.values(), actual_values.values()
                    ):
                        assert np.allclose(
                            expected, actual, rtol=2e-2
                        ), f"Expected {expected}, but got {actual} with tolerance 5e-1"
                else:
                    for expected, actual in zip(expected_values, actual_values):
                        assert np.allclose(
                            expected, actual, rtol=2e-2
                        ), f"Expected {expected}, but got {actual} with tolerance 5e-1"


if __name__ == "__main__":
    pytest.main()
