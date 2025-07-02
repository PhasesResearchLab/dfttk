"""
Tests for the Configuration class.
"""

# Standard library imports
import os
import json
import pickle

# Third-party library imports
import numpy as np

# DFTTK imports
from dfttk.configuration import Configuration

#TODO: write tests for a magnetic example
#TODO: think of other tests for Configuration class

current_dir = os.path.dirname(os.path.abspath(__file__))
conv_test_path = os.path.join(current_dir, "vasp_data/Al/conv_test")
vasp_cmd = ["mpirun", "/opt/packages/VASP/VASP6/6.4.3/ONEAPI/vasp_std"]
config_Al_conv = Configuration(conv_test_path, "config_Al", vasp_cmd)

config_Al_path = os.path.join(current_dir, "vasp_data/Al/config_Al")
config_Al = Configuration(config_Al_path, "config_Al", vasp_cmd)
config_Al.process_ev_curve()

number_of_atoms = 4
# TODO: rewrite some of the json files to include floats. Eg. 0.0K
temperature_range = np.arange(0, 1010, 100)

config_Al.process_phonons(number_of_atoms, temperature_range)
config_Al.process_debye(scaling_factor=0.617, gruneisen_x=2 / 3, temperatures=temperature_range)
config_Al.process_thermal_electronic(temperature_range, order=1)

volume_range = np.linspace(0.98 * 60, 1.02 * 74, 1000)
config_Al.process_qha("debye", volume_range, P=0)
config_Al.process_qha("debye_thermal_electronic", volume_range, P=0)
config_Al.process_qha("phonons", volume_range, P=0)
config_Al.process_qha("phonons_thermal_electronic", volume_range, P=0)

# Load the expected results
with open(os.path.join(current_dir, "test_configuration_data/config_Al.pkl"), "rb") as f:
    expected_config_Al = pickle.load(f)

def test_analyze_encut_conv():
    encut_conv_df, fig = config_Al_conv.analyze_encut_conv(plot=False)

    encut = encut_conv_df["encut"].values
    kpoints_grid = encut_conv_df["kpoint_grid"].values[0]
    kppa = encut_conv_df["kppa"].values[0]
    energy = encut_conv_df["energy"].values
    number_of_atoms = encut_conv_df["number_of_atoms"].values[0]
    energy_per_atom = encut_conv_df["energy_per_atom"].values
    difference_mev_per_atom = encut_conv_df["difference_mev_per_atom"].values

    expected_encut = np.array([270, 320, 370, 420, 470, 520, 570, 620, 670, 720, 770, 820])
    expected_kpoints_grid = [9, 9, 9]
    expected_kppa = 2916
    expected_energy = np.array([-14.952094, -14.966816, -14.973924, -14.975857, -14.976588, -14.97668, -14.976812, -14.977072, -14.977261, -14.977352, -14.977392, -14.977423])
    expected_number_of_atoms = 4
    expected_energy_per_atom = np.array([-3.7380235, -3.741704, -3.743481, -3.74396425, -3.744147, -3.74417, -3.744203, -3.744268, -3.74431525, -3.744338, -3.744348, -3.74435575])
    expected_difference_mev_per_atom = np.array([np.nan, -3.6805, -1.777, -0.48325, -0.18275, -0.023, -0.033, -0.065, -0.04725, -0.02275, -0.01, -0.00775])

    assert np.array_equal(encut, expected_encut), f"Expected {expected_encut}, but got {encut}"
    assert np.array_equal(kpoints_grid, expected_kpoints_grid), f"Expected {expected_kpoints_grid}, but got {kpoints_grid}"
    assert kppa == expected_kppa, f"Expected {expected_kppa}, but got {kppa}"
    assert np.array_equal(energy, expected_energy), f"Expected {expected_energy}, but got {energy}"
    assert number_of_atoms == expected_number_of_atoms, f"Expected {expected_number_of_atoms}, but got {number_of_atoms}"
    assert np.array_equal(energy_per_atom, expected_energy_per_atom), f"Expected {expected_energy_per_atom}, but got {energy_per_atom}"
    assert np.allclose(difference_mev_per_atom, expected_difference_mev_per_atom, equal_nan=True), f"Expected {expected_difference_mev_per_atom}, but got {difference_mev_per_atom}"


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
    expected_kpoints_grid = np.array([[6, 6, 6], [7, 7, 7], [9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12], [13, 13, 13]])
    expected_kppa = np.array([864, 1372, 2916, 4000, 5324, 6912, 8788])
    expected_energy = np.array([-15.058461, -14.990809, -14.97668, -14.973059, -14.98156, -14.980429, -14.987405])
    expected_number_of_atoms = 4
    expected_energy_per_atom = np.array([-3.76461525, -3.74770225, -3.74417, -3.74326475, -3.74539, -3.74510725, -3.74685125])
    expected_difference_mev_per_atom = np.array([np.nan, 16.913, 3.53225, 0.90525, -2.12525, 0.28275, -1.744])

    assert encut == expected_encut, f"Expected {expected_encut}, but got {encut}"
    assert np.array_equal(kpoints_grid, expected_kpoints_grid), f"Expected {expected_kpoints_grid}, but got {kpoints_grid}"
    assert np.array_equal(kppa, expected_kppa), f"Expected {expected_kppa}, but got {kppa}"
    assert np.array_equal(energy, expected_energy), f"Expected {expected_energy}, but got {energy}"
    assert number_of_atoms == expected_number_of_atoms, f"Expected {expected_number_of_atoms}, but got {number_of_atoms}"
    assert np.array_equal(energy_per_atom, expected_energy_per_atom), f"Expected {expected_energy_per_atom}, but got {energy_per_atom}"
    assert np.allclose(difference_mev_per_atom, expected_difference_mev_per_atom, equal_nan=True), f"Expected {expected_difference_mev_per_atom}, but got {difference_mev_per_atom}"


def test_process_ev_curve():
    ev_curve = config_Al.ev_curve
    expected_ev_curve = expected_config_Al.ev_curve

    # Test the values of the ev_curve attributes. 
    assert ev_curve.atomic_masses == expected_ev_curve.atomic_masses, (f"Expected {expected_ev_curve.atomic_masses}, but got {ev_curve.atomic_masses}")
    assert ev_curve.average_mass == expected_ev_curve.average_mass, (f"Expected {expected_ev_curve.average_mass}, but got {ev_curve.average_mass}")
    assert np.allclose(ev_curve.energies, expected_ev_curve.energies, equal_nan=True), (f"Expected {expected_ev_curve.energies}, but got {ev_curve.energies}")
    assert ev_curve.eos_parameters['eos_name'] == expected_ev_curve.eos_parameters['eos_name'], (f"Expected {expected_ev_curve.eos_parameters['eos_name']}, but got {ev_curve.eos_parameters['eos_name']}")
    for key in expected_ev_curve.eos_parameters:
        if key != 'eos_name':
            # Increased tolerance to rtol=5e-4 for EOS parameter comparison
            assert np.isclose(ev_curve.eos_parameters[key], expected_ev_curve.eos_parameters[key], rtol=5e-4), (f"Expected {expected_ev_curve.eos_parameters[key]}, but got {ev_curve.eos_parameters[key]}")
    assert ev_curve.incars == expected_ev_curve.incars, (f"Expected {expected_ev_curve.incars}, but got {ev_curve.incars}")
    assert ev_curve.initial_poscar == expected_ev_curve.initial_poscar, (f"Expected {expected_ev_curve.initial_poscar}, but got {ev_curve.initial_poscar}")
    assert ev_curve.kpoints == expected_ev_curve.kpoints, (f"Expected {expected_ev_curve.kpoints}, but got {ev_curve.kpoints}")
    assert ev_curve.mag_data == expected_ev_curve.mag_data, (f"Expected {expected_ev_curve.mag_data}, but got {ev_curve.mag_data}")
    assert ev_curve.magnetic_ordering == expected_ev_curve.magnetic_ordering, (f"Expected {expected_ev_curve.magnetic_ordering}, but got {ev_curve.magnetic_ordering}")
    assert ev_curve.number_of_atoms == expected_ev_curve.number_of_atoms, (f"Expected {expected_ev_curve.number_of_atoms}, but got {ev_curve.number_of_atoms}")
    assert ev_curve.relaxed_structures == expected_ev_curve.relaxed_structures, (f"Expected {expected_ev_curve.relaxed_structures}, but got {ev_curve.relaxed_structures}")
    assert ev_curve.total_magnetic_moment == expected_ev_curve.total_magnetic_moment, (f"Expected {expected_ev_curve.total_magnetic_moment}, but got {ev_curve.total_magnetic_moment}")
    assert np.allclose(ev_curve.volumes, expected_ev_curve.volumes, rtol=1e-4), (f"Expected {expected_ev_curve.volumes}, but got {ev_curve.volumes}")


def test_process_phonons():
    phonons = config_Al.phonons
    expected_phonons = expected_config_Al.phonons

    # Test the values of the phonons attributes.
    assert np.allclose(phonons.entropy, expected_phonons.entropy, rtol=1e-4), (f"Expected {expected_phonons.entropy}, but got {phonons.entropy}")
    assert np.allclose(phonons.entropy_fit, expected_phonons.entropy_fit, rtol=1e-4), (f"Expected {expected_phonons.entropy_fit}, but got {phonons.entropy_fit}")
    assert np.allclose(phonons.entropy_poly_coeffs, expected_phonons.entropy_poly_coeffs, rtol=1e-4), (f"Expected {expected_phonons.entropy_poly_coeffs}, but got {phonons.entropy_poly_coeffs}")
    assert np.allclose(phonons.helmholtz_energy, expected_phonons.helmholtz_energy, rtol=1e-4), (f"Expected {expected_phonons.helmholtz_energy}, but got {phonons.helmholtz_energy}")
    assert np.allclose(phonons.helmholtz_energy_fit, expected_phonons.helmholtz_energy_fit, rtol=1e-4), (f"Expected {expected_phonons.helmholtz_energy_fit}, but got {phonons.helmholtz_energy_fit}")
    assert np.allclose(phonons.helmholtz_energy_poly_coeffs, expected_phonons.helmholtz_energy_poly_coeffs, rtol=1e-4), (f"Expected {expected_phonons.helmholtz_energy_poly_coeffs}, but got {phonons.helmholtz_energy_poly_coeffs}")
    assert np.allclose(phonons.heat_capacity, expected_phonons.heat_capacity, rtol=1e-4), (f"Expected {expected_phonons.heat_capacity}, but got {phonons.heat_capacity}")
    assert np.allclose(phonons.heat_capacity_fit, expected_phonons.heat_capacity_fit, rtol=1e-4), (f"Expected {expected_phonons.heat_capacity_fit}, but got {phonons.heat_capacity_fit}")
    assert np.allclose(phonons.heat_capacity_poly_coeffs, expected_phonons.heat_capacity_poly_coeffs, rtol=1e-4), (f"Expected {expected_phonons.heat_capacity_poly_coeffs}, but got {phonons.heat_capacity_poly_coeffs}")
    assert phonons.incars == expected_phonons.incars, (f"Expected {expected_phonons.incars}, but got {phonons.incars}")
    assert np.allclose(phonons.internal_energy, expected_phonons.internal_energy, rtol=1e-4), (f"Expected {expected_phonons.internal_energy}, but got {phonons.internal_energy}")
    assert phonons.kpoints == expected_phonons.kpoints, (f"Expected {expected_phonons.kpoints}, but got {phonons.kpoints}")
    assert phonons.number_of_atoms == expected_phonons.number_of_atoms, (f"Expected {expected_phonons.number_of_atoms}, but got {phonons.number_of_atoms}")
    assert phonons.phonon_structures == expected_phonons.phonon_structures, (f"Expected {expected_phonons.phonon_structures}, but got {phonons.phonon_structures}")
    assert np.allclose(phonons.temperatures, expected_phonons.temperatures, rtol=1e-4), (f"Expected {expected_phonons.temperatures}, but got {phonons.temperatures}")
    assert np.allclose(phonons.volumes, expected_phonons.volumes, rtol=1e-4), (f"Expected {expected_phonons.volumes}, but got {phonons.volumes}")
    assert np.allclose(phonons.volumes_fit, expected_phonons.volumes_fit, rtol=1e-4), (f"Expected {expected_phonons.volumes_fit}, but got {phonons.volumes_fit}")


def test_process_debye():
    debye = config_Al.debye
    expected_debye = expected_config_Al.debye
    
    # Test the values of the debye attributes.
    assert np.isclose(debye.B, expected_debye.B, rtol=1e-4), f"Expected {expected_debye.B}, but got {debye.B}"
    assert np.isclose(debye.BP, expected_debye.BP, rtol=1e-4), f"Expected {expected_debye.BP}, but got {debye.BP}"
    assert np.isclose(debye.V0, expected_debye.V0, rtol=1e-4), f"Expected {expected_debye.V0}, but got {debye.V0}"
    assert np.isclose(debye.atomic_mass, expected_debye.atomic_mass, rtol=1e-4), f"Expected {expected_debye.atomic_mass}, but got {debye.atomic_mass}"
    assert np.isclose(debye.gruneisen_x, expected_debye.gruneisen_x, rtol=1e-4), f"Expected {expected_debye.gruneisen_x}, but got {debye.gruneisen_x}"
    assert np.allclose(debye.entropies, expected_debye.entropies, rtol=1e-4), f"Expected {expected_debye.entropies}, but got {debye.entropies}"
    assert np.allclose(debye.heat_capacities, expected_debye.heat_capacities, rtol=1e-4), f"Expected {expected_debye.heat_capacities}, but got {debye.heat_capacities}"
    assert np.allclose(debye.helmholtz_energies, expected_debye.helmholtz_energies, rtol=2e-3), f"Expected {expected_debye.helmholtz_energies}, but got {debye.helmholtz_energies}"
    assert number_of_atoms == expected_debye.number_of_atoms, f"Expected {expected_debye.number_of_atoms}, but got {debye.number_of_atoms}"
    assert np.isclose(debye.scaling_factor, expected_debye.scaling_factor, rtol=1e-4), f"Expected {expected_debye.scaling_factor}, but got {debye.scaling_factor}"
    assert np.allclose(debye.temperatures, expected_debye.temperatures, rtol=1e-4), f"Expected {expected_debye.temperatures}, but got {debye.temperatures}"
    assert np.allclose(debye.volumes, expected_debye.volumes, rtol=1e-4), f"Expected {expected_debye.volumes}, but got {debye.volumes}"


def test_process_thermal_electronic():
    thermal_electronic_files_and_attributes = [
        ("test_configuration_data/expected_thermal_electronic_incars.json", "incars"),
        #("test_configuration_data/expected_thermal_electronic_kpoints.json", "kpoints"),
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
            "test_configuration_data/expected_thermal_electronic_helmholtz_energy.json",
            "helmholtz_energy",
        ),
        (
            "test_configuration_data/expected_thermal_electronic_internal_energy.json",
            "internal_energy",
        ),
        ("test_configuration_data/expected_thermal_electronic_entropy.json", "entropy"),
        (
            "test_configuration_data/expected_thermal_electronic_heat_capacity.json",
            "heat_capacity",
        ),
        (
            "test_configuration_data/expected_thermal_electronic_helmholtz_energy_fit.json",
            "helmholtz_energy_fit",
        ),
        (
            "test_configuration_data/expected_thermal_electronic_entropy_fit.json",
            "entropy_fit",
        ),
        (
            "test_configuration_data/expected_thermal_electronic_heat_capacity_fit.json",
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
    assert np.array_equal(
        config_Al.qha.temperatures, expected_temperatures
    ), f"Expected {expected_temperatures}, but got {config_Al.qha.temperatures}"

    expected_volumes = np.linspace(0.98 * 60, 1.02 * 74, 1000)
    assert np.allclose(
        config_Al.qha.volumes, expected_volumes, atol=1e-6
    ), f"Expected {expected_volumes}, but got {config_Al.qha.volumes}"

    files_and_attributes = [
        ("test_configuration_data/expected_qha_debye.json", "debye"),
        (
            "test_configuration_data/expected_qha_debye_thermal_electronic.json",
            "debye_thermal_electronic",
        ),
        ("test_configuration_data/expected_qha_phonons.json", "phonons"),
        (
            "test_configuration_data/expected_qha_phonons_thermal_electronic.json",
            "phonons_thermal_electronic",
        ),
    ]

    methods_copy = config_Al.qha.methods
    for filename, attribute in files_and_attributes:
        with open(os.path.join(current_dir, filename), "r") as f:
            expected_data = json.load(f)

        properties = ["helmholtz_energy_pv", "entropy", "heat_capacity"]
        for property in properties:
            if property == "helmholtz_energy_pv":
                expected_property_data = expected_data["0.00_GPa"][property][
                    "eos_constants"
                ]
                actual_property_data = methods_copy[attribute]["0.00_GPa"][property][
                    "eos_constants"
                ]
                expected_property_data.pop("eos_name", None)
                actual_property_data.pop("eos_name", None)
            else:
                expected_property_data = expected_data[property]["poly_coeffs"]
                actual_property_data = methods_copy[attribute][property]["poly_coeffs"]

            for temp, expected_values in expected_property_data.items():
                actual_values = actual_property_data[temp]
                if property == "helmholtz_energy_pv":
                    for expected, actual in zip(
                        expected_values.values(), actual_values.values()
                    ):
                        assert np.allclose(
                            expected, actual, rtol=2e-2
                        ), f"Expected {expected}, but got {actual} with tolerance 2e-2"
                else:
                    for expected, actual in zip(expected_values, actual_values):
                        assert np.allclose(
                            expected, actual, rtol=2e-2
                        ), f"Expected {expected}, but got {actual} with tolerance 2e-2"
