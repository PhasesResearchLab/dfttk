"""
Tests for the aggregate_extraction module.
"""

# Standard library imports
import os

# Third-party library imports
import numpy as np

# DFTTK imports
from dfttk.aggregate_extraction import extract_configuration_data, extract_convergence_data

# TODO: add some more tests

current_dir = os.path.dirname(os.path.abspath(__file__))
Al_path = os.path.join(current_dir, "vasp_data/Al")
Fe3Pt_path = os.path.join(current_dir, "vasp_data/Fe3Pt")
CoNi_path = os.path.join(current_dir, "vasp_data/CoNi")


# Tests for extract_configuration_data function
def test_extract_configuration_data_Al_non_magnetic():
    extract_config_path = os.path.join(Al_path, "config_Al")
    (
        number_of_atoms,
        volumes,
        energies,
        atomic_masses,
        average_mass,
        mag_data_array,
        total_magnetic_moments,
        magnetic_orderings,
    ) = extract_configuration_data(extract_config_path)

    expected_number_of_atoms = 4
    expected_volumes = np.array([74, 72, 70, 68, 66, 64, 62, 60])
    expected_energies = np.array([-14.787067, -14.863567, -14.92244, -14.960229, -14.973035, -14.955434, -14.902786, -14.808673])
    expected_atomic_masses = {"Al": 26.981}
    expected_average_mass = 26.981
    expected_mag_data_array = np.array([])
    expected_total_magnetic_moments = np.array([])
    expected_magnetic_orderings = np.array([])

    assert number_of_atoms == expected_number_of_atoms, f"Expected number of atoms: {expected_number_of_atoms}, got {number_of_atoms}"
    assert np.array_equal(volumes, expected_volumes), f"Expected volumes: {expected_volumes}, got {volumes}"
    assert np.array_equal(energies, expected_energies), f"Expected energies: {expected_energies}, got {energies}"
    assert atomic_masses == expected_atomic_masses, f"Expected atomic masses: {expected_atomic_masses}, got {atomic_masses}"
    assert average_mass == expected_average_mass, f"Expected average mass: {expected_average_mass}, got {average_mass}"
    assert np.array_equal(mag_data_array, expected_mag_data_array), f"Expected mag data list: {expected_mag_data_array}, got {mag_data_array}"
    assert np.array_equal(total_magnetic_moments, expected_total_magnetic_moments), f"Expected total magnetic moments: {expected_total_magnetic_moments}, got {total_magnetic_moments}"
    assert np.array_equal(magnetic_orderings, expected_magnetic_orderings), f"Expected magnetic orderings: {expected_magnetic_orderings}, got {magnetic_orderings}"


def test_extract_configuration_data_Fe3Pt_magnetic():
    # ISPIN=2 and LORBIT=11 set in INCAR files
    extract_config_path = os.path.join(Fe3Pt_path, "config_1")
    (
        number_of_atoms,
        volumes,
        energies,
        atomic_masses,
        average_mass,
        mag_data_array,
        total_magnetic_moments,
        magnetic_orderings,
    ) = extract_configuration_data(extract_config_path, collect_mag_data=True)

    expected_number_of_atoms = 12
    expected_volumes = np.array([172.0, 169.0, 166.0, 163.0, 160.0, 157.0, 154.0, 151.0, 148.0, 145.0, 142.0, 139.0])
    expected_energies = np.array([-92.467716, -92.723711, -92.940105, -93.113167, -93.236764, -92.985001, -93.301162, -93.268978, -93.171998, -93.005946, -92.761316, -92.429074])
    expected_atomic_masses = {"Fe": 55.847, "Pt": 195.08}
    expected_average_mass = 76.34895077638512
    expected_mag_data_array = np.array(
        [
            [[1, -2.791, "Fe"], [2, 2.974, "Fe"], [3, 2.974, "Fe"], [4, 2.783, "Fe"], [5, 2.783, "Fe"], [6, 2.924, "Fe"], [7, 2.584, "Fe"], [8, 2.841, "Fe"], [9, 2.841, "Fe"], [10, 0.124, "Pt"], [11, 0.124, "Pt"], [12, 0.303, "Pt"]],
            [[1, -2.752, "Fe"], [2, 2.951, "Fe"], [3, 2.95, "Fe"], [4, 2.754, "Fe"], [5, 2.755, "Fe"], [6, 2.902, "Fe"], [7, 2.552, "Fe"], [8, 2.819, "Fe"], [9, 2.819, "Fe"], [10, 0.127, "Pt"], [11, 0.127, "Pt"], [12, 0.306, "Pt"]],
            [[1, -2.705, "Fe"], [2, 2.922, "Fe"], [3, 2.923, "Fe"], [4, 2.719, "Fe"], [5, 2.72, "Fe"], [6, 2.879, "Fe"], [7, 2.519, "Fe"], [8, 2.796, "Fe"], [9, 2.796, "Fe"], [10, 0.129, "Pt"], [11, 0.13, "Pt"], [12, 0.307, "Pt"]],
            [[1, -2.66, "Fe"], [2, 2.896, "Fe"], [3, 2.896, "Fe"], [4, 2.687, "Fe"], [5, 2.688, "Fe"], [6, 2.858, "Fe"], [7, 2.485, "Fe"], [8, 2.771, "Fe"], [9, 2.771, "Fe"], [10, 0.13, "Pt"], [11, 0.13, "Pt"], [12, 0.31, "Pt"]],
            [[1, -2.606, "Fe"], [2, 2.862, "Fe"], [3, 2.862, "Fe"], [4, 2.647, "Fe"], [5, 2.647, "Fe"], [6, 2.829, "Fe"], [7, 2.463, "Fe"], [8, 2.747, "Fe"], [9, 2.747, "Fe"], [10, 0.132, "Pt"], [11, 0.132, "Pt"], [12, 0.311, "Pt"]],
            [[1, -2.535, "Fe"], [2, 2.781, "Fe"], [3, 2.78, "Fe"], [4, 2.632, "Fe"], [5, 2.633, "Fe"], [6, 2.791, "Fe"], [7, 0.025, "Fe"], [8, 2.705, "Fe"], [9, 2.704, "Fe"], [10, 0.111, "Pt"], [11, 0.111, "Pt"], [12, 0.297, "Pt"]],
            [[1, -2.378, "Fe"], [2, 2.627, "Fe"], [3, 2.628, "Fe"], [4, 2.563, "Fe"], [5, 2.562, "Fe"], [6, 2.738, "Fe"], [7, -2.377, "Fe"], [8, 2.628, "Fe"], [9, 2.629, "Fe"], [10, 0.061, "Pt"], [11, 0.061, "Pt"], [12, 0.296, "Pt"]],
            [[1, -2.306, "Fe"], [2, 2.58, "Fe"], [3, 2.58, "Fe"], [4, 2.497, "Fe"], [5, 2.497, "Fe"], [6, 2.693, "Fe"], [7, -2.306, "Fe"], [8, 2.58, "Fe"], [9, 2.58, "Fe"], [10, 0.058, "Pt"], [11, 0.058, "Pt"], [12, 0.293, "Pt"]],
            [[1, -2.225, "Fe"], [2, 2.514, "Fe"], [3, 2.514, "Fe"], [4, 2.41, "Fe"], [5, 2.41, "Fe"], [6, 2.626, "Fe"], [7, -2.225, "Fe"], [8, 2.514, "Fe"], [9, 2.514, "Fe"], [10, 0.052, "Pt"], [11, 0.052, "Pt"], [12, 0.282, "Pt"]],
            [[1, -2.167, "Fe"], [2, 2.444, "Fe"], [3, 2.444, "Fe"], [4, 2.266, "Fe"], [5, 2.266, "Fe"], [6, 2.549, "Fe"], [7, -2.167, "Fe"], [8, 2.444, "Fe"], [9, 2.444, "Fe"], [10, 0.032, "Pt"], [11, 0.032, "Pt"], [12, 0.266, "Pt"]],
            [[1, -2.082, "Fe"], [2, 2.376, "Fe"], [3, 2.376, "Fe"], [4, 2.118, "Fe"], [5, 2.118, "Fe"], [6, 2.479, "Fe"], [7, -2.082, "Fe"], [8, 2.376, "Fe"], [9, 2.376, "Fe"], [10, 0.019, "Pt"], [11, 0.019, "Pt"], [12, 0.255, "Pt"]],
            [[1, -1.981, "Fe"], [2, 2.3, "Fe"], [3, 2.3, "Fe"], [4, 1.985, "Fe"], [5, 1.985, "Fe"], [6, 2.4, "Fe"], [7, -1.981, "Fe"], [8, 2.3, "Fe"], [9, 2.3, "Fe"], [10, 0.015, "Pt"], [11, 0.015, "Pt"], [12, 0.242, "Pt"]],
        ],
        dtype=object,
    )
    expected_total_magnetic_moments = np.array([20.464, 20.31, 20.135, 19.962, 19.773, 17.035, 14.038, 13.804, 13.438, 12.853, 12.348, 11.88])
    expected_magnetic_orderings = np.array(["SF", "SF", "SF", "SF", "SF", "SF", "SF", "SF", "SF", "SF", "SF", "SF"], dtype="<U32")

    assert number_of_atoms == expected_number_of_atoms, f"Expected number of atoms: {expected_number_of_atoms}, got {number_of_atoms}"
    assert np.array_equal(volumes, expected_volumes), f"Expected volumes: {expected_volumes}, got {volumes}"
    assert np.array_equal(energies, expected_energies), f"Expected energies: {expected_energies}, got {energies}"
    assert atomic_masses == expected_atomic_masses, f"Expected atomic masses: {expected_atomic_masses}, got {atomic_masses}"
    assert average_mass == expected_average_mass, f"Expected average mass: {expected_average_mass}, got {average_mass}"
    np.testing.assert_array_equal(mag_data_array, expected_mag_data_array, err_msg="Mag data array does not match expected values")
    assert np.allclose(total_magnetic_moments, expected_total_magnetic_moments), f"Expected total magnetic moments: {expected_total_magnetic_moments}, got {total_magnetic_moments}"
    assert np.array_equal(magnetic_orderings, expected_magnetic_orderings), f"Expected magnetic orderings: {expected_magnetic_orderings}, got {magnetic_orderings}"


def test_extract_configuration_data_CoNi_magnetic():
    # ISPIN=2 and LORBIT=0 set in INCAR files
    (
        number_of_atoms,
        volumes,
        energies,
        atomic_masses,
        average_mass,
        mag_data_array,
        total_magnetic_moments,
        magnetic_orderings,
    ) = extract_configuration_data(CoNi_path, outcar_name="OUTCAR", oszicar_name="OSZICAR", contcar_name="CONTCAR", collect_mag_data=True)

    expected_number_of_atoms = 12
    expected_volumes = np.array([106.657558, 109.990369, 113.391896, 116.862839, 120.4039, 124.01578, 127.69918, 131.454801])
    expected_energies = np.array([-86.136464, -86.89351, -87.412736, -87.709766, -87.810335, -87.733064, -87.507118, -87.143013])
    expected_atomic_masses = {"Co": 58.933, "Ni": 58.69}
    expected_average_mass = 58.81137449507534
    expected_mag_data_array = np.array([])
    expected_total_magnetic_moments = np.array([12.0585095, 12.3217673, 12.5185976, 12.7169581, 12.9088164, 13.1099944, 13.3005653, 13.4927865])
    expected_magnetic_orderings = np.array([])

    assert number_of_atoms == expected_number_of_atoms, f"Expected number of atoms: {expected_number_of_atoms}, got {number_of_atoms}"
    assert np.array_equal(volumes, expected_volumes), f"Expected volumes: {expected_volumes}, got {volumes}"
    assert np.array_equal(energies, expected_energies), f"Expected energies: {expected_energies}, got {energies}"
    assert atomic_masses == expected_atomic_masses, f"Expected atomic masses: {expected_atomic_masses}, got {atomic_masses}"
    assert average_mass == expected_average_mass, f"Expected average mass: {expected_average_mass}, got {average_mass}"
    assert np.array_equal(mag_data_array, expected_mag_data_array), f"Expected mag data list: {expected_mag_data_array}, got {mag_data_array}"
    assert np.array_equal(total_magnetic_moments, expected_total_magnetic_moments), f"Expected total magnetic moments: {expected_total_magnetic_moments}, got {total_magnetic_moments}"
    assert np.array_equal(magnetic_orderings, expected_magnetic_orderings), f"Expected magnetic orderings: {expected_magnetic_orderings}, got {magnetic_orderings}"


# Tests for extract_convergence_data function
def test_extract_convergence_data_Al_non_magnetic():
    encut_conv_data = os.path.join(Al_path, "conv_test/encut_conv")
    df = extract_convergence_data(encut_conv_data)

    encut = df["encut"].values
    kpoint_grid = df["kpoint_grid"].values
    kppa = df["kppa"].values
    energy = df["energy"].values
    number_of_atoms = df["number_of_atoms"].values[0]
    energy_per_atom = df["energy_per_atom"].values
    difference_mev_per_atom = df["difference_mev_per_atom"].values

    expected_encut = np.array([270, 320, 370, 420, 470, 520, 570, 620, 670, 720, 770, 820])

    kpoint_list = [list([9.0, 9.0, 9.0]) for _ in range(12)]
    expected_kpoint_grid = np.empty(12, dtype=object)
    expected_kpoint_grid[:] = kpoint_list

    expected_kppa = np.array([2916.0, 2916.0, 2916.0, 2916.0, 2916.0, 2916.0, 2916.0, 2916.0, 2916.0, 2916.0, 2916.0, 2916.0])
    expected_energy = np.array([-14.952094, -14.966816, -14.973924, -14.975857, -14.976588, -14.97668, -14.976812, -14.977072, -14.977261, -14.977352, -14.977392, -14.977423])
    expected_number_of_atoms = 4
    expected_energy_per_atom = np.array([-3.7380235, -3.741704, -3.743481, -3.74396425, -3.744147, -3.74417, -3.744203, -3.744268, -3.74431525, -3.744338, -3.744348, -3.74435575])
    expected_difference_mev_per_atom = np.array([np.nan, -3.6805, -1.777, -0.48325, -0.18275, -0.023, -0.033, -0.065, -0.04725, -0.02275, -0.01, -0.00775])

    assert np.array_equal(encut, expected_encut), f"Expected encut: {expected_encut}, got {encut}"
    assert np.array_equal(kpoint_grid, expected_kpoint_grid), f"Expected kpoint grid: {expected_kpoint_grid}, got {kpoint_grid}"
    assert np.array_equal(kppa, expected_kppa), f"Expected kppa: {expected_kppa}, got {kppa}"
    assert np.array_equal(energy, expected_energy), f"Expected energy: {expected_energy}, got {energy}"
    assert number_of_atoms == expected_number_of_atoms, f"Expected number of atoms: {expected_number_of_atoms}, got {number_of_atoms}"
    assert np.array_equal(energy_per_atom, expected_energy_per_atom), f"Expected energy per atom: {expected_energy_per_atom}, got {energy_per_atom}"
    assert np.allclose(difference_mev_per_atom, expected_difference_mev_per_atom, equal_nan=True), f"Expected difference mev per atom: {expected_difference_mev_per_atom}, got {difference_mev_per_atom}"
