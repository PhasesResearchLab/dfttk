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
    extract_config_path = os.path.join(Fe3Pt_path, "config_28")
    (
        number_of_atoms,
        volumes,
        energies,
        atomic_masses,
        average_mass,
        mag_data_array,
        total_magnetic_moments,
        magnetic_orderings,
    ) = extract_configuration_data(
        extract_config_path,
        collect_mag_data=True,
        magmom_tolerance=0.3,  # Ignore magnetic moments on Pt atoms
        total_magnetic_moment_tolerance=0.01,
    )

    expected_number_of_atoms = 12
    expected_volumes = np.array([172.0, 169.0, 166.0, 163.0, 160.0, 157.0, 154.0, 151.0, 148.0, 145.0, 142.0, 139.0])
    expected_energies = np.array([-92.49406, -92.74686, -92.965225, -93.145906, -93.284249, -93.374201, -93.410732, -93.386833, -93.296155, -93.133266, -92.894763, -92.567931])
    expected_atomic_masses = {"Fe": 55.847, "Pt": 195.08}
    expected_average_mass = 76.34895077638512
    expected_mag_data_array = np.array(
        [
            [[1, -2.798, "Fe"], [2, 2.815, "Fe"], [3, 2.816, "Fe"], [4, -2.836, "Fe"], [5, -2.835, "Fe"], [6, 2.954, "Fe"], [7, -2.798, "Fe"], [8, 2.815, "Fe"], [9, 2.816, "Fe"], [10, -0.077, "Pt"], [11, -0.077, "Pt"], [12, 0.253, "Pt"]],
            [[1, -2.764, "Fe"], [2, 2.775, "Fe"], [3, 2.775, "Fe"], [4, -2.798, "Fe"], [5, -2.798, "Fe"], [6, 2.928, "Fe"], [7, -2.764, "Fe"], [8, 2.775, "Fe"], [9, 2.775, "Fe"], [10, -0.078, "Pt"], [11, -0.078, "Pt"], [12, 0.251, "Pt"]],
            [[1, -2.723, "Fe"], [2, 2.730, "Fe"], [3, 2.730, "Fe"], [4, -2.758, "Fe"], [5, -2.758, "Fe"], [6, 2.893, "Fe"], [7, -2.723, "Fe"], [8, 2.730, "Fe"], [9, 2.730, "Fe"], [10, -0.079, "Pt"], [11, -0.079, "Pt"], [12, 0.247, "Pt"]],
            [[1, -2.680, "Fe"], [2, 2.681, "Fe"], [3, 2.683, "Fe"], [4, -2.719, "Fe"], [5, -2.715, "Fe"], [6, 2.858, "Fe"], [7, -2.680, "Fe"], [8, 2.681, "Fe"], [9, 2.683, "Fe"], [10, -0.082, "Pt"], [11, -0.081, "Pt"], [12, 0.245, "Pt"]],
            [[1, -2.636, "Fe"], [2, 2.630, "Fe"], [3, 2.631, "Fe"], [4, -2.676, "Fe"], [5, -2.676, "Fe"], [6, 2.817, "Fe"], [7, -2.636, "Fe"], [8, 2.630, "Fe"], [9, 2.631, "Fe"], [10, -0.083, "Pt"], [11, -0.083, "Pt"], [12, 0.240, "Pt"]],
            [[1, -2.592, "Fe"], [2, 2.578, "Fe"], [3, 2.575, "Fe"], [4, -2.632, "Fe"], [5, -2.633, "Fe"], [6, 2.773, "Fe"], [7, -2.592, "Fe"], [8, 2.578, "Fe"], [9, 2.575, "Fe"], [10, -0.086, "Pt"], [11, -0.086, "Pt"], [12, 0.237, "Pt"]],
            [[1, -2.546, "Fe"], [2, 2.520, "Fe"], [3, 2.520, "Fe"], [4, -2.585, "Fe"], [5, -2.585, "Fe"], [6, 2.725, "Fe"], [7, -2.546, "Fe"], [8, 2.520, "Fe"], [9, 2.520, "Fe"], [10, -0.088, "Pt"], [11, -0.088, "Pt"], [12, 0.232, "Pt"]],
            [[1, -2.499, "Fe"], [2, 2.455, "Fe"], [3, 2.454, "Fe"], [4, -2.535, "Fe"], [5, -2.536, "Fe"], [6, 2.671, "Fe"], [7, -2.499, "Fe"], [8, 2.455, "Fe"], [9, 2.454, "Fe"], [10, -0.091, "Pt"], [11, -0.091, "Pt"], [12, 0.226, "Pt"]],
            [[1, -2.450, "Fe"], [2, 2.377, "Fe"], [3, 2.376, "Fe"], [4, -2.482, "Fe"], [5, -2.482, "Fe"], [6, 2.595, "Fe"], [7, -2.450, "Fe"], [8, 2.377, "Fe"], [9, 2.376, "Fe"], [10, -0.094, "Pt"], [11, -0.094, "Pt"], [12, 0.215, "Pt"]],
            [[1, -2.393, "Fe"], [2, 2.288, "Fe"], [3, 2.288, "Fe"], [4, -2.419, "Fe"], [5, -2.418, "Fe"], [6, 2.476, "Fe"], [7, -2.393, "Fe"], [8, 2.288, "Fe"], [9, 2.288, "Fe"], [10, -0.094, "Pt"], [11, -0.094, "Pt"], [12, 0.194, "Pt"]],
            [[1, -2.320, "Fe"], [2, 2.208, "Fe"], [3, 2.206, "Fe"], [4, -2.331, "Fe"], [5, -2.331, "Fe"], [6, 2.358, "Fe"], [7, -2.320, "Fe"], [8, 2.208, "Fe"], [9, 2.206, "Fe"], [10, -0.087, "Pt"], [11, -0.087, "Pt"], [12, 0.179, "Pt"]],
            [[1, -2.241, "Fe"], [2, 2.122, "Fe"], [3, 2.120, "Fe"], [4, -2.232, "Fe"], [5, -2.233, "Fe"], [6, 2.256, "Fe"], [7, -2.241, "Fe"], [8, 2.122, "Fe"], [9, 2.121, "Fe"], [10, -0.080, "Pt"], [11, -0.080, "Pt"], [12, 0.168, "Pt"]],
        ],
        dtype=object,
    )
    expected_total_magnetic_moments = np.array([3.048, 2.999, 2.94, 2.874, 2.789, 2.695, 2.599, 2.464, 2.264, 2.011, 1.889, 1.802])
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
