"""
Tests for the aggregate_extraction module.
"""

# Standard library imports
import os

# Third-party library imports
import pytest
import numpy as np

# DFTTK imports
from dfttk.aggregate_extraction import (
    extract_configuration_data,
    extract_convergence_data,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, "vasp_data/Al")


def test_extract_configuration_data():
    extract_config_path = os.path.join(path, "config_Al")
    (
        number_of_atoms,
        volumes,
        energies,
        atomic_masses,
        average_mass,
        mag_data_list,
        total_magnetic_moments,
        magnetic_orderings,
    ) = extract_configuration_data(extract_config_path)

    expected_number_of_atoms = 4
    expected_volumes = np.array([74, 72, 70, 68, 66, 64, 62, 60])
    expected_energies = np.array(
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
    )
    expected_atomic_masses = {"Al": 26.981}
    expected_average_mass = 26.981
    expected_mag_data_list = np.array([])
    expected_total_magnetic_moments = np.array([])
    expected_magnetic_orderings = np.array([])

    assert (
        number_of_atoms == expected_number_of_atoms
    ), f"Expected number of atoms: {expected_number_of_atoms}, got {number_of_atoms}"
    assert np.array_equal(
        volumes, expected_volumes
    ), f"Expected volumes: {expected_volumes}, got {volumes}"
    assert np.array_equal(
        energies, expected_energies
    ), f"Expected energies: {expected_energies}, got {energies}"
    assert (
        atomic_masses == expected_atomic_masses
    ), f"Expected atomic masses: {expected_atomic_masses}, got {atomic_masses}"
    assert (
        average_mass == expected_average_mass
    ), f"Expected average mass: {expected_average_mass}, got {average_mass}"
    assert np.array_equal(
        mag_data_list, expected_mag_data_list
    ), f"Expected mag data list: {expected_mag_data_list}, got {mag_data_list}"
    assert np.array_equal(
        total_magnetic_moments, expected_total_magnetic_moments
    ), f"Expected total magnetic moments: {expected_total_magnetic_moments}, got {total_magnetic_moments}"
    assert np.array_equal(
        magnetic_orderings, expected_magnetic_orderings
    ), f"Expected magnetic orderings: {expected_magnetic_orderings}, got {magnetic_orderings}"


def test_extract_convergence_data():
    encut_conv_data = os.path.join(path, "conv_test/encut_conv")
    df = extract_convergence_data(encut_conv_data)

    encut = df["encut"].values
    kpoint_grid = df["kpoint_grid"].values
    kppa = df["kppa"].values
    energy = df["energy"].values
    number_of_atoms = df["number_of_atoms"].values[0]
    energy_per_atom = df["energy_per_atom"].values
    difference_mev_per_atom = df["difference_mev_per_atom"].values

    expected_encut = np.array(
        [270, 320, 370, 420, 470, 520, 570, 620, 670, 720, 770, 820]
    )

    kpoint_list = [list([9.0, 9.0, 9.0]) for _ in range(12)]
    expected_kpoint_grid = np.empty(12, dtype=object)
    expected_kpoint_grid[:] = kpoint_list

    expected_kppa = np.array(
        [
            2916.0,
            2916.0,
            2916.0,
            2916.0,
            2916.0,
            2916.0,
            2916.0,
            2916.0,
            2916.0,
            2916.0,
            2916.0,
            2916.0,
        ]
    )
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
    ), f"Expected encut: {expected_encut}, got {encut}"
    assert np.array_equal(
        kpoint_grid, expected_kpoint_grid
    ), f"Expected kpoint grid: {expected_kpoint_grid}, got {kpoint_grid}"
    assert np.array_equal(
        kppa, expected_kppa
    ), f"Expected kppa: {expected_kppa}, got {kppa}"
    assert np.array_equal(
        energy, expected_energy
    ), f"Expected energy: {expected_energy}, got {energy}"
    assert (
        number_of_atoms == expected_number_of_atoms
    ), f"Expected number of atoms: {expected_number_of_atoms}, got {number_of_atoms}"
    assert np.array_equal(
        energy_per_atom, expected_energy_per_atom
    ), f"Expected energy per atom: {expected_energy_per_atom}, got {energy_per_atom}"
    assert np.allclose(
        difference_mev_per_atom, expected_difference_mev_per_atom, equal_nan=True
    ), f"Expected difference mev per atom: {expected_difference_mev_per_atom}, got {difference_mev_per_atom}"


if __name__ == "__main__":
    pytest.main()
