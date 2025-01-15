# Standard library imports
import os

# Third-party library imports
import pytest
import numpy as np

# DFTTK imports
from dfttk.aggregate_extraction import extract_configuration_data

current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, "vasp_data/Al/config_Al")


def test_extract_configuration_data():
    (
        number_of_atoms,
        volumes,
        energies,
        atomic_masses,
        average_mass,
        mag_data_list,
        total_magnetic_moments,
        magnetic_orderings,
    ) = extract_configuration_data(path)

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
    
if __name__ == "__main__":
    pytest.main()
