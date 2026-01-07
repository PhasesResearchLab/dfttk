"""
Tests for the data_extraction module.
"""

# Standard library imports
import os

# Third-party library imports
import pytest
import numpy as np

# DFTTK imports
from dfttk.data_extraction import (
    extract_atomic_masses,
    extract_average_mass,
    extract_mag_data,
    extract_tot_mag_data,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
vasp_data_path = os.path.join(current_dir, "vasp_data/Al")
test_data_extraction_data_path = os.path.join(current_dir, "test_data_extraction_data")

outcar_path = os.path.join(vasp_data_path, "config_Al/vol_0/OUTCAR.3static")
contcar_path = os.path.join(vasp_data_path, "config_Al/vol_0/CONTCAR.3static")
doscar_path = os.path.join(vasp_data_path, "config_Al/elec_0/DOSCAR.elec_dos")
vasprun_path = os.path.join(vasp_data_path, "config_Al/elec_0/vasprun.xml.elec_dos")


def test_extract_atomic_masses():
    atomic_masses = extract_atomic_masses(outcar_path)
    expected_atomic_masses = {"Al": 26.981}
    assert (
        atomic_masses == expected_atomic_masses
    ), f"Expected atomic masses: {expected_atomic_masses}, got {atomic_masses}"


def test_extract_average_mass():
    arithmetic = extract_average_mass(contcar_path, outcar_path, average="arithmetic")
    geometric = extract_average_mass(contcar_path, outcar_path, average="geometric")
    harmonic = extract_average_mass(contcar_path, outcar_path, average="harmonic")

    expected_arithmetic = 26.981
    expected_geometric = 26.981
    expected_harmonic = 26.981

    assert (
        arithmetic == expected_arithmetic
    ), f"Expected arithmetic average: {expected_arithmetic}, got {arithmetic}"
    assert (
        geometric == expected_geometric
    ), f"Expected geometric average: {expected_geometric}, got {geometric}"
    assert (
        harmonic == expected_harmonic
    ), f"Expected harmonic average: {expected_harmonic}, got {harmonic}"


def test_extract_mag_data():
    with pytest.raises(ValueError):
        extract_mag_data(outcar_path)


def test_extract_tot_mag_data():
    with pytest.raises(ValueError):
        extract_tot_mag_data(outcar_path, contcar_path)
