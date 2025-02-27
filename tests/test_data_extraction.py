"""
Tests for the aggregate_extraction module.
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
    parse_doscar
)

current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, "vasp_data/Al")

outcar_path = os.path.join(path, "config_Al/vol_0/OUTCAR.3static")
contcar_path = os.path.join(path, "config_Al/vol_0/CONTCAR.3static")

def test_extract_atomic_masses():
    atomic_masses = extract_atomic_masses(outcar_path)
    expected_atomic_masses = {"Al": 26.981}
    assert atomic_masses == expected_atomic_masses, f"Expected atomic masses: {expected_atomic_masses}, got {atomic_masses}"


def test_extract_average_mass():
    arithmetic = extract_average_mass(contcar_path, outcar_path, average="arithmetic")
    geometric = extract_average_mass(contcar_path, outcar_path, average="geometric")
    harmonic = extract_average_mass(contcar_path, outcar_path, average="harmonic")
    
    expected_arithmetic = 26.981
    expected_geometric = 26.981
    expected_harmonic = 26.981
    
    assert arithmetic == expected_arithmetic, f"Expected arithmetic average: {expected_arithmetic}, got {arithmetic}"
    assert geometric == expected_geometric, f"Expected geometric average: {expected_geometric}, got {geometric}"
    assert harmonic == expected_harmonic, f"Expected harmonic average: {expected_harmonic}, got {harmonic}"
    