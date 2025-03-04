"""
Tests for the magnetism module.
"""

# Standard library imports
import os

# Third-party library imports
import pytest

# DFTTK imports
from dfttk.magnetism import determine_magnetic_ordering, get_magnetic_structure

current_dir = os.path.dirname(os.path.abspath(__file__))
vasp_data_path = os.path.join(current_dir, "vasp_data/Al")

contcar_path = os.path.join(vasp_data_path, "config_Al/vol_0/CONTCAR.3static")
outcar_path = os.path.join(vasp_data_path, "config_Al/vol_0/OUTCAR.3static")


def test_determine_magnetic_ordering():
    pass


def test_get_magnetic_structure():
    with pytest.raises(ValueError):
        get_magnetic_structure(contcar_path, outcar_path)
