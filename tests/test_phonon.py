"""
Tests for the dfttk.phonon module.
"""

# Standard Library Imports
import os
import pickle

# Third-Party Library Imports
import numpy as np
import plotly.graph_objects as go

# DFTTK Imports
from dfttk.phonon.harmonic_phonon_yphon import HarmonicPhononYphon
from dfttk.phonon.yphon_phonon_data import YphonPhononData

current_dir = os.path.dirname(os.path.abspath(__file__))
config_Al_path = os.path.join(current_dir, "vasp_data/Al/config_Al")
number_of_atoms = 4
temperatures = np.arange(0, 1010, 100)

RTOL = 1e-5  # Relative tolerance for all np.allclose comparisons

def test_YphonPhononData():
    """
    Test that YphonPhononData produces expected results for a reference dataset.
    """
    with open(os.path.join(current_dir, "test_phonon_data/yphon_phonon_data.pkl"), "rb") as f:
        expected_phonon_data = pickle.load(f)

    phonon_data = YphonPhononData(config_Al_path)
    phonon_data.get_vasp_input()
    phonon_data.get_harmonic_data(number_of_atoms=number_of_atoms, temperatures=temperatures, order=2)

    assert phonon_data.incars == expected_phonon_data.incars, "INCARS do not match expected data."
    assert phonon_data.kpoints == expected_phonon_data.kpoints, "KPOINTS do not match expected data."
    assert phonon_data.phonon_structures == expected_phonon_data.phonon_structures, "Phonon structures do not match expected data."
    assert phonon_data.number_of_atoms == expected_phonon_data.number_of_atoms, "Number of atoms does not match expected data."
    assert np.allclose(phonon_data.volumes, expected_phonon_data.volumes, rtol=RTOL), "Volumes do not match expected data."
    assert np.allclose(phonon_data.temperatures, expected_phonon_data.temperatures, rtol=RTOL), "Temperatures do not match expected data."
    assert np.allclose(phonon_data.helmholtz_energy, expected_phonon_data.helmholtz_energy, rtol=RTOL), "Helmholtz energies do not match expected data."
    assert np.allclose(phonon_data.internal_energy, expected_phonon_data.internal_energy, rtol=RTOL), "Internal energies do not match expected data."
    assert np.allclose(phonon_data.entropy, expected_phonon_data.entropy, rtol=RTOL), "Entropies do not match expected data."
    assert np.allclose(phonon_data.heat_capacity, expected_phonon_data.heat_capacity, rtol=RTOL), "Heat capacities do not match expected data."
    assert np.allclose(phonon_data.volumes_fit, expected_phonon_data.volumes_fit, rtol=RTOL), "Volumes fit do not match expected data."
    assert np.allclose(phonon_data.helmholtz_energy_fit, expected_phonon_data.helmholtz_energy_fit, rtol=RTOL), "Helmholtz energy fit do not match expected data."
    assert np.allclose(phonon_data.entropy_fit, expected_phonon_data.entropy_fit, rtol=RTOL), "Entropy fit do not match expected data."
    assert np.allclose(phonon_data.heat_capacity_fit, expected_phonon_data.heat_capacity_fit, rtol=RTOL), "Heat capacity fit do not match expected data."
    assert np.allclose(phonon_data.helmholtz_energy_poly_coeffs, expected_phonon_data.helmholtz_energy_poly_coeffs, rtol=RTOL), "Helmholtz energy polynomial coefficients do not match expected data."
    assert np.allclose(phonon_data.entropy_poly_coeffs, expected_phonon_data.entropy_poly_coeffs, rtol=RTOL), "Entropy polynomial coefficients do not match expected data."
    assert np.allclose(phonon_data.heat_capacity_poly_coeffs, expected_phonon_data.heat_capacity_poly_coeffs, rtol=RTOL), "Heat capacity polynomial coefficients do not match expected data."


def test_YphonPhononData_plot_smoke():
    """
    Smoke tests for plotting methods of YphonPhononData.
    Ensures that plot functions run without error after get_harmonic_data().
    Also tests the selected_temperatures argument for plot_harmonic.
    """
    phonon_data = YphonPhononData(config_Al_path)
    phonon_data.get_vasp_input()
    phonon_data.get_harmonic_data(number_of_atoms=number_of_atoms, temperatures=temperatures, order=2)

    # plot_scaled_dos should return a Plotly Figure
    fig_scaled = phonon_data.plot_scaled_dos(number_of_atoms=number_of_atoms, plot=True)
    assert isinstance(fig_scaled, go.Figure)

    # plot_multiple_dos should return a Plotly Figure
    fig_multi = phonon_data.plot_multiple_dos()
    assert isinstance(fig_multi, go.Figure)

    # plot_harmonic for all valid properties should return a tuple of two Plotly Figures
    for prop in ["helmholtz_energy", "entropy", "heat_capacity"]:
        # Default: all temperatures
        figs = phonon_data.plot_harmonic(property=prop)
        assert isinstance(figs, tuple) and len(figs) == 2
        assert all(isinstance(f, go.Figure) for f in figs)
        # With selected_temperatures
        selected_temperatures = np.array([0, 300, 600, 900])
        figs_sel = phonon_data.plot_harmonic(property=prop, selected_temperatures=selected_temperatures)
        assert isinstance(figs_sel, tuple) and len(figs_sel) == 2
        assert all(isinstance(f, go.Figure) for f in figs_sel)
