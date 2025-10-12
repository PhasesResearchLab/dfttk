"""
Tests for the dfttk.phonon module.
"""

# Standard library imports
import os
import re
import pickle

# Third-party imports
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# DFTTK imports
from dfttk.phonon.harmonic_phonon_yphon import HarmonicPhononYphon
from dfttk.phonon.yphon_phonon_data import YphonPhononData

current_dir = os.path.dirname(os.path.abspath(__file__))
config_Al_path = os.path.join(current_dir, "vasp_data/Al/config_Al")
yphon_results_path = os.path.join(config_Al_path, "YPHON_results")
number_of_atoms = 4
volumes_fit = np.linspace(0.98 * 60, 1.02 * 74, 1000)
temperatures = np.arange(0, 1010, 100)
RTOL = 1e-5  # Relative tolerance for all np.allclose comparisons


# Tests for HarmonicPhononYphon
def test_load_dos_raises_on_no_vdos_or_volph_files(tmp_path):
    """Raise FileNotFoundError if no vdos_ or volph_ files are found."""
    hp = HarmonicPhononYphon()
    with pytest.raises(FileNotFoundError, match="No vdos_ or volph_ files found in the specified directory."):
        hp.load_dos(str(tmp_path))


def test_load_dos_raises_on_mismatched_file_count(tmp_path):
    """Raise ValueError if the number of vdos and volph files do not match."""
    (tmp_path / "vdos_0.dat").write_text("dummy")
    (tmp_path / "volph_0.dat").write_text("dummy")
    (tmp_path / "volph_1.dat").write_text("dummy")
    hp = HarmonicPhononYphon()
    with pytest.raises(ValueError, match="number of vdos files does not match"):
        hp.load_dos(str(tmp_path))


def test_load_dos_raises_on_mismatched_indexes(tmp_path):
    """Raise ValueError if the indexes of vdos and volph files do not match."""
    (tmp_path / "vdos_0.dat").write_text("dummy")
    (tmp_path / "vdos_1.dat").write_text("dummy")
    (tmp_path / "volph_1.dat").write_text("dummy")
    (tmp_path / "volph_2.dat").write_text("dummy")
    hp = HarmonicPhononYphon()
    with pytest.raises(ValueError, match="indexes of vdos files do not match"):
        hp.load_dos(str(tmp_path))


def test_run_time_errors_harmonicphononyphon():
    """Raise RuntimeError for missing data in HarmonicPhononYphon workflow."""
    hp = HarmonicPhononYphon()
    with pytest.raises(RuntimeError, match=re.escape("Phonon DOS data not loaded. Call load_dos() before scale_dos().")):
        hp.scale_dos(number_of_atoms=4)
    hp.load_dos(yphon_results_path)
    with pytest.raises(RuntimeError, match=re.escape("Scaled phonon DOS data not calculated. Call scale_dos() before plot_dos().")):
        hp.plot_dos()
    with pytest.raises(RuntimeError, match=re.escape("Scaled phonon DOS data not calculated. Call scale_dos() before calculate_harmonic().")):
        hp.calculate_harmonic(temperatures=temperatures)
    hp.scale_dos(number_of_atoms=number_of_atoms)
    with pytest.raises(RuntimeError, match=re.escape("Thermodynamic properties not calculated. Call calculate_harmonic() before fit_harmonic().")):
        hp.fit_harmonic(volumes_fit=volumes_fit, order=2)
    with pytest.raises(RuntimeError, match=re.escape("Thermodynamic properties not calculated. Call calculate_harmonic() before fit_harmonic().")):
        hp.plot_harmonic("helmholtz_energy")
    hp.calculate_harmonic(temperatures=temperatures)
    with pytest.raises(RuntimeError, match=re.escape("Fitted thermodynamic properties not calculated. Call fit_harmonic() before plot_fit_harmonic().")):
        hp.plot_fit_harmonic("helmholtz_energy")


def test_plot_harmonic_valueerror():
    """Raise ValueError for invalid property in plot_harmonic."""
    hp = HarmonicPhononYphon()
    hp.load_dos(yphon_results_path)
    hp.scale_dos(number_of_atoms=number_of_atoms)
    hp.calculate_harmonic(temperatures=temperatures)
    hp.fit_harmonic(volumes_fit=volumes_fit, order=2)
    with pytest.raises(ValueError, match="property must be one of"):
        hp.plot_harmonic("not_a_property")


def test_plot_fit_harmonic_valueerror():
    """Raise ValueError for invalid property in plot_fit_harmonic."""
    hp = HarmonicPhononYphon()
    hp.load_dos(yphon_results_path)
    hp.scale_dos(number_of_atoms=number_of_atoms)
    hp.calculate_harmonic(temperatures=temperatures)
    hp.fit_harmonic(volumes_fit=volumes_fit, order=2)
    with pytest.raises(ValueError, match="property must be one of"):
        hp.plot_fit_harmonic("not_a_property")


def test_plot_scale_dos_smoke():
    """Smoke test for plot_scaled_dos: should run without error and not actually show the plot."""
    hp = HarmonicPhononYphon()
    hp.load_dos(yphon_results_path)
    hp.scale_dos(number_of_atoms=number_of_atoms, plot=True)


def test_plot_dos_smoke():
    """Smoke test for plot_dos: returns a plotly Figure."""
    hp = HarmonicPhononYphon()
    hp.load_dos(yphon_results_path)
    hp.scale_dos(number_of_atoms=number_of_atoms)
    fig = hp.plot_dos()
    assert isinstance(fig, go.Figure)


def test_plot_harmonic_and_fit_smoke():
    """Smoke test for plot_harmonic and plot_fit_harmonic for all valid properties."""
    hp = HarmonicPhononYphon()
    hp.load_dos(yphon_results_path)
    hp.scale_dos(number_of_atoms=number_of_atoms)
    hp.calculate_harmonic(temperatures=temperatures)
    hp.fit_harmonic(volumes_fit=volumes_fit, order=2)
    for prop in ["helmholtz_energy", "entropy", "heat_capacity"]:
        fig = hp.plot_harmonic(prop)
        assert isinstance(fig, go.Figure)
        fig_fit = hp.plot_fit_harmonic(prop)
        assert isinstance(fig_fit, go.Figure)
        selected_temperatures = np.array([0, 600])
        fig_fit_sel = hp.plot_fit_harmonic(prop, selected_temperatures=selected_temperatures)
        assert isinstance(fig_fit_sel, go.Figure)


def test_harmonicphononyphon():
    """Check HarmonicPhononYphon produces expected results for a reference dataset."""
    with open(os.path.join(current_dir, "test_phonon_data/harmonic_phonon_data.pkl"), "rb") as f:
        expected = pickle.load(f)
    hp = HarmonicPhononYphon()
    hp.load_dos(yphon_results_path)
    hp.scale_dos(number_of_atoms=number_of_atoms)
    hp.calculate_harmonic(temperatures=temperatures)
    hp.fit_harmonic(volumes_fit, order=2)
    pd.testing.assert_frame_equal(hp.phonon_dos, expected.phonon_dos)
    pd.testing.assert_frame_equal(hp.scaled_phonon_dos, expected.scaled_phonon_dos)
    assert hp.number_of_atoms == expected.number_of_atoms
    assert np.allclose(hp.volumes_per_atom, expected.volumes_per_atom, rtol=RTOL)
    assert np.allclose(hp.volumes, expected.volumes, rtol=RTOL)
    assert np.allclose(hp.temperatures, expected.temperatures, rtol=RTOL)
    assert np.allclose(hp.helmholtz_energies, expected.helmholtz_energies, rtol=RTOL)
    assert np.allclose(hp.entropies, expected.entropies, rtol=RTOL)
    assert np.allclose(hp.heat_capacities, expected.heat_capacities, rtol=RTOL)
    assert np.allclose(hp.volumes_fit, expected.volumes_fit, rtol=RTOL)
    assert np.allclose(hp.helmholtz_energies_fit, expected.helmholtz_energies_fit, rtol=RTOL)
    assert np.allclose(hp.entropies_fit, expected.entropies_fit, rtol=RTOL)
    assert np.allclose(hp.heat_capacities_fit, expected.heat_capacities_fit, rtol=RTOL)
    assert np.allclose(hp.helmholtz_energies_poly_coeffs, expected.helmholtz_energies_poly_coeffs, rtol=RTOL)
    assert np.allclose(hp.entropies_poly_coeffs, expected.entropies_poly_coeffs, rtol=RTOL)
    assert np.allclose(hp.heat_capacities_poly_coeffs, expected.heat_capacities_poly_coeffs, rtol=RTOL)


def test_harmonicphononyphon_subset():
    """Check HarmonicPhononYphon produces expected results for a subset of selected volumes."""
    with open(os.path.join(current_dir, "test_phonon_data/harmonic_phonon_data_subset.pkl"), "rb") as f:
        expected = pickle.load(f)
    hp = HarmonicPhononYphon()
    hp.load_dos(yphon_results_path)
    hp.scale_dos(number_of_atoms=number_of_atoms)
    selected_volumes = np.array([60.0, 62.0, 66.0, 68.0, 72.0, 74.0])
    hp.calculate_harmonic(temperatures=temperatures, selected_volumes=selected_volumes)
    hp.fit_harmonic(volumes_fit=volumes_fit, order=2)
    pd.testing.assert_frame_equal(hp.phonon_dos, expected.phonon_dos)
    pd.testing.assert_frame_equal(hp.scaled_phonon_dos, expected.scaled_phonon_dos)
    assert hp.number_of_atoms == expected.number_of_atoms
    assert np.allclose(hp.volumes_per_atom, expected.volumes_per_atom, rtol=RTOL)
    assert np.allclose(hp.volumes, expected.volumes, rtol=RTOL)
    assert np.allclose(hp.temperatures, expected.temperatures, rtol=RTOL)
    assert np.allclose(hp.helmholtz_energies, expected.helmholtz_energies, rtol=RTOL)
    assert np.allclose(hp.entropies, expected.entropies, rtol=RTOL)
    assert np.allclose(hp.heat_capacities, expected.heat_capacities, rtol=RTOL)
    assert np.allclose(hp.volumes_fit, expected.volumes_fit, rtol=RTOL)
    assert np.allclose(hp.helmholtz_energies_fit, expected.helmholtz_energies_fit, rtol=RTOL)
    assert np.allclose(hp.entropies_fit, expected.entropies_fit, rtol=RTOL)
    assert np.allclose(hp.heat_capacities_fit, expected.heat_capacities_fit, rtol=RTOL)
    assert np.allclose(hp.helmholtz_energies_poly_coeffs, expected.helmholtz_energies_poly_coeffs, rtol=RTOL)
    assert np.allclose(hp.entropies_poly_coeffs, expected.entropies_poly_coeffs, rtol=RTOL)
    assert np.allclose(hp.heat_capacities_poly_coeffs, expected.heat_capacities_poly_coeffs, rtol=RTOL)


# Tests for YphonPhononData
def test_yphonphonondata():
    """Check YphonPhononData produces expected results for a reference dataset."""
    with open(os.path.join(current_dir, "test_phonon_data/yphon_phonon_data.pkl"), "rb") as f:
        expected = pickle.load(f)
    pd_obj = YphonPhononData(config_Al_path)
    pd_obj.get_vasp_input()
    pd_obj.get_harmonic_data(number_of_atoms=number_of_atoms, volumes_fit=volumes_fit, temperatures=temperatures, order=2)
    assert pd_obj.incars == expected.incars
    assert pd_obj.kpoints == expected.kpoints
    assert pd_obj.phonon_structures == expected.phonon_structures
    assert pd_obj.number_of_atoms == expected.number_of_atoms
    assert np.allclose(pd_obj.volumes, expected.volumes, rtol=RTOL)
    assert np.allclose(pd_obj.temperatures, expected.temperatures, rtol=RTOL)
    assert np.allclose(pd_obj.helmholtz_energies, expected.helmholtz_energies, rtol=RTOL)
    assert np.allclose(pd_obj.internal_energies, expected.internal_energies, rtol=RTOL)
    assert np.allclose(pd_obj.entropies, expected.entropies, rtol=RTOL)
    assert np.allclose(pd_obj.heat_capacities, expected.heat_capacities, rtol=RTOL)
    assert np.allclose(pd_obj.volumes_fit, expected.volumes_fit, rtol=RTOL)
    assert np.allclose(pd_obj.helmholtz_energies_fit, expected.helmholtz_energies_fit, rtol=RTOL)
    assert np.allclose(pd_obj.entropies_fit, expected.entropies_fit, rtol=RTOL)
    assert np.allclose(pd_obj.heat_capacities_fit, expected.heat_capacities_fit, rtol=RTOL)
    assert np.allclose(pd_obj.helmholtz_energies_poly_coeffs, expected.helmholtz_energies_poly_coeffs, rtol=RTOL)
    assert np.allclose(pd_obj.entropies_poly_coeffs, expected.entropies_poly_coeffs, rtol=RTOL)
    assert np.allclose(pd_obj.heat_capacities_poly_coeffs, expected.heat_capacities_poly_coeffs, rtol=RTOL)


def test_yphonphonondata_subset():
    """Check YphonPhononData produces expected results for a subset of selected volumes."""
    with open(os.path.join(current_dir, "test_phonon_data/yphon_phonon_data_subset.pkl"), "rb") as f:
        expected = pickle.load(f)
    pd_obj = YphonPhononData(config_Al_path)
    selected_phonon_volumes = np.array([592.0, 544.0, 512.0])
    pd_obj.get_vasp_input(selected_phonon_volumes=selected_phonon_volumes)
    selected_volumes = selected_phonon_volumes / 32 * number_of_atoms
    selected_volumes_fit = np.linspace(0.98 * min(selected_volumes), 1.02 * max(selected_volumes), 1000)
    pd_obj.get_harmonic_data(number_of_atoms=number_of_atoms, volumes_fit=selected_volumes_fit, temperatures=temperatures, order=2, selected_volumes=selected_volumes)
    assert pd_obj.incars == expected.incars
    assert pd_obj.kpoints == expected.kpoints
    assert pd_obj.phonon_structures == expected.phonon_structures
    assert pd_obj.number_of_atoms == expected.number_of_atoms
    assert np.allclose(pd_obj.volumes, expected.volumes, rtol=RTOL)
    assert np.allclose(pd_obj.temperatures, expected.temperatures, rtol=RTOL)
    assert np.allclose(pd_obj.helmholtz_energies, expected.helmholtz_energies, rtol=RTOL)
    assert np.allclose(pd_obj.internal_energies, expected.internal_energies, rtol=RTOL)
    assert np.allclose(pd_obj.entropies, expected.entropies, rtol=RTOL)
    assert np.allclose(pd_obj.heat_capacities, expected.heat_capacities, rtol=RTOL)
    assert np.allclose(pd_obj.volumes_fit, expected.volumes_fit, rtol=RTOL)
    assert np.allclose(pd_obj.helmholtz_energies_fit, expected.helmholtz_energies_fit, rtol=RTOL)
    assert np.allclose(pd_obj.entropies_fit, expected.entropies_fit, rtol=RTOL)
    assert np.allclose(pd_obj.heat_capacities_fit, expected.heat_capacities_fit, rtol=RTOL)
    assert np.allclose(pd_obj.helmholtz_energies_poly_coeffs, expected.helmholtz_energies_poly_coeffs, rtol=RTOL)
    assert np.allclose(pd_obj.entropies_poly_coeffs, expected.entropies_poly_coeffs, rtol=RTOL)
    assert np.allclose(pd_obj.heat_capacities_poly_coeffs, expected.heat_capacities_poly_coeffs, rtol=RTOL)


def test_yphonphonondata_plot_smoke():
    """Smoke tests for plotting methods of YphonPhononData."""
    pd_obj = YphonPhononData(config_Al_path)
    pd_obj.get_vasp_input()
    pd_obj.get_harmonic_data(number_of_atoms=number_of_atoms, volumes_fit=volumes_fit, temperatures=temperatures, order=2)
    fig_scaled = pd_obj.plot_scaled_dos(number_of_atoms=number_of_atoms, plot=True)
    fig_multi = pd_obj.plot_multiple_dos()
    assert isinstance(fig_multi, go.Figure)
    for prop in ["helmholtz_energy", "entropy", "heat_capacity"]:
        figs = pd_obj.plot_harmonic(property=prop)
        assert isinstance(figs, tuple) and len(figs) == 2
        assert all(isinstance(f, go.Figure) for f in figs)
        selected_temperatures = np.array([0, 300, 600, 900])
        figs_sel = pd_obj.plot_harmonic(property=prop, selected_temperatures=selected_temperatures)
        assert isinstance(figs_sel, tuple) and len(figs_sel) == 2
        assert all(isinstance(f, go.Figure) for f in figs_sel)


def test_run_time_errors_yphonphonondata():
    """Raise RuntimeError for missing data in YphonPhononData workflow."""
    pd_obj = YphonPhononData(config_Al_path)
    with pytest.raises(RuntimeError, match=re.escape("Call get_harmonic_data() before plotting.")):
        pd_obj.plot_scaled_dos(number_of_atoms=number_of_atoms, plot=True)
    with pytest.raises(RuntimeError, match=re.escape("Call get_harmonic_data() before plotting.")):
        pd_obj.plot_multiple_dos()
    with pytest.raises(RuntimeError, match=re.escape("Call get_harmonic_data() before plotting.")):
        pd_obj.plot_harmonic(property="helmholtz_energy")
