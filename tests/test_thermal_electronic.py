"""
Tests for the ThermalElectronic class. 
"""

# Standard library imports
import os
import re
import pickle
import pytest

# Third-party library imports
import numpy as np

# DFTTK imports
from dfttk.thermal_electronic.functions import ThermalElectronic

current_dir = os.path.dirname(os.path.abspath(__file__))
config_Al_path = os.path.join(current_dir, "vasp_data/Al/config_Al")


# Load the expected results from a pickle file (for direct data tests)
with open(os.path.join(current_dir, "test_thermal_electronic_data", "expected_energies_list.pkl"), "rb") as f:
    expected_energies_list = pickle.load(f)
with open(os.path.join(current_dir, "test_thermal_electronic_data", "expected_dos_list.pkl"), "rb") as f:
    expected_dos_list = pickle.load(f)


# Pytest fixture to cache DOS data (read from disk only once per module)
@pytest.fixture(scope="module")
def cached_dos_data():
    te = ThermalElectronic()
    te.read_total_electron_dos(path=config_Al_path)
    return {
        "number_of_atoms": te.number_of_atoms,
        "volumes": te.volumes,
        "energies_list": te.energies_list,
        "dos_list": te.dos_list,
    }


def test_read_total_electron_dos():
    # Test all volumes
    # This test is specifically for the read_total_electron_dos method, so keep the direct call
    thermal_electronic = ThermalElectronic()
    thermal_electronic.read_total_electron_dos(path=config_Al_path)

    assert thermal_electronic.path == config_Al_path
    assert thermal_electronic.number_of_atoms == 4
    assert np.all(thermal_electronic.volumes == np.array([60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0]))

    # Check that the energies_list and dos_list match expected values
    for i, (energies_list, dos_list) in enumerate(zip(thermal_electronic.energies_list, thermal_electronic.dos_list)):
        np.testing.assert_allclose(energies_list, expected_energies_list[i])
        np.testing.assert_allclose(dos_list, expected_dos_list[i])

    # Test selected_volumes
    selected_volumes = np.array([60.0, 66.0])
    thermal_electronic.read_total_electron_dos(path=config_Al_path, selected_volumes=selected_volumes)

    assert thermal_electronic.path == config_Al_path
    assert thermal_electronic.number_of_atoms == 4
    assert np.all(thermal_electronic.volumes == np.array([60.0, 66.0]))

    # Check that the energies_list and dos_list match expected values
    for i, (energies_list, dos_list) in enumerate(zip(thermal_electronic.energies_list, thermal_electronic.dos_list)):
        np.testing.assert_allclose(energies_list, expected_energies_list[i * 3])  # 0 and 3 indices
        np.testing.assert_allclose(dos_list, expected_dos_list[i * 3])  # 0 and 3 indices

    # Test invalid selected_volumes (should raise an error)
    invalid_volumes = np.array([61.0, 67.0])
    with pytest.raises(ValueError, match="The following selected volumes were not found:"):
        thermal_electronic.read_total_electron_dos(path=config_Al_path, selected_volumes=invalid_volumes)

    # TODO: Test inconsistent number of atoms
    # TODO: Test inconsistent number of electrons


def test_set_total_electron_dos(cached_dos_data):
    thermal_electronic = ThermalElectronic()
    thermal_electronic.set_total_electron_dos(
        number_of_atoms=cached_dos_data["number_of_atoms"], volumes=cached_dos_data["volumes"], energies_list=cached_dos_data["energies_list"], dos_list=cached_dos_data["dos_list"]
    )

    assert thermal_electronic.path == None
    assert thermal_electronic.number_of_atoms == cached_dos_data["number_of_atoms"]
    assert np.all(thermal_electronic.volumes == cached_dos_data["volumes"])

    for i, (energies_list, dos_list) in enumerate(zip(thermal_electronic.energies_list, thermal_electronic.dos_list)):
        np.testing.assert_allclose(energies_list, cached_dos_data["energies_list"][i])
        np.testing.assert_allclose(dos_list, cached_dos_data["dos_list"][i])

    # For different lengths of volumes, energies, and dos_list (should raise an error)
    with pytest.raises(ValueError, match="Lengths of volumes, energies_list, and dos_list must be the same."):
        thermal_electronic.set_total_electron_dos(
            number_of_atoms=cached_dos_data["number_of_atoms"],
            volumes=cached_dos_data["volumes"][:-1],
            energies_list=cached_dos_data["energies_list"],
            dos_list=cached_dos_data["dos_list"],
        )  # Remove one volume to create mismatch


def test_process(cached_dos_data):
    temperatures = np.array([0, 300, 600, 900])

    thermal_electronic = ThermalElectronic()

    # Process without setting the DOS (should raise an error)
    with pytest.raises(
        ValueError, match=re.escape("DOS data not found. Please read or set the total electron DOS first using read_total_electron_dos() or set_total_electron_dos().")
    ):
        thermal_electronic.process(temperatures=temperatures)

    # Now set the DOS and process
    thermal_electronic.set_total_electron_dos(
        number_of_atoms=cached_dos_data["number_of_atoms"], volumes=cached_dos_data["volumes"], energies_list=cached_dos_data["energies_list"], dos_list=cached_dos_data["dos_list"]
    )
    thermal_electronic.process(temperatures=temperatures)

    assert np.all(thermal_electronic.temperatures == temperatures)
    assert np.allclose(thermal_electronic.helmholtz_energies, expected_helmholtz_energies)
    assert np.allclose(thermal_electronic.internal_energies, expected_internal_energies)
    assert np.allclose(thermal_electronic.entropies, expected_entropies)
    assert np.allclose(thermal_electronic.heat_capacities, expected_heat_capacities)


# Large expected arrays for test_process
expected_helmholtz_energies = np.array(
    [
        [-6.73648444e-08, -8.49219646e-08, -6.37930199e-07, -2.47784314e-07, -4.17356560e-07, -1.94288120e-07, -8.37631909e-07, -7.06400130e-07],
        [-1.86812194e-03, -1.94212095e-03, -2.01034372e-03, -2.04885416e-03, -2.07876817e-03, -2.11227531e-03, -2.14827701e-03, -2.18280849e-03],
        [-7.43625295e-03, -7.70784476e-03, -7.95363672e-03, -8.13724063e-03, -8.28299483e-03, -8.41321496e-03, -8.54203925e-03, -8.65734626e-03],
        [-1.65370414e-02, -1.71079727e-02, -1.76229463e-02, -1.80374507e-02, -1.83793761e-02, -1.86755333e-02, -1.89589683e-02, -1.92030741e-02],
    ]
)

expected_internal_energies = np.array(
    [
        [-6.73648444e-08, -8.49219646e-08, -6.37930199e-07, -2.47784314e-07, -4.17356560e-07, -1.94288120e-07, -8.37631909e-07, -7.06400130e-07],
        [1.87288067e-03, 1.93643996e-03, 1.99433637e-03, 2.05074681e-03, 2.09172870e-03, 2.12162543e-03, 2.15042343e-03, 2.17618076e-03],
        [7.29105411e-03, 7.52653664e-03, 7.73358755e-03, 7.93143942e-03, 8.10121120e-03, 8.23313062e-03, 8.34962180e-03, 8.44030084e-03],
        [1.59001038e-02, 1.63515358e-02, 1.67643207e-02, 1.71580707e-02, 1.75173170e-02, 1.78240169e-02, 1.81024403e-02, 1.83270521e-02],
    ]
)

expected_entropies = np.array(
    [
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        [1.24700087e-05, 1.29285363e-05, 1.33489336e-05, 1.36653366e-05, 1.39016562e-05, 1.41130025e-05, 1.43290015e-05, 1.45299642e-05],
        [2.45455118e-05, 2.53906357e-05, 2.61453738e-05, 2.67811334e-05, 2.73070100e-05, 2.77439093e-05, 2.81527684e-05, 2.84960785e-05],
        [3.60412725e-05, 3.71772316e-05, 3.82080744e-05, 3.91061349e-05, 3.98852145e-05, 4.05550557e-05, 4.11793428e-05, 4.17001402e-05],
    ]
)

expected_heat_capacities = np.array(
    [
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        [1.23895008e-05, 1.28039582e-05, 1.31599552e-05, 1.35335246e-05, 1.38546462e-05, 1.40612784e-05, 1.42335131e-05, 1.43632407e-05],
        [2.35292884e-05, 2.42063750e-05, 2.48167634e-05, 2.53797514e-05, 2.59075475e-05, 2.63707678e-05, 2.67862119e-05, 2.71075556e-05],
        [3.37636167e-05, 3.45191398e-05, 3.52743536e-05, 3.60181369e-05, 3.67568460e-05, 3.74607514e-05, 3.81281327e-05, 3.87085897e-05],
    ]
)


def test_fit(cached_dos_data):
    temperatures = np.array([0, 300, 600, 900])
    volumes_fit = np.linspace(0.98 * 60, 1.02 * 74, 10)
    order = 1

    thermal_electronic = ThermalElectronic()
    # Use set_total_electron_dos with cached data
    thermal_electronic.set_total_electron_dos(
        number_of_atoms=cached_dos_data["number_of_atoms"], volumes=cached_dos_data["volumes"], energies_list=cached_dos_data["energies_list"], dos_list=cached_dos_data["dos_list"]
    )

    # Call fit without process (should raise an error)
    with pytest.raises(ValueError, match=re.escape("Thermodynamic properties not yet calculated. Please call process() first")):
        thermal_electronic.fit(volumes_fit=volumes_fit, order=order)

    # Now process and fit
    thermal_electronic.process(temperatures=temperatures)
    thermal_electronic.fit(volumes_fit=volumes_fit, order=order)

    assert np.all(thermal_electronic.volumes_fit == volumes_fit)
    assert np.allclose(thermal_electronic.helmholtz_energies_fit, expected_helmholtz_energies_fit)
    assert np.allclose(thermal_electronic.entropies_fit, expected_entropies_fit)
    assert np.allclose(thermal_electronic.heat_capacities_fit, expected_heat_capacities_fit)

    # Set volume to only one value and test fit (should raise an error)
    thermal_electronic.volumes = np.array([60.0])
    with pytest.raises(ValueError, match=re.escape("Only one volume found. Need at least two volumes to perform fitting.")):
        thermal_electronic.fit(volumes_fit=volumes_fit, order=order)


# Large expected arrays for test_fit()
expected_helmholtz_energies_fit = np.array(
    [
        [-5.38607649e-08, -1.31915252e-07, -2.09969739e-07, -2.88024227e-07, -3.66078714e-07, -4.44133201e-07, -5.22187688e-07, -6.00242175e-07, -6.78296662e-07, -7.56351150e-07],
        [-1.87473063e-03, -1.91410618e-03, -1.95348174e-03, -1.99285730e-03, -2.03223285e-03, -2.07160841e-03, -2.11098396e-03, -2.15035952e-03, -2.18973508e-03, -2.22911063e-03],
        [-7.44612166e-03, -7.60324808e-03, -7.76037450e-03, -7.91750092e-03, -8.07462735e-03, -8.23175377e-03, -8.38888019e-03, -8.54600661e-03, -8.70313303e-03, -8.86025946e-03],
        [-1.65318518e-02, -1.68784350e-02, -1.72250181e-02, -1.75716013e-02, -1.79181845e-02, -1.82647677e-02, -1.86113509e-02, -1.89579341e-02, -1.93045173e-02, -1.96511004e-02],
    ]
)

expected_entropies_fit = np.array(
    [
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        [1.24917905e-05, 1.27560068e-05, 1.30202230e-05, 1.32844393e-05, 1.35486556e-05, 1.38128719e-05, 1.40770881e-05, 1.43413044e-05, 1.46055207e-05, 1.48697370e-05],
        [2.45366947e-05, 2.50528272e-05, 2.55689597e-05, 2.60850923e-05, 2.66012248e-05, 2.71173573e-05, 2.76334898e-05, 2.81496224e-05, 2.86657549e-05, 2.91818874e-05],
        [3.59397145e-05, 3.66837248e-05, 3.74277350e-05, 3.81717452e-05, 3.89157554e-05, 3.96597657e-05, 4.04037759e-05, 4.11477861e-05, 4.18917964e-05, 4.26358066e-05],
    ]
)

expected_heat_capacities_fit = np.array(
    [
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        [1.23790583e-05, 1.26436993e-05, 1.29083404e-05, 1.31729814e-05, 1.34376225e-05, 1.37022635e-05, 1.39669046e-05, 1.42315456e-05, 1.44961867e-05, 1.47608277e-05],
        [2.34075428e-05, 2.38834177e-05, 2.43592927e-05, 2.48351676e-05, 2.53110425e-05, 2.57869175e-05, 2.62627924e-05, 2.67386674e-05, 2.72145423e-05, 2.76904172e-05],
        [3.34021901e-05, 3.40636280e-05, 3.47250659e-05, 3.53865038e-05, 3.60479416e-05, 3.67093795e-05, 3.73708174e-05, 3.80322553e-05, 3.86936931e-05, 3.93551310e-05],
    ]
)


def test_plot_total_dos(cached_dos_data):
    thermal_electronic = ThermalElectronic()

    # Should raise error if DOS not set
    with pytest.raises(
        ValueError, match=re.escape("DOS data not found. Please read or set the total electron DOS first using read_total_electron_dos() or set_total_electron_dos().")
    ):
        thermal_electronic.plot_total_dos()

    # Set DOS and plot (smoke test, check no error and fig is not None)
    thermal_electronic.set_total_electron_dos(
        number_of_atoms=cached_dos_data["number_of_atoms"], volumes=cached_dos_data["volumes"], energies_list=cached_dos_data["energies_list"], dos_list=cached_dos_data["dos_list"]
    )
    fig = thermal_electronic.plot_total_dos()
    assert fig is not None


def test_plot_vs_temp(cached_dos_data):
    thermal_electronic = ThermalElectronic()

    # Set DOS
    thermal_electronic.set_total_electron_dos(
        number_of_atoms=cached_dos_data["number_of_atoms"], volumes=cached_dos_data["volumes"], energies_list=cached_dos_data["energies_list"], dos_list=cached_dos_data["dos_list"]
    )

    # Try to plot without process (should raise an error)
    with pytest.raises(ValueError, match=re.escape("Thermodynamic properties not yet calculated. Please call process() first.")):
        thermal_electronic.plot_vs_temp(property="helmholtz_energy")

    # Now process and plot (smoke test)
    temperatures = np.array([0, 300, 600, 900])
    thermal_electronic.process(temperatures=temperatures)

    for property in ["helmholtz_energy", "entropy", "heat_capacity"]:
        fig = thermal_electronic.plot_vs_temp(property=property)
        assert fig is not None

    # Test invalid property (should raise an error)
    with pytest.raises(ValueError, match=re.escape("property must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'")):
        thermal_electronic.plot_vs_temp(property="invalid_property")


def test_plot_vs_volume(cached_dos_data):
    thermal_electronic = ThermalElectronic()

    # Set DOS
    thermal_electronic.set_total_electron_dos(
        number_of_atoms=cached_dos_data["number_of_atoms"], volumes=cached_dos_data["volumes"], energies_list=cached_dos_data["energies_list"], dos_list=cached_dos_data["dos_list"]
    )

    # Try to plot without process (should raise an error)
    with pytest.raises(ValueError, match=re.escape("Thermodynamic properties not yet calculated. Please call process() first.")):
        thermal_electronic.plot_vs_volume(property="helmholtz_energy")

    # Now process and plot (smoke test)
    temperatures = np.array([0, 300, 600, 900])
    thermal_electronic.process(temperatures=temperatures)

    for property in ["helmholtz_energy", "entropy", "heat_capacity"]:
        fig = thermal_electronic.plot_vs_volume(property=property)
        assert fig is not None

    # Fit and plot fitted curves as well (smoke test)
    volumes_fit = np.linspace(0.98 * 60, 1.02 * 74, 10)
    order = 1
    thermal_electronic.fit(volumes_fit=volumes_fit, order=order)

    for property in ["helmholtz_energy", "entropy", "heat_capacity"]:
        fig = thermal_electronic.plot_vs_volume(property=property)
        assert fig is not None

    # Test plot for selected_temperatures
    selected_temperatures = np.array([300, 900])

    for property in ["helmholtz_energy", "entropy", "heat_capacity"]:
        fig = thermal_electronic.plot_vs_volume(property=property, selected_temperatures=selected_temperatures)
        assert fig is not None

    # Test invalid property (should raise an error)
    with pytest.raises(ValueError, match=re.escape("property must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'")):
        thermal_electronic.plot_vs_volume(property="invalid_property")
