"""
Tests for the dfttk.eos module.
"""

# Standard library imports
import re
import os
import pickle

# Third-party library imports
import pytest
import numpy as np

# DFTTK imports
from dfttk.eos.functions import *
from dfttk.eos.fit import EOSFitter
from dfttk.eos.ev_curve_data import EvCurveData

number_of_atoms = 4
volumes = np.array([74.0, 72.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0])
energies = np.array([-14.787067, -14.863567, -14.92244, -14.960229, -14.973035, -14.955434, -14.902786, -14.808673])
fitter = EOSFitter("Al", number_of_atoms, volumes, energies)


def assert_eos_results(
    eos_parameters,
    volume_range,
    energy_eos,
    pressure_eos,
    expected_eos_parameters,
    expected_volume_range,
    expected_energy_eos,
    expected_pressure_eos,
):
    """Helper function to assert EOS results with a consistent tolerance."""
    assert np.allclose(eos_parameters, expected_eos_parameters, rtol=3e-2), f"Expected {expected_eos_parameters}, got {eos_parameters}"
    assert np.allclose(volume_range, expected_volume_range, rtol=3e-2), f"Expected {expected_volume_range}, got {volume_range}"
    assert np.allclose(energy_eos, expected_energy_eos, rtol=3e-2), f"Expected {expected_energy_eos}, got {energy_eos}"
    assert np.allclose(pressure_eos, expected_pressure_eos, rtol=3e-2), f"Expected {expected_pressure_eos}, got {pressure_eos}"


# EOSFitter and eos.functions tests
# Error handling tests
def test_EOSFitter_errors():
    """Test that EOSFitter raises errors for invalid inputs."""
    volumes_unequal = np.array([1, 2])
    energies_unequal = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="Volumes and energies must have the same length."):
        EOSFitter("Al", number_of_atoms, volumes_unequal, energies_unequal)

    with pytest.raises(ValueError, match="EOS function 'unknown_eos' not recognized."):
        fitter.fit(eos_name="unknown_eos", num_volumes=10)

    with pytest.raises(ValueError, match=re.escape("You must call fit() before plot().")):
        fitter.plot()

    fitter.fit(eos_name="BM4", num_volumes=10)
    with pytest.raises(ValueError, match="cmap must be 'plotly' or 'distinctipy'"):
        fitter.plot(cmap="invalid_cmap")


# Plotting smoke test
def test_EOSFitter_plot_smoke():
    """Test that EOSFitter can plot without errors (smoke test)."""
    fitter.fit(eos_name="BM4", volume_min=60.0, volume_max=74.0, num_volumes=10)

    # Check if the plot method runs without errors
    try:
        fitter.plot()
    except Exception as e:
        pytest.fail(f"Plotting failed with error: {e}")


# mBM4 EOS fitting test
def test_mBM4():
    """Test the mBM4 equation of state fitting."""
    __, eos_parameters, volume_range, energy_eos, pressure_eos = mBM4(volumes, energies, num_volumes=10)  # Test the mBM4 equation of state fitting.
    fitter.fit(eos_name="mBM4", num_volumes=10)  # Test EOSFitter with mBM4 EOS.

    eos_parameters2 = fitter.eos_parameters
    volume_range2 = fitter.volume_range
    energy_eos2 = fitter.energy_eos
    pressure_eos2 = fitter.pressure_eos

    expected_eos_parameters = np.array([66.10232407335344, -14.972805616293215, 78.04074181821356, 4.603971403722543, -0.08199458053345686])
    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)
    expected_energy_eos = np.array([-14.80858914, -14.88584888, -14.93687147, -14.96489423, -14.97275283, -14.96293628, -14.93763348, -14.89877287, -14.84805627, -14.78698777])
    expected_pressure_eos = np.array([9.43148922, 6.54701637, 4.01879926, 1.8023087, -0.14083532, -1.84380962, -3.3353728, -4.64051925, -5.78102603, -6.775912])
    expected_energy_derivatve2 = np.array([0.00372272, 0.0044418, 0.00528349, 0.00627107, 0.00743295, 0.00880393, 0.0104269, 0.01235499])

    # Use the helper function for assertions
    assert_eos_results(
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )

    # Tests not covered by eos functions or EOSFitter
    V0 = eos_parameters[0]
    E0 = eos_parameters[1]
    B = eos_parameters[2]
    BP = eos_parameters[3]
    a, b, c, d = mBM4_eos_constants(V0, E0, B, BP)
    V0, E0, B, BP, B2P = mBM4_eos_parameters(a, b, c, d)
    energy_derivative2 = mBM4_derivative2(volumes, b, c, d)

    assert np.allclose(
        np.array([V0, E0, B, BP, B2P]),
        expected_eos_parameters,
        rtol=3e-2,
    ), f"Expected {expected_eos_parameters}, got {np.array([V0,E0,B,BP, B2P])}"
    assert np.allclose(
        energy_derivative2,
        expected_energy_derivatve2,
        rtol=3e-2,
    ), f"Expected {expected_energy_derivatve2}, got {energy_derivative2}"


# mBM5 EOS fitting test
def test_mBM5():
    """Test the mBM5 equation of state fitting."""
    __, eos_parameters, volume_range, energy_eos, pressure_eos = mBM5(volumes, energies, num_volumes=10)  # Test the mBM5 equation of state fitting.
    fitter.fit(eos_name="mBM5", num_volumes=10)  # Test EOSFitter with mBM5 EOS.

    eos_parameters2 = fitter.eos_parameters
    volume_range2 = fitter.volume_range
    energy_eos2 = fitter.energy_eos
    pressure_eos2 = fitter.pressure_eos

    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)

    expected_eos_parameters = np.array([66.10542344565768, -14.972938667152562, 78.53582491557574, 4.549077485105092, -0.16682566975204927])
    expected_energy_eos = np.array([-14.80866964, -14.88567265, -14.9367749, -14.96494777, -14.97288825, -14.96304204, -14.93762528, -14.8986448, -14.84791694, -14.78708477])
    expected_pressure_eos = np.array([9.37451174, 6.54374803, 4.03398164, 1.81572153, -0.13808295, -1.85217081, -3.34911634, -4.64948339, -5.7719805, -6.73361204])
    expected_energy_derivative2 = np.array([0.00355687, 0.00435729, 0.00526821, 0.00630408, 0.007481, 0.00881677, 0.01033085, 0.01204411])

    # Use the helper function for assertions
    assert_eos_results(
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )

    # Tests not covered by eos functions or EOSFitter
    V0 = eos_parameters[0]
    E0 = eos_parameters[1]
    B = eos_parameters[2]
    BP = eos_parameters[3]
    B2P = eos_parameters[4]
    a, b, c, d, e = mBM5_eos_constants(V0, E0, B, BP, B2P)
    V0, E0, B, BP, B2P = mBM5_eos_parameters(volume_range, a, b, c, d, e)
    energy_derivative2 = mBM5_derivative2(volumes, b, c, d, e)

    assert np.allclose(
        np.array([V0, E0, B, BP, B2P]),
        expected_eos_parameters,
        rtol=3e-2,
    ), f"Expected {expected_eos_parameters}, got {np.array([V0,E0,B,BP, B2P])}"
    assert np.allclose(
        energy_derivative2,
        expected_energy_derivative2,
        rtol=3e-2,
    ), f"Expected {expected_energy_derivative2}, got {energy_derivative2}"


# BM4 EOS fitting test
def test_BM4():
    """Test the BM4 equation of state fitting."""
    __, eos_parameters, volume_range, energy_eos, pressure_eos = BM4(volumes, energies, num_volumes=10)  # Test the BM4 equation of state fitting.
    fitter.fit(eos_name="BM4", num_volumes=10)  # Test EOSFitter with BM4 EOS.

    eos_parameters2 = fitter.eos_parameters
    volume_range2 = fitter.volume_range
    energy_eos2 = fitter.energy_eos
    pressure_eos2 = fitter.pressure_eos

    expected_eos_parameters = np.array([66.10191547034127, -14.972775074363833, 77.92792067011315, 4.612739661291564, -0.06258448064264342])
    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)
    expected_energy_eos = np.array([-14.80857541, -14.88588549, -14.93688813, -14.96487866, -14.97272201, -14.96291517, -14.93763964, -14.89880545, -14.84808841, -14.78696169])
    expected_pressure_eos = np.array([9.44321537, 6.54729341, 4.01539266, 1.79956554, -0.14110841, -1.84166929, -3.33221706, -4.63868592, -5.78348475, -6.78602896])
    expected_energy_derivative2 = np.array([0.00376055, 0.00446175, 0.00528792, 0.00626431, 0.00742211, 0.00880004, 0.01044646, 0.01242216])

    # Use the helper function for assertions
    assert_eos_results(
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )

    # Tests not covered by eos functions or EOSFitter
    V0 = eos_parameters[0]
    E0 = eos_parameters[1]
    B = eos_parameters[2]
    BP = eos_parameters[3]
    a, b, c, d = BM4_eos_constants(V0, E0, B, BP)
    V0, E0, B, BP, B2P = BM4_eos_parameters(a, b, c, d)
    energy_derivative2 = BM4_derivative2(volumes, b, c, d)

    assert np.allclose(
        np.array([V0, E0, B, BP, B2P]),
        expected_eos_parameters,
        rtol=3e-2,
    ), f"Expected {expected_eos_parameters}, got {np.array([V0,E0,B,BP, B2P])}"
    assert np.allclose(
        energy_derivative2,
        expected_energy_derivative2,
        rtol=3e-2,
    ), f"Expected {expected_energy_derivative2}, got {energy_derivative2}"


# BM5 EOS fitting test
def test_BM5():
    """Test the BM5 equation of state fitting."""
    __, eos_parameters, volume_range, energy_eos, pressure_eos = BM5(volumes, energies, num_volumes=10)  # Test the BM5 equation of state fitting.
    fitter.fit(eos_name="BM5", num_volumes=10)  # Test EOSFitter with BM5 EOS.

    eos_parameters2 = fitter.eos_parameters
    volume_range2 = fitter.volume_range
    energy_eos2 = fitter.energy_eos
    pressure_eos2 = fitter.pressure_eos

    expected_eos_parameters = np.array([66.10494560451701, -14.97294336567709, 78.55818945461328, 4.566872980385188, -0.1715488686904988])
    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)
    expected_energy_eos = np.array([-14.8086726, -14.88566133, -14.93677519, -14.96494334, -14.97288522, -14.96304321, -14.93762865, -14.89864603, -14.84791413, -14.78708483])
    expected_pressure_eos = np.array([9.3696372, 6.5444211, 4.03516087, 1.81589736, -0.13868335, -1.85275913, -3.34908499, -4.64884651, -5.77160844, -6.73532668])
    expected_energy_derivative2 = np.array([0.00357019, 0.00435943, 0.00526553, 0.00630216, 0.00748309, 0.00882116, 0.01032664, 0.01200455])

    # Use the helper function for assertions
    assert_eos_results(
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )

    # Tests not covered by eos functions or EOSFitter
    V0 = eos_parameters[0]
    E0 = eos_parameters[1]
    B = eos_parameters[2]
    BP = eos_parameters[3]
    B2P = eos_parameters[4]
    a, b, c, d, e = BM5_eos_constants(V0, E0, B, BP, B2P)
    V0, E0, B, BP, B2P = BM5_eos_parameters(volume_range, a, b, c, d, e)
    energy_derivative2 = BM5_derivative2(volumes, b, c, d, e)

    assert np.allclose(
        np.array([V0, E0, B, BP, B2P]),
        expected_eos_parameters,
        rtol=3e-2,
    ), f"Expected {expected_eos_parameters}, got {np.array([V0,E0,B,BP, B2P])}"
    assert np.allclose(
        energy_derivative2,
        expected_energy_derivative2,
        rtol=3e-2,
    ), f"Expected {expected_energy_derivative2}, got {energy_derivative2}"


# LOG4 EOS fitting test
def test_LOG4():
    """Test the LOG4 equation of state fitting."""
    __, eos_parameters, volume_range, energy_eos, pressure_eos = LOG4(volumes, energies, num_volumes=10)  # Test the LOG4 equation of state fitting.
    fitter.fit(eos_name="LOG4", num_volumes=10)  # Test EOSFitter with LOG4 EOS.

    eos_parameters2 = fitter.eos_parameters
    volume_range2 = fitter.volume_range
    energy_eos2 = fitter.energy_eos
    pressure_eos2 = fitter.pressure_eos

    expected_eos_parameters = np.array([66.10417220805562, -14.972881468407536, 78.32086632030024, 4.5678527771002395, -0.12974474286845175])
    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)
    expected_energy_eos = np.array([-14.80863335, -14.8857512, -14.93681727, -14.96492436, -14.97283011, -14.96299734, -14.93762891, -14.89869789, -14.8479738, -14.78704547])
    expected_pressure_eos = np.array([9.40029153, 6.54508695, 4.02723604, 1.80992917, -0.1391734, -1.84855594, -3.34336671, -4.64585903, -5.7757668, -6.75062551])
    expected_energy_derivative2 = np.array([0.00362194, 0.00439183, 0.00527518, 0.00629056, 0.00746013, 0.0088104, 0.01037327, 0.0121873])

    # Use the helper function for assertions
    assert_eos_results(
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )

    # Tests not covered by eos functions or EOSFitter
    V0 = eos_parameters[0]
    E0 = eos_parameters[1]
    B = eos_parameters[2]
    BP = eos_parameters[3]
    a, b, c, d = LOG4_eos_constants(V0, E0, B, BP)
    V0, E0, B, BP, B2P = LOG4_eos_parameters(volume_range, a, b, c, d)
    energy_derivative2 = LOG4_derivative2(volumes, b, c, d)

    assert np.allclose(
        np.array([V0, E0, B, BP, B2P]),
        expected_eos_parameters,
        rtol=3e-2,
    ), f"Expected {expected_eos_parameters}, got {np.array([V0,E0,B,BP, B2P])}"
    assert np.allclose(
        energy_derivative2,
        expected_energy_derivative2,
        rtol=3e-2,
    ), f"Expected {expected_energy_derivative2}, got {energy_derivative2}"


# LOG5 EOS fitting test
def test_LOG5():
    """Test the LOG5 equation of state fitting."""
    __, eos_parameters, volume_range, energy_eos, pressure_eos = LOG5(volumes, energies, num_volumes=10)  # Test the LOG5 equation of state fitting.
    fitter.fit(eos_name="LOG5", num_volumes=10)  # Test EOSFitter with LOG5 EOS.

    eos_parameters2 = fitter.eos_parameters
    volume_range2 = fitter.volume_range
    energy_eos2 = fitter.energy_eos
    pressure_eos2 = fitter.pressure_eos

    expected_eos_parameters = np.array([66.10571794361299, -14.972935374888266, 78.51990653243391, 4.538051436151986, -0.16321711742191786])
    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)
    expected_energy_eos = np.array([-14.80866759, -14.88568006, -14.93677519, -14.96494334, -14.97288522, -14.96304321, -14.93762865, -14.89864603, -14.84791413, -14.78708483])
    expected_pressure_eos = np.array([9.37760388, 6.54335988, 4.03321021, 1.81557029, -0.13770931, -1.8517691, -3.34910711, -4.64989492, -5.77225095, -6.73247751])
    expected_energy_derivative2 = np.array([0.0035477, 0.00435601, 0.00527006, 0.00630523, 0.0074795, 0.00881404, 0.0103338, 0.01206839])

    # Use the helper function for assertions
    assert_eos_results(
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )

    # Tests not covered by eos functions or EOSFitter
    V0 = eos_parameters[0]
    E0 = eos_parameters[1]
    B = eos_parameters[2]
    BP = eos_parameters[3]
    B2P = eos_parameters[4]
    a, b, c, d, e = LOG5_eos_constants(V0, E0, B, BP, B2P)
    V0, E0, B, BP, B2P = LOG5_eos_parameters(volume_range, a, b, c, d, e)
    energy_derivative2 = LOG5_derivative2(volumes, b, c, d, e)

    assert np.allclose(
        np.array([V0, E0, B, BP, B2P]),
        expected_eos_parameters,
        rtol=3e-2,
    ), f"Expected {expected_eos_parameters}, got {np.array([V0,E0,B,BP, B2P])}"
    assert np.allclose(
        energy_derivative2,
        expected_energy_derivative2,
        rtol=3e-2,
    ), f"Expected {expected_energy_derivative2}, got {energy_derivative2}"


# Murnaghan EOS fitting test
def test_murnaghan():
    """Test the murnaghan equation of state fitting."""
    __, eos_parameters, volume_range, energy_eos, pressure_eos = murnaghan(volumes, energies, num_volumes=10)  # Test the murnaghan equation of state fitting.
    fitter.fit(eos_name="murnaghan", num_volumes=10)  # Test EOSFitter with murnaghan EOS.

    eos_parameters2 = fitter.eos_parameters
    volume_range2 = fitter.volume_range
    energy_eos2 = fitter.energy_eos
    pressure_eos2 = fitter.pressure_eos

    expected_eos_parameters = np.array([66.10051541316386, -14.972678601682624, 77.57033054075912, 4.638920125800259, 0.0])
    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)
    expected_energy_eos = np.array([-14.80852844, -14.88600543, -14.93694421, -14.96483083, -14.97262454, -14.96284734, -14.93765668, -14.89890482, -14.84818745, -14.78688374])
    expected_pressure_eos = np.array([9.48202844, 6.54833124, 4.00441242, 1.79071425, -0.14208703, -1.83502591, -3.3223889, -4.63294293, -5.7909193, -6.816806])

    # Use the helper function for assertions
    assert_eos_results(
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )


# Vinet EOS fitting test
def test_vinet():
    """Test the vinet equation of state fitting."""
    __, eos_parameters, volume_range, energy_eos, pressure_eos = vinet(volumes, energies, num_volumes=10)  # Test the vinet equation of state fitting.
    fitter.fit(eos_name="vinet", num_volumes=10)  # Test EOSFitter with vinet EOS.

    eos_parameters2 = fitter.eos_parameters
    volume_range2 = fitter.volume_range
    energy_eos2 = fitter.energy_eos
    pressure_eos2 = fitter.pressure_eos

    expected_eos_parameters = np.array([66.102714503444, -14.972818017637815, 78.08775964234005, 4.598758024396886, -0.09039502616658532])
    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)
    expected_energy_eos = np.array([-14.80859905, -14.88582997, -14.93685998, -14.96489803, -14.97276554, -14.9629472, -14.93763448, -14.89876297, -14.8480444, -14.78699431])
    expected_pressure_eos = np.array([9.42521358, 6.54654636, 4.02031868, 1.8037025, -0.14046339, -1.84450265, -3.3365967, -4.64136987, -5.78038347, -6.77255427])

    # Use the helper function for assertions
    assert_eos_results(
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )


# Morse EOS fitting test
def test_morse():
    """Test the morse equation of state fitting."""
    __, eos_parameters, volume_range, energy_eos, pressure_eos = morse(volumes, energies, num_volumes=10)  # Test the morse equation of state fitting.
    fitter.fit(eos_name="morse", num_volumes=10)  # Test EOSFitter with morse EOS.

    eos_parameters2 = fitter.eos_parameters
    volume_range2 = fitter.volume_range
    energy_eos2 = fitter.energy_eos
    pressure_eos2 = fitter.pressure_eos

    expected_eos_parameters = np.array([66.10258535018146, -14.972811004254211, 78.0617287696514, 4.601191170503221, -0.08589765672017415])
    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)
    expected_energy_eos = np.array([-14.80859533, -14.88583885, -14.93686446, -14.96489484, -14.97275843, -14.96294198, -14.93763539, -14.89877005, -14.84805172, -14.7869888])
    expected_pressure_eos = np.array([9.4280737, 6.5466657, 4.01952541, 1.80303261, -0.14056754, -1.84403695, -3.33587398, -4.64092543, -5.78090175, -6.77481233])

    # Use the helper function for assertions
    assert_eos_results(
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )


# EvCurveData tests
# RuntimeError
def test_fit_energy_volume_data_runtime_error():
    """Test that fit_energy_volume_data raises RuntimeError if energy-volume data is not loaded."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_Al_path = os.path.join(current_dir, "vasp_data/Al/config_Al")
    ev_data = EvCurveData(config_Al_path, "Al")

    # Do NOT call get_energy_volume_data
    with pytest.raises(RuntimeError, match="You must call get_energy_volume_data\\(\\) before fit_energy_volume_data\\(\\)."):
        ev_data.fit_energy_volume_data()


# Helper function to assert EvCurveData matches expected data
def assert_ev_data_matches_expected(ev_data, expected_ev_data):
    assert ev_data.incars == expected_ev_data.incars, f"Expected {expected_ev_data.incars}, but got {ev_data.incars}"
    assert ev_data.kpoints == expected_ev_data.kpoints, f"Expected {expected_ev_data.kpoints}, but got {ev_data.kpoints}"
    assert ev_data.initial_poscar == expected_ev_data.initial_poscar, f"Expected {expected_ev_data.initial_poscar}, but got {ev_data.initial_poscar}"
    assert ev_data.relaxed_structures == expected_ev_data.relaxed_structures, f"Expected {expected_ev_data.relaxed_structures}, but got {ev_data.relaxed_structures}"
    assert ev_data.number_of_atoms == expected_ev_data.number_of_atoms, f"Expected {expected_ev_data.number_of_atoms}, but got {ev_data.number_of_atoms}"
    assert np.allclose(ev_data.volumes, expected_ev_data.volumes, rtol=1e-5), f"Expected {expected_ev_data.volumes}, but got {ev_data.volumes}"
    assert np.allclose(ev_data.energies, expected_ev_data.energies, rtol=1e-5), f"Expected {expected_ev_data.energies}, but got {ev_data.energies}"
    assert ev_data.atomic_masses == expected_ev_data.atomic_masses, f"Expected {expected_ev_data.atomic_masses}, but got {ev_data.atomic_masses}"
    assert ev_data.average_mass == expected_ev_data.average_mass, f"Expected {expected_ev_data.average_mass}, but got {ev_data.average_mass}"
    # Use == for lists, np.allclose for arrays
    if isinstance(ev_data.total_magnetic_moment, list) and isinstance(expected_ev_data.total_magnetic_moment, list):
        assert ev_data.total_magnetic_moment == expected_ev_data.total_magnetic_moment, f"Expected {expected_ev_data.total_magnetic_moment}, but got {ev_data.total_magnetic_moment}"
    else:
        assert np.allclose(ev_data.total_magnetic_moment, expected_ev_data.total_magnetic_moment, rtol=1e-5), f"Expected {expected_ev_data.total_magnetic_moment}, but got {ev_data.total_magnetic_moment}"
    # Use == for lists, np.array_equal for numeric arrays
    if isinstance(ev_data.magnetic_ordering, list) and isinstance(expected_ev_data.magnetic_ordering, list):
        assert ev_data.magnetic_ordering == expected_ev_data.magnetic_ordering, f"Expected {expected_ev_data.magnetic_ordering}, but got {ev_data.magnetic_ordering}"
    else:
        assert np.array_equal(ev_data.magnetic_ordering, expected_ev_data.magnetic_ordering), f"Expected {expected_ev_data.magnetic_ordering}, but got {ev_data.magnetic_ordering}"
    assert ev_data.mag_data == expected_ev_data.mag_data, f"Expected {expected_ev_data.mag_data}, but got {ev_data.mag_data}"
    assert ev_data.eos_parameters["eos_name"] == expected_ev_data.eos_parameters["eos_name"], f"Expected {expected_ev_data.eos_parameters['eos_name']}, but got {ev_data.eos_parameters['eos_name']}"
    # Don't test the eos constants a, b, c, d, and e
    for key in ["V0", "E0", "B", "BP", "B2P"]:
        if key in ev_data.eos_parameters and key in expected_ev_data.eos_parameters:
            assert np.isclose(
                ev_data.eos_parameters[key],
                expected_ev_data.eos_parameters[key],
                rtol=3e-2,
            ), f"Expected {expected_ev_data.eos_parameters[key]} for '{key}', but got {ev_data.eos_parameters[key]}"


# EvCurveData tests (non-magnetic Al)
def test_EvCurveData_Al_non_magnetic():
    """Test EvCurveData for a non-magnetic system and compare with reference pickle data."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_Al_path = os.path.join(current_dir, "vasp_data/Al/config_Al")

    # Test with collect_mag_data=False (default)
    ev_data = EvCurveData(config_Al_path, "Al")
    ev_data.get_vasp_input()
    ev_data.get_energy_volume_data()
    ev_data.fit_energy_volume_data()
    with open(os.path.join(current_dir, "test_eos_data/ev_data_Al.pkl"), "rb") as f:
        expected_ev_data = pickle.load(f)
    assert_ev_data_matches_expected(ev_data, expected_ev_data)

    # Test with collect_mag_data=True
    ev_data_mag = EvCurveData(config_Al_path, "Al")
    ev_data_mag.get_vasp_input()
    ev_data_mag.get_energy_volume_data(collect_mag_data=True)
    ev_data_mag.fit_energy_volume_data()
    with open(os.path.join(current_dir, "test_eos_data/ev_data_Al.pkl"), "rb") as f:
        expected_ev_data = pickle.load(f)
    assert_ev_data_matches_expected(ev_data_mag, expected_ev_data)

    # Test for a subset of selected volumes with collect_mag_data=False
    ev_data_subset = EvCurveData(config_Al_path, "Al")
    selected_volumes = np.array([74.0, 72.0, 70.0, 68.0, 64.0, 62.0])
    ev_data_subset.get_vasp_input(selected_volumes=selected_volumes)
    ev_data_subset.get_energy_volume_data(selected_volumes=selected_volumes)
    ev_data_subset.fit_energy_volume_data()
    with open(os.path.join(current_dir, "test_eos_data/ev_data_subset_Al.pkl"), "rb") as f:
        expected_ev_data_subset = pickle.load(f)
    assert_ev_data_matches_expected(ev_data_subset, expected_ev_data_subset)

    # Test for a subset of selected volumes with collect_mag_data=True
    ev_data_subset_mag = EvCurveData(config_Al_path, "Al")
    ev_data_subset_mag.get_vasp_input(selected_volumes=selected_volumes)
    ev_data_subset_mag.get_energy_volume_data(selected_volumes=selected_volumes, collect_mag_data=True)
    ev_data_subset_mag.fit_energy_volume_data()
    with open(os.path.join(current_dir, "test_eos_data/ev_data_subset_Al.pkl"), "rb") as f:
        expected_ev_data_subset = pickle.load(f)
    assert_ev_data_matches_expected(ev_data_subset_mag, expected_ev_data_subset)


# EvCurveData tests (magnetic Fe3Pt)
def test_EvCurveData_Fe3Pt_magnetic():
    """Test EvCurveData for a magnetic Fe3Pt system (ISPIN=2, LORBIT=11) and compare with reference pickle data."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_Fe3Pt_path = os.path.join(current_dir, "vasp_data/Fe3Pt/config_28")

    # Test with collect_mag_data=True
    ev_data_mag = EvCurveData(config_Fe3Pt_path, "Fe3Pt")
    ev_data_mag.get_vasp_input()
    ev_data_mag.get_energy_volume_data(collect_mag_data=True)
    ev_data_mag.fit_energy_volume_data()
    with open(os.path.join(current_dir, "test_eos_data/ev_data_Fe3Pt.pkl"), "rb") as f:
        expected_ev_data = pickle.load(f)
    assert_ev_data_matches_expected(ev_data_mag, expected_ev_data)

    # Test for a subset of selected volumes with collect_mag_data=True
    ev_data_subset_mag = EvCurveData(config_Fe3Pt_path, "Fe3Pt")
    selected_volumes = np.array([160.0, 157.0, 154.0, 151.0, 148.0])
    ev_data_subset_mag = EvCurveData(config_Fe3Pt_path, "config_28")
    ev_data_subset_mag.get_vasp_input(selected_volumes=selected_volumes)
    ev_data_subset_mag.get_energy_volume_data(selected_volumes=selected_volumes, collect_mag_data=True)
    ev_data_subset_mag.fit_energy_volume_data()
    with open(os.path.join(current_dir, "test_eos_data/ev_data_subset_Fe3Pt.pkl"), "rb") as f:
        expected_ev_data_subset = pickle.load(f)
    assert_ev_data_matches_expected(ev_data_subset_mag, expected_ev_data_subset)

    # Test for a subset of selected volumes with different ordering of volumes (should give the same result) with collect_mag_data=True
    ev_data_subset_mag = EvCurveData(config_Fe3Pt_path, "Fe3Pt")
    selected_volumes = np.array([151.0, 160.0, 148.0, 157.0, 154.0])
    ev_data_subset_mag = EvCurveData(config_Fe3Pt_path, "config_28")
    ev_data_subset_mag.get_vasp_input(selected_volumes=selected_volumes)
    ev_data_subset_mag.get_energy_volume_data(selected_volumes=selected_volumes, collect_mag_data=True)
    ev_data_subset_mag.fit_energy_volume_data()
    with open(os.path.join(current_dir, "test_eos_data/ev_data_subset_Fe3Pt.pkl"), "rb") as f:
        expected_ev_data_subset = pickle.load(f)
    assert_ev_data_matches_expected(ev_data_subset_mag, expected_ev_data_subset)
