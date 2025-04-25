"""
Tests for the dfttk.eos module.    
"""

# Third-party library imports
import pytest
import numpy as np

# DFTTK imports
from dfttk.eos.functions import *
from dfttk.eos.fit import fit_to_eos

# TODO: Write tests for the 2nd derivatives of the equations of state
# The first derivative is already covered by the pressure test

# No tests for EvCurveData as test_config.py should be sufficient

# Conversion factor
EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3  = 160.21766208 GPa

volumes = np.array([74.0, 72.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0])
energies = np.array(
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


def assert_eos_results(
    eos_constants,
    eos_parameters,
    volume_range,
    energy_eos,
    pressure_eos,
    expected_eos_constants,
    expected_eos_parameters,
    expected_volume_range,
    expected_energy_eos,
    expected_pressure_eos,
):
    """Helper function to assert EOS results."""
    assert np.allclose(
        eos_constants, expected_eos_constants
    ), f"Expected {expected_eos_constants}, got {eos_constants}"
    assert np.allclose(
        eos_parameters, expected_eos_parameters
    ), f"Expected {expected_eos_parameters}, got {eos_parameters}"
    assert np.allclose(
        volume_range, expected_volume_range
    ), f"Expected {expected_volume_range}, got {volume_range}"
    assert np.allclose(
        energy_eos, expected_energy_eos
    ), f"Expected {expected_energy_eos}, got {energy_eos}"
    assert np.allclose(
        pressure_eos, expected_pressure_eos
    ), f"Expected {expected_pressure_eos}, got {pressure_eos}"


def test_mBM4():
    eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = mBM4(
        volumes, energies, num_volumes=10
    )
    eos_constants2, eos_parameters2, volume_range2, energy_eos2, pressure_eos2 = (
        fit_to_eos(volumes, energies, eos_name="mBM4", num_volumes=10)
    )

    V0 = eos_parameters[0]
    E0 = eos_parameters[1]
    B = eos_parameters[2]
    BP = eos_parameters[3]
    a, b, c, d = mBM4_eos_constants(V0, E0, B, BP)

    expected_eos_constants = np.array(
        [
            -102.48259086791732,
            1647.332314906896,
            -9029.447536308013,
            15362.206257426327,
            0.0,
        ]
    )
    expected_eos_parameters = np.array(
        [
            66.10232407335344,
            -14.972805616293215,
            78.04074181821356,
            4.603971403722543,
            -0.08199458053345686,
        ]
    )

    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)

    expected_energy_eos = np.array(
        [
            -14.80858914,
            -14.88584888,
            -14.93687147,
            -14.96489423,
            -14.97275283,
            -14.96293628,
            -14.93763348,
            -14.89877287,
            -14.84805627,
            -14.78698777,
        ]
    )

    expected_pressure_eos = np.array(
        [
            9.43148922,
            6.54701637,
            4.01879926,
            1.8023087,
            -0.14083532,
            -1.84380962,
            -3.3353728,
            -4.64051925,
            -5.78102603,
            -6.775912,
        ]
    )

    # Use the helper function for assertions
    assert_eos_results(
        eos_constants,
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_constants2,
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert np.allclose(
        np.array([a, b, c, d, 0]), expected_eos_constants
    ), f"Expected {expected_eos_constants}, got {np.array([a,b,c,d, 0])}"


def test_mBM5():
    eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = mBM5(
        volumes, energies, num_volumes=10
    )
    eos_constants2, eos_parameters2, volume_range2, energy_eos2, pressure_eos2 = (
        fit_to_eos(volumes, energies, eos_name="mBM5", num_volumes=10)
    )

    V0 = eos_parameters[0]
    E0 = eos_parameters[1]
    B = eos_parameters[2]
    BP = eos_parameters[3]
    B2P = eos_parameters[4]
    a, b, c, d, e = mBM5_eos_constants(V0, E0, B, BP, B2P)

    expected_eos_constants = np.array(
        [
            -853.0124885112899,
            13819.945959859902,
            -83047.67221528212,
            215356.98925097284,
            -202599.15453877283,
        ]
    )
    expected_eos_parameters = np.array(
        [
            66.10542344565768,
            -14.972938667152562,
            78.53582491557574,
            4.549077485105092,
            -0.16682566975204927,
        ]
    )

    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)

    expected_energy_eos = np.array(
        [
            -14.80866964,
            -14.88567265,
            -14.9367749,
            -14.96494777,
            -14.97288825,
            -14.96304204,
            -14.93762528,
            -14.8986448,
            -14.84791694,
            -14.78708477,
        ]
    )

    expected_pressure_eos = np.array(
        [
            9.37451174,
            6.54374803,
            4.03398164,
            1.81572153,
            -0.13808295,
            -1.85217081,
            -3.34911634,
            -4.64948339,
            -5.7719805,
            -6.73361204,
        ]
    )

    # Use the helper function for assertions
    assert_eos_results(
        eos_constants,
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_constants2,
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert np.allclose(
        np.array([a, b, c, d, e]), expected_eos_constants
    ), f"Expected {expected_eos_constants}, got {np.array([a,b,c,d, e])}"


def test_BM4():
    eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = BM4(
        volumes, energies, num_volumes=10
    )
    eos_constants2, eos_parameters2, volume_range2, energy_eos2, pressure_eos2 = (
        fit_to_eos(volumes, energies, eos_name="BM4", num_volumes=10)
    )

    V0 = eos_parameters[0]
    E0 = eos_parameters[1]
    B = eos_parameters[2]
    BP = eos_parameters[3]
    a, b, c, d = BM4_eos_constants(V0, E0, B, BP)

    expected_eos_constants = np.array(
        [
            10.115871836386141,
            -639.1561876497228,
            781.9858370675397,
            48419.8400405475,
            0.0,
        ]
    )
    expected_eos_parameters = np.array(
        [
            66.10191547034127,
            -14.972775074363833,
            77.92792067011315,
            4.612739661291564,
            -0.06258448064264342,
        ]
    )

    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)

    expected_energy_eos = np.array(
        [
            -14.80857541,
            -14.88588549,
            -14.93688813,
            -14.96487866,
            -14.97272201,
            -14.96291517,
            -14.93763964,
            -14.89880545,
            -14.84808841,
            -14.78696169,
        ]
    )

    expected_pressure_eos = np.array(
        [
            9.44321537,
            6.54729341,
            4.01539266,
            1.79956554,
            -0.14110841,
            -1.84166929,
            -3.33221706,
            -4.63868592,
            -5.78348475,
            -6.78602896,
        ]
    )

    # Use the helper function for assertions
    assert_eos_results(
        eos_constants,
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_constants2,
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert np.allclose(
        np.array([a, b, c, d, 0]), expected_eos_constants
    ), f"Expected {expected_eos_constants}, got {np.array([a,b,c,d, 0])}"


def test_BM5():
    eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = BM5(
        volumes, energies, num_volumes=10
    )
    eos_constants2, eos_parameters2, volume_range2, energy_eos2, pressure_eos2 = (
        fit_to_eos(volumes, energies, eos_name="BM5", num_volumes=10)
    )

    V0 = eos_parameters[0]
    E0 = eos_parameters[1]
    B = eos_parameters[2]
    BP = eos_parameters[3]
    B2P = eos_parameters[4]
    a, b, c, d, e = BM5_eos_constants(V0, E0, B, BP, B2P)

    expected_eos_constants = np.array(
        [
            -48.32264004043502,
            3204.2596680358665,
            -93928.45938357542,
            1084818.520079709,
            -4249309.924239315,
        ]
    )
    expected_eos_parameters = np.array(
        [
            66.10494560451701,
            -14.97294336567709,
            78.55818945461328,
            4.566872980385188,
            -0.1715488686904988,
        ]
    )

    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)

    expected_energy_eos = np.array(
        [
            -14.8086726,
            -14.88566133,
            -14.9367749,
            -14.96495465,
            -14.97289253,
            -14.9630399,
            -14.93762014,
            -14.89864324,
            -14.84792136,
            -14.78708454,
        ]
    )

    expected_pressure_eos = np.array(
        [
            9.3696372,
            6.5444211,
            4.03516087,
            1.81589736,
            -0.13868335,
            -1.85275913,
            -3.34908499,
            -4.64884651,
            -5.77160844,
            -6.73532668,
        ]
    )

    # Use the helper function for assertions
    assert_eos_results(
        eos_constants,
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_constants2,
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert np.allclose(
        np.array([a, b, c, d, e]), expected_eos_constants
    ), f"Expected {expected_eos_constants}, got {np.array([a,b,c,d, e])}"


def test_LOG4():
    eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = LOG4(
        volumes, energies, num_volumes=10
    )
    eos_constants2, eos_parameters2, volume_range2, energy_eos2, pressure_eos2 = (
        fit_to_eos(volumes, energies, eos_name="LOG4", num_volumes=10)
    )

    V0 = eos_parameters[0]
    E0 = eos_parameters[1]
    B = eos_parameters[2]
    BP = eos_parameters[3]
    a, b, c, d = LOG4_eos_constants(V0, E0, B, BP)

    expected_eos_constants = np.array(
        [
            1287.0672157843683,
            -864.2557274641168,
            190.04846366479103,
            -13.82976606339218,
            0.0,
        ]
    )
    expected_eos_parameters = np.array(
        [
            66.10417220805562,
            -14.972881468407536,
            78.32086632030024,
            4.5678527771002395,
            -0.12974474286845175,
        ]
    )

    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)

    expected_energy_eos = np.array(
        [
            -14.80863335,
            -14.8857512,
            -14.93681727,
            -14.96492436,
            -14.97283011,
            -14.96299734,
            -14.93762891,
            -14.89869789,
            -14.8479738,
            -14.78704547,
        ]
    )

    expected_pressure_eos = np.array(
        [
            9.40029153,
            6.54508695,
            4.02723604,
            1.80992917,
            -0.1391734,
            -1.84855594,
            -3.34336671,
            -4.64585903,
            -5.7757668,
            -6.75062551,
        ]
    )

    # Use the helper function for assertions
    assert_eos_results(
        eos_constants,
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_constants2,
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert np.allclose(
        np.array([a, b, c, d, 0]), expected_eos_constants
    ), f"Expected {expected_eos_constants}, got {np.array([a,b,c,d, 0])}"


def test_LOG5():
    eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = LOG5(
        volumes, energies, num_volumes=10
    )
    eos_constants2, eos_parameters2, volume_range2, energy_eos2, pressure_eos2 = (
        fit_to_eos(volumes, energies, eos_name="LOG5", num_volumes=10)
    )

    V0 = eos_parameters[0]
    E0 = eos_parameters[1]
    B = eos_parameters[2]
    BP = eos_parameters[3]
    B2P = eos_parameters[4]
    a, b, c, d, e = LOG5_eos_constants(V0, E0, B, BP, B2P)

    expected_eos_constants = np.array(
        [
            97.21090483634063,
            269.45919413164813,
            -214.99095649480188,
            50.47764844893589,
            -3.8283263266927334,
        ]
    )
    expected_eos_parameters = np.array(
        [
            66.10571794361299,
            -14.972935374888266,
            78.51990653243391,
            4.538051436151986,
            -0.16321711742191786,
        ]
    )

    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)

    expected_energy_eos = np.array(
        [
            -14.80866759,
            -14.88568006,
            -14.93677519,
            -14.96494334,
            -14.97288522,
            -14.96304321,
            -14.93762865,
            -14.89864603,
            -14.84791413,
            -14.78708483,
        ]
    )

    expected_pressure_eos = np.array(
        [
            9.37760388,
            6.54335988,
            4.03321021,
            1.81557029,
            -0.13770931,
            -1.8517691,
            -3.34910711,
            -4.64989492,
            -5.77225095,
            -6.73247751,
        ]
    )

    # Use the helper function for assertions
    assert_eos_results(
        eos_constants,
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_constants2,
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert np.allclose(
        np.array([a, b, c, d, e]), expected_eos_constants
    ), f"Expected {expected_eos_constants}, got {np.array([a,b,c,d, e])}"


def test_murnaghan():
    eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = murnaghan(
        volumes, energies, num_volumes=10
    )
    eos_constants2, eos_parameters2, volume_range2, energy_eos2, pressure_eos2 = (
        fit_to_eos(volumes, energies, eos_name="murnaghan", num_volumes=10)
    )

    expected_eos_constants = np.array(
        [
            0,
            0,
            0,
            0,
            0,
        ]
    )
    expected_eos_parameters = np.array(
        [
            66.10051541316386,
            -14.972678601682624,
            77.57033054075912,
            4.638920125800259,
            0.0,
        ]
    )

    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)

    expected_energy_eos = np.array(
        [
            -14.80852844,
            -14.88600543,
            -14.93694421,
            -14.96483083,
            -14.97262454,
            -14.96284734,
            -14.93765668,
            -14.89890482,
            -14.84818745,
            -14.78688374,
        ]
    )

    expected_pressure_eos = np.array(
        [
            9.48202844,
            6.54833124,
            4.00441242,
            1.79071425,
            -0.14208703,
            -1.83502591,
            -3.3223889,
            -4.63294293,
            -5.7909193,
            -6.816806,
        ]
    )

    # Use the helper function for assertions
    assert_eos_results(
        eos_constants,
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_constants2,
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )


def test_vinet():
    eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = vinet(
        volumes, energies, num_volumes=10
    )
    eos_constants2, eos_parameters2, volume_range2, energy_eos2, pressure_eos2 = (
        fit_to_eos(volumes, energies, eos_name="vinet", num_volumes=10)
    )

    expected_eos_constants = np.array(
        [
            0,
            0,
            0,
            0,
            0,
        ]
    )
    expected_eos_parameters = np.array(
        [
            66.102714503444,
            -14.972818017637815,
            78.08775964234005,
            4.598758024396886,
            -0.09039502616658532,
        ]
    )

    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)

    expected_energy_eos = np.array(
        [
            -14.80859905,
            -14.88582997,
            -14.93685998,
            -14.96489803,
            -14.97276554,
            -14.9629472,
            -14.93763448,
            -14.89876297,
            -14.8480444,
            -14.78699431,
        ]
    )

    expected_pressure_eos = np.array(
        [
            9.42521358,
            6.54654636,
            4.02031868,
            1.8037025,
            -0.14046339,
            -1.84450265,
            -3.3365967,
            -4.64136987,
            -5.78038347,
            -6.77255427,
        ]
    )

    # Use the helper function for assertions
    assert_eos_results(
        eos_constants,
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_constants2,
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )


def test_morse():
    eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = morse(
        volumes, energies, num_volumes=10
    )
    eos_constants2, eos_parameters2, volume_range2, energy_eos2, pressure_eos2 = (
        fit_to_eos(volumes, energies, eos_name="morse", num_volumes=10)
    )

    expected_eos_constants = np.array(
        [
            -3.7973259801377353,
            -818.9810070970155,
            15004.49171866411,
            -0.8906492259402166,
            0.0,
        ]
    )
    expected_eos_parameters = np.array(
        [
            66.10258535018146,
            -14.972811004254211,
            78.0617287696514,
            4.601191170503221,
            -0.08589765672017415,
        ]
    )

    volume_min = min(volumes)
    volume_max = max(volumes)
    expected_volume_range = np.linspace(volume_min, volume_max, 10)

    expected_energy_eos = np.array(
        [
            -14.80859533,
            -14.88583885,
            -14.93686446,
            -14.96489484,
            -14.97275843,
            -14.96294198,
            -14.93763539,
            -14.89877005,
            -14.84805172,
            -14.7869888,
        ]
    )

    expected_pressure_eos = np.array(
        [
            9.4280737,
            6.5466657,
            4.01952541,
            1.80303261,
            -0.14056754,
            -1.84403695,
            -3.33587398,
            -4.64092543,
            -5.78090175,
            -6.77481233,
        ]
    )

    # Use the helper function for assertions
    assert_eos_results(
        eos_constants,
        eos_parameters,
        volume_range,
        energy_eos,
        pressure_eos,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )
    assert_eos_results(
        eos_constants2,
        eos_parameters2,
        volume_range2,
        energy_eos2,
        pressure_eos2,
        expected_eos_constants,
        expected_eos_parameters,
        expected_volume_range,
        expected_energy_eos,
        expected_pressure_eos,
    )


if __name__ == "__main__":
    pytest.main()
