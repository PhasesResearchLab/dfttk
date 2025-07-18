"""
Tests for the dfttk.debye module.
"""

# Related third party imports
import numpy as np
import pytest

# DFTTK imports
from dfttk.debye.debye_gruneisen import DebyeGruneisen

# Test parameters
number_of_atoms = 4
volumes = np.linspace(0.98 * 60, 1.02 * 74, 10)
temperatures = np.arange(0, 1010, 100)
atomic_mass = 26.981
V0 = 66.10191547034127
B = 77.92792067011315
BP = 4.612739661291564
scaling_factor = 0.617
gruneisen_x = 2 / 3

# Expected results
expected_gruneisen_parameter = 2.1397031639791155
expected_debye_temperatures = np.array([625.79542565, 585.59163116, 549.06704229, 515.79032759, 485.39120448, 457.55039483, 431.99145197, 408.47407193, 386.78858907, 366.75142293])
expected_debye_integral = np.array([0.06996584, 0.26773252, 0.42488624, 0.53228596, 0.60755784, 0.662579, 0.70433899, 0.73703332, 0.76328807, 0.78481708])

expected_helmholtz_energies = np.array(
    [
        [0.24267095, 0.22708072, 0.21291722, 0.20001317, 0.188225, 0.17742889, 0.16751764, 0.15839807, 0.14998888, 0.14221887],
        [0.24006104, 0.22394907, 0.20919411, 0.19562628, 0.18310022, 0.17149113, 0.16069159, 0.1506089, 0.14116288, 0.132284],
        [0.21495901, 0.19552038, 0.17729185, 0.16012371, 0.14389, 0.12848428, 0.11381615, 0.09980853, 0.08639536, 0.07351978],
        [0.15759857, 0.13299223, 0.10958466, 0.08724416, 0.06585962, 0.04533669, 0.02559472, 0.00656437, -0.0118144, -0.0295938],
        [0.07219976, 0.04166913, 0.0124138, -0.0156899, -0.04274739, -0.06884916, -0.09407349, -0.1184885, -0.1421539, -0.16512231],
        [-0.03625184, -0.07305896, -0.10846809, -0.14260142, -0.17556426, -0.20744806, -0.23833276, -0.26828876, -0.29737841, -0.32565726],
        [-0.16393316, -0.20720485, -0.24892994, -0.28923313, -0.32822295, -0.36599456, -0.40263193, -0.43820967, -0.47279442, -0.50644603],
        [-0.30800338, -0.35785085, -0.40598742, -0.45254279, -0.49763077, -0.54135193, -0.58379572, -0.62504212, -0.66516307, -0.70422355],
        [-0.4663115, -0.52280571, -0.57741454, -0.63027434, -0.68150559, -0.73121539, -0.77949955, -0.82644424, -0.8721273, -0.91661939],
        [-0.63718655, -0.70037542, -0.7614975, -0.82069694, -0.87810179, -0.93382649, -0.98797384, -1.04063669, -1.09189923, -1.14183817],
        [-0.81929879, -0.88921617, -0.95688041, -1.02244429, -1.08604412, -1.14780219, -1.20782877, -1.26622374, -1.32307799, -1.37847448],
    ]
)
expected_entropies = np.array(
    [
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
        [9.84493663e-05, 1.16371781e-04, 1.36100975e-04, 1.57574618e-04, 1.80705750e-04, 2.05388273e-04, 2.31502418e-04, 2.58919842e-04, 2.87508100e-04, 3.17134404e-04],
        [4.15416511e-04, 4.60612022e-04, 5.06574141e-04, 5.53070159e-04, 5.99897798e-04, 6.46883369e-04, 6.93879342e-04, 7.40761641e-04, 7.87426911e-04, 8.33789882e-04],
        [7.22940954e-04, 7.79424027e-04, 8.35486090e-04, 8.91002239e-04, 9.45876077e-04, 1.00003471e-03, 1.05342447e-03, 1.10600731e-03, 1.15775780e-03, 1.20866065e-03],
        [9.76604225e-04, 1.03800110e-03, 1.09835515e-03, 1.15762438e-03, 1.21578475e-03, 1.27282617e-03, 1.32874918e-03, 1.38356245e-03, 1.43728072e-03, 1.48992324e-03],
        [1.18610897e-03, 1.24998927e-03, 1.31249036e-03, 1.37362050e-03, 1.43339800e-03, 1.49184846e-03, 1.54900250e-03, 1.60489415e-03, 1.65955960e-03, 1.71303621e-03],
        [1.36283321e-03, 1.42812315e-03, 1.49183659e-03, 1.55401279e-03, 1.61469582e-03, 1.67393263e-03, 1.73177167e-03, 1.78826187e-03, 1.84345186e-03, 1.89738951e-03],
        [1.51502047e-03, 1.58118198e-03, 1.64564275e-03, 1.70846212e-03, 1.76970065e-03, 1.82941893e-03, 1.88767670e-03, 1.94453224e-03, 2.00004201e-03, 2.05426040e-03],
        [1.64837946e-03, 1.71511560e-03, 1.78006814e-03, 1.84331005e-03, 1.90491303e-03, 1.96494681e-03, 2.02347867e-03, 2.08057314e-03, 2.13629188e-03, 2.19069360e-03],
        [1.76692036e-03, 1.83405459e-03, 1.89934740e-03, 1.96288135e-03, 2.02473599e-03, 2.08498746e-03, 2.14370830e-03, 2.20096741e-03, 2.25683006e-03, 2.31135799e-03],
        [1.87353338e-03, 1.94095444e-03, 2.00649221e-03, 2.07023625e-03, 2.13227180e-03, 2.19267966e-03, 2.25153622e-03, 2.30891352e-03, 2.36487944e-03, 2.41949789e-03],
    ]
)
expected_heat_capacities = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.00025215, 0.00028806, 0.00032492, 0.00036222, 0.00039949, 0.0004363, 0.00047231, 0.00050723, 0.00054083, 0.00057295],
        [0.00066319, 0.00069776, 0.00072925, 0.00075781, 0.00078364, 0.00080695, 0.00082795, 0.00084684, 0.00086384, 0.00087912],
        [0.00083989, 0.00086104, 0.00087965, 0.00089602, 0.00091043, 0.00092313, 0.00093433, 0.00094423, 0.00095299, 0.00096075],
        [0.00091781, 0.00093122, 0.00094285, 0.00095296, 0.00096177, 0.00096947, 0.0009762, 0.00098211, 0.0009873, 0.00099189],
        [0.00095741, 0.00096649, 0.00097432, 0.00098108, 0.00098695, 0.00099204, 0.00099649, 0.00100038, 0.00100379, 0.00100679],
        [0.00097995, 0.00098646, 0.00099204, 0.00099686, 0.00100102, 0.00100463, 0.00100777, 0.00101051, 0.00101291, 0.00101502],
        [0.00099391, 0.00099878, 0.00100295, 0.00100654, 0.00100964, 0.00101232, 0.00101465, 0.00101669, 0.00101846, 0.00102003],
        [0.00100312, 0.0010069, 0.00101013, 0.0010129, 0.00101529, 0.00101736, 0.00101916, 0.00102073, 0.00102209, 0.00102329],
        [0.00100951, 0.00101252, 0.00101509, 0.0010173, 0.0010192, 0.00102084, 0.00102227, 0.00102351, 0.00102459, 0.00102554],
        [0.00101411, 0.00101656, 0.00101866, 0.00102045, 0.001022, 0.00102334, 0.0010245, 0.0010255, 0.00102639, 0.00102716],
    ]
)


def test_DebyeGruneisen():
    debye = DebyeGruneisen()
    debye.process(
        number_of_atoms,
        volumes,
        temperatures,
        atomic_mass,
        V0,
        B,
        BP,
        scaling_factor,
        gruneisen_x,
    )

    gruneisen_parameter = debye.calculate_gruneisen_parameter()
    assert np.isclose(gruneisen_parameter, expected_gruneisen_parameter, rtol=1e-5)

    debye_temperatures = debye.calculate_debye_temperatures(gruneisen_parameter)
    assert np.allclose(debye_temperatures, expected_debye_temperatures, rtol=1e-5)

    x_array = debye_temperatures[0] / temperatures[temperatures > 0]
    debye_integral = debye.calculate_debye_integral_n3(x_array)
    assert np.allclose(debye_integral, expected_debye_integral, rtol=1e-5)

    assert np.allclose(debye.helmholtz_energies, expected_helmholtz_energies, rtol=1e-5)
    assert np.allclose(debye.entropies, expected_entropies, rtol=1e-5)
    assert np.allclose(debye.heat_capacities, expected_heat_capacities, rtol=1e-5)


def test_plot_requires_process():
    debye = DebyeGruneisen()
    with pytest.raises(RuntimeError, match="process\\(\\) must be called before plot\\(\\)"):
        debye.plot("helmholtz_energy")


def test_plot_runs_after_process():
    debye = DebyeGruneisen()
    debye.process(
        number_of_atoms,
        volumes,
        temperatures,
        atomic_mass,
        V0,
        B,
        BP,
        scaling_factor,
        gruneisen_x,
    )
    # Should not raise
    fig_t, fig_v = debye.plot("helmholtz_energy")
    assert fig_t is not None
    assert fig_v is not None


def test_plot_invalid_property():
    debye = DebyeGruneisen()
    debye.process(
        number_of_atoms,
        volumes,
        temperatures,
        atomic_mass,
        V0,
        B,
        BP,
        scaling_factor,
        gruneisen_x,
    )
    with pytest.raises(ValueError, match="property must be one of"):
        debye.plot("invalid_property")


def test_plot_selected_volumes():
    debye = DebyeGruneisen()
    debye.process(
        number_of_atoms,
        volumes,
        temperatures,
        atomic_mass,
        V0,
        B,
        BP,
        scaling_factor,
        gruneisen_x,
    )
    # Pick a subset of volumes to plot
    selected_volumes = np.array([74, 72, 70, 68, 66, 64, 62, 60])
    fig_t, fig_v = debye.plot("helmholtz_energy", selected_volumes=selected_volumes)
    # Check that the correct number of traces are present
    assert len(fig_t.data) == len(selected_volumes)
    assert fig_t is not None and fig_v is not None


def test_plot_selected_temperatures():
    debye = DebyeGruneisen()
    debye.process(
        number_of_atoms,
        volumes,
        temperatures,
        atomic_mass,
        V0,
        B,
        BP,
        scaling_factor,
        gruneisen_x,
    )
    # Pick a subset of temperatures to plot
    selected_temperatures = np.arange(0, 1010, 100)
    fig_t, fig_v = debye.plot("helmholtz_energy", selected_temperatures=selected_temperatures)
    # Check that the correct number of traces are present
    assert len(fig_v.data) == len(selected_temperatures)
    assert fig_t is not None and fig_v is not None
