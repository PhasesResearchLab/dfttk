"""
Tests for the Configuration class.
"""

# Standard library imports
import os
import bson
import pickle

# Third-party library imports
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Kpoints

# DFTTK imports
from dfttk.configuration import Configuration

# TODO: write tests for a magnetic example. Include smoke test for plot_multiple_ev.
# TODO: think of other tests for Configuration class.

current_dir = os.path.dirname(os.path.abspath(__file__))
conv_test_path = os.path.join(current_dir, "vasp_data/Al/conv_test")
vasp_cmd = ["mpirun", "/opt/packages/VASP/VASP6/6.4.3/ONEAPI/vasp_std"]
config_Al_conv = Configuration(conv_test_path, "config_Al", vasp_cmd)

config_Al_path = os.path.join(current_dir, "vasp_data/Al/config_Al")
config_Al = Configuration(config_Al_path, "config_Al", vasp_cmd)
config_Al.process_ev_curve()

number_of_atoms = 4
temperature_range = np.arange(0, 1010, 100)

config_Al.process_phonons(number_of_atoms, temperature_range)
config_Al.process_debye(scaling_factor=0.617, gruneisen_x=2 / 3, temperatures=temperature_range)
config_Al.process_thermal_electronic(temperature_range, order=1)

volume_range = np.linspace(0.98 * 60, 1.02 * 74, 1000)
config_Al.process_qha("debye", volume_range, P=0)
config_Al.process_qha("debye_thermal_electronic", volume_range, P=0)
config_Al.process_qha("phonons", volume_range, P=0)
config_Al.process_qha("phonons_thermal_electronic", volume_range, P=0)

# Load the expected results
with open(os.path.join(current_dir, "test_configuration_data/config_Al.pkl"), "rb") as f:
    expected_config_Al = pickle.load(f)
with open(os.path.join(current_dir, "test_configuration_data/config_Al_document.bson"), "rb") as f:
    expected_document = bson.BSON.decode(f.read())

def test_analyze_encut_conv():
    encut_conv_df, fig = config_Al_conv.analyze_encut_conv(plot=False)

    encut = encut_conv_df["encut"].values
    kpoints_grid = encut_conv_df["kpoint_grid"].values[0]
    kppa = encut_conv_df["kppa"].values[0]
    energy = encut_conv_df["energy"].values
    number_of_atoms = encut_conv_df["number_of_atoms"].values[0]
    energy_per_atom = encut_conv_df["energy_per_atom"].values
    difference_mev_per_atom = encut_conv_df["difference_mev_per_atom"].values

    expected_encut = np.array([270, 320, 370, 420, 470, 520, 570, 620, 670, 720, 770, 820])
    expected_kpoints_grid = [9, 9, 9]
    expected_kppa = 2916
    expected_energy = np.array([-14.952094, -14.966816, -14.973924, -14.975857, -14.976588, -14.97668, -14.976812, -14.977072, -14.977261, -14.977352, -14.977392, -14.977423])
    expected_number_of_atoms = 4
    expected_energy_per_atom = np.array([-3.7380235, -3.741704, -3.743481, -3.74396425, -3.744147, -3.74417, -3.744203, -3.744268, -3.74431525, -3.744338, -3.744348, -3.74435575])
    expected_difference_mev_per_atom = np.array([np.nan, -3.6805, -1.777, -0.48325, -0.18275, -0.023, -0.033, -0.065, -0.04725, -0.02275, -0.01, -0.00775])

    assert np.array_equal(encut, expected_encut), f"Expected {expected_encut}, but got {encut}"
    assert np.array_equal(kpoints_grid, expected_kpoints_grid), f"Expected {expected_kpoints_grid}, but got {kpoints_grid}"
    assert kppa == expected_kppa, f"Expected {expected_kppa}, but got {kppa}"
    assert np.array_equal(energy, expected_energy), f"Expected {expected_energy}, but got {energy}"
    assert number_of_atoms == expected_number_of_atoms, f"Expected {expected_number_of_atoms}, but got {number_of_atoms}"
    assert np.array_equal(energy_per_atom, expected_energy_per_atom), f"Expected {expected_energy_per_atom}, but got {energy_per_atom}"
    assert np.allclose(difference_mev_per_atom, expected_difference_mev_per_atom, equal_nan=True), f"Expected {expected_difference_mev_per_atom}, but got {difference_mev_per_atom}"


def test_analyze_kpoints_conv():
    kpoints_conv_df, fig = config_Al_conv.analyze_kpoints_conv(plot=False)

    encut = kpoints_conv_df["encut"].values[0]
    kpoints_grid = np.array([list(x) for x in kpoints_conv_df["kpoint_grid"].values])
    kppa = kpoints_conv_df["kppa"].values
    energy = kpoints_conv_df["energy"].values
    number_of_atoms = kpoints_conv_df["number_of_atoms"].values[0]
    energy_per_atom = kpoints_conv_df["energy_per_atom"].values
    difference_mev_per_atom = kpoints_conv_df["difference_mev_per_atom"].values

    expected_encut = 520
    expected_kpoints_grid = np.array([[6, 6, 6], [7, 7, 7], [9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12], [13, 13, 13]])
    expected_kppa = np.array([864, 1372, 2916, 4000, 5324, 6912, 8788])
    expected_energy = np.array([-15.058461, -14.990809, -14.97668, -14.973059, -14.98156, -14.980429, -14.987405])
    expected_number_of_atoms = 4
    expected_energy_per_atom = np.array([-3.76461525, -3.74770225, -3.74417, -3.74326475, -3.74539, -3.74510725, -3.74685125])
    expected_difference_mev_per_atom = np.array([np.nan, 16.913, 3.53225, 0.90525, -2.12525, 0.28275, -1.744])

    assert encut == expected_encut, f"Expected {expected_encut}, but got {encut}"
    assert np.array_equal(kpoints_grid, expected_kpoints_grid), f"Expected {expected_kpoints_grid}, but got {kpoints_grid}"
    assert np.array_equal(kppa, expected_kppa), f"Expected {expected_kppa}, but got {kppa}"
    assert np.array_equal(energy, expected_energy), f"Expected {expected_energy}, but got {energy}"
    assert number_of_atoms == expected_number_of_atoms, f"Expected {expected_number_of_atoms}, but got {number_of_atoms}"
    assert np.array_equal(energy_per_atom, expected_energy_per_atom), f"Expected {expected_energy_per_atom}, but got {energy_per_atom}"
    assert np.allclose(difference_mev_per_atom, expected_difference_mev_per_atom, equal_nan=True), f"Expected {expected_difference_mev_per_atom}, but got {difference_mev_per_atom}"


def test_process_ev_curve():
    ev_curve = config_Al.ev_curve
    expected_ev_curve = expected_config_Al.ev_curve

    # Test the values of the ev_curve attributes
    assert ev_curve.atomic_masses == expected_ev_curve.atomic_masses, f"Expected {expected_ev_curve.atomic_masses}, but got {ev_curve.atomic_masses}"
    assert ev_curve.average_mass == expected_ev_curve.average_mass, f"Expected {expected_ev_curve.average_mass}, but got {ev_curve.average_mass}"
    assert np.allclose(ev_curve.energies, expected_ev_curve.energies, equal_nan=True), f"Expected {expected_ev_curve.energies}, but got {ev_curve.energies}"
    assert ev_curve.eos_parameters["eos_name"] == expected_ev_curve.eos_parameters["eos_name"], f"Expected {expected_ev_curve.eos_parameters['eos_name']}, but got {ev_curve.eos_parameters['eos_name']}"
    for key in expected_ev_curve.eos_parameters:
        if key != "eos_name":
            assert np.isclose(ev_curve.eos_parameters[key], expected_ev_curve.eos_parameters[key], rtol=5e-4), f"Expected {expected_ev_curve.eos_parameters[key]}, but got {ev_curve.eos_parameters[key]}"
    assert ev_curve.incars == expected_ev_curve.incars, f"Expected {expected_ev_curve.incars}, but got {ev_curve.incars}"
    assert ev_curve.initial_poscar == expected_ev_curve.initial_poscar, f"Expected {expected_ev_curve.initial_poscar}, but got {ev_curve.initial_poscar}"
    assert ev_curve.kpoints == expected_ev_curve.kpoints, f"Expected {expected_ev_curve.kpoints}, but got {ev_curve.kpoints}"
    assert ev_curve.mag_data == expected_ev_curve.mag_data, f"Expected {expected_ev_curve.mag_data}, but got {ev_curve.mag_data}"
    assert ev_curve.magnetic_ordering == expected_ev_curve.magnetic_ordering, f"Expected {expected_ev_curve.magnetic_ordering}, but got {ev_curve.magnetic_ordering}"
    assert ev_curve.number_of_atoms == expected_ev_curve.number_of_atoms, f"Expected {expected_ev_curve.number_of_atoms}, but got {ev_curve.number_of_atoms}"
    assert ev_curve.relaxed_structures == expected_ev_curve.relaxed_structures, f"Expected {expected_ev_curve.relaxed_structures}, but got {ev_curve.relaxed_structures}"
    assert ev_curve.total_magnetic_moment == expected_ev_curve.total_magnetic_moment, f"Expected {expected_ev_curve.total_magnetic_moment}, but got {ev_curve.total_magnetic_moment}"
    assert np.allclose(ev_curve.volumes, expected_ev_curve.volumes, rtol=1e-4), f"Expected {expected_ev_curve.volumes}, but got {ev_curve.volumes}"


def test_process_phonons():
    phonons = config_Al.phonons
    expected_phonons = expected_config_Al.phonons

    # Test the values of the phonons attributes
    assert np.allclose(phonons.entropies, expected_phonons.entropies, rtol=1e-4), f"Expected {expected_phonons.entropies}, but got {phonons.entropies}"
    assert np.allclose(phonons.entropies_fit, expected_phonons.entropies_fit, rtol=1e-4), f"Expected {expected_phonons.entropies_fit}, but got {phonons.entropies_fit}"
    assert np.allclose(phonons.entropies_poly_coeffs, expected_phonons.entropies_poly_coeffs, rtol=1e-4), f"Expected {expected_phonons.entropies_poly_coeffs}, but got {phonons.entropies_poly_coeffs}"
    assert np.allclose(phonons.helmholtz_energies, expected_phonons.helmholtz_energies, rtol=1e-4), f"Expected {expected_phonons.helmholtz_energies}, but got {phonons.helmholtz_energies}"
    assert np.allclose(phonons.helmholtz_energies_fit, expected_phonons.helmholtz_energies_fit, rtol=1e-4), f"Expected {expected_phonons.helmholtz_energies_fit}, but got {phonons.helmholtz_energies_fit}"
    assert np.allclose(phonons.helmholtz_energies_poly_coeffs, expected_phonons.helmholtz_energies_poly_coeffs, rtol=1e-4), f"Expected {expected_phonons.helmholtz_energies_poly_coeffs}, but got {phonons.helmholtz_energies_poly_coeffs}"
    assert np.allclose(phonons.heat_capacities, expected_phonons.heat_capacities, rtol=1e-4), f"Expected {expected_phonons.heat_capacities}, but got {phonons.heat_capacities}"
    assert np.allclose(phonons.heat_capacities_fit, expected_phonons.heat_capacities_fit, rtol=1e-4), f"Expected {expected_phonons.heat_capacities_fit}, but got {phonons.heat_capacities_fit}"
    assert np.allclose(phonons.heat_capacities_poly_coeffs, expected_phonons.heat_capacities_poly_coeffs, rtol=1e-4), f"Expected {expected_phonons.heat_capacities_poly_coeffs}, but got {phonons.heat_capacities_poly_coeffs}"
    assert phonons.incars == expected_phonons.incars, f"Expected {expected_phonons.incars}, but got {phonons.incars}"
    assert np.allclose(phonons.internal_energies, expected_phonons.internal_energies, rtol=1e-4), f"Expected {expected_phonons.internal_energies}, but got {phonons.internal_energies}"
    assert phonons.kpoints == expected_phonons.kpoints, f"Expected {expected_phonons.kpoints}, but got {phonons.kpoints}"
    assert phonons.number_of_atoms == expected_phonons.number_of_atoms, f"Expected {expected_phonons.number_of_atoms}, but got {phonons.number_of_atoms}"
    assert phonons.phonon_structures == expected_phonons.phonon_structures, f"Expected {expected_phonons.phonon_structures}, but got {phonons.phonon_structures}"
    assert np.allclose(phonons.temperatures, expected_phonons.temperatures, rtol=1e-4), f"Expected {expected_phonons.temperatures}, but got {phonons.temperatures}"
    assert np.allclose(phonons.volumes, expected_phonons.volumes, rtol=1e-4), f"Expected {expected_phonons.volumes}, but got {phonons.volumes}"
    assert np.allclose(phonons.volumes_fit, expected_phonons.volumes_fit, rtol=1e-4), f"Expected {expected_phonons.volumes_fit}, but got {phonons.volumes_fit}"


def test_process_debye():
    debye = config_Al.debye
    expected_debye = expected_config_Al.debye

    # Test the values of the debye attributes
    assert np.isclose(debye.B, expected_debye.B, rtol=1e-4), f"Expected {expected_debye.B}, but got {debye.B}"
    assert np.isclose(debye.BP, expected_debye.BP, rtol=1e-4), f"Expected {expected_debye.BP}, but got {debye.BP}"
    assert np.isclose(debye.V0, expected_debye.V0, rtol=1e-4), f"Expected {expected_debye.V0}, but got {debye.V0}"
    assert np.isclose(debye.atomic_mass, expected_debye.atomic_mass, rtol=1e-4), f"Expected {expected_debye.atomic_mass}, but got {debye.atomic_mass}"
    assert np.isclose(debye.gruneisen_x, expected_debye.gruneisen_x, rtol=1e-4), f"Expected {expected_debye.gruneisen_x}, but got {debye.gruneisen_x}"
    assert np.allclose(debye.entropies, expected_debye.entropies, rtol=1e-4), f"Expected {expected_debye.entropies}, but got {debye.entropies}"
    assert np.allclose(debye.heat_capacities, expected_debye.heat_capacities, rtol=1e-4), f"Expected {expected_debye.heat_capacities}, but got {debye.heat_capacities}"
    assert np.allclose(debye.helmholtz_energies, expected_debye.helmholtz_energies, rtol=2e-3, atol=1e-6), f"Expected {expected_debye.helmholtz_energies}, but got {debye.helmholtz_energies}"
    assert debye.number_of_atoms == expected_debye.number_of_atoms, f"Expected {expected_debye.number_of_atoms}, but got {debye.number_of_atoms}"
    assert np.isclose(debye.scaling_factor, expected_debye.scaling_factor, rtol=1e-4), f"Expected {expected_debye.scaling_factor}, but got {debye.scaling_factor}"
    assert np.allclose(debye.temperatures, expected_debye.temperatures, rtol=1e-4), f"Expected {expected_debye.temperatures}, but got {debye.temperatures}"
    assert np.allclose(debye.volumes, expected_debye.volumes, rtol=1e-4), f"Expected {expected_debye.volumes}, but got {debye.volumes}"


def test_process_thermal_electronic():
    thermal_electronic = config_Al.thermal_electronic
    expected_thermal_electronic = expected_config_Al.thermal_electronic

    # Test the values of the thermal_electronic attributes
    # Current s_el_poly, entropy, entropy_fit, heat_capacity, heat_capacity_fit, helmholtz_energy, helmholtz_energy_fit, and internal_energy are not tested
    assert thermal_electronic.number_of_atoms == expected_thermal_electronic.number_of_atoms, f"Expected {expected_thermal_electronic.number_of_atoms}, but got {thermal_electronic.number_of_atoms}"
    assert np.allclose(thermal_electronic.s_el, expected_thermal_electronic.s_el, rtol=1e-4), f"Expected {expected_thermal_electronic.s_el}, but got {thermal_electronic.s_el}"
    assert np.allclose(np.vstack(thermal_electronic.s_el_fit), np.vstack(expected_thermal_electronic.s_el_fit), rtol=1e-4), f"Expected {expected_thermal_electronic.s_el_fit}, but got {thermal_electronic.s_el_fit}"
    assert thermal_electronic.structures == expected_thermal_electronic.structures, f"Expected {expected_thermal_electronic.structures}, but got {thermal_electronic.structures}"
    assert np.allclose(thermal_electronic.temperatures, expected_thermal_electronic.temperatures, rtol=1e-4), f"Expected {expected_thermal_electronic.temperatures}, but got {thermal_electronic.temperatures}"
    assert np.allclose(thermal_electronic.volume_fit, expected_thermal_electronic.volume_fit, rtol=1e-4), f"Expected {expected_thermal_electronic.volume_fit}, but got {thermal_electronic.volume_fit}"
    assert np.allclose(thermal_electronic.cv_el, expected_thermal_electronic.cv_el, rtol=1e-4), f"Expected {expected_thermal_electronic.cv_el}, but got {thermal_electronic.cv_el}"
    assert np.allclose(np.vstack(thermal_electronic.cv_el_fit), np.vstack(expected_thermal_electronic.cv_el_fit), rtol=1e-4), f"Expected {expected_thermal_electronic.cv_el_fit}, but got {thermal_electronic.cv_el_fit}"
    assert np.allclose(thermal_electronic.e_el, expected_thermal_electronic.e_el, rtol=1e-4), f"Expected {expected_thermal_electronic.e_el}, but got {thermal_electronic.e_el}"
    pd.testing.assert_frame_equal(thermal_electronic.electron_dos_data, expected_thermal_electronic.electron_dos_data)
    assert np.allclose(thermal_electronic.f_el, expected_thermal_electronic.f_el, rtol=1e-4), f"Expected {expected_thermal_electronic.f_el}, but got {thermal_electronic.f_el}"
    assert np.allclose(thermal_electronic.f_el_fit, expected_thermal_electronic.f_el_fit, rtol=1e-4), f"Expected {expected_thermal_electronic.f_el_fit}, but got {thermal_electronic.f_el_fit}"
    assert thermal_electronic.incars == expected_thermal_electronic.incars, f"Expected {expected_thermal_electronic.incars}, but got {thermal_electronic.incars}"
    assert thermal_electronic.kpoints == expected_thermal_electronic.kpoints, f"Expected {expected_thermal_electronic.kpoints}, but got {thermal_electronic.kpoints}"


def test_process_qha():
    qha = config_Al.qha
    expected_qha = expected_config_Al.qha

    # Test the values of the qha attributes
    assert qha.number_of_atoms == expected_qha.number_of_atoms, f"Expected {expected_qha.number_of_atoms}, but got {qha.number_of_atoms}"
    assert np.allclose(qha.temperatures, expected_qha.temperatures, rtol=1e-4), f"Expected {expected_qha.temperatures}, but got {qha.temperatures}"
    assert np.allclose(qha.volumes, expected_qha.volumes, rtol=1e-4), f"Expected {expected_qha.volumes}, but got {qha.volumes}"

    methods = ["debye", "debye_thermal_electronic", "phonons", "phonons_thermal_electronic"]
    for method in methods:
        assert np.allclose(qha.methods[method]["helmholtz_energy"]["values"], expected_qha.methods[method]["helmholtz_energy"]["values"], rtol=1e-4), f"Expected {expected_qha.methods[method]['helmholtz_energy']['values']}, but got {qha.methods[method]['helmholtz_energy']['values']}"
        assert np.allclose(qha.methods[method]["entropy"]["values"], expected_qha.methods[method]["entropy"]["values"], rtol=1e-4), f"Expected {expected_qha.methods[method]['entropy']['values']}, but got {qha.methods[method]['entropy']['values']}"
        assert np.allclose(qha.methods[method]["heat_capacity"]["values"], expected_qha.methods[method]["heat_capacity"]["values"], rtol=1e-4), f"Expected {expected_qha.methods[method]['heat_capacity']['values']}, but got {qha.methods[method]['heat_capacity']['values']}"
        assert np.allclose(qha.methods[method]["0.00_GPa"]["helmholtz_energy_pv"]["values"], expected_qha.methods[method]["0.00_GPa"]["helmholtz_energy_pv"]["values"], rtol=1e-4), f"Expected {expected_qha.methods[method]['0.00_GPa']['helmholtz_energy_pv']['values']}, but got {qha.methods[method]['0.00_GPa']['helmholtz_energy_pv']['values']}"
        assert np.allclose(qha.methods[method]["0.00_GPa"]["V0"], expected_qha.methods[method]["0.00_GPa"]["V0"], rtol=1e-4), f"Expected {expected_qha.methods[method]['0.00_GPa']['V0']}, but got {qha.methods[method]['0.00_GPa']['V0']}"
        assert np.allclose(qha.methods[method]["0.00_GPa"]["G0"], expected_qha.methods[method]["0.00_GPa"]["G0"], rtol=1e-4), f"Expected {expected_qha.methods[method]['0.00_GPa']['G0']}, but got {qha.methods[method]['0.00_GPa']['G0']}"
        assert np.allclose(qha.methods[method]["0.00_GPa"]["S0"], expected_qha.methods[method]["0.00_GPa"]["S0"], rtol=1e-4), f"Expected {expected_qha.methods[method]['0.00_GPa']['S0']}, but got {qha.methods[method]['0.00_GPa']['S0']}"
        assert np.allclose(qha.methods[method]["0.00_GPa"]["H0"], expected_qha.methods[method]["0.00_GPa"]["H0"], rtol=1e-4), f"Expected {expected_qha.methods[method]['0.00_GPa']['H0']}, but got {qha.methods[method]['0.00_GPa']['H0']}"
        assert np.allclose(qha.methods[method]["0.00_GPa"]["B"], expected_qha.methods[method]["0.00_GPa"]["B"], rtol=1e-4), f"Expected {expected_qha.methods[method]['0.00_GPa']['B']}, but got {qha.methods[method]['0.00_GPa']['B']}"
        assert np.allclose(qha.methods[method]["0.00_GPa"]["BP"], expected_qha.methods[method]["0.00_GPa"]["BP"], rtol=1e-4), f"Expected {expected_qha.methods[method]['0.00_GPa']['BP']}, but got {qha.methods[method]['0.00_GPa']['BP']}"
        assert np.allclose(qha.methods[method]["0.00_GPa"]["CTE"], expected_qha.methods[method]["0.00_GPa"]["CTE"], rtol=1e-4), f"Expected {expected_qha.methods[method]['0.00_GPa']['CTE']}, but got {qha.methods[method]['0.00_GPa']['CTE']}"
        assert np.allclose(qha.methods[method]["0.00_GPa"]["LCTE"], expected_qha.methods[method]["0.00_GPa"]["LCTE"], rtol=1e-4), f"Expected {expected_qha.methods[method]['0.00_GPa']['LCTE']}, but got {qha.methods[method]['0.00_GPa']['LCTE']}"
        assert np.allclose(qha.methods[method]["0.00_GPa"]["Cp"], expected_qha.methods[method]["0.00_GPa"]["Cp"], rtol=1e-4), f"Expected {expected_qha.methods[method]['0.00_GPa']['Cp']}, but got {qha.methods[method]['0.00_GPa']['Cp']}"

def test_to_mongodb():
    connection_string = "mongodb+srv://admin:og7MRdgE2wY2KWiw@dfttk.3cdhgac.mongodb.net/?retryWrites=true&w=majority&appName=DFTTK"
    db_name = "DFTTK"
    collection_name = "community"
    document = config_Al.to_mongodb(connection_string, db_name, collection_name, insert=False)

    # Compare metadata
    assert document['metadata']['vaspVersion'] == expected_document['metadata']['vaspVersion']
    assert document['metadata']['parentDatabase'] == expected_document['metadata']['parentDatabase']
    assert document['metadata']['parentDatabaseId'] == expected_document['metadata']['parentDatabaseId']
    assert document['metadata']['parentDatabaseURL'] == expected_document['metadata']['parentDatabaseURL']
    assert document['metadata']['affiliation'] == expected_document['metadata']['affiliation']
    assert document['metadata']['comment'] == expected_document['metadata']['comment']
    assert type(document['metadata']['created']) == type(expected_document['metadata']['created'])
    assert type(document['metadata']['lastModified']) == type(expected_document['metadata']['lastModified'])
    
    # Compare configuration
    assert document['configuration']['name'] == expected_document['configuration']['name']
    assert document['configuration']['alias'] == expected_document['configuration']['alias']
    assert document['configuration']['multiplicity'] == expected_document['configuration']['multiplicity']
    assert document['configuration']['reducedFormula'] == expected_document['configuration']['reducedFormula']
    assert document['configuration']['nComponents'] == expected_document['configuration']['nComponents']
    assert document['configuration']['numberOfAtoms'] == expected_document['configuration']['numberOfAtoms']
    
    # Compare evCurve
    ## input
    initial_poscar = document['evCurve']['input']['initialPoscar']
    initial_structure = Structure.from_dict(initial_poscar)
    expected_initial_poscar = expected_document['evCurve']['input']['initialPoscar']
    expected_initial_structure = Structure.from_dict(expected_initial_poscar)
    assert initial_structure == expected_initial_structure, f"Expected {expected_initial_structure}, but got {initial_structure}"
    
    assert document['evCurve']['input']['incars'] == expected_document['evCurve']['input']['incars'], f"Expected {expected_document['evCurve']['input']['incars']}, but got {document['evCurve']['input']['incars']}"
    
    kpoints_objects = document['evCurve']['input']['kpoints']
    expected_kpoints_objects = expected_document['evCurve']['input']['kpoints']
    for kpoints, expected_kpoints in zip(kpoints_objects, expected_kpoints_objects):
        kpoints_1relax = Kpoints.from_dict(kpoints['1relax'])
        kpoints_2relax = Kpoints.from_dict(kpoints['2relax'])
        kpoints_3static = Kpoints.from_dict(kpoints['3static'])
        expected_kpoints_1relax = Kpoints.from_dict(expected_kpoints['1relax'])
        expected_kpoints_2relax = Kpoints.from_dict(expected_kpoints['2relax'])
        expected_kpoints_3static = Kpoints.from_dict(expected_kpoints['3static'])
        assert kpoints_1relax == expected_kpoints_1relax, f"Expected {expected_kpoints_1relax}, but got {kpoints_1relax}"
        assert kpoints_2relax == expected_kpoints_2relax, f"Expected {expected_kpoints_2relax}, but got {kpoints_2relax}"
        assert kpoints_3static == expected_kpoints_3static, f"Expected {expected_kpoints_3static}, but got {kpoints_3static}"
    
    assert document['evCurve']['input']['potcar'] == expected_document['evCurve']['input']['potcar'], f"Expected {expected_document['evCurve']['input']['potcar']}, but got {document['evCurve']['input']['potcar']}"
    
    ## output
    assert document['evCurve']['output']['scaleAtoms'] == expected_document['evCurve']['output']['scaleAtoms']
    assert document['evCurve']['output']['volumes'] == expected_document['evCurve']['output']['volumes']
    assert document['evCurve']['output']['energies'] == expected_document['evCurve']['output']['energies']

    relaxed_structures = document['evCurve']['output']['relaxedStructures']
    expected_relaxed_structures = expected_document['evCurve']['output']['relaxedStructures']
    for relaxed_structure, expected_relaxed_structure in zip(relaxed_structures, expected_relaxed_structures):
        structure = Structure.from_dict(relaxed_structure)
        expected_structure = Structure.from_dict(expected_relaxed_structure)
        assert structure == expected_structure, f"Expected {expected_structure}, but got {structure}"

    assert document['evCurve']['output']['totalMagneticMoments'] == expected_document['evCurve']['output']['totalMagneticMoments'], f"Expected {expected_document['evCurve']['output']['totalMagneticMoments']}, but got {document['evCurve']['output']['totalMagneticMoments']}"
    assert document['evCurve']['output']['magneticOrderings'] == expected_document['evCurve']['output']['magneticOrderings'], f"Expected {expected_document['evCurve']['output']['magneticOrderings']}, but got {document['evCurve']['output']['magneticOrderings']}"
    assert document['evCurve']['output']['magData'] == expected_document['evCurve']['output']['magData'], f"Expected {expected_document['evCurve']['output']['magData']}, but got {document['evCurve']['output']['magData']}"
    assert document['evCurve']['output']['eosParameters']['eosName'] == expected_document['evCurve']['output']['eosParameters']['eosName'], f"Expected {expected_document['evCurve']['output']['eosParameters']['eosName']}, but got {document['evCurve']['output']['eosParameters']['eosName']}"
    # Only compare the eos parameters, not eos constants
    assert np.isclose(document['evCurve']['output']['eosParameters']['V0'], expected_document['evCurve']['output']['eosParameters']['V0']), f"Expected {expected_document['evCurve']['output']['eosParameters']['V0']}, but got {document['evCurve']['output']['eosParameters']['V0']}"
    assert np.isclose(document['evCurve']['output']['eosParameters']['E0'], expected_document['evCurve']['output']['eosParameters']['E0']), f"Expected {expected_document['evCurve']['output']['eosParameters']['E0']}, but got {document['evCurve']['output']['eosParameters']['E0']}"
    assert np.isclose(document['evCurve']['output']['eosParameters']['B'], expected_document['evCurve']['output']['eosParameters']['B']), f"Expected {expected_document['evCurve']['output']['eosParameters']['B']}, but got {document['evCurve']['output']['eosParameters']['B']}"
    assert np.isclose(document['evCurve']['output']['eosParameters']['BP'], expected_document['evCurve']['output']['eosParameters']['BP']), f"Expected {expected_document['evCurve']['output']['eosParameters']['BP']}, but got {document['evCurve']['output']['eosParameters']['BP']}"
    assert np.isclose(document['evCurve']['output']['eosParameters']['B2P'], expected_document['evCurve']['output']['eosParameters']['B2P']), f"Expected {expected_document['evCurve']['output']['eosParameters']['B2P']}, but got {document['evCurve']['output']['eosParameters']['B2P']}"