'''
This module contains the PhononData class, which is used to process and analyze phonon data from VASP calculations.
'''
from plotly import graph_objects as go
import os
from natsort import natsorted
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar
from dfttk.workflows import process_phonon_dos_YPHON
from dfttk.phonon_functions import (
    harmonic,
    fit_harmonic,
    scale_phonon_dos,
    plot_phonon_dos,
    plot_harmonic,
    plot_fit_harmonic,
)

class PhononData:
    def __init__(self, path: str):
        self.path = path
        self.incars: list[dict] = []
        self.kpoints: Kpoints = None
        self.potcar: Potcar = None
        self.phonon_structures: list[Structure] = []
        self.number_of_atoms: int = None
        self.temperatures: np.ndarray = None
        self.volumes: np.ndarray = None
        self.helmholtz_energy: np.ndarray = None
        self.internal_energy: np.ndarray = None
        self.entropy: np.ndarray = None
        self.heat_capacity: np.ndarray = None
        self.helmholtz_energy_fit: dict = None
        self.entropy_fit: dict = None
        self.heat_capacity_fit: dict = None
        self.f_vib: np.ndarray = None
        self.s_vib: np.ndarray = None
        self.cv_vib: np.ndarray = None
        self.f_vib_fit: np.ndarray = None
        self.s_vib_fit: np.ndarray = None
        self.cv_vib_fit: np.ndarray = None
        self.volume_fit: np.ndarray = None

    def process_phonon_dos(self):
        process_phonon_dos_YPHON(self.path)

    def _get_phonon_folders(self):
        return natsorted([f for f in os.listdir(self.path) if f.startswith("phonon_")])

    def get_vasp_input(self, volumes: list[float] = None):
        phonon_folders = self._get_phonon_folders()
        incar_keys = ["1relax", "2phonons"]

        if volumes is not None:
            volumes_set = {round(volume, 2) for volume in volumes}
            phonon_folders = [
                phonon_folder
                for phonon_folder in phonon_folders
                if os.path.exists(
                    os.path.join(self.path, phonon_folder, "CONTCAR.2phonons")
                )
                and round(
                    Structure.from_file(
                        os.path.join(self.path, phonon_folder, "CONTCAR.2phonons")
                    ).volume,
                    2,
                )
                in volumes_set
            ]

        for phonon_folder in phonon_folders:
            incar_data = {}
            for key in incar_keys:
                try:
                    incar_data[key] = Incar.from_file(
                        os.path.join(self.path, phonon_folder, f"INCAR.{key}")
                    )
                except FileNotFoundError:
                    if key == "1relax":
                        continue
                    else:
                        raise
            self.incars.append(incar_data)

            structure = Structure.from_file(
                os.path.join(self.path, phonon_folder, "CONTCAR.2phonons")
            )
            self.phonon_structures.append(structure)

        if phonon_folders:
            self.kpoints = Kpoints.from_file(
                os.path.join(self.path, phonon_folders[0], "KPOINTS.2phonons")
            )

        try:
            self.potcar = Potcar.from_file(os.path.join(self.path, "POTCAR"))
        except FileNotFoundError:
            self.potcar = None

    @staticmethod
    def pad_arrays(arrays, pad_value=0, pad_type="constant"):
        max_length = max(len(arr) for arr in arrays)
        padded_arrays = []
        for arr in arrays:
            if pad_type == "constant":
                padded_arr = np.pad(
                    arr, (0, max_length - len(arr)), constant_values=pad_value
                )
            elif pad_type == "increasing":
                increment = arr[-1] - arr[-2]
                pad_values = np.arange(
                    arr[-1] + increment,
                    arr[-1] + increment * (max_length - len(arr) + 1),
                    increment,
                )
                padded_arr = np.concatenate([arr, pad_values])
            padded_arrays.append(padded_arr)
        return np.column_stack(padded_arrays)

    def get_harmonic_data(
        self,
        scale_atoms: int,
        temperatures: np.ndarray,
        order: int,
    ) -> None:

        yphon_results_path = os.path.join(self.path, "YPHON_results")
        vdos_data_scaled = scale_phonon_dos(yphon_results_path)
        volumes_per_atom = np.sort(vdos_data_scaled["volume_per_atom"].unique())
        frequency_array = []
        dos_array = []

        frequency_array = [
            vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom][
                "frequency_hz"
            ].values
            for volume_per_atom in volumes_per_atom
        ]
        dos_array = [
            vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom][
                "dos_1_per_hz"
            ].values
            for volume_per_atom in volumes_per_atom
        ]

        frequency_array = self.pad_arrays(frequency_array, pad_type="increasing")
        dos_array = self.pad_arrays(dos_array, pad_value=0, pad_type="constant")

        self.temperatures = temperatures
        volumes, f_vib, e_vib, s_vib, cv_vib = harmonic(
            scale_atoms,
            volumes_per_atom,
            temperatures,
            frequency_array,
            dos_array,
        )

        self.number_of_atoms = scale_atoms
        self.volumes = volumes
        self.f_vib = f_vib
        self.s_vib = s_vib
        self.cv_vib = cv_vib

        (
            volume_fit,
            f_vib_fit,
            s_vib_fit,
            cv_vib_fit,
            f_vib_poly,
            s_vib_poly,
            cv_vib_poly,
        ) = fit_harmonic(
            self.volumes, self.temperatures, self.f_vib, self.s_vib, self.cv_vib, order
        )

        self.f_vib_fit = f_vib_fit
        self.s_vib_fit = s_vib_fit
        self.cv_vib_fit = cv_vib_fit
        self.volume_fit = volume_fit

        self.helmholtz_energy = {
            f"{temp}K": f_vib[i] for i, temp in enumerate(self.temperatures)
        }
        self.internal_energy = {
            f"{temp}K": e_vib[i] for i, temp in enumerate(self.temperatures)
        }
        self.entropy = {
            f"{temp}K": s_vib[i] for i, temp in enumerate(self.temperatures)
        }
        self.heat_capacity = {
            f"{temp}K": cv_vib[i] for i, temp in enumerate(self.temperatures)
        }

        self.helmholtz_energy_fit = {
            "polynomial_coefficients": {
                f"{temp}K": coeff for temp, coeff in zip(self.temperatures, f_vib_poly)
            }
        }
        self.entropy_fit = {
            "polynomial_coefficients": {
                f"{temp}K": coeff for temp, coeff in zip(self.temperatures, s_vib_poly)
            }
        }
        self.heat_capacity_fit = {
            "polynomial_coefficients": {
                f"{temp}K": coeff for temp, coeff in zip(self.temperatures, cv_vib_poly)
            }
        }

    def plot_scaled_dos(self, number_of_atoms: int, plot: bool = True) -> None:
        yphon_results_path = os.path.join(self.path, "YPHON_results")
        scale_phonon_dos(yphon_results_path, number_of_atoms, plot)

    def plot_multiple_dos(self, number_of_atoms: int) -> None:
        yphon_results_path = os.path.join(self.path, "YPHON_results")
        plot_phonon_dos(yphon_results_path, number_of_atoms)

    def plot_harmonic(
        self, property_to_plot: str, selected_temperatures_plot: np.ndarray = None
    ) -> tuple[go.Figure, go.Figure]:

        property_mapping = {
            "helmholtz_energy": "f_vib",
            "entropy": "s_vib",
            "heat_capacity": "cv_vib",
        }

        if property_to_plot not in property_mapping:
            raise ValueError(f"Invalid property_to_plot: {property_to_plot}")

        property_name = property_mapping[property_to_plot]
        property_data = getattr(self, property_name)
        property_fit_data = getattr(self, f"{property_name}_fit")

        fig_harmonic = plot_harmonic(
            scale_atoms=self.number_of_atoms,
            volumes=self.volumes,
            temperatures=self.temperatures,
            property=property_data,
            property_name=property_to_plot,
        )

        fig_fit_harmonic = plot_fit_harmonic(
            scale_atoms=self.number_of_atoms,
            volumes=self.volumes,
            temperatures=self.temperatures,
            property_name=property_to_plot,
            property=property_data,
            volume_fit=self.volume_fit,
            property_fit=property_fit_data,
            selected_temperatures_plot=selected_temperatures_plot,
        )
        return fig_harmonic, fig_fit_harmonic

