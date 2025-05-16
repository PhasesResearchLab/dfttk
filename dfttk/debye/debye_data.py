"""
Debye-Grüneisen module to calculate the vibrational contribution to the Helmholtz energy, entropy, and heat capacity.    
"""

# Related third party imports
import numpy as np
import plotly.graph_objects as go

# DFTTK imports
from dfttk.debye.functions import(
    process_debye_gruneisen,
    plot,
)


# TODO: add docstrings
class DebyeData:
    def __init__(self):
        self.number_of_atoms: int = None
        self.scaling_factor: float = None
        self.gruneisen_x: float = None
        self.temperatures: np.ndarray = None
        self.volumes: np.ndarray = None
        self.free_energy: np.ndarray = None
        self.entropy: np.ndarray = None
        self.heat_capacity: np.ndarray = None

    def get_debye_gruneisen_data(
        self,
        number_of_atoms: int,
        volumes: np.ndarray,
        average_mass: float,
        volume_0: float,
        bulk_modulus: float,
        bulk_modulus_prime: float,
        scaling_factor: float = 0.617,
        gruneisen_x: float = 1,
        temperatures: np.ndarray = np.linspace(0, 1000, 101),
    ) -> None:
        volumes = np.linspace(0.98 * min(volumes), 1.02 * max(volumes), 1000) # Temporary for testing
        (
            f_vib,
            s_vib,
            cv_vib,
        ) = process_debye_gruneisen(
            number_of_atoms,
            volumes,
            temperatures,
            average_mass,
            volume_0,
            bulk_modulus,
            bulk_modulus_prime,
            scaling_factor,
            gruneisen_x,
        )

        self.number_of_atoms = number_of_atoms
        self.scaling_factor = scaling_factor
        self.gruneisen_x = gruneisen_x
        self.temperatures = temperatures
        self.volumes = volumes
        self.free_energy = f_vib
        self.entropy = s_vib
        self.heat_capacity = cv_vib

    def plot(
        self,
        property: str,
        temperatures: np.ndarray = None,
        volumes: np.ndarray = None,
    ) -> tuple[go.Figure, go.Figure]:

        fig_t, fig_v = plot(
            property=property,
            number_of_atoms=self.number_of_atoms,
            temperatures=self.temperatures,
            volumes=self.volumes,
            helmholtz_energies=self.free_energy,
            entropies=self.entropy,
            heat_capacities=self.heat_capacity,
            selected_temperatures=temperatures,
            selected_volumes=volumes,
        )

        return fig_t, fig_v
    