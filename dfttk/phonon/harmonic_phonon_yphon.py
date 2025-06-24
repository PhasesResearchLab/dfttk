"""
HarmonicPhononYphon class for computing thermodynamic properties using phonon DOS data from VASP and YPHON.
"""

# Standard library imports
import os

# Third-party imports
import numpy as np
import pandas as pd
import scipy.constants
import plotly.graph_objects as go
from typing import Optional

# DFTTK imports
from dfttk.plotly_format import plot_format

EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3 = 160.21766208 GPa
BOLTZMANN_CONSTANT = scipy.constants.Boltzmann / scipy.constants.electron_volt  # eV/K
PLANCK_CONSTANT = scipy.constants.Planck / scipy.constants.electron_volt  # eV·s


class HarmonicPhononYphon:
    """
    Class for computing thermodynamic properties within the harmonic approximation
    using phonon density of states (DOS) data from VASP and YPHON.

    Attributes:
        phonon_dos (pd.DataFrame): Raw phonon DOS data loaded from YPHON output.
        scaled_phonon_dos (pd.DataFrame): Phonon DOS data scaled to the specified number of atoms.
        number_of_atoms (int): Number of atoms used for scaling the DOS and thermodynamic properties.
        volumes_per_atom (np.ndarray): Unique volumes per atom corresponding to the phonon DOS data.
        volumes (np.ndarray): Total volumes (per supercell) after scaling by number_of_atoms.
        helmholtz_energy (np.ndarray): Vibrational Helmholtz free energy (eV/atom); shape (temperatures, volumes).
        entropy (np.ndarray): Vibrational entropy (eV/K/atom); shape (temperatures, volumes).
        heat_capacity (np.ndarray): Vibrational heat capacity at constant volume (eV/K/atom); shape (temperatures, volumes).
        helmholtz_energy_fit (np.ndarray): Fitted Helmholtz energy (eV/atom); shape (temperatures, volumes_fit).
        entropy_fit (np.ndarray): Fitted entropy (eV/K/atom); shape (temperatures, volumes_fit).
        heat_capacity_fit (np.ndarray): Fitted heat capacity (eV/K/atom); shape (temperatures, volumes_fit).
        helmholtz_energy_poly_coeffs (np.ndarray): Polynomial coefficients for Helmholtz energy fit.
        entropy_poly_coeffs (np.ndarray): Polynomial coefficients for entropy fit.
        heat_capacity_poly_coeffs (np.ndarray): Polynomial coefficients for heat capacity fit.
    """

    def __init__(self):
        self.phonon_dos: pd.DataFrame = None
        self.scaled_phonon_dos: pd.DataFrame = None

        self.number_of_atoms: int = None
        self.volumes_per_atom: np.ndarray = None
        self.volumes: np.ndarray = None
        self.temperatures: np.ndarray = None
        self.helmholtz_energy: np.ndarray = None
        self.entropy: np.ndarray = None
        self.heat_capacity: np.ndarray = None

        self.volumes_fit: np.ndarray = None
        self.helmholtz_energy_fit: np.ndarray = None
        self.entropy_fit: np.ndarray = None
        self.heat_capacity_fit: np.ndarray = None
        self.helmholtz_energy_poly_coeffs: np.ndarray = None
        self.entropy_poly_coeffs: np.ndarray = None
        self.heat_capacity_poly_coeffs: np.ndarray = None

    def load_dos(self, path: str) -> None:
        """
        Load phonon DOS data from vdos and volph files in the specified directory.

        Args:
            path (str): Directory containing vdos_ and volph_ files.

        Returns:
            pd.DataFrame: Combined phonon DOS data.

        Raises:
            FileNotFoundError: If no vdos or volph files are found in the directory.
            ValueError: If the number of vdos files does not match the number of volph files,
                        or if the indexes of vdos and volph files do not match.
        """
        file_list = os.listdir(path)
        vdos_files = sorted([f for f in file_list if f.startswith("vdos_")])
        volph_files = sorted([f for f in file_list if f.startswith("volph_")])

        if not vdos_files or not volph_files:
            raise FileNotFoundError("No vdos_ or volph_ files found in the specified directory.")

        if len(vdos_files) != len(volph_files):
            raise ValueError("The number of vdos files does not match the number of volph files.")

        vdos_indexes = [f.split("_")[1].split(".")[0] for f in vdos_files]
        volph_indexes = [f.split("_")[1].split(".")[0] for f in volph_files]
        if vdos_indexes != volph_indexes:
            raise ValueError("The indexes of vdos files do not match the indexes of volph files.")

        dataframes = []
        for vdos_file, volph_file in zip(vdos_files, volph_files):
            with open(os.path.join(path, volph_file)) as f:
                volph_content = float(f.readline().strip())
            df = pd.read_csv(
                os.path.join(path, vdos_file),
                sep=r"\s+",
                header=None,
                names=["frequency_hz", "dos_1_per_hz"],
            )
            df.insert(0, "volume_per_atom", volph_content)
            dataframes.append(df)

        self.phonon_dos = pd.concat(dataframes, ignore_index=True)
        self.volumes_per_atom = np.sort(self.phonon_dos["volume_per_atom"].unique())

    def scale_dos(self, number_of_atoms: int, plot: bool = False) -> None:
        """
        Scale the area under the phonon DOS to 3N, where N is the number of atoms.

        Args:
            number_of_atoms (int): Number of atoms to scale the DOS to.
            plot (bool): If True, return a plotly Figure of the original and scaled DOS.

        Returns:
            Optional[go.Figure]: Plotly Figure if plot is True, otherwise None.

        Raises:
            RuntimeError: If phonon DOS data is not loaded (call load_dos() before scale_dos()).
        """
        vdos_data = self.phonon_dos
        self.number_of_atoms = number_of_atoms

        if self.phonon_dos is None:
            raise RuntimeError("Phonon DOS data not loaded. Call load_dos() before scale_dos().")

        # Count the area % of positive and negative frequencies
        area_count = []
        for volume_per_atom in self.volumes_per_atom:
            vdos_total = vdos_data[vdos_data["volume_per_atom"] == volume_per_atom]
            vdos_neg = vdos_total[vdos_total["frequency_hz"] < 0]
            vdos_pos = vdos_total[vdos_total["frequency_hz"] >= 0]
            area_neg = np.trapz(vdos_neg["dos_1_per_hz"], vdos_neg["frequency_hz"])
            area_pos = np.trapz(vdos_pos["dos_1_per_hz"], vdos_pos["frequency_hz"])
            area_tot = np.trapz(vdos_total["dos_1_per_hz"], vdos_total["frequency_hz"])
            area_count.append((volume_per_atom, area_pos / area_tot * 100, area_neg / area_tot * 100))

        # Count the original number of atoms before scaling
        original_atoms = [round(np.trapz(vdos_data[vdos_data["volume_per_atom"] == volume_per_atom]["dos_1_per_hz"], vdos_data[vdos_data["volume_per_atom"] == volume_per_atom]["frequency_hz"]) / 3) for volume_per_atom in self.volumes_per_atom]

        # Remove negative frequencies
        vdos_data_scaled = vdos_data[vdos_data["frequency_hz"] > 0].reset_index(drop=True)

        # Add a row of zero frequency and DOS to the beginning of each volume_per_atom
        dfs = []
        for volume_per_atom in self.volumes_per_atom:
            filtered_df = vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom]
            new_row = pd.DataFrame([[volume_per_atom, 0, 0]], columns=vdos_data_scaled.columns)
            filtered_df = pd.concat([new_row, filtered_df], ignore_index=True)
            dfs.append(filtered_df)
        vdos_data_scaled = pd.concat(dfs, ignore_index=True)

        # Add number_of_atoms column
        vdos_data_scaled.insert(1, "number_of_atoms", number_of_atoms)

        # Scale the phonon DOS to 3N
        num_atoms_3N = number_of_atoms * 3
        for volume_per_atom in self.volumes_per_atom:
            mask = vdos_data_scaled["volume_per_atom"] == volume_per_atom
            freq = vdos_data_scaled.loc[mask, "frequency_hz"]
            dos = vdos_data_scaled.loc[mask, "dos_1_per_hz"]
            area = np.trapz(dos, freq)
            vdos_data_scaled.loc[mask, "dos_1_per_hz"] = dos * num_atoms_3N / area

        if plot:
            for i, volume_per_atom in enumerate(self.volumes_per_atom):
                fig = go.Figure()
                freq = vdos_data[vdos_data["volume_per_atom"] == volume_per_atom]["frequency_hz"]
                dos = vdos_data[vdos_data["volume_per_atom"] == volume_per_atom]["dos_1_per_hz"]
                scaled_freq = vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom]["frequency_hz"]
                scaled_dos = vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom]["dos_1_per_hz"]

                fig.add_trace(go.Scatter(x=freq / 1e12, y=dos * 1e12, mode="lines", name=f"Original - {original_atoms[i]} atoms", showlegend=True))
                fig.add_trace(go.Scatter(x=scaled_freq / 1e12, y=scaled_dos * 1e12, mode="lines", name=f"Scaled - {number_of_atoms} atoms", showlegend=True))
                plot_format(fig, "Frequency (THz)", "DOS (1/THz)")
                fig.update_layout(
                    title=dict(
                        text=f"Original volume: {volume_per_atom * original_atoms[i]} Å³/{original_atoms[i]} atoms" f"<br> Scaled volume: {volume_per_atom * number_of_atoms} Å³/{number_of_atoms} atoms" f"<br> (Area: {area_count[i][1]:.1f}% positive, {area_count[i][2]:.1f}% negative)",
                        font=dict(size=20, color="rgb(0,0,0)"),
                    ),
                    margin=dict(t=130),
                )
                fig.show()

        self.volumes = self.volumes_per_atom * number_of_atoms
        self.scaled_phonon_dos = vdos_data_scaled

    def plot_dos(self) -> go.Figure:
        """
        Plot the scaled phonon density of states (DOS) for multiple volumes.

        Returns:
            go.Figure: Plotly Figure object showing the scaled phonon DOS for each volume.

        Raises:
            RuntimeError: If scaled phonon DOS data or volumes_per_atom is not set.
        """
        if self.scaled_phonon_dos is None or self.volumes_per_atom is None:
            raise RuntimeError("Scaled phonon DOS data not calculated. Call scale_dos() before plot_dos().")

        vdos_data_scaled = self.scaled_phonon_dos
        volumes_per_atom = self.volumes_per_atom

        fig = go.Figure()
        for volume_per_atom in volumes_per_atom:
            freq = vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom]["frequency_hz"]
            dos = vdos_data_scaled[vdos_data_scaled["volume_per_atom"] == volume_per_atom]["dos_1_per_hz"]
            fig.add_trace(go.Scatter(x=freq / 1e12, y=dos * 1e12, mode="lines", name=f"{volume_per_atom * self.number_of_atoms} Å³", showlegend=True))
        plot_format(fig, "Frequency (THz)", "DOS (1/THz)")
        fig.update_layout(
            title=dict(
                text=f"Number of atoms = {self.number_of_atoms}",
                font=dict(size=24, color="rgb(0,0,0)"),
            )
        )
        return fig

    def calculate_harmonic(
        self,
        temperatures: np.ndarray,
    ) -> None:
        """
        Calculate vibrational thermodynamic properties (Helmholtz energy, internal energy, entropy, heat capacity)
        at the specified temperatures using the scaled phonon DOS and the harmonic approximation.

        Args:
            temperatures (np.ndarray): 1D array of temperatures in Kelvin.

        Raises:
            RuntimeError: If scaled phonon DOS data or volumes_per_atom is not set (call scale_dos() before calculate_harmonic()).
        """
        if self.scaled_phonon_dos is None or self.volumes_per_atom is None:
            raise RuntimeError("Scaled phonon DOS data not calculated. Call scale_dos() before calculate_harmonic().")

        self.temperatures = temperatures
        frequency_array = [self.scaled_phonon_dos[self.scaled_phonon_dos["volume_per_atom"] == volume_per_atom]["frequency_hz"].values for volume_per_atom in self.volumes_per_atom]
        dos_array = [self.scaled_phonon_dos[self.scaled_phonon_dos["volume_per_atom"] == volume_per_atom]["dos_1_per_hz"].values for volume_per_atom in self.volumes_per_atom]
        frequency_array = self.pad_arrays(frequency_array, pad_type="increasing")
        dos_array = self.pad_arrays(dos_array, pad_value=0, pad_type="constant")

        num_volumes = len(self.volumes_per_atom)
        num_temps = len(temperatures)

        # Preallocate output arrays
        helmholtz_energy = np.zeros((num_temps, num_volumes))
        internal_energy = np.zeros((num_temps, num_volumes))
        entropy = np.zeros((num_temps, num_volumes))
        heat_capacity = np.zeros((num_temps, num_volumes))

        # Precompute midpoints and differences for all volumes
        freq_mid = (frequency_array[1:, :] + frequency_array[:-1, :]) / 2
        dos_mid = (dos_array[1:, :] + dos_array[:-1, :]) / 2
        freq_diff = frequency_array[1:, :] - frequency_array[:-1, :]

        for i in range(num_volumes):
            f_mid = freq_mid[:, i]
            d_mid = dos_mid[:, i]
            df = freq_diff[:, i]

            for j, T in enumerate(temperatures):
                if T == 0:
                    integrand = PLANCK_CONSTANT / 2 * f_mid * d_mid * df
                    F = np.sum(integrand)
                    helmholtz_energy[j, i] = F
                    internal_energy[j, i] = F
                    entropy[j, i] = 0
                    heat_capacity[j, i] = 0
                else:
                    ratio = (PLANCK_CONSTANT * f_mid) / (2 * BOLTZMANN_CONSTANT * T)
                    sinh_ratio = np.sinh(ratio)
                    log_term = np.log(2 * sinh_ratio)
                    # Helmholtz free energy
                    integrand_F = log_term * d_mid * df
                    F = BOLTZMANN_CONSTANT * T * np.sum(integrand_F)
                    helmholtz_energy[j, i] = F
                    # Internal energy
                    integrand_E = f_mid * np.cosh(ratio) / sinh_ratio * d_mid * df
                    E = PLANCK_CONSTANT / 2 * np.sum(integrand_E)
                    internal_energy[j, i] = E
                    # Entropy
                    integrand_S = (ratio * np.cosh(ratio) / sinh_ratio - log_term) * d_mid * df
                    S = BOLTZMANN_CONSTANT * np.sum(integrand_S)
                    entropy[j, i] = S
                    # Heat capacity
                    integrand_C = (ratio**2) * (1 / sinh_ratio) ** 2 * d_mid * df
                    Cv = BOLTZMANN_CONSTANT * np.sum(integrand_C)
                    heat_capacity[j, i] = Cv

        self.helmholtz_energy = helmholtz_energy
        self.internal_energy = internal_energy
        self.entropy = entropy
        self.heat_capacity = heat_capacity

    def fit_harmonic(
        self,
        order: int = 2,
        min_volume: float = None,
        max_volume: float = None,
        num_volumes: int = 1000,
    ) -> None:
        """
        Fit the Helmholtz energy, entropy, and heat capacity as a function of volume
        using polynomial regression for each temperature.

        Args:
            order (int, optional): Polynomial order for fitting. Defaults to 2.
            min_volume (float, optional): Minimum volume for fitting. Defaults to None.
            max_volume (float, optional): Maximum volume for fitting. Defaults to None.
            num_volumes (int, optional): Number of volumes for fitting. Defaults to 1000.

        Raises:
            RuntimeError: If required thermodynamic properties are not set (call calculate_harmonic() before fit_harmonic()).
        """
        if self.helmholtz_energy is None or self.entropy is None or self.heat_capacity is None or self.temperatures is None:
            raise RuntimeError("Thermodynamic properties not calculated. Call calculate_harmonic() before fit_harmonic().")

        helmholtz_energy_fit_list = []
        entropy_fit_list = []
        heat_capacity_fit_list = []
        helmholtz_energy_poly_coeffs_list = []
        entropy_poly_coeffs_list = []
        heat_capacity_poly_coeffs_list = []

        for i in range(len(self.temperatures)):
            helmholtz_energy_coefficients = np.polyfit(self.volumes, self.helmholtz_energy[i], order)
            entropy_coefficients = np.polyfit(self.volumes, self.entropy[i], order)
            heat_capacity_coefficients = np.polyfit(self.volumes, self.heat_capacity[i], order)

            helmholtz_energy_polynomial = np.poly1d(helmholtz_energy_coefficients)
            entropy_polynomial = np.poly1d(entropy_coefficients)
            heat_capacity_polynomial = np.poly1d(heat_capacity_coefficients)
            helmholtz_energy_poly_coeffs_list.append(helmholtz_energy_coefficients)
            entropy_poly_coeffs_list.append(entropy_coefficients)
            heat_capacity_poly_coeffs_list.append(heat_capacity_coefficients)

            if min_volume is None:
                min_volume = min(self.volumes) * 0.98
            if max_volume is None:
                max_volume = max(self.volumes) * 1.02
            volumes_fit = np.linspace(min_volume, max_volume, num_volumes)

            helmholtz_energy_fit = helmholtz_energy_polynomial(volumes_fit)
            entropy_fit = entropy_polynomial(volumes_fit)
            heat_capacity_fit = heat_capacity_polynomial(volumes_fit)

            helmholtz_energy_fit_list.append(helmholtz_energy_fit)
            entropy_fit_list.append(entropy_fit)
            heat_capacity_fit_list.append(heat_capacity_fit)

        self.volumes_fit = volumes_fit
        self.helmholtz_energy_fit = np.array(helmholtz_energy_fit_list)
        self.entropy_fit = np.array(entropy_fit_list)
        self.heat_capacity_fit = np.array(heat_capacity_fit_list)

        self.helmholtz_energy_poly_coeffs = np.array(helmholtz_energy_poly_coeffs_list)
        self.entropy_poly_coeffs = np.array(entropy_poly_coeffs_list)
        self.heat_capacity_poly_coeffs = np.array(heat_capacity_poly_coeffs_list)

    def plot_harmonic(
        self,
        property: str,
    ) -> go.Figure:
        """
        Plot a thermodynamic property (Helmholtz energy, entropy, or heat capacity)
        as a function of temperature for each volume.

        Args:
            property (str): Name of the property to plot. Must be one of
                'helmholtz_energy', 'entropy', or 'heat_capacity'.

        Returns:
            go.Figure: Plotly Figure object with the property vs. temperature curves.

        Raises:
            RuntimeError: If required thermodynamic properties are not set (call calculate_harmonic() before fit_harmonic()).
            ValueError: If property is not one of 'helmholtz_energy', 'entropy', or 'heat_capacity'.
        """
        if self.helmholtz_energy is None or self.entropy is None or self.heat_capacity is None or self.temperatures is None:
            raise RuntimeError("Thermodynamic properties not calculated. Call calculate_harmonic() before fit_harmonic().")

        valid_properties = {
            "helmholtz_energy": ("helmholtz_energy", f"F<sub>vib</sub> (eV/{self.number_of_atoms} atoms)"),
            "entropy": ("entropy", f"S<sub>vib</sub> (eV/K/{self.number_of_atoms} atoms)"),
            "heat_capacity": ("heat_capacity", f"C<sub>vib</sub> (eV/K/{self.number_of_atoms} atoms)"),
        }

        if property not in valid_properties:
            raise ValueError("property must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'")

        attr_name, y_title = valid_properties[property]
        property_data = getattr(self, attr_name)

        fig = go.Figure()
        for i, volume in enumerate(self.volumes):
            fig.add_trace(
                go.Scatter(
                    x=self.temperatures,
                    y=property_data[:, i],
                    mode="lines",
                    name=f"{volume} Å³",
                    showlegend=True,
                )
            )
        plot_format(fig, "Temperature (K)", y_title)

        return fig

    def plot_fit_harmonic(
        self,
        property: str,
        selected_temperatures: np.ndarray = None,
    ) -> go.Figure:
        """
        Plot fitted thermodynamic properties (Helmholtz energy, entropy, or heat capacity)
        as a function of volume for selected temperatures.

        Args:
            property (str): Name of the property to plot. Must be one of
                'helmholtz_energy', 'entropy', or 'heat_capacity'.
            selected_temperatures (np.ndarray, optional): Array of temperatures (in K) to plot.
                If None, five temperatures are selected evenly from the range.

        Returns:
            go.Figure: Plotly Figure object with the fitted property vs. volume curves.

        Raises:
            RuntimeError: If fitted thermodynamic properties are not set (call fit_harmonic() before plot_fit_harmonic()).
            ValueError: If property is not one of 'helmholtz_energy', 'entropy', or 'heat_capacity'.
        """
        if self.volumes_fit is None or self.helmholtz_energy_fit is None or self.entropy_fit is None or self.heat_capacity_fit is None or self.temperatures is None:
            raise RuntimeError("Fitted thermodynamic properties not calculated. Call fit_harmonic() before plot_fit_harmonic().")

        valid_properties = {
            "helmholtz_energy": ("helmholtz_energy", "helmholtz_energy_fit", f"F<sub>vib</sub> (eV/{self.number_of_atoms} atoms)"),
            "entropy": ("entropy", "entropy_fit", f"S<sub>vib</sub> (eV/K/{self.number_of_atoms} atoms)"),
            "heat_capacity": ("heat_capacity", "heat_capacity_fit", f"C<sub>vib</sub> (eV/K/{self.number_of_atoms} atoms)"),
        }

        if property not in valid_properties:
            raise ValueError("property must be one of 'helmholtz_energy', 'entropy', or 'heat_capacity'")

        attr_name, fit_attr_name, y_title = valid_properties[property]
        property_data = getattr(self, attr_name)
        property_fit = getattr(self, fit_attr_name)

        if selected_temperatures is None:
            indices = np.linspace(0, len(self.temperatures) - 1, 5, dtype=int)
            selected_temperatures = np.array([self.temperatures[j] for j in indices])

        fig = go.Figure()
        colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
        colors = [f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 1)" for color in colors]

        for i, temperature in enumerate(selected_temperatures):
            index = np.where(self.temperatures == temperature)[0][0]
            x = self.volumes
            y = property_data[index]
            x_fit = self.volumes_fit
            y_fit = property_fit[index]
            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    line=dict(color=color),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    line=dict(color=color),
                    name=f"{temperature} K",
                    showlegend=True,
                )
            )

        plot_format(fig, f"Volume (Å³/{self.number_of_atoms} atoms)", y_title)

        return fig

    @staticmethod
    def pad_arrays(arrays, pad_value=0, pad_type="constant"):
        """
        Pad a list of 1D arrays to the same length for stacking.

        Args:
            arrays (list of np.ndarray): List of 1D arrays to pad.
            pad_value (float, optional): Value to use for constant padding. Default is 0.
            pad_type (str, optional): Padding strategy:
                - "constant": pad with pad_value (default)
                - "increasing": pad with increasing values based on the last increment

        Returns:
            np.ndarray: 2D array with each input array padded to the maximum length.
        """
        max_length = max(len(arr) for arr in arrays)
        padded_arrays = []
        for arr in arrays:
            if pad_type == "constant":
                padded_arr = np.pad(arr, (0, max_length - len(arr)), constant_values=pad_value)
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
