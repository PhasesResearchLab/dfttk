"""
EOSFitter class for fitting and plotting energy-volume data using various equations of state (EOS).
"""

# Related third party imports
import plotly.express as px
import plotly.graph_objects as go
from distinctipy import get_colors

# DFTTK imports
from dfttk.eos.functions import (
    mBM4,
    mBM5,
    BM4,
    BM5,
    LOG4,
    LOG5,
    murnaghan,
    vinet,
    morse,
)


def assign_colors_to_configs(unique_configs: list[str], alpha: float = 1, cmap: str = "plotly") -> dict:
    """
    Assign colors to configurations based on the specified color map.

    Supports two color maps: "plotly" and "distinctipy".

    Args:
        unique_configs (list[str]): List of unique configuration names.
        alpha (float, optional): Alpha value for the color. Defaults to 1.
        cmap (str, optional): Color map to use. Defaults to "plotly".

    Returns:
        dict: Dictionary mapping configurations to colors.

    Raises:
        ValueError: If cmap is not 'plotly' or 'distinctipy'.
    """
    if cmap == "plotly":
        # Use a fixed set of Plotly qualitative colors
        colors = px.colors.qualitative.Plotly
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
        # Convert hex colors to rgba strings with the specified alpha
        colors = [f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {alpha})" for color in colors]
    elif cmap == "distinctipy":
        # Use distinctipy to generate visually distinct colors
        colors = get_colors(len(unique_configs))
        colors = [f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha})" for color in colors]
    else:
        raise ValueError("cmap must be 'plotly' or 'distinctipy'")

    # Assign colors to each config, cycling if there are more configs than colors
    config_colors = {config: colors[i % len(colors)] for i, config in enumerate(unique_configs)}
    return config_colors


def assign_marker_symbols_to_configs(unique_configs: list[str]) -> dict:
    """
    Assign marker symbols to configurations.

    Args:
        unique_configs (list[str]): List of unique configuration names.

    Returns:
        dict: Dictionary mapping configurations to marker symbols.
    """
    # List of marker symbols to cycle through
    symbols = [
        "circle",
        "square",
        "diamond",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "pentagon",
        "hexagon",
        "octagon",
        "star",
        "hexagram",
        "star-triangle-up",
        "star-triangle-down",
        "star-square",
        "star-diamond",
        "diamond-tall",
        "diamond-wide",
        "hourglass",
        "bowtie",
        "circle-cross",
        "circle-x",
        "square-cross",
        "square-x",
        "diamond-cross",
        "diamond-x",
        "arrow-left",
        "arrow-right",
        "arrow",
        "circle-open",
        "square-open",
        "diamond-open",
        "x-open",
        "triangle-up-open",
        "triangle-down-open",
        "triangle-left-open",
        "triangle-right-open",
        "pentagon-open",
    ]
    # Assign symbols to each config, cycling if there are more configs than symbols
    config_symbols = {config: symbols[i % len(symbols)] for i, config in enumerate(unique_configs)}
    return config_symbols


class EOSFitter:
    """
    Class for fitting and plotting energy-volume data using various equations of state (EOS).
    """

    def __init__(
        self,
        name: str,
        number_of_atoms: int,
        volumes: list[float],
        energies: list[float],
    ):
        """
        Initialize the EOSFitter.

        Args:
            name (str): Name of the configuration.
            number_of_atoms (int): Number of atoms in the supercell.
            volumes (np.ndarray): Array of volumes.
            energies (np.ndarray): Array of energies.
        """
        self.name = name
        self.number_of_atoms = number_of_atoms
        self.volumes = volumes
        self.energies = energies
        self.eos_name = None
        self.eos_constants = None
        self.eos_parameters = None
        self.volume_range = None
        self.energy_eos = None
        self.pressure_eos = None

        # Ensure input data is consistent
        if len(self.volumes) != len(self.energies):
            raise ValueError("Volumes and energies must have the same length.")

    def fit(
        self,
        eos_name: str = "BM4",
        volume_min: float = None,
        volume_max: float = None,
        num_volumes: int = 1000,
    ):
        """
        Fit the given energy-volume data to an equation of state (EOS) using the specified EOS function.

        The EOS function should be defined in the dfttk.eos.functions module.

        Args:
            eos_name (str, optional): Name of the EOS function to use. Defaults to "BM4".
            volume_min (float, optional): Minimum volume to consider for fitting. Defaults to None.
            volume_max (float, optional): Maximum volume to consider for fitting. Defaults to None.
            num_volumes (int, optional): Number of volumes to generate for the EOS fit. Defaults to 1000.

        Raises:
            ValueError: If the specified EOS function is not recognized.
            RuntimeError: If an unexpected error occurs during fitting.
        """
        # Dictionary mapping EOS function names to their implementations
        eos_functions = {
            "mBM4": mBM4,
            "mBM5": mBM5,
            "BM4": BM4,
            "BM5": BM5,
            "LOG4": LOG4,
            "LOG5": LOG5,
            "murnaghan": murnaghan,
            "vinet": vinet,
            "morse": morse,
        }
        # Select the EOS function based on the provided name
        eos_function = eos_functions.get(eos_name)
        if eos_function is None:
            raise ValueError(f"EOS function '{eos_name}' not recognized.")

        try:
            # Perform the EOS fit using the selected function
            (
                eos_constants,
                eos_parameters,
                volume_range,
                energy_eos,
                pressure_eos,
            ) = eos_function(self.volumes, self.energies, volume_min, volume_max, num_volumes)
        except Exception as e:
            # Raise a RuntimeError for any unexpected error during fitting
            raise RuntimeError(f"Error fitting config: {e}") from e

        # Store the fit results as attributes
        self.eos_name = eos_name
        self.eos_constants = eos_constants
        self.eos_parameters = eos_parameters
        self.volume_range = volume_range
        self.energy_eos = energy_eos
        self.pressure_eos = pressure_eos

    def plot(
        self,
        highlight_minimum: bool = True,
        per_atom: bool = False,
        title: str = None,
        cmap: str = "plotly",
        marker_alpha: int = 1,
        marker_size: int = 10,
    ) -> go.Figure:
        """
        Plot the energy vs volume curves for each configuration.

        Args:
            highlight_minimum (bool, optional): Whether to highlight the minimum energy. Must be True or False. Defaults to True.
            per_atom (bool, optional): Whether to plot energy and volume per atom. Must be True or False. Defaults to False.
            title (str, optional): Title of the plot. Defaults to None.
            cmap (str, optional): Color map to use. Defaults to "plotly".
            marker_alpha (int, optional): Alpha value for the marker color. Defaults to 1.
            marker_size (int, optional): Size of the markers. Defaults to 10.

        Returns:
            go.Figure: A Plotly figure object containing the plot.

        Raises:
            ValueError: If fit() has not been called before plot().
        """
        # Ensure fit() has been called before plotting
        if not hasattr(self, "eos_name") or self.eos_name is None:
            raise ValueError("You must call fit() before plot().")

        # Prepare color and symbol assignments for plotting
        unique_configs = [self.name]
        config_colors = assign_colors_to_configs(unique_configs, alpha=marker_alpha, cmap=cmap)
        config_symbols = assign_marker_symbols_to_configs(unique_configs)

        # Create a new Plotly figure
        fig = go.Figure()
        fig.update_layout(
            font=dict(
                family="Devaju Sans",
            )
        )

        # Prepare x and y data for plotting
        if isinstance(per_atom, bool):
            x = self.volumes
            y = self.energies

            # Convert to per atom if requested
            if per_atom:
                x = x / self.number_of_atoms
                y = y / self.number_of_atoms

        # Plot the raw energy vs volume data points
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=config_colors[self.name],
                    symbol=config_symbols[self.name],
                ),
                legendgroup=self.name,
                name=self.name,
                showlegend=True,
            )
        )

        # Set axis labels
        if isinstance(per_atom, bool):
            atom_suffix = "/atom" if per_atom else ""
            fig.update_xaxes(
                title=dict(
                    text=f"Volume (Å<sup>3</sup>{atom_suffix})",
                    font=dict(size=22, color="rgb(0,0,0)"),
                )
            )
            fig.update_yaxes(
                title=dict(
                    text=f"Energy (eV{atom_suffix})",
                    font=dict(size=22, color="rgb(0,0,0)"),
                )
            )

        # Plot the EOS fit curve
        if self.eos_name is not None:
            x = self.volume_range
            y = self.energy_eos
            if per_atom:
                num_atoms = self.number_of_atoms
                x = x / num_atoms
                y = y / num_atoms

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"{self.eos_name} fit",
                    line=dict(width=1.75, color=config_colors[self.name]),
                    legendgroup=self.name,
                    showlegend=False,
                )
            )

            # Optionally highlight the minimum energy point
            if highlight_minimum is True:
                x = self.eos_parameters[0]
                y = self.eos_parameters[1]

                if per_atom:
                    num_atoms = self.number_of_atoms
                    x = x / num_atoms
                    y = y / num_atoms

                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode="markers",
                        name="minimum",
                        marker=dict(color="black", size=marker_size, symbol="cross"),
                        legendgroup=self.name,
                        showlegend=False,
                    )
                )
            elif highlight_minimum is False:
                pass

        # Update layout with common parameters
        axis_params = dict(
            showline=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            mirror="allticks",
            tickwidth=1,
            tickcolor="black",
            showgrid=False,
            tickfont=dict(color="rgb(0,0,0)", size=20),
        )

        fig.update_layout(
            plot_bgcolor="white",
            width=840,
            height=600,
            legend=dict(font=dict(size=20, color="black")),
            xaxis=axis_params,
            yaxis=axis_params,
        )

        # Set the title if provided
        if title is not None:
            fig.update_layout(title=dict(text=title, x=0.5, font=dict(color="rgb(0,0,0)", size=30)))

        return fig
