"""
Module for fitting equations of state (EOS) to energy-volume data and plotting the results.
"""

# Related third party imports
import numpy as np
import pandas as pd
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


def fit_to_eos(
    volumes: np.ndarray,
    energies: np.ndarray,
    eos_name: str = "BM4",
    volume_min: float = None,
    volume_max: float = None,
    num_volumes: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fits the given energy-volume data to an equation of state (EOS) using the specified EOS function.
    The EOS function should be defined in the dfttk.eos.functions module.

    Args:
        volumes (np.ndarray): array of volumes.
        energies (np.ndarray): array of energies.
        eos_name (str, optional): name of the EOS function to use. Defaults to "BM4".
        volume_min (float, optional): minimum volume to consider for fitting. Defaults to None.
        volume_max (float, optional): maximum volume to consider for fitting. Defaults to None.
        num_volumes (int, optional): number of volumes to generate for the EOS fit. Defaults to 1000.

    Raises:
        ValueError: if the specified EOS function is not recognized.

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray): tuple containing:
            - eos_constants: array of EOS constants.
            - eos_parameters: array of EOS parameters.
            - volume_range: array of volumes for the EOS fit.
            - energy_eos: array of energies for the EOS fit.
            - pressure_eos: array of pressures for the EOS fit.
    """

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
    eos_function = eos_functions.get(eos_name)
    if eos_function is None:
        raise ValueError(f"EOS function '{eos_name}' not recognized.")

    try:
        (
            eos_constants,
            eos_parameters,
            volume_range,
            energy_eos,
            pressure_eos,
        ) = eos_function(volumes, energies, volume_min, volume_max, num_volumes)
        eos_name = eos_function.__name__
    except Exception as e:
        print(f"Error fitting config: {e}")

    return eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos


def assign_colors_to_configs(
    unique_configs: list[str], alpha: float = 1, cmap: str = "plotly"
) -> dict:
    """Assigns colors to configurations based on the specified color map.
    The function supports two color maps: "plotly" and "distinctipy".

    Args:
        unique_configs (list[str]): List of unique configurations.
        alpha (float, optional): Alpha value for the color. Defaults to 1.
        cmap (str, optional): Color map to use. Defaults to "plotly".

    Returns:
        dict: dictionary mapping configurations to colors.
    """

    if cmap == "plotly":
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
        colors = [
            f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {alpha})"
            for color in colors
        ]

    elif cmap == "distinctipy":
        colors = get_colors(len(unique_configs))
        colors = [
            f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha})" for color in colors
        ]
    else:
        print("cmap must be 'plotly' or 'distinctipy'")

    config_colors = {
        config: colors[i % len(colors)] for i, config in enumerate(unique_configs)
    }
    return config_colors


def assign_marker_symbols_to_configs(unique_configs: list[str]) -> dict:
    """Assigns marker symbols to configurations.

    Args:
        unique_configs (list[str]): List of unique configurations.

    Returns:
        dict: Dictionary mapping configurations to marker symbols.
    """

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
    config_symbols = {
        config: symbols[i % len(symbols)] for i, config in enumerate(unique_configs)
    }
    return config_symbols


def plot_ev(
    name: str,
    number_of_atoms: int,
    volumes: np.ndarray,
    energies: np.ndarray,
    volume_min: float = None,
    volume_max: float = None,
    num_volumes: int = 1000,
    eos_name: str = "BM4",
    highlight_minimum: bool = True,
    per_atom: bool = False,
    title: str = None,
    show_fig: bool = True,
    cmap: str = "plotly",
    marker_alpha: int = 1,
    marker_size: int = 10,
) -> go.Figure:
    """Plot the energy vs volume curves for each configuration.

    Args:
        name (str): Name of the configuration.
        number_of_atoms (int): Number of atoms in the configuration.
        volumes (np.ndarray): Array of volumes.
        energies (np.ndarray): Array of energies.
        volume_min (float, optional): Minimum volume to plot for EOS. Defaults to None.
        volume_max (float, optional): Maximum volume to plot for EOS. Defaults to None.
        num_volumes (int, optional): Number of volumes to plot for EOS. Defaults to None.
        eos_name (str, optional): EOS name. Defaults to "BM4".
        highlight_minimum (bool, optional): Whether to highlight the minimum energy. Defaults to True.
        per_atom (bool, optional): Whether to plot energy and volume per atom. Defaults to False.
        title (_type_, optional): Title of the plot. Defaults to None.
        show_fig (bool, optional): Whether to show the figure. Defaults to True.
        cmap (str, optional): Color map to use. Defaults to "plotly".
        marker_alpha (int, optional): Alpha value for the marker color. Defaults to 1.
        marker_size (int, optional): Size of the markers. Defaults to 10.

    Returns:
        fig (plotly.graph_objs._figure.Figure): A Plotly figure object containing the plot.
    """

    if eos_name != None:
        eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = (
            fit_to_eos(
                volumes,
                energies,
                eos_name=eos_name,
                volume_min=volume_min,
                volume_max=volume_max,
                num_volumes=num_volumes,
            )
        )

    unique_configs = [name]
    config_colors = assign_colors_to_configs(
        unique_configs, alpha=marker_alpha, cmap=cmap
    )
    config_symbols = assign_marker_symbols_to_configs(unique_configs)

    # First, plot the energy vs volume data points.
    fig = go.Figure()
    fig.update_layout(
        font=dict(
            family="Devaju Sans",
        )
    )

    if isinstance(per_atom, bool):
        x = volumes
        y = energies

        if per_atom:
            x = x / number_of_atoms
            y = y / number_of_atoms
    else:
        raise ValueError("per_atom must be True or False")

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=config_colors[name],
                symbol=config_symbols[name],
            ),
            legendgroup=name,
            name=name,
            showlegend=True,
        )
    )

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
                text=f"Energy (eV{atom_suffix})", font=dict(size=22, color="rgb(0,0,0)")
            )
        )

    # Second, plot the EOS fit.
    if eos_name != None:

        x = volume_range
        y = energy_eos
        if per_atom:
            num_atoms = number_of_atoms

            x = x / num_atoms
            y = y / num_atoms

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{eos_name} fit",
                line=dict(width=1.75, color=config_colors[name]),
                legendgroup=name,
                showlegend=False,
            )
        )

        if highlight_minimum == True:

            x = eos_parameters[0]
            y = eos_parameters[1]

            if per_atom:
                num_atoms = number_of_atoms
                x = x / num_atoms
                y = y / num_atoms

            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    name=f"minimum",
                    marker=dict(color="black", size=marker_size, symbol="cross"),
                    legendgroup=name,
                    showlegend=False,
                )
            )

        elif highlight_minimum == False:
            pass

        else:
            raise ValueError("highlight_minimum must be True or False")

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

    if title != None:
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(color="rgb(0,0,0)", size=30))
        )

    if show_fig:
        fig.show()

    return fig
