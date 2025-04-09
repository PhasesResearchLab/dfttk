"""
Module for fitting equations of state (EOS) to energy-volume data and plotting the results.
"""

# Related third party imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from distinctipy import get_colors
from scipy.optimize import fsolve, curve_fit
from collections import namedtuple

# DFTTK imports
from dfttk.eos_functions import (
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
) -> tuple[pd.DataFrame, pd.DataFrame]:

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
    unique_configs, alpha: float = 1, cmap: str = "plotly"
) -> dict:

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


def assign_marker_symbols_to_configs(unique_configs):

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
    name,
    number_of_atoms,
    volumes,
    energies,
    volume_min=None,
    volume_max=None,
    num_volumes=None,
    eos_name="BM4",
    highlight_minimum=True,
    per_atom=False,
    title=None,
    show_fig=True,
    cmap="plotly",
    marker_alpha=1,
    marker_size=10,
):
    """Plot the energy vs volume curves for each configuration.

    Args:
        data (pandas.DataFrame, list of pandas.DataFrame, or list of str): Data must be a pandas
        DataFrame or a list of pandas DataFrames.
        eos_name (str, optional): EOS name. Defaults to "BM4".
        highlight_minimum (bool, optional): Defaults to True.
        per_atom (bool, optional):Defaults to False.
        title (_type_, optional): Defaults to None.
        show_fig (bool, optional): Defaults to True.
        left_col (str, optional): Defaults to "volume".
        right_col (str, optional): Defaults to "energy".
        cmap (str, optional): Defaults to 'plotly'.
        marker_alpha (int, optional): Defaults to 1.
        marker_size (int, optional): Defaults to 10.

    Returns:
        fig (plotly.graph_objs._figure.Figure): A Plotly figure.
    """

    if eos_name != None:
        eos_constants, eos_parameters, volume_range, energy_eos, pressure_eos = (
            fit_to_eos(volumes, energies, eos_name=eos_name, volume_min=volume_min, volume_max=volume_max, num_volumes=num_volumes)
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
