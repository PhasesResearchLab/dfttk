"""
Plotting formats using plotly.   
"""

# Related third party imports
import plotly.graph_objects as go


def plot_format(
    fig: go.Figure, xtitle: str, ytitle: str, width: int = 840, height: int = 600
):
    """Plot format using plotly

    Args:
        fig (go.Figure): plotly figure
        xtitle (str): title of x-axis
        ytitle (str): title of y-axis
        width (int, optional): plot width. Defaults to 840.
        height (int, optional): plot height. Defaults to 600.
    """

    fig.update_layout(
        font=dict(
            family="Devaju Sans",
        )
    )
    fig.update_xaxes(
        title=dict(
            text=xtitle,
            font=dict(size=22, color="rgb(0,0,0)"),
        )
    )
    fig.update_yaxes(title=dict(text=ytitle, font=dict(size=22, color="rgb(0,0,0)")))
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
        width=width,
        height=height,
        legend=dict(font=dict(size=20, color="black")),
        xaxis=axis_params,
        yaxis=axis_params,
    )
