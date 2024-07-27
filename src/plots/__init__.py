from typing import Union, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_hist_density(
    x: dict[str, Union[np.ndarray, Iterable, int, float]],
    xlabel: str,
    ylabel: str,
    plot_context: dict = None,
    ax: Axes = None,
    xmin: float = 1.0,
    xmax: float = 5.0,
):
    colors = ["r", "b", "g", "#0C5DA5", "#FF2C00", "#474747", "#9e9e9e"]
    if plot_context is None:
        plot_context = PLOT_CONTEXT
    with mpl.rc_context(plot_context):
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        else:
            fig = ax.get_figure()

        xmin, xmax = max(xmin, min([min(x[label]) for label in x])), min(
            xmax, max([max(x[label]) for label in x])
        )
        bins = np.linspace(xmin, xmax, 100 + 1)
        for i, label in enumerate(x):
            binned_value = np.histogram(x[label], bins, density=True)[0]
            ax.plot(bins[1:], binned_value, c=colors[i], label=label, linewidth=1.0, linestyle="-")

        ax.legend(loc="best", fontsize="small", frameon=False)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=xmin, right=xmax)
        fig.tight_layout()
    return fig, ax


def plot_stacked_bar(
    xdict,  # xlabel : [values to stack vertically]
    ylabel: str,
    labels: list,
    plot_context: dict = None,
):
    if plot_context is None:
        plot_context = PLOT_CONTEXT

    import matplotlib as mpl

    norm = mpl.colors.Normalize(vmin=0, vmax=5)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])

    with mpl.rc_context(plot_context):
        fig, ax = plt.subplots(figsize=(3.2, 4.2))
        x = list(xdict.keys())
        prev_y = np.array([0.0 for _ in x])
        width = 0.5
        for i, label in enumerate(labels):
            y = np.array([xdict[_x][i] for _x in xdict])
            ax.bar(x, y, width, bottom=prev_y, label=rf"{label}", color=cmap.to_rgba(i + 1))
            prev_y += y
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", frameon=False, ncol=3, fontsize="small")
    plt.tight_layout()
    return fig, ax


def plot_multi_bar(
    xticks,
    y: dict[str, list],
    xlabel: str,
    ylabel: str,
    plot_context: dict = None,
):
    from matplotlib.colors import to_rgb

    colors = ["g", "b", "r", "#0C5DA5", "#FF2C00", "#474747", "#9e9e9e"]
    if plot_context is None:
        plot_context = PLOT_CONTEXT
    with mpl.rc_context(plot_context):
        fig, ax = plt.subplots(figsize=(4.2, 2.1))
        x = np.arange(len(xticks))
        width = 0.9 / len(y)
        for i, label in enumerate(y, start=1):
            ax.bar(
                x + (i - len(y)) * width / 2,
                y[label],
                width,
                label=label,
                color=to_rgb(colors[i - 1]),
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(
            x,
        )
        ax.set_xticklabels(xticks, fontsize=10, rotation=45)
        ax.legend(frameon=False)
    plt.tight_layout()
    return fig, ax


PLOT_CONTEXT = {
    # Matplotlib style for scientific plotting
    # This is the base style for "SciencePlots"
    # see: https://github.com/garrettj403/SciencePlots
    # Set color cycle: blue, green, yellow, red, violet, gray
    "axes.prop_cycle": "cycler('ls', ['-','--', '-.', ':']) * cycler('color', ['0C5DA5', '00B945', 'FF9500', 'FF2C00', '845B97', '474747', '9e9e9e'])",
    # Set default figure size
    "figure.figsize": "3.5, 2.625",
    # Set x axis
    "xtick.direction": "in",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "xtick.top": True,
    # Set y axis
    "ytick.direction": "in",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": True,
    "ytick.right": True,
    # Set line widths
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    # Remove legend frame
    "legend.frameon": True,
    # Always save as 'tight'
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # Use serif fonts
    "font.family": "sans-serif",
    # Use LaTeX for math formatting
    "text.usetex": False,
    # "text.latex.preamble": "\\usepackage{amsmath, amssymb, sfmath}",
    # Grid lines
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.linestyle": "--",
    "grid.color": "k",
    "grid.alpha": 0.25,
    # Legend
    "legend.framealpha": 1.0,
    "legend.fancybox": False,
    "legend.numpoints": 1,
}