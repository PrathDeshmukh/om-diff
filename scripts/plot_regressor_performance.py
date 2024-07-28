"""

Script producing table for report.

"""
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.plots

MCS = {77: "Ir"}

DUMMY_PRED = {77: 11.97}


def get_bins(x):
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    binwidth = 2 * iqr / len(x) ** (1 / 3)
    return np.arange(min(x), max(x) + binwidth, binwidth)


def parse_cmd():
    parser = argparse.ArgumentParser(
        description="Script for plotting geometry statistics as a function of training progress."
    )
    parser.add_argument(
        "--fold_path",
        type=str,
        nargs="+",
    )

    return parser.parse_args()


def load_evaluations(paths):
    dfs = []
    for dir_path in paths:
        path = os.path.join(dir_path, "evaluations.csv")
        df = pd.read_csv(path)
        dfs.append(df)
    concat_df = pd.concat(dfs, axis=0)

    return concat_df


if __name__ == "__main__":
    cmd_args = parse_cmd()

    df = load_evaluations(cmd_args.fold_path)

    key = "pred"

    with mpl.rc_context(src.plots.PLOT_CONTEXT):
        xlabel = "Residual [kcal/mol]"
        ylabel = "# Complexes"
        fig, axs = plt.subplots(
            nrows=len(MCS) + 1,
            ncols=1,
            # sharex="col",
            # sharey="row",
            figsize=(2.625, len(MCS) * 2.625),
            gridspec_kw={"hspace": 0.15},
        )

        gt = np.array(df["target"])
        y = np.array(df[key])

        residuals = gt - y

        axs[0].hist(residuals, bins=get_bins(residuals))
        axs[0].set_ylabel(ylabel)

        for i, mc in enumerate(MCS, start=1):
            ax = axs[i]
            ax.set_ylabel(ylabel)
            ax.text(
                0.1,
                0.9,
                MCS[mc],
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

            mask_mc = df["mc"] == mc

            gt = np.array(df.loc[mask_mc, "target"])
            y = np.array(df.loc[mask_mc, key])

            residuals = gt - y

            axs[i].hist(residuals, bins=get_bins(residuals))
        axs[-1].set_xlabel(xlabel)
        fig.tight_layout()
        fig.savefig("regressor_performance_residuals.pdf")

        xlabel = "True [kcal/mol]"
        ylabel = "Predicted [kcal/mol]"
        fig, axs = plt.subplots(
            nrows=len(MCS),
            ncols=1,
            # sharex="col",
            # sharey="row",
            figsize=(2.625, len(MCS) * 2.625),
            gridspec_kw={"hspace": 0.15},
        )
        min_xx, max_xx = 0, 0
        for i, mc in enumerate(MCS):
            ax = axs[i]
            ax.set_ylabel(ylabel)
            ax.text(
                0.1,
                0.9,
                MCS[mc],
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.margins(0.02)

            mask_mc = df["mc"] == mc

            x = np.array(df.loc[mask_mc, "target"])
            y = np.array(df.loc[mask_mc, key])

            min_x = np.min(x)
            max_x = np.max(x)

            min_xx = np.minimum(min_x, np.min(y))
            max_xx = np.maximum(max_x, np.max(y))

            ax.scatter(x, y, s=0.75)
            xx_line = np.linspace(min_xx, max_xx, 1000)
            ax.plot(xx_line, xx_line, "--k", lw=0.75)
            ax.set_xticks(ax.get_yticks())
            # ax.set_xlim(ax.get_ylim())
            ax.set_aspect("equal")
            if min_xx < -32.1:
                ax.axvspan(min_xx, -32.1, alpha=0.2, color="grey")
                ax.axhspan(min_xx, -32.1, alpha=0.2, color="grey")
            if max_xx > -23.0:
                ax.axvspan(np.maximum(min_xx, -23.0), max_xx, alpha=0.2, color="grey")
                ax.axhspan(np.maximum(min_xx, -23.0), max_xx, alpha=0.2, color="grey")

        axs[-1].set_xlabel(xlabel)
        fig.tight_layout()
        fig.savefig("regressor_performance_xx.pdf")

        xlabel = "Binding Energy [kcal/mol]"
        ylabel = "Absolute Error [kcal/mol]"
        fig, axs = plt.subplots(
            nrows=2,
            ncols=len(MCS),
            sharex="col",
            sharey="row",
            gridspec_kw=dict(height_ratios=(4, 1), bottom=0.1, top=0.9, hspace=0.0, wspace=0.05),
            figsize=(len(MCS) * 3.5, 3.3),
        )

        for i, mc in enumerate(MCS):
            ax_hist = axs[1, i]

            if i == 0:
                ax_hist.set_ylabel("Count [-]")
            ax_hist.set_xlabel(xlabel)
            ax_hist.grid(False)

            ax = axs[0, i]
            if i == 0:
                ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", labelbottom=False)
            ax.text(
                0.1,
                0.9,
                MCS[mc],
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

            mask_mc = df["mc"] == mc

            x = np.array(df.loc[mask_mc, "target"])
            y = np.abs(np.array(df.loc[mask_mc, key]) - x)
            y_dummy = np.abs(x - DUMMY_PRED[mc])

            sorted_idx = np.argsort(x)
            bins_idx = np.array_split(sorted_idx, 100)
            min_xx, max_xx = np.min(x), np.max(x)
            bins = np.linspace(min_xx, max_xx, num=int((max_xx - min_xx) / 2.0))

            x_bins = np.digitize(x, bins=bins)
            y_binned = [[] for _ in range(np.max(x_bins) + 1)]
            for i, bin in enumerate(x_bins):
                y_binned[bin].append(y[i])

            current_bin = []
            fixed_y_binned = []
            fixed_bins = []
            for bin, y_bin in zip(bins, y_binned):
                current_bin.extend(y_bin)
                if len(current_bin) > 3:
                    fixed_y_binned.append(current_bin)
                    fixed_bins.append(bin)
                    current_bin = []

            if len(current_bin):
                fixed_y_binned[-1].extend(current_bin)

            # x_binned = (bins[1:] + bins[:-1]) / 2
            y_mean_binned = np.array([np.mean(y_bin) for y_bin in fixed_y_binned])
            y_std_binned = np.array([np.std(y_bin) for y_bin in fixed_y_binned])

            # y_mean_binned = np.array([np.mean([y[idx] for idx in bin_idx]) for bin_idx in bins_idx])
            # y_std_binned = np.array([np.std([y[idx] for idx in bin_idx]) for bin_idx in bins_idx])

            # y_dummy_binned = [np.mean([y_dummy[idx] for idx in bin_idx]) for bin_idx in bins_idx]
            x_binned = [np.mean([x[idx] for idx in bin_idx]) for bin_idx in bins_idx]
            x_binned = fixed_bins

            # ax.plot(x_binned, y_dummy_binned, linewidth=0.5, c="r")
            ax.plot(x_binned, y_mean_binned, "tab:red", drawstyle="steps-mid", label="steps-mid")
            ax.fill_between(
                x_binned,
                y_mean_binned - y_std_binned,
                y_mean_binned + y_std_binned,
                color="tab:red",
                step="mid",
                alpha=0.1,
            )

            x_min = np.min(x_binned)
            x_max = np.max(x_binned)
            rng = x_max - x_min
            x_min = x_min - (80.0 - rng) / 2
            x_max = x_max + (80.0 - rng) / 2
            ax.set_xlim(x_min, x_max)

            if x_min < -32.1:
                ax.axvspan(x_min, -32.1, alpha=0.1, color="grey")
                ax_hist.axvspan(x_min, -32.1, alpha=0.1, color="grey")
            if x_max > -23.0:
                ax.axvspan(-23.0, x_max, alpha=0.1, color="grey")
                ax_hist.axvspan(-23.0, x_max, alpha=0.1, color="grey")

            x_dummy = np.linspace(np.min(x_binned), np.max(x_binned), 1000)
            error_dummy = np.abs(DUMMY_PRED[mc] - x_dummy)
            ax.plot(x_dummy, error_dummy, linestyle="--", c="k", zorder=-1, linewidth=0.6)

            ax.set_ylim(0.0, 18)

            ax_hist.bar(x_binned, list(map(len, fixed_y_binned)), color="tab:red")

        # gs.tight_layout(fig)
        fig.savefig("regressor_performance.pdf")