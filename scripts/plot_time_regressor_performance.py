"""

Script producing table for report.

"""
import argparse
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
sys.path.append('/home/energy/s222491/om-diff/')
import src.plots


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
        path = os.path.join(dir_path, "evaluation.csv")
        df = pd.read_csv(path)
        dfs.append(df)
    concat_df = pd.concat(dfs, axis=0)

    return concat_df


if __name__ == "__main__":
    cmd_args = parse_cmd()

    df = load_evaluations(cmd_args.fold_path)

    key = "pred"
    color_error = "tab:blue"
    color_r2 = "tab:blue"

    ts = sorted(df["t"].unique())
    folds = sorted(df["fold_idx"].unique())

    with mpl.rc_context(src.plots.PLOT_CONTEXT):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 3.5), sharex="col")

        mean_errors, std_errors, r2_scores = [], [], []

        for t in ts:
            mask = df["t"] == t

            targets = df.loc[mask, "target"]
            inputs = df.loc[mask, "pred"]

            ae = np.abs(targets - inputs)
            mae = np.mean(ae)
            std_ae = np.std(ae)

            mean_errors.append(mae)
            std_errors.append(std_ae)
            r2_scores.append(r2_score(targets, inputs))

        mean_errors, std_errors, r2_scores = (
            np.array(mean_errors),
            np.array(std_errors),
            np.array(r2_scores),
        )

        ax = axs[0]
        ax.grid(False)

        ax.plot(ts, mean_errors, "o-", c=color_error, markersize=3.0, label="Ir")
        ax.fill_between(
            ts,
            (mean_errors - std_errors),
            (mean_errors + std_errors),
            alpha=0.1,
            color=color_error,
        )
        ax.set_ylabel("MAE [kcal/mol]")
        ax.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower left",
            mode="expand",
            borderaxespad=0,
            ncol=1,
            frameon=False,
        )

        # ax.yaxis.label.set_color(color_error)
        # ax.tick_params(axis="y", labelcolor=color_error)

        ax.set_ylim(0)

        ax = axs[1]
        ax.set_xlabel("diffusion time step")
        ax.plot(ts, r2_scores, "o-", c=color_r2, markersize=3.0)
        ax.invert_xaxis()

        ax.set_ylabel("$R^{2}$")
        # ax.yaxis.label.set_color(color_r2)
        # ax.tick_params(axis="y", labelcolor=color_r2)
        ax.set_ylim(0.0, 1)
        ax.grid(False)

        fig.tight_layout()
        fig.savefig("time_regressor_performance_vaskas.pdf")