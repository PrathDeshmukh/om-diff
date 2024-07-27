
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import src.plots


def parse_cmd():
    parser = argparse.ArgumentParser(
        description="Script for error of denoiser as a function of diffusion time index."
    )
    parser.add_argument(
        "--dirname",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--name",
        type=str,
        nargs="+",
    )

    return parser.parse_args()


def load_evaluations(names, paths):
    dfs = {}
    for name, dir_path in zip(names, paths):
        path = os.path.join(dir_path, "evaluation.csv")
        df = pd.read_csv(path)
        dfs[name] = df

    return dfs


if __name__ == "__main__":
    cmd_args = parse_cmd()

    dfs = load_evaluations(cmd_args.name, cmd_args.dirname)

    with mpl.rc_context(src.plots.PLOT_CONTEXT):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(2 * 3.5, 2 * 2.625), sharex="col")

        # ax_in = axs[1, 1].inset_axes([0.05, 0.05, 0.45, 0.45])

        for name in dfs:
            df = dfs[name]
            time_means = df.groupby("t").median()
            print(name, time_means)

            ts = time_means.index.to_numpy()[:-1]

            ax = axs[0, 0]
            ax.set_ylabel("MSE $\epsilon_h$")
            ax.plot(
                ts,
                time_means["eps_node_features"].to_numpy()[:-1],
                "--",
                linewidth=0.75,
                markersize=1.5,
                label=name,
            )

            ax = axs[1, 0]
            ax.set_ylabel("MSE $h$")
            ax.plot(
                ts,
                time_means["x_node_features"].to_numpy()[:-1],
                "--",
                linewidth=0.75,
                markersize=1.5,
                label=name,
            )
            ax.set_yscale("log")

            ax = axs[0, 1]
            ax.set_ylabel("MSE $\epsilon_x$")
            ax.plot(
                ts,
                time_means["eps_node_positions"].to_numpy()[:-1],
                "--",
                linewidth=0.75,
                markersize=1.5,
                label=name,
            )
            ax = axs[1, 1]
            ax.set_ylabel("MSE $x$")
            ax.plot(
                ts,
                time_means["x_node_positions"].to_numpy()[:-1],
                "--",
                linewidth=0.75,
                markersize=1.5,
                label=name,
            )
            ax.set_yscale("log")

            # ax_in.plot(
            #    ts,
            #    time_means["x_node_positions"].to_numpy()[:-1],
            #    "--",
            #   linewidth=0.75,
            #    markersize=1.5,
            #    label=name,
            # )

        axs[1, 0].invert_xaxis()
        axs[1, 0].set_xlabel("diffusion time step")

        axs[1, 1].invert_xaxis()
        axs[1, 1].set_xlabel("diffusion time step")

        # ax_in.set_xlim(0, 400)
        # ax_in.set_yscale("log")
        # ax_in.set_ylim(0, 1)
        # ax_in.invert_xaxis()

        # ax_in.set_xticks([])
        # ax_in.set_yticks([])
        # ax_in.tick_params(axis="x", labelbottom=False)
        # ax_in.tick_params(axis="y", labelbottom=False)

        # axs[1, 1].indicate_inset_zoom(ax_in, edgecolor="black")

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(0.15, 1.0, 0.7, 0.04),
            mode="expand",
            loc="lower left",
            borderaxespad=0,
            ncol=4,
            frameon=False,
        )

        fig.tight_layout()
        fig.savefig("plots/denoiser_performance.pdf")
