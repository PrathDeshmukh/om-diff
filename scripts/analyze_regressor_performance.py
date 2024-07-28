"""

Script producing table for report.

"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

MCS = {77: "Ir"}


def parse_cmd():
    parser = argparse.ArgumentParser(
        description="Script for plotting geometry statistics as a function of training progress."
    )
    parser.add_argument(
        "--fold_path",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--min_target_value",
        type=float,
        default=-32.1,
    )
    parser.add_argument(
        "--max_target_value",
        type=float,
        default=-23.0,
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


def print_latex(latex_dict):
    col_names = list(latex_dict.keys())
    row_names = list(latex_dict[col_names[0]].keys())

    latex = "\\toprule \n & " + " & ".join(col_names) + "\\\\ \midrule \n"
    for row_name in row_names:
        latex += (
            f"{row_name} & "
            + " & ".join(
                [
                    f"${np.mean(latex_dict[col_name][row_name]):2.3f}\pm{np.std(latex_dict[col_name][row_name]):2.3f}$"
                    for col_name in col_names
                ]
            )
            + "\\\\ \n"
        )
    latex += "\\bottomrule"
    print(latex)


def mae(inputs, targets):
    return np.mean(np.abs(inputs - targets))


def rmse(inputs, targets):
    return np.sqrt(np.mean(np.square(inputs - targets)))


def maxe(inputs, targets):
    return np.max(np.abs(inputs - targets))


def r2(inputs, targets):
    return r2_score(targets, inputs)


if __name__ == "__main__":
    cmd_args = parse_cmd()

    df = load_evaluations(cmd_args.fold_path)

    key = "pred"

    for metric_name, metric in zip(
        ["MAE", "RMSE", "Max error", "R-squared"], [mae, rmse, maxe, r2]
    ):
        agg_results = {
            "GNN": {mc: [] for mc in ["All"] + list(MCS.values())},
        }

        for fold_idx in range(df["fold_idx"].max() + 1):
            mask_fold = df["fold_idx"] == fold_idx

            agg_results["GNN"]["All"].append(
                metric(df.loc[mask_fold, key], df.loc[mask_fold, "target"])
            )

            for mc in MCS:
                mask_mc = df["mc"] == mc

                agg_results["GNN"][MCS[mc]].append(
                    metric(df.loc[mask_fold & mask_mc, key], df.loc[mask_fold & mask_mc, "target"])
                )

        print(metric_name)
        print_latex(agg_results)
        print()

    print("<Error in range of interest>")
    mask_range = (df["target"] >= cmd_args.min_target_value) & (
        df["target"] <= cmd_args.max_target_value
    )
    for metric_name, metric in zip(
        ["MAE", "RMSE", "Max error", "R-squared"], [mae, rmse, maxe, r2]
    ):
        agg_results = {
            "GNN": {mc: [] for mc in ["All"] + list(MCS.values())},
        }

        agg_results["GNN"]["All"].append(
            metric(df.loc[mask_range, key], df.loc[mask_range, "target"])
        )

        for mc in MCS:
            mask_mc = df["mc"] == mc

            agg_results["GNN"][MCS[mc]].append(
                metric(
                    df.loc[mask_mc & mask_range, key],
                    df.loc[mask_mc & mask_range, "target"],
                )
            )

        print(metric_name)
        print_latex(agg_results)
        print()