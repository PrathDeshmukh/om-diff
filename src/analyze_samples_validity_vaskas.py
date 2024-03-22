import argparse

import ase.data
import numpy as np


def parse_cmd():
  parser = argparse.ArgumentParser(
    description="Script for plotting geometry statistics as a function of training progress."
  )
  parser.add_argument(
    "--samples_path",
    type=str,
    nargs="+",
    default=["/Users/frjc/phd/projects/diffusion4md/data/vaskas.db"],
  )
  parser.add_argument("--name", type=str, nargs="+", default=["Dataset"])
  
  return parser.parse_args()

if __name__ == "__main__":
    cmd_args = parse_cmd()

    mc_filter = OneMetalCenterFilter()

    filters = {
        "$d_{ij} \in [d_{\min}, d_{\max}]$": DistanceFilter(min_distance=0.8, max_distance=50.0),
        "RDKit w. bond inf.": RDKitFilter(),
    }

    MCs = [77]

    xs, filtered, mcs = {}, {}, {}
    for name, filename in zip(cmd_args.name, cmd_args.samples_path):
        f_acc, _mcs = mc_filter(filename=filename, return_mcs=True)

        mcs[name] = {ase.data.chemical_symbols[mc]: np.mean(_mcs == mc) for mc in MCs}

        filtered[name] = {"Exactly one MC": np.mean(f_acc)}
        for filter_name in filters:
            f_acc = filters[filter_name](valid_so_far=f_acc, filename=filename)
            filtered[name][f"{filter_name} (All)"] = np.mean(f_acc)
            for mc in MCs:
                filtered[name][f"{filter_name} ({ase.data.chemical_symbols[mc]})"] = np.mean(
                    f_acc[_mcs == mc]
                )