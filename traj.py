"""Plot trajectories from experiment_goal_*_positions.csv files.

Expected CSV schema (semicolon-delimited):
trial;repeat;measurement;timestamp_s;x;y;theta
"""

from argparse import ArgumentParser
from glob import glob
import math
import os

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "trial",
    "repeat",
    "measurement",
    "timestamp_s",
    "x",
    "y",
    "theta",
}


def _extract_goal_index(path):
    basename = os.path.basename(path)
    # experiment_goal_<N>_positions.csv -> N
    parts = basename.split("_")
    if len(parts) >= 4 and parts[0] == "experiment" and parts[1] == "goal":
        return parts[2]
    return "?"


def _load_files(input_paths):
    data = []
    for path in input_paths:
        df = pd.read_csv(path, sep=";")
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )
        data.append((path, df))
    return data


def _build_figure(files_with_data):
    n = len(files_with_data)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

    if not isinstance(axes, (list, tuple)):
        try:
            axes = axes.flatten()
        except AttributeError:
            axes = [axes]
    else:
        axes = list(axes)

    for ax_idx, (path, df) in enumerate(files_with_data):
        ax = axes[ax_idx]
        goal_idx = _extract_goal_index(path)

        grouped = df.sort_values("measurement").groupby(["trial", "repeat"])
        for (trial, repeat), run_df in grouped:
            label = f"trial {int(trial)} repeat {int(repeat)}"
            ax.plot(run_df["x"], run_df["y"], linewidth=1.2, alpha=0.9, label=label)
            ax.plot(run_df["x"].iloc[0], run_df["y"].iloc[0], "o", markersize=4)
            ax.plot(run_df["x"].iloc[-1], run_df["y"].iloc[-1], "x", markersize=6)

        ax.set_title(f"Goal {goal_idx}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(len(files_with_data), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Trajectories Per Goal", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="*",
        help="Optional explicit CSV files. Default: experiment_goal_*_positions.csv",
    )
    parser.add_argument(
        "--output",
        default="trajectories.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    input_paths = args.inputs or sorted(glob("experiment_goal_*_positions.csv"))
    if not input_paths:
        raise FileNotFoundError(
            "No input files found. Provide --inputs or place experiment_goal_*_positions.csv in CWD."
        )

    files_with_data = _load_files(input_paths)
    fig = _build_figure(files_with_data)
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()