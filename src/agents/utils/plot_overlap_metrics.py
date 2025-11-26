#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot MSE and mean_cos over RTC overlap metric files."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Path to directory containing *overlap_metrics.json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="overlap_metrics.png",
        help="Output image filename (PNG, PDF, etc.). Default: overlap_metrics.png",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {log_dir}")

    files = sorted(log_dir.glob("*overlap_metrics.json"))

    if not files:
        raise RuntimeError(f"No *overlap_metrics.json files found in {log_dir}")

    mse_vals = []
    cos_vals = []

    for f in files:
        with f.open("r") as fp:
            data = json.load(fp)

        rmse = data["rmse"]
        mse = rmse ** 2

        mse_vals.append(mse)
        cos_vals.append(data["mean_cos"])

    x = list(range(len(mse_vals)))  # chunk index

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left axis: MSE
    ax1.plot(x, mse_vals, marker="o", linestyle="-", color="tab:blue", label="MSE")
    ax1.set_xlabel("RTC chunk index")
    ax1.set_ylabel("MSE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # X ticks: maybe fewer if many chunks
    step = max(len(x) // 10, 1)
    ax1.set_xticks(x[::step])

    # Right axis: mean_cos
    ax2 = ax1.twinx()
    ax2.plot(x, cos_vals, marker="x", linestyle="-", color="tab:orange", label="mean_cos")
    ax2.set_ylabel("mean_cos", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Grid on main axis
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = log_dir / output_path

    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
