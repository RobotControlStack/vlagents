#!/usr/bin/env python3
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

def load_chunk_csv(path: str) -> np.ndarray:
    # CSV format: t,d0,...,d7
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[:, 1:]  # (T,8)

def plot_8dims(A: np.ndarray, out_png: str, title: str = None, s=25, d=10, shade ="tail", first=False):
    T, D = A.shape

    assert D == 8, f"Expected 8 dims, got {D}"
    t = np.arange(T)

    fig, axes = plt.subplots(8, 1, figsize=(12, 10), sharex=True)
    for dim in range(8):
        ax = axes[dim]
        ax.plot(t, A[:, dim], linewidth=1)
        # inside plot_8dims(...):
        if shade == "tail":
            if first:
                ax.axvspan(s - 0.5, s + d - 0.5, alpha=0.12)
            else:
                # overlap tail for k≥2 within this chunk: [d+s, d+s+d)
                ax.axvspan(d + s - 0.5, d + s + d - 0.5, alpha=0.12)
        else:  # "head"
            ax.axvspan(0 - 0.5, d - 0.5, alpha=0.12)  # optional: start at -0.5 for symmetry

        ax.grid(True, alpha=0.3)
        ax.set_ylabel(f"d{dim}")
    axes[-1].set_xlabel("step")
    if title:
        fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")

def main(root_or_csv: str):
    if root_or_csv.endswith(".csv") and os.path.isfile(root_or_csv):
        csvs = [root_or_csv]
        outdir = os.path.dirname(root_or_csv) or "."
    else:
        outdir = root_or_csv
        csvs = sorted(glob.glob(os.path.join(root_or_csv, "*.csv")))
        if not csvs:
            # try pointer
            ptr = os.path.join(root_or_csv, "chunk_latest_path.txt")
            if os.path.exists(ptr):
                with open(ptr) as f:
                    p = f.read().strip()
                if os.path.exists(p):
                    csvs = [p]
    if not csvs:
        print(f"No CSVs found in '{root_or_csv}'")
        return

    for i in range(len(csvs)-1):
        p = csvs[i]
        c = csvs[i+1]
        prev = load_chunk_csv(p)
        curr = load_chunk_csv(c)
        # inside main(), compute distinct bases and filenames
        prev_base = os.path.splitext(os.path.basename(p))[0]
        curr_base = os.path.splitext(os.path.basename(c))[0]
        prev_out_png = os.path.join(outdir, prev_base + "_tail_8dims.png")
        curr_out_png = os.path.join(outdir, curr_base + "_head_8dims.png")

        if i == 0:
            plot_8dims(prev, prev_out_png, title=prev_base, shade='tail', first=True)
            plot_8dims(curr, curr_out_png, title=curr_base, shade='head')
        else:
            plot_8dims(prev, prev_out_png, title=prev_base, shade='tail')
            plot_8dims(curr, curr_out_png, title=curr_base, shade='head')


if __name__ == "__main__":
    # Usage:
    #   python tools/chunk_plot_dims.py dbg_chunks
    #   python tools/chunk_plot_dims.py dbg_chunks/20251023-153012_c001.csv
    target = sys.argv[1] if len(sys.argv) > 1 else "dbg_chunks"
    main(target)

