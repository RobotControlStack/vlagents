#!/usr/bin/env python3
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt

def load_chunk_csv(path: str) -> np.ndarray:
    # CSV format: t,d0,...,d7
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[:, 1:]  # (T,8)

def plot_8dims(A: np.ndarray, out_png: str, title: str = None, shade_tail: int = 10):
    T, D = A.shape
    assert D == 8, f"Expected 8 dims, got {D}"
    t = np.arange(T)

    fig, axes = plt.subplots(8, 1, figsize=(12, 10), sharex=True)
    for d in range(8):
        ax = axes[d]
        ax.plot(t, A[:, d], linewidth=1)
        if shade_tail > 0 and T > shade_tail:
            ax.axvspan(T - shade_tail - 0.5, T - 0.5, alpha=0.12)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(f"d{d}")
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

    for p in csvs:
        A = load_chunk_csv(p)
        base = os.path.splitext(os.path.basename(p))[0]
        out_png = os.path.join(outdir, base + "_8dims.png")
        plot_8dims(A, out_png, title=base)

if __name__ == "__main__":
    # Usage:
    #   python tools/chunk_plot_dims.py dbg_chunks
    #   python tools/chunk_plot_dims.py dbg_chunks/20251023-153012_c001.csv
    target = sys.argv[1] if len(sys.argv) > 1 else "dbg_chunks"
    main(target)

