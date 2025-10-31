#!/usr/bin/env python3
import os, sys, glob, numpy as np
import matplotlib.pyplot as plt

def _latest_csv(root: str) -> str:
    ptr = os.path.join(root, "chunk_latest_path.txt")
    if not os.path.exists(ptr):
        raise FileNotFoundError(f"Pointer not found: {ptr}")
    with open(ptr) as f:
        p = f.read().strip()
    if not os.path.exists(p):
        raise FileNotFoundError(f"CSV path from pointer missing: {p}")
    return p

def _load_csv(path: str) -> np.ndarray:
    # format: t,d0,...,d7
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 1:]  # drop t

def _print_stats(A: np.ndarray):
    dA = np.diff(A, axis=0)
    ddA = np.diff(dA, axis=0)
    print("[chunk] shape:", A.shape)
    print("[chunk] min per dim:", A.min(0))
    print("[chunk] max per dim:", A.max(0))
    print("[chunk] mean per dim:", A.mean(0))
    print("[Δ] max|Δ| per dim:", np.abs(dA).max(0))
    print("[Δ²] max|Δ²| per dim:", np.abs(ddA).max(0))
    # tail instability scores
    H = A.shape[0]
    head = A[:max(5, H//2)]
    tail = A[-10:]
    def _stable_var(x): return np.var(x, axis=0) + 1e-9
    score = _stable_var(tail) / _stable_var(head)
    print("[tail/head var ratio] per dim:", score)

def _plot_series(A: np.ndarray, out: str, title: str):
    T, D = A.shape
    t = np.arange(T)
    plt.figure(figsize=(10,5))
    for d in range(D):
        plt.plot(t, A[:, d], label=f"d{d}")
    plt.title(title); plt.xlabel("step"); plt.ylabel("value"); plt.grid(True)
    plt.legend(ncol=4, fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[plot] {out}")

def main(root_or_csv="dbg_chunks", csv_path=None):
    print(f"[analyze] root_or_csv='{root_or_csv}' csv_path='{csv_path}'")
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

    for i,csv_path in enumerate(csvs):
        print(f"\n=== Analyzing chunk {i+1}/{len(csvs)}: {csv_path} ===")
        A = _load_csv(csv_path)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        full_png = os.path.join(outdir, base + "_chunk_full.png")
        diff_png = os.path.join(outdir, base + "_chunk_diff.png")
        tail_png = os.path.join(outdir, base + "_chunk_tail.png")

        _print_stats(A)
        _plot_series(A, full_png, "Chunk (all dims)")
        _plot_series(np.diff(A, axis=0), diff_png, "Δ Chunk (per-step change)")
        _plot_series(A[-10:], tail_png, "Tail (last 10 steps)")

    # if csv_path is None:
    #     csv_path = _latest_csv(root)
    # A = _load_csv(csv_path)
    # _print_stats(A)
    # _plot_series(A, os.path.join(root, "chunk_full.png"), "Chunk (all dims)")
    # _plot_series(np.diff(A, axis=0), os.path.join(root, "chunk_diff.png"), "Δ Chunk (per-step change)")
    # _plot_series(A[-10:], os.path.join(root, "chunk_tail.png"), "Tail (last 10 steps)")

if __name__ == "__main__":
    # usage: python tools/chunk_quicklook.py [root] [optional_csv]
    root = sys.argv[1] if len(sys.argv) > 1 else "dbg_chunks"
    csv = sys.argv[2] if len(sys.argv) > 2 else None
    main(root, csv)
