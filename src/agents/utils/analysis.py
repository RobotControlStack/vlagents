#!/usr/bin/env python3
import os, sys, numpy as np
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

def main(root="dbg_chunks", csv_path=None):
    if csv_path is None:
        csv_path = _latest_csv(root)
    A = _load_csv(csv_path)
    _print_stats(A)
    _plot_series(A, os.path.join(root, "chunk_full.png"), "Chunk (all dims)")
    _plot_series(np.diff(A, axis=0), os.path.join(root, "chunk_diff.png"), "Δ Chunk (per-step change)")
    _plot_series(A[-10:], os.path.join(root, "chunk_tail.png"), "Tail (last 10 steps)")

if __name__ == "__main__":
    # usage: python tools/chunk_quicklook.py [root] [optional_csv]
    root = sys.argv[1] if len(sys.argv) > 1 else "dbg_chunks"
    csv = sys.argv[2] if len(sys.argv) > 2 else None
    main(root, csv)
