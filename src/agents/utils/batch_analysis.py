#!/usr/bin/env python3
import os, glob, csv, numpy as np
import matplotlib.pyplot as plt

def load_csv(path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[:,1:]  # (T,D)

def stats_for_chunk(A: np.ndarray):
    T, D = A.shape
    dA = np.diff(A, axis=0)
    head = A[:max(5, T//2)]
    tail = A[-10:] if T >= 10 else A[-T:]
    def svar(x): return np.var(x, axis=0) + 1e-9
    tail_var_ratio = svar(tail) / svar(head)
    tail_jump_max = np.max(np.abs(dA[-10:,:]), axis=0) if T > 1 else np.zeros(D)
    # aggregate risk: emphasize big ratios without letting 1 dim dominate
    tail_outlier_score = float(np.mean(np.clip(tail_var_ratio, 0, 10)))
    return {
        "T": T,
        "D": D,
        "min": A.min(0),
        "max": A.max(0),
        "mean": A.mean(0),
        "tail_var_ratio": tail_var_ratio,
        "tail_jump_max": tail_jump_max,
        "score": tail_outlier_score,
    }

def plot_if_weird(A: np.ndarray, outbase: str, score: float, score_thresh=1.5):
    if score < score_thresh: 
        return
    t = np.arange(A.shape[0])
    # full
    plt.figure(figsize=(10,5))
    for d in range(A.shape[1]): plt.plot(t, A[:,d], label=f"d{d}")
    plt.title(f"Chunk (score={score:.2f})"); plt.xlabel("step"); plt.ylabel("value"); plt.grid(True)
    plt.legend(ncol=4, fontsize=8, frameon=False); plt.tight_layout()
    plt.savefig(outbase+"_full.png", dpi=150); plt.close()
    # diff
    dA = np.diff(A, axis=0)
    plt.figure(figsize=(10,5))
    for d in range(dA.shape[1]): plt.plot(np.arange(dA.shape[0]), dA[:,d], label=f"d{d}")
    plt.title("Δ per step"); plt.xlabel("step"); plt.ylabel("Δ"); plt.grid(True)
    plt.legend(ncol=4, fontsize=8, frameon=False); plt.tight_layout()
    plt.savefig(outbase+"_diff.png", dpi=150); plt.close()
    # tail
    tail = A[-10:] if A.shape[0] >= 10 else A
    plt.figure(figsize=(10,5))
    for d in range(tail.shape[1]): plt.plot(np.arange(tail.shape[0]), tail[:,d], label=f"d{d}")
    plt.title("Tail (last 10)"); plt.xlabel("tail step"); plt.ylabel("value"); plt.grid(True)
    plt.legend(ncol=4, fontsize=8, frameon=False); plt.tight_layout()
    plt.savefig(outbase+"_tail.png", dpi=150); plt.close()

def main(root="dbg_chunks", score_thresh=1.5):
    os.makedirs(root, exist_ok=True)
    csvs = sorted(glob.glob(os.path.join(root, "*.csv")))
    if not csvs:
        print(f"No CSVs in {root}")
        return
    rows = []
    for p in csvs:
        A = load_csv(p)
        S = stats_for_chunk(A)
        rows.append({
            "path": p,
            "T": S["T"],
            "D": S["D"],
            "score": S["score"],
            "tail_var_ratio_mean": float(np.mean(S["tail_var_ratio"])),
            "tail_var_ratio_max": float(np.max(S["tail_var_ratio"])),
            "tail_jump_max_max": float(np.max(S["tail_jump_max"])),
        })
        base = os.path.splitext(p)[0]
        plot_if_weird(A, base, S["score"], score_thresh)

    # write summary
    summ_path = os.path.join(root, "summary.csv")
    with open(summ_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[summary] {summ_path}")

    # top risk list
    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    top_path = os.path.join(root, "top_tail_risk.csv")
    with open(top_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows_sorted[:20])
    print(f"[top] {top_path}")

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "dbg_chunks"
    score_thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    main(root, score_thresh)
