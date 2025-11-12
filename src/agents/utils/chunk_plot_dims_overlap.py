#!/usr/bin/env python3
import os, sys, glob, json, argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------- IO ----------
def load_chunk_csv(path: str) -> np.ndarray:
    # CSV format: t,d0,...,d7
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    A = arr[:, 1:]  # (T,8)
    assert A.ndim == 2 and A.shape[1] == 8, f"Expected (T,8), got {A.shape} from {path}"
    return A

# ---------- plotting ----------
def plot_8dims(
    A: np.ndarray,
    out_png: str,
    title: str = None,
    s: int = 25,
    d: int = 10,
    shade: str = "tail",           # "tail" or "head"
    tail_phase: str = "later"      # "first" or "later"  (only used when shade=="tail")
):
    """
    Tail shading rule:
      - first  prev chunk: [s, s+d)
      - later  prev chunks: [d+s, d+s+d)
    Head shading (if used): [0, d)
    """
    T, D = A.shape
    t = np.arange(T)
    fig, axes = plt.subplots(8, 1, figsize=(12, 10), sharex=True)

    for dim in range(8):
        ax = axes[dim]
        ax.plot(t, A[:, dim], linewidth=1)

        if shade == "tail":
            if tail_phase == "first":
                start = s
            else:
                start = d + s
            end = start + d
            # clamp to timeline
            start_v = max(0, min(T, start)) - 0.5
            end_v   = max(0, min(T, end)) - 0.5
            if end_v > start_v:
                ax.axvspan(start_v, end_v, alpha=0.12)
        else:  # shade head [0, d)
            start_v = -0.5
            end_v   = max(0, min(T, d)) - 0.5
            if end_v > start_v:
                ax.axvspan(start_v, end_v, alpha=0.12)

        ax.grid(True, alpha=0.3)
        ax.set_ylabel(f"d{dim}")

    axes[-1].set_xlabel("step")
    if title:
        fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")

def plot_overlap_preview(prev_tail: np.ndarray, curr_head: np.ndarray, out_png: str, title: str):
    ov, D = prev_tail.shape
    t = np.arange(ov)
    fig, axes = plt.subplots(8, 1, figsize=(12, 10), sharex=True)
    for dim in range(8):
        ax = axes[dim]
        ax.plot(t, prev_tail[:, dim], linewidth=1, label="prev_tail")
        ax.plot(t, curr_head[:, dim], linewidth=1, linestyle="--", label="curr_head")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(f"d{dim}")
        if dim == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("overlap step (0..ov-1)")
    fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_png}")

# ---------- metrics ----------
def overlap_metrics(prev_tail: np.ndarray, curr_head: np.ndarray):
    diff = curr_head - prev_tail
    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    num = np.sum(curr_head * prev_tail, axis=-1)
    den = (np.linalg.norm(curr_head, axis=-1) + 1e-9) * (np.linalg.norm(prev_tail, axis=-1) + 1e-9)
    mean_cos = float(np.mean(num / den))
    per_dim_rmse = np.sqrt(np.mean(diff ** 2, axis=0))
    return {"mae": mae, "rmse": rmse, "mean_cos": mean_cos, "per_dim_rmse": per_dim_rmse.tolist()}

# ---------- main ----------
def resolve_targets(root_or_csv: str):
    if root_or_csv.endswith(".csv") and os.path.isfile(root_or_csv):
        csvs = [root_or_csv]
        outdir = os.path.dirname(root_or_csv) or "."
    else:
        outdir = root_or_csv
        csvs = sorted(glob.glob(os.path.join(os.path.join(root_or_csv, "chunks"), "*.csv")))
        if not csvs:
            ptr = os.path.join(root_or_csv, "chunk_latest_path.txt")
            if os.path.exists(ptr):
                with open(ptr) as f:
                    p = f.read().strip()
                if os.path.exists(p):
                    csvs = [p]
    return outdir, csvs

def main():
    ap = argparse.ArgumentParser(description="Plot 8 dims per chunk and compute overlap metrics between consecutive chunks.")
    ap.add_argument("target", nargs="?", default="dbg_chunks", help="CSV file or directory containing chunk CSVs")
    ap.add_argument("--s", type=int, default=25, help="start index of tail in previous chunk (first only)")
    ap.add_argument("--d", type=int, default=10, help="overlap length; also used for head shading if enabled")
    ap.add_argument("--ov", type=int, default=None, help="explicit overlap length (default: same as --d)")
    ap.add_argument("--overlap-figs", action="store_true", help="emit extra figures overlaying prev tail vs curr head")
    ap.add_argument("--shade-head", action="store_true", help="also produce head-shaded plots for current chunks")
    args = ap.parse_args()

    outdir, csvs = resolve_targets(args.target)
    outdir = os.path.join(outdir, "overlap_analysis")
    os.makedirs(outdir, exist_ok=True)
    if not csvs:
        print(f"No CSVs found in '{args.target}'")
        return

    ov = args.ov if args.ov is not None else args.d

    prevA = None
    prev_base = None

    for i, p in enumerate(csvs):
        A = load_chunk_csv(p)
        base = os.path.splitext(os.path.basename(p))[0]

        # Tail shading for "previous" chunk logic:
        # - i == 0   -> first prev: [s, s+d)
        # - i >= 1   -> later prev: [d+s, d+s+d)
        tail_phase = "first" if i == 0 else "later"
        tail_png = os.path.join(outdir, base + "_tail_8dims.png")
        plot_8dims(A, tail_png, title=base, s=args.s, d=args.d, shade="tail", tail_phase=tail_phase)

        # Optional head-shaded version (useful for seeing the skipped [0:d) region)
        if args.shade_head and i > 0:
            head_png = os.path.join(outdir, base + "_head_8dims.png")
            plot_8dims(A, head_png, title=base, s=args.s, d=args.d, shade="head")

        # ----- overlap metrics vs previous -----
        if prevA is not None:
            s_clamped = max(0, min(args.s, prevA.shape[0]))
            ov_eff = max(0, min(ov, prevA.shape[0] - s_clamped, A.shape[0]))
            if ov_eff > 0:
                prev_tail = prevA[s_clamped:s_clamped+ov_eff, :]
                curr_head = A[:ov_eff, :]
                m = overlap_metrics(prev_tail, curr_head)

                metrics_path = os.path.join(outdir, base + "_overlap_metrics.json")
                payload = {
                    "prev_chunk": prev_base,
                    "curr_chunk": base,
                    "s": s_clamped,
                    "ov": ov_eff,
                    **m
                }
                with open(metrics_path, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"[metrics] {metrics_path}  MAE={m['mae']:.6f}  RMSE={m['rmse']:.6f}  meanCos={m['mean_cos']:.6f}")

                if args.overlap_figs or args.overlap_figs:  # accept both spellings if you typo'd earlier
                    ov_png = os.path.join(outdir, base + "_overlap_preview.png")
                    plot_overlap_preview(prev_tail, curr_head, ov_png,
                        title=f"{prev_base} (tail s={s_clamped},ov={ov_eff}) vs {base} (head)")

        prevA = A
        prev_base = base

if __name__ == "__main__":
    main()
# python chunk_plot_dims_overlap.py saved_chunks/{dbg_rtc_07} --s 12 --d 10 --overlap-figs --shade-head