#!/usr/bin/env python3
# tools/combine_chunk_panels.py
import os, re, glob
from PIL import Image, ImageDraw, ImageFont

# ---------- text & drawing helpers ----------
def _load_font(size=18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:
        return draw.textsize(text, font=font)

def _add_label(img: Image.Image, text: str, pad=8):
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font(18)
    tw, th = _measure_text(draw, text, font)
    box = (pad - 4, pad - 4, pad + tw + 8, pad + th + 4)
    draw.rectangle(box, fill=(0, 0, 0, 128))
    draw.text((pad, pad), text, fill=(255, 255, 255, 255), font=font)
    return img

def _letterbox(img: Image.Image, target_w: int, target_h: int, bg=(255, 255, 255)) -> Image.Image:
    scale = min(target_w / img.width, target_h / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized = img if (new_w == img.width and new_h == img.height) else img.resize((new_w, new_h), Image.BICUBIC)
    out = Image.new("RGB", (target_w, target_h), bg)
    out.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return out

def _cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    scale = max(target_w / img.width, target_h / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))

def _basename(path):  # without extension
    return os.path.splitext(os.path.basename(path))[0]

# ---------- parse "<prefix>_c###[_...]" ----------
_rx_ci = re.compile(r"^(?P<prefix>.*)_c(?P<idx>\d+)(?P<rest>.*)?$")

def _parse_prefix_idx(stem: str):
    m = _rx_ci.match(stem)
    if not m:
        return stem, -1
    return m.group("prefix"), int(m.group("idx"))

# ---------- pairing logic ----------
def collect_pairs(dir_8dims: str, dir_overlap: str):
    """
    Returns list of tuples: (prev_tail_path, overlap_path, curr_head_path, curr_base)
      - 8dims: <prefix>_cXXX_tail_8dims.png and <prefix>_cXXX_head_8dims.png
      - overlap: <prefix>_cXXX_overlap_preview.png  (for current chunk XXX)
    """
    tails = sorted(glob.glob(os.path.join(dir_8dims, "*_tail_8dims.png")))
    heads = sorted(glob.glob(os.path.join(dir_8dims, "*_head_8dims.png")))
    overs = sorted(glob.glob(os.path.join(dir_overlap, "*_overlap_preview.png")))

    # Maps
    tail_by_pref_idx = {}   # (prefix, idx) -> path
    tails_by_idx = {}       # idx -> list[(prefix, path)]  (for cross-prefix fallback)
    for p in tails:
        stem = _basename(p)
        pref, idx = _parse_prefix_idx(stem)
        if idx < 0: 
            continue
        tail_by_pref_idx[(pref, idx)] = p
        tails_by_idx.setdefault(idx, []).append((pref, p))

    # Map "<prefix>_cXXX" -> overlap path
    overlap_map = {}
    suf = "_overlap_preview"
    for p in overs:
        b = _basename(p)
        if b.endswith(suf):
            curr_base = b[:-len(suf)]  # "<prefix>_cXXX"
            overlap_map[curr_base] = p

    # Build head entries with parsed prefix/idx
    head_entries = []
    for p in heads:
        stem = _basename(p)  # "..._c002_head_8dims"
        pref, idx = _parse_prefix_idx(stem)
        head_entries.append((pref, idx, p, stem))
    # Sort naturally by (idx, then prefix) so c1..cN order holds even if prefixes differ
    head_entries.sort(key=lambda x: (x[1], x[0]))

    def _pick_prev_tail(curr_pref: str, curr_idx: int):
        prev_idx = curr_idx - 1
        # 1) Same-prefix ideal match
        path = tail_by_pref_idx.get((curr_pref, prev_idx))
        if path:
            return path
        # 2) Fallback: any tail with idx==prev_idx. Choose the one whose prefix is <= curr_pref and closest,
        #    otherwise the lexicographically largest (latest) prefix.
        cands = tails_by_idx.get(prev_idx, [])
        if not cands:
            return None
        # split by <= curr_pref
        le = [ (pref,p) for pref,p in cands if pref <= curr_pref ]
        if le:
            # choose the max prefix among <=
            pref, p = max(le, key=lambda t: t[0])
            return p
        # else choose the max prefix overall
        pref, p = max(cands, key=lambda t: t[0])
        return p

    pairs = []
    for pref, idx, curr_head_path, curr_stem in head_entries:
        if idx <= 0:
            continue  # ignore weird negatives
        prev_tail_path = _pick_prev_tail(pref, idx)
        if not prev_tail_path:
            print(f"[warn] No tail image for {_basename(curr_stem).split('_head_')[0].rsplit('_',1)[0].replace('_c'+str(idx), f'_c{idx-1:03d}')} ; skipping {os.path.basename(curr_head_path)}")
            continue

        # "<prefix>_cXXX" base for overlap lookup should use current head's prefix+idx
        curr_base = f"{pref}_c{idx:03d}"
        ovp = overlap_map.get(curr_base)
        if not ovp:
            print(f"[warn] No overlap image for {curr_base}; skipping pair.")
            continue

        pairs.append((prev_tail_path, ovp, curr_head_path, curr_base))

    return pairs

# ---------- 2x1 layout with strict equal row heights ----------
def _compose_equal_rows(im_prev: Image.Image,
                        im_curr: Image.Image,
                        im_overlap: Image.Image,
                        gap: int,
                        bg,
                        col_width: int | None,
                        row_height: int | None,
                        bottom_fit: str = "contain") -> Image.Image:
    # Decide column width
    if col_width is None:
        col_width = max(im_prev.width, im_curr.width)
    # Heights at that width
    h_prev = int(round(im_prev.height * col_width / im_prev.width))
    h_curr = int(round(im_curr.height * col_width / im_curr.width))
    H_row = max(1, row_height) if row_height is not None else max(h_prev, h_curr)
    # Top cells
    prev_box = _letterbox(im_prev, col_width, H_row, bg=bg)
    curr_box = _letterbox(im_curr, col_width, H_row, bg=bg)
    # Bottom cell spans both columns
    total_w = col_width * 2 + gap
    bottom_box = _cover(im_overlap, total_w, H_row) if bottom_fit == "cover" else _letterbox(im_overlap, total_w, H_row, bg=bg)
    # Compose
    out = Image.new("RGB", (total_w, H_row + gap + H_row), bg)
    out.paste(prev_box, (0, 0))
    out.paste(curr_box, (col_width + gap, 0))
    out.paste(bottom_box, (0, H_row + gap))
    return out

# ---------- combine ----------
def combine_all(dir_8dims="8dims",
                dir_overlap="overlap",
                outdir="combined",
                gap=16,
                col_width=None,
                row_height=None,
                bottom_fit="contain",
                bg=(255, 255, 255)):
    os.makedirs(outdir, exist_ok=True)
    pairs = collect_pairs(dir_8dims, dir_overlap)
    if not pairs:
        print("[err] No pairs found.")
        return

    for prev_tail, ovp, curr_head, curr_base in pairs:
        im_prev = Image.open(prev_tail).convert("RGB")
        im_ov   = Image.open(ovp).convert("RGB")
        im_curr = Image.open(curr_head).convert("RGB")

        im_prev = _add_label(im_prev, f"PREV (tail): {_basename(prev_tail)}")
        im_curr = _add_label(im_curr, f"CURR (head): {_basename(curr_head)}")
        im_ov   = _add_label(im_ov,   f"OVERLAP: {_basename(ovp)}")

        combined = _compose_equal_rows(
            im_prev, im_curr, im_ov,
            gap=gap, bg=bg, col_width=col_width, row_height=row_height, bottom_fit=bottom_fit
        )

        out_path = os.path.join(outdir, f"{curr_base}_combo.png")
        combined.save(out_path, optimize=True)
        print(f"[write] {out_path}")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Combine prev tail (cN-1) | curr head (cN) on top, overlap (cN) on bottom.")
    ap.add_argument("--d8", default="8dims", help="Directory containing *_tail_8dims.png and *_head_8dims.png")
    ap.add_argument("--ov", default="overlap", help="Directory containing *_overlap_preview.png")
    ap.add_argument("--out", default="combined", help="Output directory for combined images")
    ap.add_argument("--gap", type=int, default=16, help="Gap (pixels) between rows and between top columns")
    ap.add_argument("--col-width", type=int, default=None, help="Force column width (pixels) for the top panels")
    ap.add_argument("--row-height", type=int, default=None, help="Force exact row height (pixels) for BOTH rows")
    ap.add_argument("--bottom-fit", choices=["contain", "cover"], default="contain",
                    help="Fit bottom overlap into its row: 'contain' (letterbox) or 'cover' (crop)")
    args = ap.parse_args()
    combine_all(args.d8, args.ov, args.out, gap=args.gap,
                col_width=args.col_width, row_height=args.row_height,
                bottom_fit=args.bottom_fit)
# python combine_chunk_panels.py --d8 saved_chunks/{dbg_rtc_07}/overlap_analysis --ov saved_chunks/{dbg_rtc_07}/overlap_analysis --out saved_chunks/{dbg_rtc_07}/combined