"""
build_index.py — Ver10
=======================
Same as Ver09: T=50, stride=12 (75% overlap)
"""

import os
import re
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATASET_ROOT  = os.environ.get("DATASET_ROOT", "./embryo_dataset")
OUT_CSV       = "index.csv"
T             = 50
WINDOW_STRIDE = T // 4   # stride=12, 75% overlap

run_pat = re.compile(r'RUN[_\- ]?(\d+)', re.I)
num_pat = re.compile(r'(\d+)')


def parse_sort_key(p: Path):
    name    = p.name
    run_m   = run_pat.search(name)
    run_idx = int(run_m.group(1)) if run_m else 10**9
    nums    = tuple(int(x) for x in num_pat.findall(name))
    return (run_idx, nums, p.stat().st_mtime_ns)


def list_frames(cell_dir: Path) -> list:
    exts   = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    frames = []
    for ext in exts:
        frames += [Path(p) for p in glob.glob(str(cell_dir / ext))]
    frames = [p for p in frames if p.exists() and p.stat().st_size > 0]
    frames.sort(key=parse_sort_key)
    return frames


def main():
    root      = Path(DATASET_ROOT)
    cell_dirs = [p for p in root.iterdir() if p.is_dir()]
    rows      = []
    skipped   = 0

    for cell in tqdm(sorted(cell_dirs, key=lambda x: x.name), desc="Indexing"):
        frames = list_frames(cell)
        if len(frames) < T:
            skipped += 1
            continue
        for start in range(0, len(frames) - T + 1, WINDOW_STRIDE):
            seq = frames[start:start + T]
            rows.append({
                "cell_id":   cell.name,
                "start_idx": start,
                "paths":     "|".join(str(p) for p in seq),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nDataset summary:")
    print(f"  Sequences : {len(df)}")
    print(f"  Skipped   : {skipped} cells (< {T} frames)")
    print(f"  seq_len   : {T} ({T * 12} min = {T * 12 / 60:.1f} hours per window)")
    print(f"  stride    : {WINDOW_STRIDE} (75% overlap)")
    print(f"  Output    : {OUT_CSV}")


if __name__ == "__main__":
    main()
