# build_index_embryo.py - Build index for complete embryo sequences
import re
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATASET_ROOT = "./embryo_dataset"
OUT_CSV = "index_embryo.csv"

# Regex patterns for sorting frames
run_pat = re.compile(r'RUN[_\- ]?(\d+)', re.I)
num_pat = re.compile(r'(\d+)')


def parse_sort_key(p: Path):
    """Generate a sort key for frame paths based on run number, numeric parts, and modification time."""
    name = p.name
    run_m = run_pat.search(name)
    run_idx = int(run_m.group(1)) if run_m else 10**9
    nums = [int(x) for x in num_pat.findall(name)]
    nums = tuple(nums) if nums else ()
    mtime = p.stat().st_mtime_ns
    return (run_idx, nums, mtime)


def list_frames(cell_dir: Path):
    """List all image frames in a directory, sorted by name."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    frames = []
    for ext in exts:
        frames += [Path(p) for p in glob.glob(str(cell_dir / ext))]

    # Filter out non-existent or empty files
    frames = [p for p in frames if p.exists() and p.stat().st_size > 0]

    # Sort by the parse key
    frames.sort(key=parse_sort_key)

    return frames


def main(limit=None):
    if limit == None:
        return
    """Build an index CSV where each row represents one complete embryo sequence.

    Args:
        limit: Maximum number of embryos to process (None = process all)
    """
    grades_df = pd.read_csv("embryo_dataset_grades.csv")
    a_mask = (grades_df["TE"] == "A") & (grades_df["ICM"] == "A")
    b_mask = (~ a_mask) & ((grades_df["TE"] == "B") | (grades_df["ICM"] == "B")) 
    c_mask = (~ a_mask) & (~ b_mask) & ((grades_df["TE"] == "C") | (grades_df["ICM"] == "C")) 
    na_mask = (~ a_mask) & (~ b_mask) & (~ c_mask)
    a_df = grades_df[a_mask].head(limit // 4)
    b_df = grades_df[b_mask].head(limit // 4)
    c_df = grades_df[c_mask].head(limit // 4)
    na_df = grades_df[na_mask].head(limit // 4)
    grades_df = pd.concat([a_df, b_df, c_df, na_df])
    root = Path(DATASET_ROOT)

    if not root.exists():
        print(f"Error: Dataset root '{DATASET_ROOT}' does not exist")
        return

    #embryo_dirs = [p for p in root.iterdir() if p.is_dir() and p.name in grades_df['video_name'].values]
    embryo_dirs = [p for p in root.iterdir()]
    if not embryo_dirs:
        print(f"Warning: No directories found in '{DATASET_ROOT}'")
        return

    # Sort and limit embryos
    #embryo_dirs = sorted(embryo_dirs, key=lambda x: grades_df.loc[grades_df['video_name'] == x.name][0]["TE"] + grades_df.loc[grades_df['video_name'] == x.name][0]["ICM"])
    #if limit is not None and limit > 0:
    #embryo_dirs = embryo_dirs[:limit]
    #print(f"Processing first {len(embryo_dirs)} embryos (limited by --limit {limit})")

    rows = []

    for embryo_dir in tqdm(embryo_dirs, desc="Processing embryos"):
        frames = list_frames(embryo_dir)

        if not frames:
            print(f"Warning: No frames found in {embryo_dir.name}")
            continue

        # Create one row per embryo with all frames
        rows.append({
            "embryo_id": embryo_dir.name,
            "num_frames": len(frames),
            "embryo_paths": "|".join([str(p) for p in frames])
        })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nDataset summary:")
    print(f"  Total embryos: {len(df)}")
    print(f"  Total frames: {df['num_frames'].sum()}")
    print(f"  Avg frames per embryo: {df['num_frames'].mean():.1f}")
    print(f"  Min frames: {df['num_frames'].min()}")
    print(f"  Max frames: {df['num_frames'].max()}")
    print(f"  Wrote index to: {OUT_CSV}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build index for complete embryo sequences.")
    parser.add_argument("--limit", type=int, default=25,
                       help="Maximum number of embryos to process (default: 25, use 0 for all)")

    args = parser.parse_args()

    # Convert 0 to None for processing all embryos
    limit = None if args.limit == 0 else args.limit

    main(limit=limit)
