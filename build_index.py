#!/usr/bin/env python3
"""
Build index.csv for IVF dataset (JobAccounting style)
Scans data directory and creates index.csv with sequences for training

Usage:
    python build_index.py --root data --out index.csv
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from tqdm import tqdm

# Sequence parameters
T = 16                 # Sequence length (frames)
SUBSAMPLE = 3          # Take every 3rd frame
WINDOW_STRIDE = T // 2   # 50% overlap

# Pattern for sorting
run_pat = re.compile(r'RUN[_\- ]?(\d+)', re.I)
num_pat = re.compile(r'(\d+)')


def parse_sort_key(p: Path):
    """Parse sorting key from path name"""
    name = p.name
    # Extract RUN number
    run_m = run_pat.search(name)
    run_idx = int(run_m.group(1)) if run_m else 10**9  # No RUN number goes last
    # Extract all numbers from filename
    nums = [int(x) for x in num_pat.findall(name)]
    nums = tuple(nums) if nums else ()
    # File modification time (nanoseconds)
    mtime = p.stat().st_mtime_ns
    return (run_idx, nums, mtime)


def list_frames(cell_dir: Path):
    """List all image frames in a cell directory"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    frames = []
    for ext in exts:
        frames += [Path(p) for p in cell_dir.glob(ext)]
    frames = [p for p in frames if p.exists() and p.stat().st_size > 0]
    frames.sort(key=parse_sort_key)
    return frames


def parse_args():
    """Parse command line arguments"""
    p = argparse.ArgumentParser(
        description="Build IVF image index.csv from data directory"
    )
    p.add_argument(
        "--root",
        type=str,
        default="data",
        help="Root directory containing cell folders (default: data)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="index.csv",
        help="Output CSV file path (default: index.csv)",
    )
    p.add_argument(
        "--seq_len",
        type=int,
        default=T,
        help=f"Sequence length in frames (default: {T})",
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=SUBSAMPLE,
        help=f"Take every Nth frame (default: {SUBSAMPLE})",
    )
    return p.parse_args()


def build_index(root_dir: Path, out_path: Path, seq_len: int, subsample: int):
    """
    Build index.csv from data directory
    
    Args:
        root_dir: Root directory containing cell folders
        out_path: Output CSV file path
        seq_len: Sequence length
        subsample: Subsample rate
    """
    root_dir = root_dir.resolve()
    
    print(f"[build_index] Scanning directory: {root_dir}", flush=True)
    print(f"[build_index] Current directory: {Path.cwd()}", flush=True)
    print(f"[build_index] Root exists: {root_dir.exists()}", flush=True)
    print(f"[build_index] Root is directory: {root_dir.is_dir()}", flush=True)
    
    # Check if root exists
    if not root_dir.exists():
        print(f"[build_index] ERROR: Root directory '{root_dir}' does not exist!", flush=True)
        print(f"[build_index] Current directory: {Path.cwd()}", flush=True)
        print(f"[build_index] Available files: {list(Path('.').iterdir())[:10]}", flush=True)
        sys.exit(1)
    
    if not root_dir.is_dir():
        print(f"[build_index] ERROR: '{root_dir}' exists but is not a directory!", flush=True)
        sys.exit(1)
    
    # Find cell directories
    cell_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not cell_dirs:
        print(f"[build_index] ERROR: No cell directories found in '{root_dir}'!", flush=True)
        print(f"[build_index] Contents: {list(root_dir.iterdir())[:10]}", flush=True)
        sys.exit(1)
    
    print(f"[build_index] Found {len(cell_dirs)} cell directories", flush=True)
    
    # CSV fieldnames (matching IVFSequenceDataset expectations)
    fieldnames = ["cell_id", "start_idx", "paths"]
    rows = []
    
    # Process each cell directory
    for cell in tqdm(sorted(cell_dirs, key=lambda x: x.name), desc="Processing cells"):
        frames = list_frames(cell)
        if not frames:
            continue
        
        # Temporal subsampling
        frames = frames[::subsample]
        if len(frames) < seq_len:
            continue
        
        # Sliding window
        window_stride = seq_len // 2  # 50% overlap
        for start in range(0, len(frames) - seq_len + 1, window_stride):
            seq = frames[start:start + seq_len]
            rows.append({
                "cell_id": cell.name,
                "start_idx": start,
                "paths": "|".join(str(p) for p in seq)
            })
    
    if not rows:
        print(f"[build_index] ERROR: No valid sequences found!", flush=True)
        sys.exit(1)
    
    # Write CSV (JobAccounting style with csv.DictWriter)
    out_path = out_path.resolve()
    print(f"[build_index] Writing {len(rows)} sequences to {out_path}", flush=True)
    
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[build_index] ✓ Successfully wrote {out_path} with {len(rows)} sequences", flush=True)


def main(root=None, out=None, seq_len=None, subsample=None):
    """
    Main entry point - can be called as module or script
    
    Args:
        root: Root directory (default: "data")
        out: Output CSV path (default: "index.csv")
        seq_len: Sequence length (default: T)
        subsample: Subsample rate (default: SUBSAMPLE)
    """
    if root is None:
        # Called as script, parse args
        args = parse_args()
        root = Path(args.root)
        out = Path(args.out)
        seq_len = args.seq_len
        subsample = args.subsample
    else:
        # Called as module with parameters
        root = Path(root) if root else Path("data")
        out = Path(out) if out else Path("index.csv")
        seq_len = seq_len if seq_len else T
        subsample = subsample if subsample else SUBSAMPLE
    
    build_index(root, out, seq_len, subsample)


if __name__ == "__main__":
    main()
