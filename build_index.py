import re, os, glob, pandas as pd
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

DATASET_ROOT = "./embryo_dataset"
OUT_CSV = "index.csv"
T = 50                 # Sequence length (frames)
WINDOW_STRIDE = T//2   # 50% overlap
EMPTY_IMAGE_PATHS_FILE = "empty_images.txt"
NUM_TEMPORAL_SAMPLES = 2   # Number of temporal contrastive pairs per sequence
TEMPORAL_DISTANCES = [1, 5, 10, 20]  # Frame distances for temporal sampling

run_pat = re.compile(r'RUN[_\- ]?(\d+)', re.I)
num_pat = re.compile(r'(\d+)')

def parse_sort_key(p: Path):
    name = p.name
    run_m = run_pat.search(name)
    run_idx = int(run_m.group(1)) if run_m else 10**9  
    nums = [int(x) for x in num_pat.findall(name)]
    nums = tuple(nums) if nums else ()
    mtime = p.stat().st_mtime_ns
    return (run_idx, nums, mtime)

def list_frames(cell_dir: Path):
    exts = ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG")
    frames = []
    for ext in exts:
        frames += [Path(p) for p in glob.glob(str(cell_dir / ext))]
    frames = [p for p in frames if p.exists() and p.stat().st_size > 0]
    frames.sort(key=parse_sort_key)
    return frames

def load_empty_image_paths(filepath):
    """Load known empty image paths from a text file."""
    if not filepath or not os.path.exists(filepath):
        return []

    empty_paths = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                empty_paths.append(line)
    return empty_paths


def contains_empty_image(frame_paths, empty_image_set):
    """Check if any frame in the sequence is in the empty image set."""
    for frame_path in frame_paths:
        if str(frame_path) in empty_image_set:
            return True
    return False


def sample_frames_from_dataset(root_path, num_samples=100):
    """Randomly sample frame paths from the entire dataset using os.walk()."""
    all_frames = []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

    for dirpath, dirnames, filenames in os.walk(root_path):
        for ext in exts:
            pattern = os.path.join(dirpath, ext)
            frames = glob.glob(pattern)
            all_frames.extend(frames)

    if not all_frames:
        return []

    # Filter for non-empty files
    all_frames = [p for p in all_frames if os.path.exists(p) and os.path.getsize(p) > 0]

    # Sample randomly
    sample_size = min(num_samples, len(all_frames))
    return random.sample(all_frames, sample_size)


def main():
    root = Path(DATASET_ROOT)
    cell_dirs = [p for p in root.iterdir() if p.is_dir()]
    rows = []

    
    empty_images = load_empty_image_paths(EMPTY_IMAGE_PATHS_FILE)
    # Sample temporal contrastive frames from the dataset
    print(f"Sampling frames for temporal contrastive learning...")
    dataset_frames = sample_frames_from_dataset(DATASET_ROOT, num_samples=5000)
    print(f"Found {len(dataset_frames)} total frames in dataset")

    for cell in tqdm(sorted(cell_dirs, key=lambda x: x.name)):
        frames = list_frames(cell)
        if not frames:
            continue
        if len(frames) < T:
            continue

        for start in range(0, len(frames)-T+1, WINDOW_STRIDE):
            seq = frames[start:start+T]

            # Check if sequence contains any known empty images
            contains_empty = contains_empty_image(seq, empty_image)

            # Sample temporal contrastive frame pairs
            np.random.shuffle(dataset_frames)
            np.random.shuffle(empty_images) 
            rows.append({
                "cell_id": cell.name,
                "start_idx": start,
                "embryo_paths": "|".join([str(p) for p in seq]),
                "empty_well_paths": "|".join([str(p) for p in empty_images[:T]]),
                "sample_paths": "|".join([str(p) for p in dataset_frames[:T]])
            })

    # Create DataFrame and save to CSV using pandas
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nDataset summary:")
    print(f"  Wrote {OUT_CSV} with {len(df)} sequences")
    print(f"  Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()

