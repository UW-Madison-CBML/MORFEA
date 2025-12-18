import re, os, glob, pandas as pd
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

DATASET_ROOT = "./embryo_dataset"
OUT_CSV = "index.csv"
T = 32                 # Sequence length (frames)
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


def build_empty_well_sequences_map(empty_images_list, root_path):
    """
    Build a mapping from empty image paths to their containing sequences.
    For each empty image, finds the cell directory it belongs to and stores
    all sequential frames from that directory.

    Returns:
        dict: mapping from empty image path to list of sequential frames
    """
    empty_to_sequence = {}
    root = Path(root_path)

    for empty_path_str in tqdm(empty_images_list, desc="Building empty well sequences"):
        empty_path = Path(empty_path_str)

        # Find the cell directory this image belongs to
        # Assuming structure: root_path/cell_id/image.jpg
        if not empty_path.exists():
            continue

        # Get the parent directory (cell directory)
        cell_dir = empty_path.parent

        # Get all frames in this cell directory
        frames = list_frames(cell_dir)

        if not frames:
            continue

        # Find the index of this empty image in the sequence
        try:
            empty_idx = frames.index(empty_path)
        except ValueError:
            # Image not found in sorted list, skip
            continue

        # Store the sequence for this empty image
        empty_to_sequence[empty_path_str] = frames

    return empty_to_sequence


def get_empty_well_sequence(empty_images_list, empty_to_sequence, seq_len):
    """
    Pick a random empty well image and extract a sequence containing it.

    Args:
        empty_images_list: list of empty image paths
        empty_to_sequence: dict mapping empty images to their full sequences
        seq_len: length of sequence to extract

    Returns:
        list of paths forming a sequence, or None if unable to create sequence
    """
    # Shuffle and try to find a valid sequence
    random.shuffle(empty_images_list)

    for empty_path in empty_images_list:
        if empty_path not in empty_to_sequence:
            continue

        frames = empty_to_sequence[empty_path]

        # Find index of the empty image
        try:
            empty_idx = frames.index(Path(empty_path))
        except ValueError:
            continue

        # Try to build a sequence around this frame
        # Prefer centering the empty image, but adjust if near boundaries
        if len(frames) < seq_len:
            # Not enough frames in this sequence
            continue

        # Try to center the empty image in the sequence
        start_idx = max(0, empty_idx - seq_len // 2)
        end_idx = start_idx + seq_len

        # Adjust if we go past the end
        if end_idx > len(frames):
            end_idx = len(frames)
            start_idx = end_idx - seq_len

        # Extract sequence
        seq = frames[start_idx:end_idx]

        if len(seq) == seq_len:
            return [str(p) for p in seq]

    # Fallback: if no valid sequence found, return None
    return None


def main():
    root = Path(DATASET_ROOT)
    cell_dirs = [p for p in root.iterdir() if p.is_dir()]
    rows = []


    empty_images = load_empty_image_paths(EMPTY_IMAGE_PATHS_FILE)
    print(f"Loaded {len(empty_images)} empty well images")

    # Build mapping from empty images to their sequences
    print(f"Building empty well sequence mappings...")
    empty_to_sequence = build_empty_well_sequences_map(empty_images, DATASET_ROOT)
    print(f"Built sequences for {len(empty_to_sequence)} empty images")

    # Sample temporal contrastive frames from the dataset
    print(f"Sampling frames for temporal contrastive learning...")
    dataset_frames = sample_frames_from_dataset(DATASET_ROOT, num_samples=5000)
    print(f"Found {len(dataset_frames)} total frames in dataset")

    for cell in tqdm(sorted(cell_dirs, key=lambda x: x.name), desc="Processing cells"):
        frames = list_frames(cell)
        if not frames:
            continue
        if len(frames) < T:
            continue

        for start in range(0, len(frames)-T+1, WINDOW_STRIDE):
            seq = frames[start:start+T]

            # Get a proper empty well sequence
            empty_seq = get_empty_well_sequence(empty_images, empty_to_sequence, T)

            # Fallback to random empty images if sequence creation fails
            if empty_seq is None:
                np.random.shuffle(empty_images)
                empty_seq = empty_images[:T]

            # Sample random frames for temporal contrastive learning
            np.random.shuffle(dataset_frames)

            rows.append({
                "cell_id": cell.name,
                "start_idx": start,
                "embryo_paths": "|".join([str(p) for p in seq]),
                "empty_well_paths": "|".join(empty_seq),
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

