import re, os, glob, pandas as pd
from pathlib import Path
from tqdm import tqdm
from detect_empty_wells import EmptyWellDetector

DATASET_ROOT = "./embryo_dataset"  # ← 改這裡
OUT_CSV = "index.csv"
T = 50                 # 序列長度（幀）
WINDOW_STRIDE = T//2   # 50% 重疊
DETECT_EMPTY_WELLS = True  # Enable empty well detection
EMPTY_WELL_THRESHOLD = 0.75  # Probability threshold for empty classification
EMPTY_WELL_MODEL = None    # Path to trained model (optional)

run_pat = re.compile(r'RUN[_\- ]?(\d+)', re.I)
num_pat = re.compile(r'(\d+)')

def parse_sort_key(p: Path):
    name = p.name
    # 取 RUN 編號
    run_m = run_pat.search(name)
    run_idx = int(run_m.group(1)) if run_m else 10**9  # 沒有 RUN 放最後
    # 2) 檔名裡所有數字
    nums = [int(x) for x in num_pat.findall(name)]
    nums = tuple(nums) if nums else ()
    # 3) 檔案修改時間（奈秒）
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

def main():
    root = Path(DATASET_ROOT)
    cell_dirs = [p for p in root.iterdir() if p.is_dir()]
    rows = []

    # Initialize empty well detector if enabled
    detector = None
    if DETECT_EMPTY_WELLS:
        detector = EmptyWellDetector(model_path=EMPTY_WELL_MODEL)

    for cell in tqdm(sorted(cell_dirs, key=lambda x: x.name)):
        frames = list_frames(cell)
        if not frames:
            continue
        # 下採樣（時間）
        if len(frames) < T:
            continue

        # Detect empty well status (cache per sequence since frames are same)
        empty_well = False
        
        # 滑動視窗
        for start in range(0, len(frames)-T+1, WINDOW_STRIDE):
            seq = frames[start:start+T]
            if detector and frames:
                try:
                    # Use first frame to determine if well is empty
                    prob = min(detector.predict(seq[0]), detector.predict(seq[T//2]),detector.predict(seq[T-1]))
                    empty_well = prob >= EMPTY_WELL_THRESHOLD
                except Exception as e:
                    print(f"Warning: empty well detection failed for {cell.name}: {e}")
                    empty_well = False
            rows.append({
                "cell_id": cell.name,
                "start_idx": start,
                "paths": "|".join(str(p) for p in seq),
                "empty_well": empty_well,
            })
    # 輸出 CSV
    with open(OUT_CSV, "w", newline="") as f:
        fieldnames = ["cell_id", "start_idx", "paths", "empty_well"] if DETECT_EMPTY_WELLS else ["cell_id", "start_idx", "paths"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {OUT_CSV} with {len(rows)} sequences")

    if DETECT_EMPTY_WELLS:
        num_empty = sum(1 for r in rows if r.get("empty_well", False))
        print(f"Empty wells: {num_empty}/{len(rows)} ({100*num_empty/len(rows):.1f}%)")

if __name__ == "__main__":
    main()

