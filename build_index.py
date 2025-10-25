import re, os, glob, csv
from pathlib import Path
from tqdm import tqdm

DATASET_ROOT = "/Users/grnho/Desktop/Project IVF/embryo_dataset"  # ← 改這裡
OUT_CSV = "index.csv"
T = 16                 # 序列長度（幀）
SUBSAMPLE = 3          # 每3幀取1幀
WINDOW_STRIDE = T//2   # 50% 重疊

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
    for cell in tqdm(sorted(cell_dirs, key=lambda x: x.name)):
        frames = list_frames(cell)
        if not frames: 
            continue
        # 下採樣（時間）
        frames = frames[::SUBSAMPLE]
        if len(frames) < T: 
            continue
        # 滑動視窗
        for start in range(0, len(frames)-T+1, WINDOW_STRIDE):
            seq = frames[start:start+T]
            rows.append({
                "cell_id": cell.name,
                "start_idx": start,
                "paths": "|".join(str(p) for p in seq)
            })
    # 輸出 CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell_id","start_idx","paths"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {OUT_CSV} with {len(rows)} sequences")

if __name__ == "__main__":
    main()

