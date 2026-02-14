#!/usr/bin/env python3
"""
Verify tar.gz contents and compare with extracted dataset
Checks for missing files or incomplete extraction
"""

import os
import tarfile
import subprocess
from pathlib import Path
from collections import defaultdict

def get_tar_contents(tar_path):
    """List all files in tar.gz without extracting"""
    print(f"Reading tar.gz contents from: {tar_path}")
    files_in_tar = []
    total_size = 0
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    files_in_tar.append(member.name)
                    total_size += member.size
        print(f"  Found {len(files_in_tar)} files in tar.gz")
        print(f"  Total uncompressed size: {total_size / 1024**3:.2f} GB")
        return files_in_tar, total_size
    except Exception as e:
        print(f"  ❌ Error reading tar.gz: {e}")
        return [], 0

def get_extracted_files(data_dir):
    """List all files in extracted directory"""
    print(f"Scanning extracted directory: {data_dir}")
    files_extracted = []
    total_size = 0
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"  ❌ Directory does not exist: {data_dir}")
        return [], 0
    
    for file_path in data_path.rglob('*'):
        if file_path.is_file():
            rel_path = file_path.relative_to(data_path)
            files_extracted.append(str(rel_path))
            total_size += file_path.stat().st_size
    
    print(f"  Found {len(files_extracted)} files in extracted directory")
    print(f"  Total size: {total_size / 1024**3:.2f} GB")
    return files_extracted, total_size

def compare_files(tar_files, extracted_files):
    """Compare files in tar vs extracted"""
    print("\n=== Comparing Files ===")
    
    tar_set = set(tar_files)
    extracted_set = set(extracted_files)
    
    missing = tar_set - extracted_set
    extra = extracted_set - tar_set
    
    print(f"Files in tar.gz: {len(tar_set)}")
    print(f"Files extracted: {len(extracted_set)}")
    print(f"Missing files: {len(missing)}")
    print(f"Extra files: {len(extra)}")
    
    if missing:
        print(f"\n⚠️  Missing files (first 20):")
        for f in sorted(list(missing))[:20]:
            print(f"  - {f}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    
    if extra:
        print(f"\n⚠️  Extra files (not in tar.gz, first 10):")
        for f in sorted(list(extra))[:10]:
            print(f"  - {f}")
    
    return missing, extra

def analyze_by_cell(tar_files, extracted_files):
    """Analyze files by cell directory"""
    print("\n=== Analysis by Cell Directory ===")
    
    tar_cells = defaultdict(list)
    extracted_cells = defaultdict(list)
    
    for f in tar_files:
        parts = f.split('/')
        if len(parts) >= 2:
            cell_id = parts[0]
            tar_cells[cell_id].append(f)
    
    for f in extracted_files:
        parts = f.split('/')
        if len(parts) >= 2:
            cell_id = parts[0]
            extracted_cells[cell_id].append(f)
    
    all_cells = set(tar_cells.keys()) | set(extracted_cells.keys())
    
    print(f"Total cell directories: {len(all_cells)}")
    print(f"  In tar.gz: {len(tar_cells)}")
    print(f"  Extracted: {len(extracted_cells)}")
    
    missing_cells = set(tar_cells.keys()) - set(extracted_cells.keys())
    if missing_cells:
        print(f"\n⚠️  Missing cell directories ({len(missing_cells)}):")
        for cell in sorted(list(missing_cells))[:10]:
            print(f"  - {cell} ({len(tar_cells[cell])} files)")
    
    # Check for cells with missing files
    incomplete_cells = []
    for cell in tar_cells.keys():
        if cell in extracted_cells:
            tar_count = len(tar_cells[cell])
            ext_count = len(extracted_cells[cell])
            if tar_count != ext_count:
                incomplete_cells.append((cell, tar_count, ext_count))
    
    if incomplete_cells:
        print(f"\n⚠️  Cells with missing files ({len(incomplete_cells)}):")
        for cell, tar_count, ext_count in sorted(incomplete_cells, key=lambda x: x[1]-x[2], reverse=True)[:10]:
            missing = tar_count - ext_count
            print(f"  - {cell}: {tar_count} in tar, {ext_count} extracted (missing {missing})")

def main():
    print("=" * 60)
    print("Dataset Extraction Verification")
    print("=" * 60)
    print()
    
    # Find tar.gz file
    tar_paths = [
        "/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset.tar.gz",
        "/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz",
        "embryo_dataset.tar.gz",
    ]
    
    tar_path = None
    for path in tar_paths:
        if os.path.exists(path):
            tar_path = path
            break
    
    if not tar_path:
        print("❌ Could not find tar.gz file")
        print("   Checked paths:")
        for p in tar_paths:
            print(f"     - {p}")
        return
    
    # Find data directory
    data_dir = None
    if os.path.islink("data"):
        data_dir = os.path.realpath("data")
    elif os.path.isdir("data"):
        data_dir = "data"
    elif os.path.isdir("data_raw/embryo_dataset"):
        data_dir = "data_raw/embryo_dataset"
    
    if not data_dir:
        print("❌ Could not find extracted data directory")
        print("   Looked for: 'data' symlink, 'data' directory, 'data_raw/embryo_dataset'")
        return
    
    print(f"Tar.gz file: {tar_path}")
    print(f"Extracted directory: {data_dir}")
    print()
    
    # Get file lists
    tar_files, tar_size = get_tar_contents(tar_path)
    extracted_files, ext_size = get_extracted_files(data_dir)
    
    if not tar_files:
        print("❌ Could not read tar.gz contents")
        return
    
    if not extracted_files:
        print("❌ Could not read extracted directory")
        return
    
    # Compare sizes
    print("\n=== Size Comparison ===")
    print(f"Tar.gz uncompressed size: {tar_size / 1024**3:.2f} GB")
    print(f"Extracted size: {ext_size / 1024**3:.2f} GB")
    diff = tar_size - ext_size
    diff_gb = diff / 1024**3
    diff_percent = (diff / tar_size) * 100 if tar_size > 0 else 0
    print(f"Difference: {diff_gb:.2f} GB ({diff_percent:.1f}%)")
    
    if diff_gb > 0.5:
        print("⚠️  Significant size difference detected!")
    
    # Compare files
    missing, extra = compare_files(tar_files, extracted_files)
    
    # Analyze by cell
    analyze_by_cell(tar_files, extracted_files)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if not missing and diff_gb < 0.5:
        print("✅ Extraction appears complete")
    else:
        print("⚠️  Extraction may be incomplete:")
        if missing:
            print(f"   - {len(missing)} files missing")
        if diff_gb > 0.5:
            print(f"   - {diff_gb:.2f} GB missing")
        print("\nRecommendation: Re-extract the tar.gz file")

if __name__ == "__main__":
    main()

