#!/usr/bin/env python3
"""
分析 extraction 日誌，查看是否有成功提取的記錄
"""
import sys
import re
from pathlib import Path

def analyze_log(log_file):
    if not Path(log_file).exists():
        return
    
    print("=" * 70)
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    processing_count = 0
    saved_count = 0
    errors = []
    embryos_processed = []
    
    for i, line in enumerate(lines):
        if 'Processing embryo' in line or 'Processing' in line and 'embryo' in line.lower():
            processing_count += 1
            match = re.search(r'embryo[_\s]+([A-Z0-9\-]+)', line, re.I)
            if match:
                embryos_processed.append(match.group(1))
        
        if 'Saved:' in line or '✓ Saved' in line or 'saved to' in line.lower():
            saved_count += 1
        
        if any(word in line.lower() for word in ['error', 'exception', 'failed', 'failed to']):
            errors.append((i+1, line.strip()))
    
    
    if embryos_processed:
        unique_embryos = list(set(embryos_processed))[:20]
        for emb in unique_embryos:
            print(f"   - {emb}")
        if len(set(embryos_processed)) > 20:
    
    if errors:
        for line_num, error_line in errors[-20:]:
    
    dir_created = False
    for line in lines:
        if 'Output directory' in line or 'Created directory' in line or 'mkdir' in line.lower():
            print(f"   {line.strip()}")
            dir_created = True
    
    if not dir_created:
    
    for line in lines[-10:]:
        print(f"   {line.rstrip()}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        print("  python3 analyze_extraction_log.py /staging/groups/bhaskar_group/rho9/logs/extract_latents_v1_baseline.out")
        sys.exit(1)
    
    analyze_log(log_file)

