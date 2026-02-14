#!/usr/bin/env python3
"""
分析 extraction 日誌，查看是否有成功提取的記錄
"""
import sys
import re
from pathlib import Path

def analyze_log(log_file):
    """分析日誌文件"""
    if not Path(log_file).exists():
        print(f"❌ 日誌文件不存在: {log_file}")
        return
    
    print(f"📋 分析日誌文件: {log_file}")
    print("=" * 70)
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # 查找關鍵信息
    processing_count = 0
    saved_count = 0
    errors = []
    embryos_processed = []
    
    for i, line in enumerate(lines):
        # 統計處理的胚胎
        if 'Processing embryo' in line or 'Processing' in line and 'embryo' in line.lower():
            processing_count += 1
            # 提取胚胎ID
            match = re.search(r'embryo[_\s]+([A-Z0-9\-]+)', line, re.I)
            if match:
                embryos_processed.append(match.group(1))
        
        # 統計保存的文件
        if 'Saved:' in line or '✓ Saved' in line or 'saved to' in line.lower():
            saved_count += 1
        
        # 收集錯誤
        if any(word in line.lower() for word in ['error', 'exception', 'failed', 'failed to']):
            errors.append((i+1, line.strip()))
    
    print(f"\n📊 統計結果：")
    print(f"  總行數: {len(lines)}")
    print(f"  處理的胚胎數: {processing_count}")
    print(f"  保存的文件數: {saved_count}")
    print(f"  錯誤數: {len(errors)}")
    
    if embryos_processed:
        print(f"\n✅ 成功處理的胚胎（前20個）：")
        unique_embryos = list(set(embryos_processed))[:20]
        for emb in unique_embryos:
            print(f"   - {emb}")
        if len(set(embryos_processed)) > 20:
            print(f"   ... 還有 {len(set(embryos_processed)) - 20} 個")
    
    if errors:
        print(f"\n❌ 錯誤信息（最後20個）：")
        for line_num, error_line in errors[-20:]:
            print(f"   行 {line_num}: {error_line[:100]}")
    
    # 查找目錄創建相關信息
    print(f"\n📁 目錄創建相關信息：")
    dir_created = False
    for line in lines:
        if 'Output directory' in line or 'Created directory' in line or 'mkdir' in line.lower():
            print(f"   {line.strip()}")
            dir_created = True
    
    if not dir_created:
        print("   ⚠️  未找到目錄創建記錄")
    
    # 查找最後幾行
    print(f"\n📝 日誌最後10行：")
    for line in lines[-10:]:
        print(f"   {line.rstrip()}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        print("使用方法: python3 analyze_extraction_log.py <log_file>")
        print("\n或者分析 staging 日誌:")
        print("  python3 analyze_extraction_log.py /staging/groups/bhaskar_group/rho9/logs/extract_latents_v1_baseline.out")
        sys.exit(1)
    
    analyze_log(log_file)

