#!/usr/bin/env python3
"""
修复index.csv中的路径，将本地路径转换为CHTC staging路径
"""
import pandas as pd
import sys
from pathlib import Path

def fix_index_paths(input_csv, output_csv=None, staging_base="/staging/groups/bhaskar_group/ivf/embryo_dataset"):
    """
    修复index.csv中的路径
    
    Args:
        input_csv: 输入的index.csv路径
        output_csv: 输出的index.csv路径（如果None，覆盖原文件）
        staging_base: staging目录的基础路径
    """
    if output_csv is None:
        output_csv = input_csv
    
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Found {len(df)} rows")
    print(f"Sample path: {df['paths'].iloc[0].split('|')[0] if len(df) > 0 else 'N/A'}")
    
    # 转换路径
    def convert_paths(path_str):
        """转换路径字符串中的所有路径"""
        paths = path_str.split("|")
        converted = []
        for p in paths:
            # 如果是本地路径，转换为staging路径
            # Generic pattern for local development paths
            if "embryo_dataset" in p and ("/Desktop/" in p or "/Users/" in p):
                # 提取相对路径
                rel_path = p.split("embryo_dataset/", 1)[1]
                new_path = f"{staging_base}/{rel_path}"
                converted.append(new_path)
            elif "embryo_dataset" in p and not p.startswith("/staging"):
                # 尝试提取cell_name和filename
                parts = Path(p).parts
                if "embryo_dataset" in parts:
                    idx = parts.index("embryo_dataset")
                    rel_parts = parts[idx+1:]
                    new_path = f"{staging_base}/{'/'.join(rel_parts)}"
                    converted.append(new_path)
                else:
                    converted.append(p)
            else:
                # 已经是staging路径或不需要转换
                converted.append(p)
        return "|".join(converted)
    
    print("Converting paths...")
    df['paths'] = df['paths'].apply(convert_paths)
    
    print(f"Writing to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    print(f"✓ Fixed {len(df)} rows")
    print(f"Sample converted path: {df['paths'].iloc[0].split('|')[0] if len(df) > 0 else 'N/A'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fix paths in index.csv for CHTC")
    parser.add_argument("--input", type=str, default="index.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file (default: overwrite input)")
    parser.add_argument("--staging-base", type=str, 
                       default="/staging/groups/bhaskar_group/ivf/embryo_dataset",
                       help="Base path for staging directory")
    
    args = parser.parse_args()
    
    fix_index_paths(args.input, args.output, args.staging_base)

