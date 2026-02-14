#!/usr/bin/env python3
"""
删除所有 Python 和 Shell 文件中的中文注释
"""

import os
import re
from pathlib import Path

# 要处理的文件类型
FILE_EXTENSIONS = ['.py', '.sh']

# 排除的目录
EXCLUDE_DIRS = {
    '.git', '__pycache__', 'checkpoints', 'logs', 'results', 
    'reconstructions', 'tphate_results', 'curvature_results',
    'model_latents', 'latents_epoch50', 'scripts'
}

def remove_chinese_comments(content, file_ext):
    """删除中文注释"""
    lines = content.split('\n')
    new_lines = []
    removed_count = 0
    
    for line in lines:
        original_line = line
        
        # 处理 Python 文件
        if file_ext == '.py':
            # 删除整行中文注释 (# 开头)
            if line.strip().startswith('#'):
                # 检查是否包含中文
                if re.search(r'[\u4e00-\u9fff]', line):
                    removed_count += 1
                    continue  # 跳过这行
            
            # 删除行尾中文注释 (# 后面的中文)
            if '#' in line:
                parts = line.split('#', 1)
                if len(parts) == 2 and re.search(r'[\u4e00-\u9fff]', parts[1]):
                    # 保留代码部分，删除注释
                    line = parts[0].rstrip()
                    removed_count += 1
        
        # 处理 Shell 文件
        elif file_ext == '.sh':
            # 删除整行中文注释 (# 开头)
            if line.strip().startswith('#'):
                if re.search(r'[\u4e00-\u9fff]', line):
                    removed_count += 1
                    continue  # 跳过这行
            
            # 删除行尾中文注释
            if '#' in line:
                parts = line.split('#', 1)
                if len(parts) == 2 and re.search(r'[\u4e00-\u9fff]', parts[1]):
                    line = parts[0].rstrip()
                    removed_count += 1
        
        new_lines.append(line)
    
    return '\n'.join(new_lines), removed_count

def process_file(file_path):
    """处理单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        file_ext = file_path.suffix
        new_content, removed = remove_chinese_comments(content, file_ext)
        
        if removed > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return removed
        
        return 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def main():
    base_dir = Path(__file__).parent
    total_removed = 0
    processed_files = []
    
    print("=" * 70)
    print("删除中文注释")
    print("=" * 70)
    print()
    
    # 扫描所有文件
    for root, dirs, files in os.walk(base_dir):
        # 排除目录
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            file_path = Path(root) / file
            
            # 只处理特定扩展名
            if file_path.suffix not in FILE_EXTENSIONS:
                continue
            
            # 跳过脚本本身
            if file_path.name == 'remove_chinese_comments.py':
                continue
            
            rel_path = file_path.relative_to(base_dir)
            removed = process_file(file_path)
            
            if removed > 0:
                processed_files.append((str(rel_path), removed))
                total_removed += removed
    
    # 报告
    print(f"处理完成！")
    print(f"  处理文件数: {len(processed_files)}")
    print(f"  删除注释行数: {total_removed}")
    print()
    
    if processed_files:
        print("处理的文件:")
        for file_path, count in processed_files[:20]:
            print(f"  {file_path}: 删除 {count} 行")
        if len(processed_files) > 20:
            print(f"  ... 还有 {len(processed_files) - 20} 个文件")

if __name__ == '__main__':
    main()

