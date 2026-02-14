#!/usr/bin/env python3
"""
删除所有文件中的中文内容（包括注释、字符串、文档字符串）
只保留核心代码文件，删除所有中文
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

# 要删除的文件（包含大量中文的文档和脚本）
FILES_TO_DELETE = [
    # 检查脚本（很多中文）
    'check_personal_info.py',
    'remove_chinese_comments.py',
    'remove_all_chinese.py',
    'PERSONAL_INFO_REPORT.txt',
    'CLEANUP_SUMMARY.md',
    'CLEANUP_COMPLETE.md',
    'FIXED_FILES_LIST.md',
]

def has_chinese(text):
    """检查是否包含中文"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def clean_python_file(content):
    """清理 Python 文件中的中文"""
    lines = content.split('\n')
    new_lines = []
    removed_count = 0
    
    for line in lines:
        original_line = line
        
        # 删除整行中文注释
        if line.strip().startswith('#'):
            if has_chinese(line):
                removed_count += 1
                continue
        
        # 删除行尾中文注释
        if '#' in line:
            parts = line.split('#', 1)
            if len(parts) == 2 and has_chinese(parts[1]):
                line = parts[0].rstrip()
                removed_count += 1
        
        # 删除中文文档字符串（但保留函数定义）
        if '"""' in line or "'''" in line:
            if has_chinese(line):
                # 如果是单独的文档字符串行，删除
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    removed_count += 1
                    continue
                # 如果是行尾，只保留代码部分
                if '"""' in line:
                    line = line.split('"""')[0].rstrip()
                elif "'''" in line:
                    line = line.split("'''")[0].rstrip()
                removed_count += 1
        
        # 删除包含中文的 print 语句（但保留代码逻辑）
        if 'print(' in line and has_chinese(line):
            # 只删除 print 语句，保留其他代码
            removed_count += 1
            continue
        
        new_lines.append(line)
    
    return '\n'.join(new_lines), removed_count

def clean_shell_file(content):
    """清理 Shell 文件中的中文"""
    lines = content.split('\n')
    new_lines = []
    removed_count = 0
    
    for line in lines:
        # 删除整行中文注释
        if line.strip().startswith('#'):
            if has_chinese(line):
                removed_count += 1
                continue
        
        # 删除行尾中文注释
        if '#' in line:
            parts = line.split('#', 1)
            if len(parts) == 2 and has_chinese(parts[1]):
                line = parts[0].rstrip()
                removed_count += 1
        
        # 删除包含中文的 echo 语句
        if 'echo' in line and has_chinese(line):
            removed_count += 1
            continue
        
        new_lines.append(line)
    
    return '\n'.join(new_lines), removed_count

def process_file(file_path):
    """处理单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        file_ext = file_path.suffix
        
        if file_ext == '.py':
            new_content, removed = clean_python_file(content)
        elif file_ext == '.sh':
            new_content, removed = clean_shell_file(content)
        else:
            return 0
        
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
    deleted_files = []
    
    print("=" * 70)
    print("删除所有中文内容")
    print("=" * 70)
    print()
    
    # 删除不需要的文件
    print("删除不需要的文件...")
    for filename in FILES_TO_DELETE:
        file_path = base_dir / filename
        if file_path.exists():
            file_path.unlink()
            deleted_files.append(filename)
            print(f"  ✓ 删除: {filename}")
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
            if file_path.name in ['remove_chinese_comments.py', 'remove_all_chinese.py', 'check_personal_info.py']:
                continue
            
            rel_path = file_path.relative_to(base_dir)
            removed = process_file(file_path)
            
            if removed > 0:
                processed_files.append((str(rel_path), removed))
                total_removed += removed
    
    # 报告
    print(f"处理完成！")
    print(f"  删除文件数: {len(deleted_files)}")
    print(f"  处理文件数: {len(processed_files)}")
    print(f"  删除中文行数: {total_removed}")
    print()
    
    if processed_files:
        print("处理的文件（前20个）:")
        for file_path, count in processed_files[:20]:
            print(f"  {file_path}: 删除 {count} 行")
        if len(processed_files) > 20:
            print(f"  ... 还有 {len(processed_files) - 20} 个文件")

if __name__ == '__main__':
    main()

