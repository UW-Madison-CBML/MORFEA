#!/usr/bin/env python3
"""
检查代码库中的人名和中文内容
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# 要检查的人名（不区分大小写）
PERSONAL_NAMES = [
    'raffael', 'grnho', 'aadhitya', 'jens', 'jlundsgaard', 
    'bhaskar', 'rho9', 'macbook', 'raffaeldemacbook'
]

# 要检查的路径模式（可能包含个人信息）
PATH_PATTERNS = [
    r'/Users/grnho',
    r'/Desktop/Project',
    r'grnho@',
    r'MacBook',
    r'Raffael'
]

# 要检查的文件类型
FILE_EXTENSIONS = ['.py', '.md', '.txt', '.sh', '.sub', '.yaml', '.yml', '.json']

# 排除的目录
EXCLUDE_DIRS = {
    '.git', '__pycache__', 'checkpoints', 'logs', 'results', 
    'reconstructions', 'tphate_results', 'curvature_results',
    'model_latents', 'latents_epoch50', 'scripts'
}

def contains_chinese(text):
    """检查是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def contains_personal_name(text, names):
    """检查是否包含人名"""
    text_lower = text.lower()
    found = []
    for name in names:
        if name.lower() in text_lower:
            found.append(name)
    return found

def contains_path_pattern(text, patterns):
    """检查是否包含路径模式"""
    found = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(pattern)
    return found

def scan_file(file_path):
    """扫描单个文件"""
    issues = {
        'chinese': [],
        'names': [],
        'paths': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            # 检查中文
            if contains_chinese(line):
                issues['chinese'].append((line_num, line.strip()))
            
            # 检查人名
            names = contains_personal_name(line, PERSONAL_NAMES)
            if names:
                issues['names'].append((line_num, line.strip(), names))
            
            # 检查路径
            paths = contains_path_pattern(line, PATH_PATTERNS)
            if paths:
                issues['paths'].append((line_num, line.strip(), paths))
    
    except Exception as e:
        return None
    
    # 只返回有问题的文件
    if any(issues.values()):
        return issues
    return None

def main():
    base_dir = Path(__file__).parent
    results = defaultdict(dict)
    
    print("=" * 70)
    print("检查代码库中的人名和中文内容")
    print("=" * 70)
    print()
    
    # 扫描所有文件
    for root, dirs, files in os.walk(base_dir):
        # 排除目录
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            file_path = Path(root) / file
            
            # 只检查特定扩展名
            if file_path.suffix not in FILE_EXTENSIONS:
                continue
            
            # 跳过脚本本身
            if file_path.name == 'check_personal_info.py':
                continue
            
            rel_path = file_path.relative_to(base_dir)
            issues = scan_file(file_path)
            
            if issues:
                results[str(rel_path)] = issues
    
    # 生成报告
    print(f"扫描完成！找到 {len(results)} 个文件包含个人信息或中文\n")
    
    # 按类别分组
    chinese_files = []
    name_files = []
    path_files = []
    
    for file_path, issues in results.items():
        if issues['chinese']:
            chinese_files.append((file_path, issues['chinese']))
        if issues['names']:
            name_files.append((file_path, issues['names']))
        if issues['paths']:
            path_files.append((file_path, issues['paths']))
    
    # 报告中文
    if chinese_files:
        print("=" * 70)
        print(f"包含中文的文件 ({len(chinese_files)} 个):")
        print("=" * 70)
        for file_path, lines in chinese_files[:20]:  # 只显示前20个
            print(f"\n{file_path}:")
            for line_num, line in lines[:3]:  # 每文件最多3行
                print(f"  行 {line_num}: {line[:80]}")
        if len(chinese_files) > 20:
            print(f"\n... 还有 {len(chinese_files) - 20} 个文件")
    
    # 报告人名
    if name_files:
        print("\n" + "=" * 70)
        print(f"包含人名的文件 ({len(name_files)} 个):")
        print("=" * 70)
        for file_path, lines in name_files[:20]:
            print(f"\n{file_path}:")
            for line_num, line, names in lines[:3]:
                print(f"  行 {line_num} (找到: {', '.join(names)}): {line[:80]}")
        if len(name_files) > 20:
            print(f"\n... 还有 {len(name_files) - 20} 个文件")
    
    # 报告路径
    if path_files:
        print("\n" + "=" * 70)
        print(f"包含个人路径的文件 ({len(path_files)} 个):")
        print("=" * 70)
        for file_path, lines in path_files[:20]:
            print(f"\n{file_path}:")
            for line_num, line, patterns in lines[:3]:
                print(f"  行 {line_num} (找到: {', '.join(patterns)}): {line[:80]}")
        if len(path_files) > 20:
            print(f"\n... 还有 {len(path_files) - 20} 个文件")
    
    # 保存详细报告
    report_file = base_dir / 'PERSONAL_INFO_REPORT.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("详细报告：包含人名和中文的文件\n")
        f.write("=" * 70 + "\n\n")
        
        for file_path, issues in sorted(results.items()):
            f.write(f"\n{'='*70}\n")
            f.write(f"文件: {file_path}\n")
            f.write(f"{'='*70}\n")
            
            if issues['chinese']:
                f.write("\n包含中文的行:\n")
                for line_num, line in issues['chinese']:
                    f.write(f"  行 {line_num}: {line}\n")
            
            if issues['names']:
                f.write("\n包含人名的行:\n")
                for line_num, line, names in issues['names']:
                    f.write(f"  行 {line_num} (找到: {', '.join(names)}): {line}\n")
            
            if issues['paths']:
                f.write("\n包含个人路径的行:\n")
                for line_num, line, patterns in issues['paths']:
                    f.write(f"  行 {line_num} (找到: {', '.join(patterns)}): {line}\n")
    
    print(f"\n\n详细报告已保存到: {report_file}")
    print(f"\n总计:")
    print(f"  包含中文的文件: {len(chinese_files)}")
    print(f"  包含人名的文件: {len(name_files)}")
    print(f"  包含个人路径的文件: {len(path_files)}")

if __name__ == '__main__':
    main()

