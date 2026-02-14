#!/usr/bin/env python3
"""
删除所有包含中文的 .md 文档文件（只保留核心代码）
"""

import os
from pathlib import Path

EXCLUDE_DIRS = {
    '.git', '__pycache__', 'checkpoints', 'logs', 'results', 
    'reconstructions', 'tphate_results', 'curvature_results',
    'model_latents', 'latents_epoch50', 'scripts'
}

def has_chinese(text):
    """检查是否包含中文"""
    import re
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def main():
    base_dir = Path(__file__).parent
    deleted_files = []
    
    print("=" * 70)
    print("删除包含中文的文档文件")
    print("=" * 70)
    print()
    
    # 保留的重要文档
    KEEP_FILES = {
        'README.md',
        '00_START_HERE.md',  # 如果需要可以保留
    }
    
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if not file.endswith('.md'):
                continue
            
            file_path = Path(root) / file
            
            # 跳过脚本本身和保留的文件
            if file_path.name in ['remove_chinese_docs.py', 'remove_all_chinese.py'] or file_path.name in KEEP_FILES:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if has_chinese(content):
                    file_path.unlink()
                    deleted_files.append(str(file_path.relative_to(base_dir)))
            except:
                pass
    
    print(f"删除文件数: {len(deleted_files)}")
    print()
    if deleted_files:
        print("删除的文件（前30个）:")
        for f in deleted_files[:30]:
            print(f"  {f}")
        if len(deleted_files) > 30:
            print(f"  ... 还有 {len(deleted_files) - 30} 个文件")

if __name__ == '__main__':
    main()

